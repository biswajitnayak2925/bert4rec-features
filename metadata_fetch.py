
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm

BASE_URL = "https://api.themoviedb.org/3"


def get_credits(movie_id, auth):
    url = f"{BASE_URL}/movie/{int(movie_id)}/credits"
    headers, params = {}, {}
    if auth["token"]:
        headers["Authorization"] = f"Bearer {auth['token']}"
    else:
        params["api_key"] = auth["key"]
    for attempt in range(5):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=15)
        except requests.exceptions.RequestException:
            time.sleep(2 ** attempt)
            continue
        if resp.status_code == 404:
            return None
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(float(resp.headers.get("Retry-After", 2 ** attempt)))
            continue
        resp.raise_for_status()
        return resp.json()
    return None


def extract_fields(data):
    if data is None:
        return {"director": None, "lead": None}
    crew = data.get("crew", [])
    cast = data.get("cast", [])
    director = next((c["name"] for c in crew if c.get("job") == "Director"), None)
    lead = ", ".join(c["name"] for c in cast[:2]) if cast else None
    return {"director": director, "lead": lead}


def fetch_one(idx, movie_id, auth):
    if pd.isna(movie_id):
        return idx, extract_fields(None)
    return idx, extract_fields(get_credits(movie_id, auth))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--links", required=True)
    p.add_argument("--out", default="link2.csv")
    p.add_argument("--id-column", default="tmdbId")
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--checkpoint", type=int, default=2000)
    args = p.parse_args()

    auth = {"token": os.environ.get("TMDB_BEARER_TOKEN", ""), "key": os.environ.get("TMDB_API_KEY", "")}
    if not auth["token"] and not auth["key"]:
        sys.exit("Set TMDB_API_KEY or TMDB_BEARER_TOKEN")

    df = pd.read_csv(args.links).reset_index(drop=True)
    if args.id_column not in df.columns:
        raise ValueError(f"Column '{args.id_column}' not found. Columns: {list(df.columns)}")

    df["director"], df["lead"] = None, None
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out)
        if {"director", "lead"} <= set(prev.columns) and len(prev) == len(df):
            df["director"], df["lead"] = prev["director"], prev["lead"]
            print("Found existing output file, resuming.")

    pending = df.index[df["director"].isna() & df["lead"].isna()].tolist()
    print(f"{len(df) - len(pending)} rows already fetched, {len(pending)} remaining.")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(fetch_one, i, df.at[i, args.id_column], auth) for i in pending]
        for n, fut in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Fetching credits"), 1):
            idx, fields = fut.result()
            df.at[idx, "director"], df.at[idx, "lead"] = fields["director"], fields["lead"]
            if n % args.checkpoint == 0:
                df.to_csv(args.out, index=False)
    df.to_csv(args.out, index=False)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
