import os
import time
import requests
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------- CONFIG ----------------
API_KEY = ""        # v3 API key (32-char string) — leave as-is if unused
READ_ACCESS_TOKEN = ""                # v4 Read Access Token (long eyJ... string) — set this if you have it instead
INPUT_FILE = r"C:\Users\KIIT\OneDrive\Desktop\recommender\links.csv"
OUTPUT_FILE = r"C:\Users\KIIT\OneDrive\Desktop\recommender\link2.csv"
ID_COLUMN = "tmdbId"          # column in links.csv holding the TMDB movie id
MAX_WORKERS = 20               # parallel requests — TMDB handles this fine, raise/lower if you hit errors
CHECKPOINT_EVERY = 2000        # save progress every N rows in case it crashes/interrupts partway
# -----------------------------------------

BASE_URL = "https://api.themoviedb.org/3"


def get_credits(movie_id, api_key):
    url = f"{BASE_URL}/movie/{int(movie_id)}/credits"
    if READ_ACCESS_TOKEN:
        headers = {"Authorization": f"Bearer {READ_ACCESS_TOKEN}"}
        resp = requests.get(url, headers=headers, timeout=15)
    else:
        resp = requests.get(url, params={"api_key": api_key}, timeout=15)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def extract_fields(data):
    if data is None:
        return {"director": None, "lead": None}

    crew = data.get("crew", [])
    cast = data.get("cast", [])

    director = next((c["name"] for c in crew if c.get("job") == "Director"), None)
    lead = ", ".join(c["name"] for c in cast[:2]) if cast else None

    return {"director": director, "lead": lead}


def fetch_one(idx, movie_id):
    if pd.isna(movie_id):
        return idx, extract_fields(None)
    try:
        data = get_credits(movie_id, API_KEY)
        return idx, extract_fields(data)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching id {movie_id}: {e}")
        return idx, extract_fields(None)


def main():
    df = pd.read_csv(INPUT_FILE).reset_index(drop=True)

    if ID_COLUMN not in df.columns:
        raise ValueError(f"Column '{ID_COLUMN}' not found. Columns available: {list(df.columns)}")

    # Resume support: if OUTPUT_FILE already exists (from a previous run),
    # load its director/lead values and only fetch rows still missing.
    if os.path.exists(OUTPUT_FILE):
        prev = pd.read_csv(OUTPUT_FILE)
        if "director" in prev.columns and "lead" in prev.columns and len(prev) == len(df):
            df["director"] = prev["director"]
            df["lead"] = prev["lead"]
            print("Found existing output file — resuming from checkpoint.")
        else:
            df["director"] = None
            df["lead"] = None
    else:
        df["director"] = None
        df["lead"] = None

    # A row still needs fetching if director AND lead are both empty.
    # (A movie legitimately missing a director but having cast, or vice versa,
    # is still treated as "done" once fetched once — see fetch_one/extract_fields.)
    pending_idx = df.index[df["director"].isna() & df["lead"].isna()].tolist()
    already_done = len(df) - len(pending_idx)
    print(f"{already_done} rows already fetched, {len(pending_idx)} remaining.")

    if not pending_idx:
        print("Nothing left to fetch.")
        return

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_one, idx, df.at[idx, ID_COLUMN]): idx
            for idx in pending_idx
        }

        done_count = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching TMDB credits"):
            idx, fields = future.result()
            df.at[idx, "director"] = fields["director"]
            df.at[idx, "lead"] = fields["lead"]

            done_count += 1
            if done_count % CHECKPOINT_EVERY == 0:
                df.to_csv(OUTPUT_FILE, index=False)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved enriched CSV to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()