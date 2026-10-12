#!/usr/bin/env python3
"""
Build the five CSV files read by the BERT4Rec code (config.py) from the raw sources.

Inputs
  --ratings  MovieLens-25M ratings.csv          (userId, movieId, rating, timestamp)
  --links2   output of fetch_credits.py         (movieId, imdbId, tmdbId, director, lead)
  --tmdb     Kaggle "TMDB Movies Dataset 2023"  (id, title, runtime, overview, genres, ...)

Outputs (written to --out)
  Final_Ratings.csv     userId, movieId, rating, timestamp
  model_genre.csv       movieId, title, genres
  model_id.csv          movieId, title
  model_metadata.csv    movieId, title, director, lead, runtime
  model_plot.csv        movieId, title, overview

Rules (as used for the paper)
  1. Join the links2 table (which carries director and lead) with the Kaggle TMDB table on the TMDB id.
  2. Drop every movie with a null in any of: title, genres, overview, runtime, director, lead.
  3. Keep only movies present in both the ratings and the cleaned movie table.
  4. Drop ratings of removed movies; users whose ratings all disappear are dropped implicitly.

Usage
  python build_dataset.py --ratings ml-25m/ratings.csv --links2 link2.csv \
      --tmdb TMDB_movie_dataset_v11.csv --out data
"""
import argparse
import os

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ratings", required=True)
    p.add_argument("--links2", required=True)
    p.add_argument("--tmdb", required=True)
    p.add_argument("--out", default="data")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ratings = pd.read_csv(args.ratings, usecols=["userId", "movieId", "rating", "timestamp"])
    links = pd.read_csv(args.links2)
    kaggle = pd.read_csv(args.tmdb, usecols=["id", "title", "runtime", "overview", "genres"])

    links = links.dropna(subset=["tmdbId"]).copy()
    links["tmdbId"] = links["tmdbId"].astype("int64")
    links = links.drop_duplicates("movieId")

    kaggle = kaggle.rename(columns={"id": "tmdbId"}).dropna(subset=["tmdbId"]).copy()
    kaggle["tmdbId"] = kaggle["tmdbId"].astype("int64")
    kaggle = kaggle.drop_duplicates("tmdbId")

    movies = links[["movieId", "tmdbId", "director", "lead"]].merge(kaggle, on="tmdbId", how="inner")
    need = ["title", "genres", "overview", "runtime", "director", "lead"]
    movies = movies.dropna(subset=need)

    common = set(movies["movieId"]) & set(ratings["movieId"])
    movies = movies[movies["movieId"].isin(common)].sort_values("movieId")
    ratings = ratings[ratings["movieId"].isin(common)]

    ratings.to_csv(os.path.join(args.out, "Final_Ratings.csv"), index=False)
    movies[["movieId", "title", "genres"]].to_csv(os.path.join(args.out, "model_genre.csv"), index=False)
    movies[["movieId", "title"]].to_csv(os.path.join(args.out, "model_id.csv"), index=False)
    movies[["movieId", "title", "director", "lead", "runtime"]].to_csv(
        os.path.join(args.out, "model_metadata.csv"), index=False)
    movies[["movieId", "title", "overview"]].to_csv(os.path.join(args.out, "model_plot.csv"), index=False)

    assert set(ratings["movieId"]) == set(movies["movieId"]), "ratings and item tables disagree"
    print(f"users: {ratings['userId'].nunique():,}")
    print(f"items: {movies['movieId'].nunique():,}")
    print(f"interactions: {len(ratings):,}")
    print(f"wrote 5 files to {args.out}")


if __name__ == "__main__":
    main()
