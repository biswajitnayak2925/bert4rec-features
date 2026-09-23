import os
import math
import pandas as pd
import numpy as np

import config as cfg


def _read_any(path):

    if os.path.exists(path):
        if path.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(path)
        return pd.read_csv(path)
    stem, _ = os.path.splitext(path)
    for ext in (".csv", ".xlsx", ".xls"):
        alt = stem + ext
        if os.path.exists(alt):
            return _read_any(alt)
    raise FileNotFoundError(f"Could not find {path} (or a .csv/.xlsx/.xls sibling)")


def load_raw_tables():
    ratings = _read_any(cfg.RATINGS_PATH)
    ids = _read_any(cfg.ID_PATH)
    genre = _read_any(cfg.GENRE_PATH)
    metadata = _read_any(cfg.METADATA_PATH)
    plot = _read_any(cfg.PLOT_PATH)

  
    for df in (ratings, ids, genre, metadata, plot):
        df.columns = [c.strip() for c in df.columns]

    required_ratings_cols = {"userId", "movieId", "timestamp"}
    missing = required_ratings_cols - set(ratings.columns)
    if missing:
        raise ValueError(f"Final_Ratings is missing columns: {missing}. "
                          f"Found columns: {list(ratings.columns)}")

    ratings = ratings.sort_values(["userId", "timestamp"]).reset_index(drop=True)
    return ratings, ids, genre, metadata, plot


def build_item_vocab(ratings, ids_df):
    
    interacted = sorted(ratings["movieId"].unique().tolist())
    item2idx = {mid: i + 1 for i, mid in enumerate(interacted)}
    idx2item = {i: mid for mid, i in item2idx.items()}
    num_items = len(interacted)
    mask_token = num_items + 1
    pad_token = 0
    return item2idx, idx2item, num_items, mask_token, pad_token


def build_user_sequences(ratings, item2idx):
    ratings = ratings.copy()
    ratings["item_idx"] = ratings["movieId"].map(item2idx)
    ratings = ratings.dropna(subset=["item_idx"])
    ratings["item_idx"] = ratings["item_idx"].astype(int)

    sequences = {}
    for uid, g in ratings.sort_values(["userId", "timestamp"]).groupby("userId"):
        seq = g["item_idx"].tolist()
        if len(seq) < cfg.MIN_USER_INTERACTIONS:
            continue
        max_allowed = getattr(cfg, "MAX_USER_INTERACTIONS", None)
        if max_allowed is not None and len(seq) > max_allowed:
            continue
        sequences[uid] = seq
    return sequences


def leave_one_out_split(sequences):
    
    split = {}
    for uid, seq in sequences.items():
        if len(seq) < 3:
            continue 
        train = seq[:-2]
        val = seq[-2]
        test = seq[-1]
        if len(train) == 0:
            continue
        split[uid] = {"train": train, "val": val, "test": test}
    return split


def apply_sparsity(split, fraction, keep="recent"):
    new_split = {}
    rng = None
    for uid, d in split.items():
        train = d["train"]
        window = train[-cfg.MAX_SEQ_LEN:]  
        n_keep = max(1, math.ceil(len(window) * fraction))
        if keep == "recent":
            new_train = window[-n_keep:]
        elif keep == "random":
            if rng is None:
                rng = np.random.RandomState(cfg.SEEDS[0])
            idx = sorted(rng.choice(len(window), size=n_keep, replace=False))
            new_train = [window[i] for i in idx]
        else:
            raise ValueError(keep)
        new_split[uid] = {"train": new_train, "val": d["val"], "test": d["test"]}
    return new_split


def max_seq_len_for_sparsity(fraction):
    
    train_budget = max(1, math.ceil(cfg.MAX_SEQ_LEN * fraction))
    return train_budget + 2   


def build_item_feature_table(ids_df, genre_df, metadata_df, plot_df, item2idx):
    
    ids_df = ids_df[["movieId", "title"]].drop_duplicates("movieId")
    genre_df = genre_df[["movieId", "genres"]].drop_duplicates("movieId")
    metadata_cols = [c for c in ["movieId", "director", "lead", "runtime"] if c in metadata_df.columns]
    metadata_df = metadata_df[metadata_cols].drop_duplicates("movieId")
    plot_df = plot_df[["movieId", "overview"]].drop_duplicates("movieId")

    merged = ids_df.merge(genre_df, on="movieId", how="left") \
                    .merge(metadata_df, on="movieId", how="left") \
                    .merge(plot_df, on="movieId", how="left")

    merged["item_idx"] = merged["movieId"].map(item2idx)
    merged = merged.dropna(subset=["item_idx"])
    merged["item_idx"] = merged["item_idx"].astype(int)
    merged = merged.set_index("item_idx").sort_index()
    return merged


if __name__ == "__main__":
    
    ratings, ids, genre, metadata, plot = load_raw_tables()
    item2idx, idx2item, num_items, mask_token, pad_token = build_item_vocab(ratings, ids)
    print(f"num_items={num_items}, mask_token={mask_token}, pad_token={pad_token}")

    sequences = build_user_sequences(ratings, item2idx)
    print(f"num users with >= {cfg.MIN_USER_INTERACTIONS} interactions: {len(sequences)}")

    split = leave_one_out_split(sequences)
    print(f"num users after leave-one-out split: {len(split)}")
    example_uid = next(iter(split))
    print("example split:", split[example_uid])

    for frac in cfg.SPARSITY_LEVELS:
        sp = apply_sparsity(split, frac)
        avg_len = np.mean([len(d["train"]) for d in sp.values()])
        print(f"sparsity={frac:.2f} -> avg train length = {avg_len:.2f}")

    feat_table = build_item_feature_table(ids, genre, metadata, plot, item2idx)
    print(feat_table.head())
    print("feature table shape:", feat_table.shape)
