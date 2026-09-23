import os, time, shutil
import numpy as np
import pandas as pd
import config as cfg
import data_utils as du
import features as feat
from train import train_one_config


def _already_done(results_df, variant, sparsity, seed, tol=1e-9):
    if results_df is None or len(results_df) == 0:
        return False
    match = results_df[
        (results_df["variant"] == variant) &
        (np.isclose(results_df["sparsity"], sparsity, atol=tol)) &
        (results_df["seed"] == seed)
    ]
    return len(match) > 0


def main():
    results_path = os.path.join(cfg.OUTPUT_DIR, "results_by_seed.csv")
    rows = []
    results_df = None
    if os.path.exists(cfg.OUTPUT_DIR):
        if os.path.exists(results_path):
            results_df = pd.read_csv(results_path)
            rows = results_df.to_dict("records")
            print(f"Found existing {results_path} with {len(rows)} completed run(s) -- resuming this experiment.")
        else:
            raise RuntimeError(
                f"Output directory already exists but has no results_by_seed.csv: {cfg.OUTPUT_DIR}\n"
                "This looks like an unrelated/foreign directory, not a previous run of this same "
                "experiment -- delete/rename it manually before proceeding, or choose a new OUTPUT_DIR.")
    else:
        os.makedirs(cfg.OUTPUT_DIR)
    os.makedirs(os.path.join(cfg.OUTPUT_DIR, "checkpoints"), exist_ok=True)

    ratings, ids, genre, metadata, plot = du.load_raw_tables()
    item2idx, idx2item, num_items, mask_token, pad_token = du.build_item_vocab(ratings, ids)
    sequences = du.build_user_sequences(ratings, item2idx)
    full_split = du.leave_one_out_split(sequences)
    feat_table = du.build_item_feature_table(ids, genre, metadata, plot, item2idx)
    genre_mat, _ = feat.encode_genres(feat_table, num_items)
    meta = feat.encode_metadata(feat_table, num_items)
    plot_mat = feat.encode_plot(feat_table, num_items)

    print(f"Users={len(full_split)}, items={num_items}")
    print(f"Variants={cfg.VARIANTS}, sparsities={cfg.SPARSITY_LEVELS}, seeds={cfg.SEEDS}")
    plot_encoder_info = getattr(feat, "LAST_PLOT_ENCODER_INFO", "unknown")
    print(f"Plot encoder actually used: {plot_encoder_info}")
    with open(os.path.join(cfg.OUTPUT_DIR, "experiment_metadata.txt"), "w") as f:
        f.write("Publication experiment\n")
        f.write(f"epochs={cfg.NUM_EPOCHS}\npatience_checks={cfg.PATIENCE}\n")
        f.write(f"eval_every={cfg.EVAL_EVERY}\nseeds={cfg.SEEDS}\n")
        f.write("Primary metrics: full-catalog Recall@10 and NDCG@10\n")
        f.write("Validation selection: sampled Recall@10\n")
        f.write(f"Plot encoder actually used: {plot_encoder_info}\n")

    for sparsity in cfg.SPARSITY_LEVELS:
        split = du.apply_sparsity(full_split, sparsity)
        avg_train_len = sum(len(d["train"]) for d in split.values()) / len(split)
        max_seq = du.max_seq_len_for_sparsity(sparsity)
        for variant in cfg.VARIANTS:
            for seed in cfg.SEEDS:
                if _already_done(results_df, variant, sparsity, seed):
                    print(f"-- skipping {variant} | sparsity={sparsity:.2f} | seed={seed} "
                          f"(already in results_by_seed.csv) --")
                    continue

                print(f"\n=== {variant} | sparsity={sparsity:.2f} | seed={seed} ===")
                t0 = time.time()
                ckpt = os.path.join(cfg.OUTPUT_DIR, "checkpoints",
                                    f"{variant}_sparsity{sparsity:.2f}_seed{seed}.pt")
                try:
                    result = train_one_config(variant, split, num_items, mask_token, pad_token,
                                              genre_mat, meta, plot_mat, verbose=True,
                                              checkpoint_path=ckpt, resume=True,
                                              max_seq_len=max_seq, seed=seed)
                except Exception as e:
                    print(f"!! {variant} | sparsity={sparsity:.2f} | seed={seed} FAILED: "
                          f"{type(e).__name__}: {e}")
                    print("!! Skipping for now -- re-run this script later to retry it.")
                    continue

                result.update(sparsity=sparsity, avg_train_len=avg_train_len,
                              elapsed_sec=round(time.time()-t0, 1))
                rows.append(result)
                results_df = pd.DataFrame(rows)
                results_df.to_csv(results_path, index=False)

    raw = pd.DataFrame(rows)
    raw.to_csv(results_path, index=False)


    metric_cols = [f"test_recall@{cfg.TOP_K}_full", f"test_ndcg@{cfg.TOP_K}_full",
                   f"test_recall@{cfg.TOP_K}_sampled", f"test_ndcg@{cfg.TOP_K}_sampled"]
    agg = raw.groupby(["variant", "sparsity"], as_index=False).agg(
        **{f"{m}_mean": (m, "mean") for m in metric_cols},
        **{f"{m}_std": (m, "std") for m in metric_cols},
        best_epoch_mean=("best_epoch", "mean"),
        best_epoch_std=("best_epoch", "std"),
        total_params=("total_params", "first"),
        trainable_params=("trainable_params", "first"))
    agg.to_csv(os.path.join(cfg.OUTPUT_DIR, "results_summary.csv"), index=False)

    r = f"test_recall@{cfg.TOP_K}_full"; n = f"test_ndcg@{cfg.TOP_K}_full"
    summary = raw.groupby(["variant", "sparsity"])[[r, n]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(os.path.join(cfg.OUTPUT_DIR, "paper_table_full_catalog.csv"), index=False)
    print("\nDONE")
    print(f"Raw results: {cfg.OUTPUT_DIR}/results_by_seed.csv")
    print(f"Mean±SD summary: {cfg.OUTPUT_DIR}/results_summary.csv")


if __name__ == "__main__":
    main()
