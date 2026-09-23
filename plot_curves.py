import os, glob
import pandas as pd
import matplotlib.pyplot as plt
import config as cfg


def main():
    out = os.path.join(cfg.OUTPUT_DIR, "curves")
    os.makedirs(out, exist_ok=True)
    seed = cfg.SEEDS[0]
    for variant in cfg.VARIANTS:
        pattern = os.path.join(cfg.OUTPUT_DIR, "checkpoints",
                               f"{variant}_sparsity1.00_seed{seed}_history.csv")
        if not os.path.exists(pattern):
            print("Missing:", pattern); continue
        df = pd.read_csv(pattern)
        fig, ax1 = plt.subplots(figsize=(7,4.5))
        ax1.plot(df["epoch"], df["train_loss"], label="Train loss")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Training loss")
        ax2 = ax1.twinx()
        ev = df.dropna(subset=["val_recall"])
        ax2.plot(ev["epoch"], ev["val_recall"], marker="o", label="Val Recall@10")
        ax2.set_ylabel("Validation Recall@10")
        ax1.set_title(f"{variant} | sparsity=1.0 | seed={seed}")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{variant}_sparsity1.00_seed{seed}.png"), dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
