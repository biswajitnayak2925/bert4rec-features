
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

METRICS = {"Recall@10": "test_recall@10_full", "NDCG@10": "test_ndcg@10_full"}
BASE = "M1_id"


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--out", default=None, help="optional folder for CSV outputs")
    args = ap.parse_args()

    df = pd.concat([pd.read_csv(p) for p in args.results], ignore_index=True)
    df["sparsity"] = df["sparsity"].round(4)
    df = df.drop_duplicates(["variant", "sparsity", "seed"], keep="last")
    if args.out:
        os.makedirs(args.out, exist_ok=True)

    for sp in sorted(df["sparsity"].unique(), reverse=True):
        d = df[df["sparsity"] == sp]
        seeds = sorted(d["seed"].unique())
        print(f"\n{'=' * 78}\nRetention {sp:g}  |  seeds {seeds}\n{'=' * 78}")
        for name, col in METRICS.items():
            piv = d.pivot(index="seed", columns="variant", values=col)
            variants = [BASE] + [v for v in piv.columns if v != BASE]
            summ = pd.DataFrame({"mean": piv[variants].mean(), "sd": piv[variants].std(ddof=1)})
            summ["rel_vs_M1_%"] = (summ["mean"] / summ.loc[BASE, "mean"] - 1) * 100
            print(f"\n{name} (full catalogue)")
            print(summ.round(4).to_string())
            print("winner per seed:", piv.idxmax(axis=1).to_dict())
            if args.out:
                summ.to_csv(os.path.join(args.out, f"summary_{name.replace('@', '')}_ret{sp:g}.csv"))

            if len(seeds) >= 2 and BASE in piv.columns:
                rows = []
                for v in variants[1:]:
                    diff = piv[v] - piv[BASE]
                    t, p = stats.ttest_rel(piv[v], piv[BASE])
                    rows.append({"variant": v, "mean_diff": diff.mean(), "sd_diff": diff.std(ddof=1),
                                 "seeds_better": f"{int((diff > 0).sum())}/{len(diff)}", "t": t, "p": p})
                pt = pd.DataFrame(rows)
                pt["p_holm"] = holm(pt["p"].to_numpy())
                print(f"\nPaired t-tests vs {BASE} (df = {len(seeds) - 1})")
                print(pt.round(4).to_string(index=False))
                if args.out:
                    pt.to_csv(os.path.join(args.out, f"paired_{name.replace('@', '')}_ret{sp:g}.csv"), index=False)
            if len(seeds) >= 3 and piv.notna().all().all():
                chi, p = stats.friedmanchisquare(*[piv[c] for c in piv.columns])
                ranks = piv.rank(axis=1, ascending=False).mean().round(2).to_dict()
                print(f"Friedman chi2 = {chi:.2f}, p = {p:.3f}, mean ranks {ranks}")


if __name__ == "__main__":
    main()
