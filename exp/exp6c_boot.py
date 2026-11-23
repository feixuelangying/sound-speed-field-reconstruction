# -*- coding: utf-8 -*-
"""EXP-6c 补充：不同 k 下 test 覆盖率的场景级 bootstrap CI（聚类稳健）。

目的：J2 的二项容差带 [0.9487,0.9513] 用 n=109367 逐点算，隐含"点独立"；
但同场景内点强相关，有效样本量远小于逐点数。用场景级 bootstrap 给出
覆盖率的不确定度，判断"是否与名义 0.95 统计一致"。
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from exp import config as C

OUT = C.result_dir("exp6c")
RNG = np.random.default_rng(C.SEED)
NB = 2000


def boot_cover(df, k, n_boot=NB):
    s = df["s"].values
    sid = df["sid"].values
    covered = (s <= k)
    g = {u: covered[sid == u] for u in np.unique(sid)}
    keys = list(g.keys())
    covs = []
    for _ in range(n_boot):
        pick = RNG.choice(keys, len(keys), replace=True)
        covs.append(np.concatenate([g[u] for u in pick]).mean())
    return float(covered.mean()), float(np.quantile(covs, 0.025)), float(np.quantile(covs, 0.975))


def main():
    te = pd.read_csv(os.path.join(OUT, "banded_test_detail.csv"))
    print("== test 覆盖率 vs k（点估计 + 场景 bootstrap 95% CI）==")
    print(f"{'k':>7} {'cover':>8} {'ci_lo':>8} {'ci_hi':>8}  0.95_in_CI  binomial_pass")
    rows = []
    for k in (1.211, 1.240, 1.290, 1.346, 1.362, 1.40):
        c, lo, hi = boot_cover(te, k)
        in_ci = bool(lo <= 0.95 <= hi)
        bin_pass = bool(0.9487 <= c <= 0.9513)
        rows.append({"k": k, "cover": c, "ci_lo": lo, "ci_hi": hi,
                     "zero95_in_ci": in_ci, "binomial_pass": bin_pass})
        print(f"{k:7.3f} {c:8.4f} {lo:8.4f} {hi:8.4f}  {str(in_ci):>9}  {str(bin_pass):>13}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "k_bootstrap_table.csv"), index=False)
    print("\nsaved:", os.path.join(OUT, "k_bootstrap_table.csv"))


if __name__ == "__main__":
    main()
