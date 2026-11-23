# -*- coding: utf-8 -*-
"""诊断（EXP-12 附）：主 k 候选在 test 场景上的事后表现与场景级 bootstrap CI。

回答"pool_boot95(1.8891) 是否应切换为 calib_boot95(1.8670)"。

**注意**：k 的定标依据是校准域（val+calib），test 为独立仲裁集；本脚本仅作
**事后稳健性核查**，结果不用于反选 k（否则构成泄漏）。选 k 仍以校准域为准。
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from exp import config as C

OUT = C.result_dir("exp12_v1v2_compare")
DET = os.path.join(C.RESULTS, "exp6c_v2", "banded_test_detail.csv")
TOL = (0.9487048195525158, 0.9512832938637797)      # 二项容差带（独立点假设）
ALPHA = 0.05
KS = [("calib_boot95", 1.8670347355385108),
      ("pool_boot90", 1.8746557918575135),
      ("pool_boot95", 1.889111312040192)]
N_BOOT, SEED = 2000, 12345


def eval_k(df, k):
    err = df["err"].values
    half = k * df["u_c"].values
    cover = np.abs(err) <= half
    width = 2.0 * half
    wink = np.where(err < -half, 2 * half + 2 / ALPHA * (-half - err),
                    np.where(err > half, 2 * half + 2 / ALPHA * (err - half), 2 * half))
    cw = df.assign(_c=cover).groupby(["prof", "band"])["_c"].mean().mean()
    return dict(cover=float(cover.mean()), cover_w=float(cw),
                width=float(width.mean()), winkler=float(wink.mean()),
                rmse=float(np.sqrt(np.mean(err ** 2))))


def boot_ci(df, k, rng):
    cover = (np.abs(df["err"].values) <= k * df["u_c"].values).astype(float)
    g = pd.DataFrame({"sid": df["sid"].values, "c": cover}).groupby("sid")["c"].agg(
        ["mean", "size"])
    m, n = g["mean"].values, g["size"].values
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, len(m), len(m))
        out[b] = (m[idx] * n[idx]).sum() / n[idx].sum()
    return float(np.quantile(out, .025)), float(np.quantile(out, .975))


def main():
    df = pd.read_csv(DET)
    print(f"test detail: n={len(df)}, scenes={df['sid'].nunique()}, "
          f"prof={df['prof'].nunique()}, bands={sorted(df['band'].unique())}")
    rng = np.random.default_rng(SEED)
    rows = []
    for name, k in KS:
        r = {"name": name, "k": k}
        r.update(eval_k(df, k))
        lo, hi = boot_ci(df, k, rng)
        r.update(ci_lo=lo, ci_hi=hi,
                 bin_pass=bool(TOL[0] <= r["cover"] <= TOL[1]),
                 ci_pass=bool(lo <= 0.95 <= hi))
        rows.append(r)
        print(f"  {name:14s} k={k:.4f} cover={r['cover']:.5f} "
              f"(bin {TOL[0]:.5f}~{TOL[1]:.5f}) bin_pass={r['bin_pass']} "
              f"CI=[{lo:.4f},{hi:.4f}] ci_pass={r['ci_pass']} "
              f"cover_w={r['cover_w']:.5f} width={r['width']:.4f} "
              f"winkler={r['winkler']:.4f} rmse={r['rmse']:.4f}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "k_candidate_test_check.csv"), index=False)
    print("saved ->", os.path.join(OUT, "k_candidate_test_check.csv"))


if __name__ == "__main__":
    main()
