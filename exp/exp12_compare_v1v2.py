# -*- coding: utf-8 -*-
"""EXP-12：v1/v2（σ² 自由 ML vs 矩估计锚定）主方案同口径对比。

背景：EXP-11 把 GP 的 σ² 从 type-II ML 自由优化改为矩估计锚定（= 跨场景池化残差
方差），修复 u_r 高估（prod_global σ²: 10.12 → 2.48）。EXP-6c/6d 在 v2 口径下重跑
得到新 k_final。本脚本在完全相同的 test B 点集（sid/seed/prof/band/z 逐一校验）
上做严格同口径对比，量化 u_r 尺度、u_c、覆盖率、宽度、Winkler、rmse 的变化。

产出：results/exp12_v1v2_compare/{summary.csv, by_band.csv, by_season.csv}
"""
import sys, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from exp import config as C

OUT = C.result_dir("exp12_v1v2_compare")
ALPHA = 0.05

VARIANTS = [
    ("v1_ml", os.path.join(C.RESULTS, "exp6c", "banded_test_detail.csv"),
     os.path.join(C.RESULTS, "frozen", "k_main_v1.json")),
    ("v2_moment", os.path.join(C.RESULTS, "exp6c_v2", "banded_test_detail.csv"),
     os.path.join(C.RESULTS, "frozen", "k_main_v2.json")),
]


def load(variant):
    tag, det, kp = variant
    df = pd.read_csv(det)
    with open(kp, encoding="utf-8") as f:
        k = json.load(f)["k_global"]
    err = df["err"].values
    half = k * df["u_c"].values
    df["k"] = k
    df["cover"] = np.abs(err) <= half
    df["width"] = 2 * half
    df["winkler"] = np.where(err < -half, 2 * half + 2 / ALPHA * (-half - err),
                             np.where(err > half, 2 * half + 2 / ALPHA * (err - half),
                                      2 * half))
    df["variant"] = tag
    return df


def agg(df):
    cover_w = float(df.groupby(["prof", "band"])["cover"].mean().mean())
    return {"u_r_mean": float(df["u_r"].mean()), "u_c_mean": float(df["u_c"].mean()),
            "rmse": float(np.sqrt(np.mean(df["err"].values ** 2))),
            "cover_pointwise": float(df["cover"].mean()),
            "cover_weighted": cover_w,
            "width_mean": float(df["width"].mean()),
            "winkler": float(df["winkler"].mean()), "n": int(len(df))}


def main():
    dfs = {v[0]: load(v) for v in VARIANTS}
    a, b = dfs["v1_ml"], dfs["v2_moment"]
    key = ["sid", "seed", "prof", "band", "z"]
    ka = a[key].round(6).astype(str).agg("|".join, axis=1)
    kb = b[key].round(6).astype(str).agg("|".join, axis=1)
    assert (ka.values == kb.values).all(), "B 点集不一致，无法同口径对比"
    print(f"B 点集一致：n={len(a)}")

    rows = []
    for tag, d in dfs.items():
        r = {"variant": tag, "k": float(d["k"].iloc[0])}
        r.update(agg(d))
        rows.append(r)
    sm = pd.DataFrame(rows)
    sm.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    print("\n== 总体（同点集） ==")
    print(sm.round(4).to_string(index=False))

    for grp in ("band", "season"):
        rr = []
        for tag, d in dfs.items():
            for g, sub in d.groupby(grp):
                r = {"variant": tag, grp: int(g)}
                r.update(agg(sub))
                rr.append(r)
        gdf = pd.DataFrame(rr)
        gdf.to_csv(os.path.join(OUT, f"by_{grp}.csv"), index=False)
        piv = gdf.pivot(index=grp, columns="variant",
                        values=["u_r_mean", "u_c_mean", "rmse", "cover_pointwise",
                                "width_mean"])
        print(f"\n== 分{grp} ==")
        print(piv.round(4).to_string())

    # 派生指标：u_r 相对实际误差尺度的校准度
    print("\n== u_r 尺度诊断 ==")
    for tag, d in dfs.items():
        band = d.groupby("band")
        ratio = (band["u_r"].mean() / band.apply(
            lambda s: np.sqrt(np.mean(s["err"].values ** 2)))).round(3)
        print(f"  {tag}: u_r/rmse 分带 = {ratio.to_dict()}")


if __name__ == "__main__":
    main()
