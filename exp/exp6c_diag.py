# -*- coding: utf-8 -*-
"""EXP-6c 诊断：calib→test 的 s 分布偏移定位 + 备选 k 方案对照。

问题：banded + k_calib=1.240 在 test 上覆盖率 0.937 < 0.949。需判断缺口是
① 全局尺度偏移（换更大的校准集即可）② 结构性（某些带/季节尾部更肥）
③ 分带超参不稳（EXP-6b vs 6c 两轮 band5 ℓ_z 差 200 倍）。

只读已落盘明细，不跑 GP。
产出：results/exp6c/k_diag.csv + 控制台摘要。
"""
import sys, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from exp import config as C

OUT = C.result_dir("exp6c")


def cq(s, q=0.95):
    n = len(s)
    r = int(np.ceil((n + 1) * q))
    r = min(max(r, 1), n)
    return float(np.sort(s)[r - 1])


def qs(s):
    return {f"p{int(q*100)}": round(float(np.quantile(s, q)), 3)
            for q in (0.5, 0.75, 0.9, 0.95, 0.99)}


def main():
    sc = pd.read_csv(os.path.join(OUT, "banded_scores_detail.csv"))
    te = pd.read_csv(os.path.join(OUT, "banded_test_detail.csv"))
    val = sc[sc["role"] == "val"]
    cal = sc[sc["role"] == "calib"]

    print("== k 需求对照（banded）==")
    k_val = cq(val["s"].values)
    k_cal = cq(cal["s"].values)
    k_both = cq(np.concatenate([val["s"].values, cal["s"].values]))
    k_test = cq(te["s"].values)
    print(f"k_val(61场景)={k_val:.3f}  k_calib(75)={k_cal:.3f}  "
          f"k_val+calib(136)={k_both:.3f}  k_test_needed(128)={k_test:.3f}")
    print(f"偏移比: test/calib={k_test/k_cal:.3f}  test/(val+calib)={k_test/k_both:.3f}")

    # 各 k 下的 test 覆盖率
    print("\n== 备选 k 下的 test 覆盖率 ==")
    for name, k in [("k_calib", k_cal), ("k_val", k_val),
                    ("k_val+calib", k_both), ("k_test_oracle", k_test)]:
        cov = float((te["s"].values <= k).mean())
        print(f"  {name:14s} k={k:.3f}  cover={cov:.4f}  "
              f"{'PASS' if 0.9487 <= cov <= 0.9513 else 'fail'}")

    # 分带 s 分布：calib vs test（偏移定位）
    print("\n== 分带 s 分布 calib vs test（结构性 vs 全局）==")
    rows = []
    for b in sorted(te["band"].unique()):
        sb_c = cal[cal["band"] == b]["s"].values
        sb_t = te[te["band"] == b]["s"].values
        kc, kt = cq(sb_c), cq(sb_t)
        rows.append({"band": int(b), "n_calib": len(sb_c), "n_test": len(sb_t),
                     "k_calib": kc, "k_test": kt, "ratio": kt / kc,
                     "s_p50_calib": float(np.quantile(sb_c, 0.5)),
                     "s_p50_test": float(np.quantile(sb_t, 0.5)),
                     "s_p95_calib": kc, "s_p95_test": kt})
        print(f"  band{int(b)}: k_cal={kc:.3f} k_test={kt:.3f} ratio={kt/kc:.3f} "
              f"| s_p50 {np.quantile(sb_c,0.5):.3f}->{np.quantile(sb_t,0.5):.3f}")

    # 分季节
    print("\n== 分季节 k（calib vs test）==")
    for s in sorted(te["season"].unique()):
        sc_ = cal[cal["season"] == s]["s"].values
        st_ = te[te["season"] == s]["s"].values
        print(f"  season{int(s)}: k_cal={cq(sc_):.3f} k_test={cq(st_):.3f} "
              f"ratio={cq(st_)/cq(sc_):.3f}  n_cal={len(sc_)} n_test={len(st_)}")

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "k_diag.csv"), index=False)
    print("\nsaved:", os.path.join(OUT, "k_diag.csv"))


if __name__ == "__main__":
    main()
