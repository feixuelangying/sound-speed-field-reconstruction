# -*- coding: utf-8 -*-
"""EXP-6b 补充诊断：test 场景 s=|err|/u_c 分布、共形 k 需求、分带 k 需求。

用途：判断 EXP-6 覆盖率缺口（0.936 < 0.949）是"整体偏移"还是"尾部驱动"，
以及 banded GP 下需要多大的 k / 是否需要分带 k。
产出：results/exp6b/k_diagnosis.csv + 控制台摘要（不改任何模型）。
"""
import sys, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from exp import config as C

OUT = C.result_dir("exp6b")


def s_quantiles(err, u_c, qs=(0.5, 0.75, 0.9, 0.95, 0.99)):
    s = np.abs(err) / u_c
    return {f"p{int(q*100)}": float(np.quantile(s, q)) for q in qs}, s


def report(tag, err, u_c, band=None, k_frozen=1.211):
    q, s = s_quantiles(err, u_c)
    cov = float((s <= k_frozen).mean())
    k95 = float(np.quantile(s, 0.95))
    print(f"[{tag}] n={len(s)}  cover@k={k_frozen:.3f}: {cov:.4f}  "
          f"k_for_0.95={k95:.3f}  s: " +
          " ".join(f"{kk}={vv:.3f}" for kk, vv in q.items()))
    rows = [{"tag": tag, "band": "all", "n": len(s), "cover_at_frozen_k": cov,
             "k_for_0.95": k95, **q}]
    if band is not None:
        for b in sorted(np.unique(band)):
            m = band == b
            qb, sb = s_quantiles(err[m], u_c[m])
            rows.append({"tag": tag, "band": int(b), "n": int(m.sum()),
                         "cover_at_frozen_k": float((sb <= k_frozen).mean()),
                         "k_for_0.95": float(np.quantile(sb, 0.95)), **qb})
            print(f"    band{int(b):>1}: n={int(m.sum()):6d} cover={rows[-1]['cover_at_frozen_k']:.4f} "
                  f"k_req={rows[-1]['k_for_0.95']:.3f} s_p50={qb['p50']:.3f} s_p95={qb['p95']:.3f}")
    return rows


def main():
    all_rows = []
    # ① 单 GP（EXP-6 test 明细，冻结 k=1.211）
    p = os.path.join(C.RESULTS, "exp6", "test_B_detail.csv")
    df = pd.read_csv(p)
    print("== single GP (EXP-6 test_B_detail) ==")
    all_rows += report("single", df["err"].values, df["u_c"].values,
                       df["band"].values)

    # ② banded vs single（EXP-6b 明细，同一冻结 k）
    p2 = os.path.join(OUT, "banded_vs_single_detail.csv")
    if os.path.exists(p2):
        d2 = pd.read_csv(p2)
        for tag in ("single", "banded"):
            d = d2[d2["tag"] == tag]
            print(f"== {tag} GP (EXP-6b detail) ==")
            all_rows += report(tag + "_6b", d["err"].values, d["u_c"].values,
                               d["band"].values)

    out = pd.DataFrame(all_rows)
    out.to_csv(os.path.join(OUT, "k_diagnosis.csv"), index=False)
    print("\nsaved:", os.path.join(OUT, "k_diagnosis.csv"))

    # ③ 尾部诊断（single）：未覆盖点的误差量级与分带归属
    err, u_c, band = df["err"].values, df["u_c"].values, df["band"].values
    bad = np.abs(err) > 1.211 * u_c
    print(f"\n[tail] uncovered n={bad.sum()} share={bad.mean():.4f}")
    print("  uncovered |err| p50/p90/p99:",
          [round(float(np.quantile(np.abs(err[bad]), q)), 2) for q in (0.5, 0.9, 0.99)])
    print("  uncovered band share:", (pd.Series(band[bad]).value_counts(normalize=True)
                                     .sort_index().round(3).to_dict()))
    print("  all       band share:", (pd.Series(band).value_counts(normalize=True)
                                     .sort_index().round(3).to_dict()))


if __name__ == "__main__":
    main()
