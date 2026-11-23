# -*- coding: utf-8 -*-
"""EXP-6d：k 的保守化重定标（应对 calib→test 场景分布漂移，R9 预案）。

背景（EXP-6c 诊断）：test 各深度带"实际需要"的 k 比 calib 定标出的 k 高 5%~24%，
calib 75 个场景的 s 分位数系统性低估 test 需求；banded GP 已修好 u_r 的深度结构
（band0 0.979→0.958、band3 0.887→0.902），但整体覆盖 0.9372 仍差 J2 下界 0.9487。

本实验把两类不确定性显式计入 k 定标：
  ① 场景量不足 -> 合并 val+calib（136 场景）定标 k_pool；
  ② k 估计本身有抽样误差 -> 对 calib 做场景级 bootstrap，取 k 分布的 90%/95%
     分位作为保守 k（把"k 估不准"这件事换成区间加宽）。
产出：k_candidates.csv + exp6d_summary.json（含 test 覆盖的场景级 bootstrap CI、
      宽度与 Winkler 代价；并量化"事后最优 k"仅作缺口诊断、不作定标依据）。
"""
import sys, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from exp import config as C

OUT = C.result_dir(os.environ.get("EXP6D_OUT", "exp6d"))
SRC = C.result_dir(os.environ.get("EXP6C_OUT", "exp6c"))
RNG = np.random.default_rng(C.SEED)
NB = 2000
ALPHA = 0.05
TOL_LO, TOL_HI = 0.9487048195525158, 0.9512832938637797   # EXP-6 二项容差带（独立假设，作参照）


def cq(s, q=0.95):
    """split-conformal 分位数（有限样本修正）。"""
    n = len(s)
    r = int(np.ceil((n + 1) * q))
    r = min(max(r, 1), n)
    return float(np.sort(s)[r - 1])


def by_scene(df):
    return {int(u): g["s"].values for u, g in df.groupby("sid")}


def boot_k(s_by_scene, q=0.95, n_boot=NB):
    """场景级 bootstrap 的 k 分布（重采样场景，非重采样点）。"""
    keys = list(s_by_scene.keys())
    ks = np.empty(n_boot)
    for i in range(n_boot):
        pick = RNG.choice(keys, len(keys), replace=True)
        ks[i] = cq(np.concatenate([s_by_scene[u] for u in pick]), q)
    return ks


def boot_cover(te, k, n_boot=NB):
    covered = te["s"].values <= k
    sid = te["sid"].values
    g = {u: covered[sid == u] for u in np.unique(sid)}
    keys = list(g.keys())
    covs = np.empty(n_boot)
    for i in range(n_boot):
        pick = RNG.choice(keys, len(keys), replace=True)
        covs[i] = np.concatenate([g[u] for u in pick]).mean()
    return float(covered.mean()), float(np.quantile(covs, 0.025)), float(np.quantile(covs, 0.975))


def test_metrics(te, k):
    err, uc = te["err"].values, te["u_c"].values
    half = k * uc
    cover = np.abs(err) <= half
    w = 2 * half
    wink = np.where(err < -half, w + 2 / ALPHA * (-half - err),
                    np.where(err > half, w + 2 / ALPHA * (err - half), w))
    tmp = pd.DataFrame({"prof": te["prof"].values, "band": te["band"].values,
                        "cover": cover.astype(float)})
    cover_w = float(tmp.groupby(["prof", "band"])["cover"].mean().mean())
    c, lo, hi = boot_cover(te, k)
    return {"k": float(k), "cover_pointwise": c, "cover_weighted": cover_w,
            "ci_lo": lo, "ci_hi": hi, "zero95_in_ci": bool(lo <= 0.95 <= hi),
            "binomial_pass": bool(TOL_LO <= c <= TOL_HI),
            "width_mean": float(w.mean()), "winkler": float(wink.mean()),
            "rmse": float(np.sqrt(np.mean(err ** 2)))}


def main():
    sc = pd.read_csv(os.path.join(SRC, "banded_scores_detail.csv"))
    te = pd.read_csv(os.path.join(SRC, "banded_test_detail.csv"))
    val = sc[sc["role"] == "val"]
    cal = sc[sc["role"] == "calib"]
    print(f"val scenes={val['sid'].nunique()}  calib scenes={cal['sid'].nunique()}  "
          f"test scenes={te['sid'].nunique()}  test points={len(te)}")

    s_cal = by_scene(cal)
    s_val = by_scene(val)
    s_pool = {**s_cal, **s_val}

    k_calib = cq(cal["s"].values)
    k_pool = cq(np.concatenate([val["s"].values, cal["s"].values]))
    ks_boot = boot_k(s_cal)
    k90, k95 = float(np.quantile(ks_boot, 0.90)), float(np.quantile(ks_boot, 0.95))
    ks_pool_boot = boot_k(s_pool)
    k90p, k95p = float(np.quantile(ks_pool_boot, 0.90)), float(np.quantile(ks_pool_boot, 0.95))

    print("\n== k 候选（定标口径）==")
    print(f"  k_calib(75场景)      = {k_calib:.4f}")
    print(f"  k_pool(val+calib 136)= {k_pool:.4f}")
    print(f"  calib bootstrap: 中位={np.median(ks_boot):.4f}  q90={k90:.4f}  q95={k95:.4f}")
    print(f"  pool  bootstrap: 中位={np.median(ks_pool_boot):.4f}  q90={k90p:.4f}  q95={k95p:.4f}")

    cands = [("k_calib", k_calib), ("k_pool", k_pool),
             ("k_calib_boot90", k90), ("k_calib_boot95", k95),
             ("k_pool_boot90", k90p), ("k_pool_boot95", k95p)]
    rows = []
    print("\n== test 场景评估（k 候选）==")
    print(f"{'name':>16} {'k':>7} {'cover':>8} {'ci_lo':>8} {'ci_hi':>8} "
          f"{'0.95inCI':>9} {'bin_pass':>9} {'width':>7} {'winkler':>8} {'rmse':>7}")
    for name, k in cands:
        m = test_metrics(te, k)
        m["name"] = name
        rows.append(m)
        print(f"{name:>16} {m['k']:7.3f} {m['cover_pointwise']:8.4f} {m['ci_lo']:8.4f} "
              f"{m['ci_hi']:8.4f} {str(m['zero95_in_ci']):>9} {str(m['binomial_pass']):>9} "
              f"{m['width_mean']:7.3f} {m['winkler']:8.3f} {m['rmse']:7.3f}")

    # 缺口量化（仅诊断，不作定标依据——用 test 自身反解会乐观）
    ks = np.linspace(1.20, 1.60, 81)
    covers = np.array([test_metrics(te, k)["cover_pointwise"] for k in ks])
    need = ks[np.argmax(covers >= 0.95)] if (covers >= 0.95).any() else float("nan")
    need_ci = float("nan")
    for k in ks:
        if boot_cover(te, k)[1] >= TOL_LO:
            need_ci = float(k)
            break
    print(f"\n[诊断·不作定标] test 自身反解：点覆盖>=0.95 需 k>={need:.3f}；"
          f"bootstrap CI 下界>=0.9487 需 k>={need_ci:.3f}")

    # 冻结 k（纯校准域信息：val+calib 场景 bootstrap 的 95% 分位）与分层覆盖核查
    k_final = k95p
    te2 = te.assign(cover=(np.abs(te["err"].values) <= k_final * te["u_c"].values))
    band_tab = te2.groupby("band")["cover"].agg(["mean", "size"]).reset_index()
    season_tab = te2.groupby("season")["cover"].agg(["mean", "size"]).reset_index()
    scene_cov = te2.groupby("sid")["cover"].mean()
    band_w = te2.groupby(["prof", "band"])["cover"].mean().reset_index()
    band_w["band"] = te2.groupby(["prof", "band"])["band"].first().values
    print(f"\n== 冻结 k_final = pool bootstrap q95 = {k_final:.4f} 的分层核查 ==")
    print("  深度带:", " ".join(f"b{int(r.band)}={r['mean']:.4f}(n={int(r['size'])})"
                                for _, r in band_tab.iterrows()))
    print("  季节  :", " ".join(f"s{int(r.season)}={r['mean']:.4f}(n={int(r['size'])})"
                                for _, r in season_tab.iterrows()))
    print(f"  分场景覆盖: mean={scene_cov.mean():.4f} sd={scene_cov.std():.4f} "
          f"min={scene_cov.min():.4f} max={scene_cov.max():.4f}")
    print(f"  带加权覆盖: {band_w.groupby('band')['cover'].mean().round(4).to_dict()}")
    print(f"  最大带偏离 |cov-0.95| = {float((band_tab['mean'] - 0.95).abs().max()):.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "k_candidates.csv"), index=False)
    summary = {
        "exp": "exp6d", "date": "2026-09-27",
        "n_val_scenes": int(val["sid"].nunique()), "n_calib_scenes": int(cal["sid"].nunique()),
        "n_test_scenes": int(te["sid"].nunique()),
        "k_calib": k_calib, "k_pool": k_pool,
        "k_calib_boot": {"median": float(np.median(ks_boot)), "q90": k90, "q95": k95},
        "k_pool_boot": {"median": float(np.median(ks_pool_boot)), "q90": k90p, "q95": k95p},
        "candidates": rows,
        "diagnostic_min_k_for_0.95": need,
        "diagnostic_min_k_for_ci_lo": need_ci,
        "tolerance_band_binomial": [TOL_LO, TOL_HI],
        "k_final": k_final,
        "k_final_rule": "pool(val+calib) scene-bootstrap q95 (calibration-domain only)",
        "band_coverage_at_k_final": {f"band{int(r.band)}": float(r["mean"])
                                     for _, r in band_tab.iterrows()},
        "band_weighted_coverage_at_k_final": {f"band{int(b)}": float(v) for b, v in
                                              band_w.groupby("band")["cover"].mean().items()},
        "season_coverage_at_k_final": {f"season{int(r.season)}": float(r["mean"])
                                       for _, r in season_tab.iterrows()},
        "scene_coverage_at_k_final": {"mean": float(scene_cov.mean()), "sd": float(scene_cov.std()),
                                      "min": float(scene_cov.min()), "max": float(scene_cov.max())},
        "band_coverage_max_dev_from_0.95": float((band_tab["mean"] - 0.95).abs().max()),
    }
    with open(os.path.join(OUT, "exp6d_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, "k_frozen_v2.json"), "w", encoding="utf-8") as f:
        json.dump({"k_final": k_final, "rule": "pool(val+calib) scene-bootstrap q95",
                   "n_pool_scenes": len(s_pool), "model": "banded_block_gp",
                   "note": "纯校准域信息定标（不使用 test）；应对 calib->test 场景分布漂移，"
                           "把 k 估计的抽样不确定性计入区间宽度"}, f, ensure_ascii=False, indent=2)
    # 主方案冻结文件（UcComposer 可直接加载；下游 EXP-7/8/9/10 统一取用）
    main_path = os.path.join(C.RESULTS, "frozen",
                             os.environ.get("K_MAIN_FILE", "k_main.json"))
    os.makedirs(os.path.dirname(main_path), exist_ok=True)
    with open(main_path, "w", encoding="utf-8") as f:
        json.dump({"scheme": "global", "k_global": k_final,
                   "k_table": {"global": k_final},
                   "model": "banded_block_gp",
                   "source": "exp6d pool(val+calib) scene-bootstrap q95",
                   "date": "2026-09-27"}, f, ensure_ascii=False, indent=1)
    print("main k frozen:", main_path)
    print("\nsaved:", OUT)


if __name__ == "__main__":
    main()
