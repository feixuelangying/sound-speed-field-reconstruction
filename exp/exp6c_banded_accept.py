# -*- coding: utf-8 -*-
"""EXP-6c：banded block GP 的完整重标定与独立验收（EXP-6 不达标的迭代回路）。

背景：EXP-6 用单 GP，test 覆盖率 0.936 < J2 容差下界 0.949，根因 u_r 无深度结构。
EXP-6b 证明分带块 GP 能造出深度结构（各带所需 k 由 [0.53,1.79] 收窄至 [1.06,1.51]）。
本脚本把 banded 走完整个"评定—校准—验收"闭环：

① 分带超参拟合（train-only 场景，缓存 results/exp4/band_kerns.json）
② val/calib 场景 banded s=|err|/u_c 明细（与 EXP-5 同口径）
③ 分层方案选择（val 场景 LOSO）→ 最终 k 在 calib 定标（冻结 results/exp5b/k_banded.json）
④ test 场景独立验收（128 场景 × N_SEEDS）：覆盖率/rmse/宽度/场景 bootstrap CI/分带
⑤ J2 判定 + single vs banded 对照
产出：results/exp6c/*.csv|json；results/exp5b/k_banded.json
"""
import sys, os, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from exp import config as C
from exp.lib import scenes as scenelib
from exp.lib import banded as bl
from exp.lib.pipeline import SceneData, load_stack, coverage_tolerance
from exp.lib.uc import UcComposer

OUT = C.result_dir(os.environ.get("EXP6C_OUT", "exp6c"))
KOUT = C.result_dir(os.environ.get("EXP5B_OUT", "exp5b"))
RNG = np.random.default_rng(C.SEED)
N_SEEDS = 3
N_BOOT = 1000
MIN_CELL = 200
IMPROVE_THR = 0.30


def conformal_quantile(s, q=0.95):
    n = len(s)
    r = int(np.ceil((n + 1) * q))
    r = min(max(r, 1), n)
    return float(np.sort(s)[r - 1])


def scene_detail(SD, uc, bk, sid, seed=0):
    """单场景 A→B 的 banded 逐点明细。"""
    A, B = SD.ab(sid, seed)
    dsA, dsB = SD.build(sid, A), SD.build(sid, B)
    fb = SD.kern_for(dsB["season"])
    pred, var = bl.predict_banded(dsA, dsB["X4"], dsB["mean"], dsB["band"], bk, fb)
    z, season = dsB["z"], dsB["season"]
    u_extra = uc.extra_std(z, season)
    u_c = np.sqrt(var + u_extra ** 2)
    err = pred - dsB["d"]
    dh, _ = SD.dist_to_inputs(dsB["X4"], dsA["X4"])
    return pd.DataFrame({
        "sid": int(sid), "seed": seed, "prof": dsB["prof"],
        "band": np.clip(dsB["band"], 0, 5), "z": z, "season": season,
        "err": err, "u_c": u_c, "u_r": np.sqrt(var), "u_extra": u_extra,
        "s": np.abs(err) / u_c, "sparse_km": dh})


def k_by_scheme(df, scheme, k_ref):
    if scheme == "global":
        k = conformal_quantile(df["s"].values)
        return np.full(len(df), k), {"global": k}
    if scheme == "band":
        keys = df["band"].values
    elif scheme == "season":
        keys = df["season"].values
    else:
        keys = np.digitize(df["sparse_km"].values, [50.0, 200.0])
    k_arr = np.full(len(df), k_ref)
    table = {}
    for kk in np.unique(keys):
        m = keys == kk
        if m.sum() >= MIN_CELL:
            table[str(int(kk))] = conformal_quantile(df["s"].values[m])
            k_arr[m] = table[str(int(kk))]
        else:
            table[str(int(kk))] = k_ref
    return k_arr, table


def cond_cov_dev(df, k_arr, scheme):
    cover = np.abs(df["err"].values) <= k_arr * df["u_c"].values
    if scheme == "global":
        return float(np.sqrt((cover.mean() - 0.95) ** 2))
    if scheme == "band":
        keys = df["band"].values
    elif scheme == "season":
        keys = df["season"].values
    else:
        keys = np.digitize(df["sparse_km"].values, [50.0, 200.0])
    devs = []
    for kk in np.unique(keys):
        m = keys == kk
        if m.sum() >= 50:
            devs.append(cover[m].mean() - 0.95)
    return float(np.sqrt(np.mean(np.square(devs)))) if devs else 0.0


def scene_bootstrap(df, n_boot=N_BOOT):
    """场景级 bootstrap（聚类稳健）：覆盖率/RMSE 的 95% CI。"""
    sids = df["sid"].unique()
    g = {s: df[df["sid"] == s][["covered", "err2"]].values for s in sids}
    covs, rmses = [], []
    for _ in range(n_boot):
        pick = RNG.choice(sids, len(sids), replace=True)
        cov = np.concatenate([g[s][:, 0] for s in pick])
        e2 = np.concatenate([g[s][:, 1] for s in pick])
        covs.append(cov.mean())
        rmses.append(np.sqrt(e2.mean()))
    return {"cover_ci95": [float(np.quantile(covs, 0.025)),
                           float(np.quantile(covs, 0.975))],
            "rmse_ci95": [float(np.quantile(rmses, 0.025)),
                          float(np.quantile(rmses, 0.975))]}


def metrics(df, label, k_pt):
    """k_pt: 与 df 同长的逐点共形因子。"""
    err = df["err"].values
    k_pt = np.asarray(k_pt, dtype=float)
    half = k_pt * df["u_c"].values
    cover = (np.abs(err) <= half).astype(float)
    w = 2 * half
    wink = np.where(err < -half, w + 2 / 0.05 * (-half - err),
                    np.where(err > half, w + 2 / 0.05 * (err - half), w))
    tmp = pd.DataFrame({"prof": df["prof"].values, "band": df["band"].values,
                        "cover": cover, "err2": err ** 2})
    grp = tmp.groupby(["prof", "band"])
    return {"label": label, "k_mean": float(k_pt.mean()), "n": int(len(df)),
            "n_groups": int(grp.ngroups),
            "cover_weighted": float(grp["cover"].mean().mean()),
            "cover_pointwise": float(cover.mean()),
            "rmse_pointwise": float(np.sqrt(np.mean(err ** 2))),
            "mae_pointwise": float(np.mean(np.abs(err))),
            "width_mean": float(np.mean(w)),
            "winkler_pointwise": float(np.mean(wink))}


def apply_scheme_k(test_df, scheme, k_global, k_table):
    """按冻结方案给出逐点 k 数组。"""
    if scheme == "global":
        return np.full(len(test_df), k_global)
    if scheme == "band":
        keys = test_df["band"].values.astype(int).astype(str)
    elif scheme == "season":
        keys = test_df["season"].values.astype(int).astype(str)
    else:
        keys = np.digitize(test_df["sparse_km"].values, [50.0, 200.0]).astype(int).astype(str)
    return np.array([k_table.get(x, k_global) for x in keys])


def main():
    t0 = time.time()
    print("== EXP-6c: banded block GP recalibration + independent acceptance ==")
    stack = load_stack()
    SD = SceneData(stack)
    scenes = stack["scenes"]
    uc = UcComposer()
    with open(os.path.join(C.RESULTS, "exp3", "val_scene_ids.json"), encoding="utf-8") as f:
        val_sids = json.load(f)
    calib_sids = scenelib.scene_ids_by_role(scenes, "calib")
    test_sids = scenelib.scene_ids_by_role(scenes, "test")
    train_sids = scenelib.scene_ids_by_role(scenes, "train")
    print(f"scenes: val={len(val_sids)}, calib={len(calib_sids)}, "
          f"test={len(test_sids)}, train={len(train_sids)}")

    # ① 分带超参（缓存复用）
    bk = bl.load_band_kerns()
    if bk is None:
        print("-- fitting per-band hyperparameters (train-only) --")
        bk = bl.fit_band_kerns(SD, train_sids)
        bl.save_band_kerns(bk)
    else:
        print(f"-- loaded band kernels from {bl.KERN_PATH} --")
    print(f"  usable bands: {sum(v is not None for v in bk.values())}/6")

    # ② val / calib 的 banded s 明细
    print("-- banded conformal scores on val & calib scenes --")
    dfs = []
    for tag, sids in (("val", val_sids), ("calib", calib_sids)):
        for i, sid in enumerate(sids):
            d = scene_detail(SD, uc, bk, int(sid), seed=0)
            d["role"] = tag
            dfs.append(d)
        print(f"  {tag}: {len(sids)} scenes done ({time.time()-t0:.0f}s)", flush=True)
    all_df = pd.concat(dfs, ignore_index=True)
    all_df.to_csv(os.path.join(OUT, "banded_scores_detail.csv"), index=False)
    val_df = all_df[all_df["role"] == "val"]
    calib_df = all_df[all_df["role"] == "calib"]
    print(f"  val n={len(val_df)}, calib n={len(calib_df)}")

    # ③ 分层方案选择（val LOSO）+ 最终 k（calib）
    print("-- scheme selection (val LOSO) --")
    loso = []
    for scheme in ("global", "band", "season", "sparse"):
        devs = []
        for sid in val_df["sid"].unique():
            din = val_df[val_df["sid"] != sid]
            dout = val_df[val_df["sid"] == sid]
            if len(din) < 100:
                continue
            k_ref = conformal_quantile(din["s"].values)
            if scheme == "global":
                k_out = np.full(len(dout), k_ref)
            else:
                _, table = k_by_scheme(din, scheme, k_ref)
                if scheme == "band":
                    kk = dout["band"].values.astype(int).astype(str)
                elif scheme == "season":
                    kk = dout["season"].values.astype(int).astype(str)
                else:
                    kk = np.digitize(dout["sparse_km"].values,
                                     [50.0, 200.0]).astype(int).astype(str)
                k_out = np.array([table.get(x, k_ref) for x in kk])
            devs.append(cond_cov_dev(dout, k_out, scheme))
        loso.append({"scheme": scheme, "cond_cov_dev_rms": float(np.mean(devs))})
        print(f"  {scheme}: dev_rms={loso[-1]['cond_cov_dev_rms']:.4f}")
    loso_df = pd.DataFrame(loso)
    loso_df.to_csv(os.path.join(OUT, "banded_k_scheme_loso.csv"), index=False)
    dev_g = loso_df[loso_df["scheme"] == "global"]["cond_cov_dev_rms"].iloc[0]
    best = loso_df[loso_df["scheme"] != "global"].sort_values("cond_cov_dev_rms").iloc[0]
    improve = 1 - best["cond_cov_dev_rms"] / dev_g if dev_g > 0 else 0
    scheme = best["scheme"] if improve >= IMPROVE_THR else "global"
    print(f"chosen scheme: {scheme} (improve={improve:.0%} vs thr {IMPROVE_THR:.0%})")

    k_banded = conformal_quantile(calib_df["s"].values)
    k_arr, k_table = k_by_scheme(calib_df, scheme, k_banded)
    kcfg = {"scheme": scheme, "k_global": k_banded, "k_table": k_table,
            "n_calib_pts": int(len(calib_df)),
            "n_calib_scenes": int(calib_df["sid"].nunique()),
            "s_quantiles": {str(q): float(np.quantile(calib_df["s"].values, q))
                            for q in (0.5, 0.75, 0.9, 0.95, 0.99)},
            "model": "banded_block_gp",
            "improve_of_best_stratified": float(improve),
            "date": time.strftime("%Y-%m-%d")}
    with open(os.path.join(KOUT, "k_banded.json"), "w", encoding="utf-8") as f:
        json.dump(kcfg, f, ensure_ascii=False, indent=1)
    print(f"  k_banded={k_banded:.3f}, scheme={scheme}, table={k_table}")

    # ④ test 场景独立验收（banded + k_banded）
    print(f"-- test scenes x {N_SEEDS} seeds (banded) --")
    tdfs = []
    for i, sid in enumerate(test_sids):
        for seed in range(N_SEEDS):
            tdfs.append(scene_detail(SD, uc, bk, int(sid), seed))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(test_sids)} scenes ({time.time()-t0:.0f}s)", flush=True)
    test_df = pd.concat(tdfs, ignore_index=True)
    test_df.to_csv(os.path.join(OUT, "banded_test_detail.csv"), index=False)
    print(f"  B points total: {len(test_df)}")

    k_pt = apply_scheme_k(test_df, scheme, k_banded, k_table)
    test_df["k"] = k_pt
    test_df["covered"] = np.abs(test_df["err"].values) <= k_pt * test_df["u_c"].values
    test_df["err2"] = test_df["err"].values ** 2
    tol_lo, tol_hi = coverage_tolerance(len(test_df))

    rows = []
    for lbl, mask in [("all", np.ones(len(test_df), bool))] + \
                     [(f"band{int(b)}", (test_df["band"] == b).values)
                      for b in sorted(test_df["band"].unique())] + \
                     [(f"season{int(s)}", (test_df["season"] == s).values)
                      for s in sorted(test_df["season"].unique())]:
        if mask.sum() == 0:
            continue
        rows.append(metrics(test_df[mask], lbl, k_pt[mask]))
    mdf = pd.DataFrame(rows)
    mdf["tol_lo"], mdf["tol_hi"] = tol_lo, tol_hi
    mdf.to_csv(os.path.join(OUT, "banded_test_metrics.csv"), index=False)
    print(mdf.to_string(index=False))

    boot = scene_bootstrap(test_df)
    print(f"  bootstrap: cover CI={boot['cover_ci95']}, rmse CI={boot['rmse_ci95']}")

    # ⑤ J2 判定（总体）
    cov = float(test_df["covered"].mean())
    pass_binomial = bool(tol_lo <= cov <= tol_hi)
    pass_boot = bool(boot["cover_ci95"][0] <= 0.95 <= boot["cover_ci95"][1])
    print(f"  J2 (binomial band [{tol_lo:.4f},{tol_hi:.4f}]): cover={cov:.4f} "
          f"-> {'PASS' if pass_binomial else 'FAIL'}")
    print(f"  J2 (scene-bootstrap CI {boot['cover_ci95']}): "
          f"0.95 {'in' if pass_boot else 'OUT of'} CI")

    # 分带覆盖偏离（系统性）
    band_cov = {r["label"]: r["cover_pointwise"] for r in rows if r["label"].startswith("band")}
    band_dev_max = max(abs(v - 0.95) for v in band_cov.values())

    summary = {
        "exp": "exp6c", "date": time.strftime("%Y-%m-%d %H:%M"),
        "model": "banded_block_gp",
        "n_test_scenes": len(test_sids), "n_seeds": N_SEEDS,
        "n_B_points": int(len(test_df)),
        "k_banded": k_banded, "scheme": scheme, "k_table": k_table,
        "k_scheme_loso": loso,
        "calib_s_quantiles": kcfg["s_quantiles"],
        "coverage": cov, "coverage_tolerance_band": [tol_lo, tol_hi],
        "pass_J2_binomial": pass_binomial,
        "pass_J2_bootstrap": pass_boot,
        "bootstrap": boot,
        "band_coverage": band_cov,
        "band_coverage_max_dev_from_0.95": float(band_dev_max),
        "metrics": mdf.to_dict("records"),
        "elapsed_s": time.time() - t0,
    }
    with open(os.path.join(OUT, "exp6c_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    print("EXP-6c DONE", time.time() - t0)


if __name__ == "__main__":
    main()
