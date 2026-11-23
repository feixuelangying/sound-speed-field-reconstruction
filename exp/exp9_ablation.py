# -*- coding: utf-8 -*-
"""EXP-9 消融与来源分解验证（WP4，framework §4-EXP9）。
① 组件消融（test 场景 B 点，seeds 0-2，统一 k=1.96 正态口径隔离组件贡献）：
   full（完整组件 + 标准正态 k=1.96，即框架"去校准"参照口径）/
   no_obs（仅背景+均值网络）/ no_meanet（GP 零均值）/
   no_bg（GP 直接拟合 c_obs，核超参重估）/ async_bin（异步分箱对齐）
   full_conformal（完整组件 + 冻结共形 k=1.3485，即主方案口径）
② 来源贡献单调性：no_obs vs full 的 u_r；引用 EXP-7 稀疏扫描
③ u_m—u_r 耦合检验（J4）：u_m 缩放 {0, 0.5, 1, 2, 4} -> 覆盖率单调性
④ 异步处理对照：4D 核内生 vs 时间分箱对齐（4 箱 3D GP）
⑤ u_m 评定方法对照：deep ensemble 内省（5×m_θ 重训）vs 随机效应 u_m 查表
产出：results/exp9/*.csv + exp9_summary.json
"""
import sys, os, time, json, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from exp import config as C
from exp.lib import physics, uobs, anchors, scenes as scenelib, meanet, features
from exp.lib import gp as gplib
from exp.lib.pipeline import SceneData, load_stack
from exp.lib.uc import UcComposer
from exp.lib import banded

OUT = C.result_dir("exp9")
RNG = np.random.default_rng(C.SEED)
SEEDS = (0, 1, 2)
MAX_SCENES = 40
N_BIN = 4            # 异步对照的时间箱数
N_ENSEMBLE = 5       # deep ensemble 成员数


def scaled_uc(uc, factor):
    """u_m 表整体缩放 factor 的 UcComposer 副本。"""
    uc2 = copy.deepcopy(uc)
    uc2.um_table = uc2.um_table * factor
    return uc2


def eval_config(SD, uc, sid, seed, cfg, band_kerns=None):
    """单场景单 seed 的消融明细。cfg 见 main 的配置表。返回 DataFrame 或 None。
    band_kerns 非空 -> full 配置用分带块 GP（主方案）；None -> 单 GP（对照）。"""
    A, B = SD.ab(sid, seed)
    dsA, dsB = SD.build(sid, A), SD.build(sid, B)
    Xq, y, zq = dsB["X4"], dsB["d"], dsB["z"]
    season = dsB["season"]
    season_arr = np.full(len(zq), season)
    dh, _ = SceneData.dist_to_inputs(Xq, dsA["X4"])
    tag = cfg["tag"]

    if cfg["kind"] == "full":
        pred, var = banded.predict_main(SD, dsA, Xq, band_kerns, mean_q=dsB["mean"])
    elif cfg["kind"] == "no_obs":
        pred = dsB["mean"].astype(float)             # 残差域 = m_θ
        var = np.full(len(zq), SD.kern_for(season).diag_var())
    elif cfg["kind"] == "no_meanet":
        g = gplib.GPFit(dsA["X4"], dsA["d"], dsA["u"],
                        np.zeros(len(dsA["d"])), SD.kern_for(season))
        pred, var = g.predict(Xq, np.zeros(len(Xq)))
    elif cfg["kind"] == "no_bg":
        cbA = dsA["d"] + SD.bg.cs_at(dsA["X4"][:, 3], dsA["z"], dsA["X4"][:, 1],
                                     dsA["X4"][:, 0])
        g = gplib.GPFit(dsA["X4"], cbA, dsA["u"], np.zeros(len(cbA)), cfg["kern"])
        cbq = SD.bg.cs_at(Xq[:, 3], zq, Xq[:, 1], Xq[:, 0])
        pbg, var = g.predict(Xq, np.zeros(len(Xq)))
        pred = pbg - cbq                              # 转回残差域
    elif cfg["kind"] == "async_bin":
        # 时间分箱对齐：场景时间跨度 N_BIN 箱，箱内 3D GP（时间列置零）
        tA, tB = dsA["X4"][:, 3], Xq[:, 3]
        lo, hi = min(tA.min(), tB.min()), max(tA.max(), tB.max())
        edges = np.linspace(lo - 1, hi + 1, N_BIN + 1)
        biA = np.clip(np.digitize(tA, edges) - 1, 0, N_BIN - 1)
        biB = np.clip(np.digitize(tB, edges) - 1, 0, N_BIN - 1)
        pred = np.full(len(zq), np.nan)
        var = np.full(len(zq), np.nan)
        for b in range(N_BIN):
            mA, mB = biA == b, biB == b
            if mA.sum() < 15:                        # 空箱：B 点无预测（NaN 剔除）
                continue
            X4b = dsA["X4"][mA].copy()
            X4b[:, 3] = 0.0
            gb = gplib.GPFit(X4b, dsA["d"][mA], dsA["u"][mA],
                             dsA["mean"][mA], SD.kern_for(season))
            Xqb = Xq[mB].copy()
            Xqb[:, 3] = 0.0
            pred[mB], var[mB] = gb.predict(Xqb, dsB["mean"][mB])
        if np.isfinite(pred).sum() < 0.5 * len(zq):
            return None
    else:
        raise ValueError(cfg["kind"])

    u_ex = uc.extra_std(zq, season_arr)
    u_c = np.sqrt(var + u_ex ** 2)
    k = cfg.get("k", 1.96)
    if cfg.get("use_conformal_k"):
        k = uc.k_at(zq, season_arr, dh)
    err = pred - y
    ok = np.isfinite(err) & np.isfinite(u_c)
    if ok.sum() < 0.5 * len(zq):
        return None
    if not isinstance(k, np.ndarray):
        k = np.full(len(zq), float(k))
    kk = k[ok]
    return pd.DataFrame({
        "sid": sid, "seed": seed, "config": tag, "band": dsB["band"][ok],
        "prof": dsB["prof"][ok],
        "z": zq[ok], "season": season, "err": err[ok], "u_c": u_c[ok],
        "u_r": np.sqrt(var[ok]), "u_m": uc.um_at(zq[ok], season_arr[ok]), "k": kk,
        "covered": np.abs(err[ok]) <= kk * u_c[ok],
        "width": 2 * kk * u_c[ok],
        "err2": err[ok] ** 2, "abserr": np.abs(err[ok])})


def summarize(df):
    grp = df.groupby(["sid", "seed", "prof", "band"])
    agg = grp.agg(cov=("covered", "mean"), e2=("err2", "mean"),
                  w=("width", "mean")).reset_index()
    m = agg.groupby("sid").agg(cov=("cov", "mean"), mse=("e2", "mean"),
                               w=("w", "mean")).reset_index()
    return {"config": df["config"].iloc[0], "n": len(df), "n_scenes": df["sid"].nunique(),
            "rmse": float(np.sqrt(m["mse"].mean())),
            "cover": float(m["cov"].mean()),
            "width_mean": float(m["w"].mean()),
            "u_r_mean": float(df["u_r"].mean())}


def train_introspection_ensemble(SD, scenes):
    """⑤ deep ensemble 内省：5×m_θ（同 final 配置、异种子）。
    返回 (members, stdrs, fcols)。"""
    with open(os.path.join(C.WEIGHTS, "exp3", "meanet_final.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    fcols, width, depth = cfg["feat_cols"], cfg["width"], cfg["depth"]
    argo, res = SD.argo, SD.res
    from exp.lib.scenes import prof_scene_index
    ps = prof_scene_index(scenes, argo.n_prof)
    prof_of_obs = np.repeat(np.arange(argo.n_prof), argo.counts)
    train_sids = scenelib.scene_ids_by_role(scenes, "train")
    with open(os.path.join(C.RESULTS, "exp3", "val_scene_ids.json"), encoding="utf-8") as f:
        val_sids = set(json.load(f))
    tr_sids = [s for s in train_sids if s not in val_sids]
    dtr = _anchor_ds(SD, ps, prof_of_obs, tr_sids)
    dva = _anchor_ds(SD, ps, prof_of_obs, list(val_sids))
    members, stdrs = [], []
    for mi in range(N_ENSEMBLE):
        m, s, h, _ = meanet.train_meanet(
            dtr["X"], dtr["d"], dtr["u"], dva["X"], dva["d"], dva["u"],
            feat_cols=fcols, width=width, depth=depth, seed=300 + mi,
            max_epoch=60, patience=6)
        members.append(m)
        stdrs.append(s)
        print(f"    ens member {mi}: val_wrmse={h['best']:.4f}")
    return members, stdrs, fcols


def _anchor_ds(SD, prof_scene, prof_of_obs, sids):
    """场景集的锚点数据集（exp3 同口径）。"""
    sset = set(int(s) for s in sids)
    sel_prof = np.where([prof_scene[i] in sset for i in range(SD.argo.n_prof)])[0]
    idx = anchors.anchor_indices(SD.argo, sel_prof)
    prof = prof_of_obs[idx]
    keep = prof_scene[prof] >= 0
    idx, prof = idx[keep], prof[keep]
    z = SD.res["z"][idx].astype(float)
    T = SD.argo.T[idx].astype(float)
    S = SD.argo.S[idx].astype(float)
    p = SD.argo.p[idx].astype(float)
    return {"X": SD.Xfeat[idx], "d": SD.res["d"][idx].astype(float),
            "u": uobs.u_obs_c(T, S, p, z), "z": z,
            "band": uobs.band_index(z), "scene": prof_scene[prof]}


def main():
    t0 = time.time()
    print("== EXP-9: ablation & source decomposition ==")
    stack = load_stack()
    SD = SceneData(stack)
    scenes = stack["scenes"]
    uc = UcComposer(k_json=C.MAIN_K_JSON)
    band_kerns = banded.load_band_kerns()
    test_sids = scenelib.scene_ids_by_role(scenes, "test")
    if len(test_sids) > MAX_SCENES:
        keep = []
        for season in range(4):
            ss = [s for s in test_sids if SD.sbi[s]["season"] == season]
            keep += list(RNG.choice(ss, min(MAX_SCENES // 4, len(ss)), replace=False))
        test_sids = sorted(set(keep))
    print(f"test scenes: {len(test_sids)} x seeds {SEEDS}")

    # no_bg 的核超参（观测域重估，train 子集一次）
    print("-- refit kernel for no_bg (c_obs domain) --")
    tr_sids = scenelib.scene_ids_by_role(scenes, "train")
    sub = list(RNG.choice(tr_sids, min(10, len(tr_sids)), replace=False))
    ds_list = []
    for sid in sub:
        dA = SD.build(int(sid))
        cbA = dA["d"] + SD.bg.cs_at(dA["X4"][:, 3], dA["z"], dA["X4"][:, 1],
                                     dA["X4"][:, 0])
        ds_list.append((dA["X4"], cbA, dA["u"], np.zeros(len(cbA))))
    kern_nb, _, _ = gplib.fit_hyperparams(ds_list, form="prod", n_iter=120)
    print(f"  no_bg kern: {kern_nb.to_dict()}")

    cfgs = [
        {"tag": "full", "kind": "full"},
        {"tag": "no_obs", "kind": "no_obs"},
        {"tag": "no_meanet", "kind": "no_meanet"},
        {"tag": "no_bg", "kind": "no_bg", "kern": kern_nb},
        {"tag": "async_bin", "kind": "async_bin"},
        {"tag": "full_conformal", "kind": "full", "use_conformal_k": True},
    ]

    # ---------- ① 消融 + ③ u_m 缩放 ----------
    print("-- ablation loop (incl. u_m scaling) --")
    scale_cfgs = {0.0: "um_x0", 0.5: "um_x05", 1.0: None, 2.0: "um_x2", 4.0: "um_x4"}
    dfs = []
    for sid in test_sids:
        for seed in SEEDS:
            for cfg in cfgs:
                df = eval_config(SD, uc, sid, seed, cfg, band_kerns)
                if df is not None:
                    dfs.append(df)
            # u_m 缩放（full 管线，k=1.96 固定以隔离 u_m 的边际效应）
            for f, tag in scale_cfgs.items():
                if tag is None:
                    continue
                uc2 = scaled_uc(uc, f)
                df = eval_config(SD, uc2, sid, seed, {"tag": tag, "kind": "full"}, band_kerns)
                if df is not None:
                    dfs.append(df)
        print(f"  scene {sid} done ({time.time() - t0:.0f}s)")
    all_df = pd.concat(dfs, ignore_index=True)
    all_df.to_csv(os.path.join(OUT, "ablation_detail.csv"), index=False)

    configs = sorted(all_df["config"].unique())
    mrows = [summarize(all_df[all_df["config"] == c]) for c in configs]
    mdf = pd.DataFrame(mrows)
    mdf.to_csv(os.path.join(OUT, "ablation_metrics.csv"), index=False)
    print(mdf.round(4).to_string(index=False))

    # 覆盖率单调性（u_m 缩放）
    sc = mdf[mdf["config"].isin(["um_x0", "um_x05", "full", "um_x2", "um_x4"])]
    sc = sc.set_index("config").reindex(["um_x0", "um_x05", "full", "um_x2", "um_x4"])
    mono = bool((sc["cover"].diff().dropna() > 0).all())
    print(f"u_m scaling coverage monotonic: {mono}")

    # u_r 对比（no_obs vs full）
    ur_noobs = float(mdf[mdf["config"] == "no_obs"]["u_r_mean"].iloc[0])
    ur_full = float(mdf[mdf["config"] == "full"]["u_r_mean"].iloc[0])

    # ---------- ⑤ 内省对照 ----------
    print("-- deep ensemble introspection --")
    members, stdrs, fcols = train_introspection_ensemble(SD, scenes)
    intro_rows = []
    for sid in test_sids[:10]:
        _, B = SD.ab(sid, 0)
        dsB = SD.build(sid, B)
        preds = np.stack([meanet.predict(m, s, dsB["feat"], fcols)
                          for m, s in zip(members, stdrs)])
        std_ens = preds.std(0)
        season = dsB["season"]
        um_tab = uc.um_at(dsB["z"], np.full(len(dsB["z"]), season))
        for bi in range(6):
            m = dsB["band"] == bi
            if m.sum() < 20:
                continue
            intro_rows.append({"sid": sid, "band": bi, "season": season,
                               "n": int(m.sum()),
                               "std_ens_mean": float(std_ens[m].mean()),
                               "um_table_mean": float(np.mean(um_tab[m]))})
    intro_df = pd.DataFrame(intro_rows)
    intro_df.to_csv(os.path.join(OUT, "introspection.csv"), index=False)
    intro_band = intro_df.groupby("band")[["std_ens_mean", "um_table_mean"]].mean()
    ratio = float((intro_df["std_ens_mean"] / intro_df["um_table_mean"]).mean())
    print(intro_band.round(4).to_string())
    print(f"introspection/um ratio (mean): {ratio:.3f}")

    # ---------- 汇总 ----------
    summary = {
        "exp": "exp9", "date": time.strftime("%Y-%m-%d %H:%M"),
        "n_test_scenes": len(test_sids), "seeds": list(SEEDS),
        "ablation": mdf.to_dict("records"),
        "um_scaling_monotonic": mono,
        "um_scaling_cover": {c: float(sc.loc[c, "cover"]) for c in sc.index},
        "ur_no_obs_vs_full": {"no_obs": ur_noobs, "full": ur_full},
        "no_bg_kern": kern_nb.to_dict(),
        "introspection_band": intro_band.round(4).reset_index().to_dict("records"),
        "introspection_um_ratio_mean": ratio,
        "note_sparse_scan_ref": "results/exp7/sparse_scan.csv（稀疏->宽度单调性）",
        "elapsed_s": time.time() - t0,
    }
    with open(os.path.join(OUT, "exp9_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    print("EXP-9 DONE", time.time() - t0)


if __name__ == "__main__":
    main()
