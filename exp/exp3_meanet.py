# -*- coding: utf-8 -*-
"""EXP-3 均值网络训练与 u_m 随机效应评定（WP2，framework §3.3.1/§3.3.4）。
① 特征/结构轻量消融（锚点降采样点集，GPU）
② held-out（calib 场景）评估：点误差 vs 零均值基线、方差削减、条件相关诊断
③ u_m：分格（深度带×季节）跨场景随机效应方差分解（ISO 5725 口径）
   组内=σ²(η+ε) / 组间=b²(模型偏差)；收缩 + 卡方上界 + 不显著不取零
④ OOD 加强：leave-one-season-out / leave-one-region-out 重训重评 -> 两档对照表
⑤ σ² 组内估计存档（EXP-4 双路 σ² 交叉验证输入）
产出：results/exp3/*.csv + weights/exp3/meanet_final.* + exp3_summary.json
"""
import sys, os, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.stats import chi2

from exp import config as C
from exp.lib import io_data, features, anchors, uobs, meanet, scenes as scenelib
from exp.lib.interp import BgField

OUT = C.result_dir("exp3")
WDIR = os.path.join(C.WEIGHTS, "exp3")
os.makedirs(WDIR, exist_ok=True)
EPOCH = np.datetime64("2000-01-01T00:00:00").astype(np.int64)
UM_FLOOR = 0.05  # m/s，u_m 全表下限（防格值 0 使 u_c 缺模型误差分量）

FEAT_SETS = {
    "F0_coord": [0, 1, 2],
    "F1_coord_season": [0, 1, 2, 3, 4],
    "F2_plus_cb": [0, 1, 2, 3, 4, 5],
    "F3_full": [0, 1, 2, 3, 4, 5, 6, 7],
}
STRUCTS = {"S64x4": (64, 4), "S128x4": (128, 4), "S128x6": (128, 6)}


# ------------------------------------------------------------ 数据准备 ----
def load_all():
    argo = io_data.load_argo()
    bg = BgField(C.CACHE)
    res = np.load(os.path.join(C.CACHE, "argo_residuals.npz"))
    feat = features.build_obs_features(argo, bg)   # 全量特征（有缓存机制）
    X = feat["X"]
    scenes, meta = scenelib.load_scenes()
    prof_scene = scenelib.prof_scene_index(scenes, argo.n_prof)
    prof_of_obs = np.repeat(np.arange(argo.n_prof), argo.counts)
    return argo, res, X, scenes, meta, prof_scene, prof_of_obs


def anchor_dataset(argo, res, X, prof_scene, prof_of_obs, scene_ids=None, roles=None,
                   scenes=None):
    """按场景集（scene_ids）或角色（roles）构建锚点降采样数据集。
    返回 dict: X, d, u, z, band, season, scene_id, prof_id"""
    sel_prof = []
    if scene_ids is not None:
        sset = set(scene_ids)
        sel_prof = np.where([prof_scene[i] in sset for i in range(argo.n_prof)])[0]
    elif roles is not None:
        sid_by_role = {}
        for s in scenes:
            sid_by_role.setdefault(s["role"], []).append(int(s["scene_id"]))
        want = set()
        for r in roles:
            want |= set(sid_by_role.get(r, []))
        sel_prof = np.where(np.isin(prof_scene, list(want)))[0]
    idx = anchors.anchor_indices(argo, sel_prof)
    prof = prof_of_obs[idx]
    sc = prof_scene[prof]
    keep = sc >= 0
    idx, prof, sc = idx[keep], prof[keep], sc[keep]
    z = res["z"][idx].astype(float)
    d = res["d"][idx].astype(float)
    T = argo.T[idx].astype(float); S = argo.S[idx].astype(float); p = argo.p[idx].astype(float)
    u = uobs.u_obs_c(T, S, p, z)
    return {"X": X[idx], "d": d, "u": u, "z": z, "band": uobs.band_index(z),
            "season": res["season"][idx], "scene_id": sc, "prof_id": prof}


# ------------------------------------------------------------ 训练 ----
def split_train_val(ds, scenes, val_frac=0.2, seed=C.SEED):
    """场景级 train/val 划分（季节分层）。返回 train/val 两个点级子集。"""
    rng = np.random.default_rng(seed)
    uniq = np.unique(ds["scene_id"])
    sbi = scenelib.scene_by_id(scenes)
    val_sids = []
    for season in range(4):
        ss = [s for s in uniq if sbi[int(s)]["season"] == season]
        if len(ss) == 0:
            continue
        k = max(1, int(round(len(ss) * val_frac)))
        val_sids += list(rng.choice(ss, k, replace=False))
    val_sids = set(int(s) for s in val_sids)
    m_val = np.isin(ds["scene_id"], list(val_sids))
    m_tr = ~m_val
    mk = lambda m: {k: v[m] for k, v in ds.items()}
    return mk(m_tr), mk(m_val), val_sids


def wrmse(pred, d, u):
    w = 1.0 / (u ** 2 + 0.02 ** 2)
    return float(np.sqrt(np.mean(w * (pred - d) ** 2) / np.mean(w)))


# ------------------------------------------------------------ u_m 评定 ----
def random_effects_um(e, u, scene_id, band, season, prof_id, tag="indist"):
    """分格跨场景随机效应方差分解（§3.3.4 ③）。
    e=d-m_theta, u=u_obs, 逐点带 scene/band/season/prof。
    返回 (per-cell DataFrame, sigma2_within 长表 for EXP-4)。"""
    def _finalize(b2_point, b2_upper, p_chi2):
        """点估计/上界 -> 最终 b²（不显著取上界、floor 保底）。"""
        use_upper = (p_chi2 is None) or (p_chi2 >= 0.05)
        b2 = max(b2_point, b2_upper) if use_upper else b2_point
        return max(b2, UM_FLOOR ** 2)

    rows = []
    for bi in range(6):
        mb = band == bi
        # 带内池化（空格回退值 + 收缩目标）；池化也不可行时回退 floor
        pooled = _one_cell(e[mb], u[mb], scene_id[mb], prof_id[mb])
        if pooled is not None:
            p_um = float(np.sqrt(_finalize(pooled["b2_point"], pooled["b2_upper"],
                                           pooled["p_chi2"])))
        else:
            p_um = UM_FLOOR
        for se in range(4):
            m = mb & (season == se)
            cell = _one_cell(e[m], u[m], scene_id[m], prof_id[m])
            if cell is None:
                rows.append({"band": bi, "season": se, "tag": tag, "n_scenes": 0,
                             "n_pts": int(m.sum()), "filled": True,
                             "b2_point": pooled["b2_point"] if pooled else None,
                             "b2_upper": pooled["b2_upper"] if pooled else None,
                             "b2_shrunk": p_um ** 2,
                             "sigma2_within": pooled["sigma2_within"] if pooled else None,
                             "um": p_um, "p_chi2": None})
                continue
            n_g = cell["n_scenes"]
            if pooled is not None:
                b2_shrunk = (n_g * cell["b2_point"] + 3.0 * pooled["b2_point"]) / (n_g + 3.0)
            else:
                b2_shrunk = cell["b2_point"]
            use_upper = (cell["p_chi2"] is None) or (cell["p_chi2"] >= 0.05)
            b2_final = max(b2_shrunk, cell["b2_upper"]) if use_upper else b2_shrunk
            b2_final = max(b2_final, UM_FLOOR ** 2)
            rows.append({"band": bi, "season": se, "tag": tag, "n_scenes": n_g,
                         "n_pts": int(m.sum()), "filled": False,
                         "b2_point": cell["b2_point"], "b2_upper": cell["b2_upper"],
                         "b2_shrunk": b2_shrunk,
                         "sigma2_within": cell["sigma2_within"],
                         "um": float(np.sqrt(b2_final)), "p_chi2": cell["p_chi2"]})
    df = pd.DataFrame(rows)
    # 组内方差长表（供 EXP-4 双路交叉验证）：band×season 的 σ²(η+ε)
    sw = df[["band", "season", "sigma2_within", "n_scenes", "tag"]].copy()
    return df, sw


def _one_cell(e, u, scene_id, prof_id):
    """单格两层分解：组内 σ²(η+ε)（扣 û²）、组间 b²（场景均值离散 - 抽样噪声）。"""
    if len(e) < 20:
        return None
    us, inv = np.unique(scene_id, return_inverse=True)
    n_g = len(us)
    if n_g < 2:
        return None
    means, within, npts, nprof, u2bar = [], [], [], [], []
    for si in range(n_g):
        m = inv == si
        n = int(m.sum())
        if n < 5:
            continue
        means.append(float(e[m].mean()))
        within.append(float(e[m].var(ddof=1) - np.mean(u[m] ** 2)))
        npts.append(n)
        nprof.append(len(np.unique(prof_id[m])))
        u2bar.append(float(np.mean(u[m] ** 2)))
    if len(means) < 2:
        return None
    means = np.array(means); within = np.array(within)
    npts = np.array(npts); nprof = np.array(nprof, dtype=float); u2bar = np.array(u2bar)
    sigma2_within = float(np.mean(within))
    V_obs = float(means.var(ddof=1))
    # 抽样噪声：以剖面为独立单元（点内相关保守保留在组内方差中）
    sigma0_2 = float(np.mean((np.maximum(sigma2_within, 0) + u2bar) / nprof))
    b2_point = max(0.0, V_obs - sigma0_2)
    if sigma0_2 > 0 and n_g >= 3:
        stat = (n_g - 1) * V_obs / sigma0_2
        p = float(chi2.sf(stat, n_g - 1))
        b2_upper = max(0.0, (n_g - 1) * V_obs / chi2.ppf(0.05, n_g - 1) - sigma0_2)
    else:
        p = None
        b2_upper = b2_point
    return {"n_scenes": n_g, "sigma2_within": sigma2_within, "b2_point": b2_point,
            "b2_upper": b2_upper, "p_chi2": p}


def cond_diagnostics(e, ds):
    """held-out 残差条件相关诊断（诊断优先原则：系统相关 -> 补特征）。"""
    names = features.FEAT_NAMES
    rows = []
    for j, name in enumerate(names):
        x = ds["X"][:, j].astype(float)
        m = np.isfinite(x) & np.isfinite(e)
        if m.sum() < 100:
            continue
        corr = float(np.corrcoef(x[m], e[m])[0, 1])
        # 分箱均值趋势（10 分位箱）
        qs = np.quantile(x[m], np.linspace(0, 1, 11))
        qs[0] -= 1e-9
        binned = np.digitize(x[m], qs[1:-1])
        bmean = [float(e[m][binned == k].mean()) for k in range(10)
                 if (binned == k).sum() > 20]
        spread = max(bmean) - min(bmean) if len(bmean) >= 2 else 0.0
        rows.append({"feature": name, "pearson_r": corr, "bin_mean_spread_ms": spread,
                     "flag": bool(abs(corr) > 0.10 or spread > 0.5)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------ 主流程 ----
def main():
    t0 = time.time()
    print("== EXP-3: mean network & u_m assessment ==")
    argo, res, X, scenes, meta, prof_scene, prof_of_obs = load_all()
    print(f"argo obs={argo.n_obs}, scenes={len(scenes)}, "
          f"roles={pd.Series([s['role'] for s in scenes]).value_counts().to_dict()}")

    # 训练池：train 场景，锚点降采样
    ds_train = anchor_dataset(argo, res, X, prof_scene, prof_of_obs, roles=["train"],
                              scenes=scenes)
    ds_calib = anchor_dataset(argo, res, X, prof_scene, prof_of_obs, roles=["calib"],
                              scenes=scenes)
    print(f"anchor pts: train={len(ds_train['d'])}, calib(held-out)={len(ds_calib['d'])}")

    # 场景级 train/val（早停 + 消融选择）
    tr, va, val_sids = split_train_val(ds_train, scenes)
    print(f"scene-level split: train={len(np.unique(tr['scene_id']))} scenes "
          f"({len(tr['d'])} pts), val={len(val_sids)} scenes ({len(va['d'])} pts)")

    base = dict(Xtr=tr["X"], dtr=tr["d"], utr=tr["u"], Xva=va["X"], dva=va["d"],
                uva=va["u"], seed=C.SEED)
    base_rmse = wrmse(np.zeros(len(va["d"])), va["d"], va["u"])  # 零均值基线

    # ① 特征消融（固定结构 S128x4）
    print("-- feature ablation --")
    rows = []
    for name, cols in FEAT_SETS.items():
        m, s, h, _ = meanet.train_meanet(**base, feat_cols=cols, width=128, depth=4)
        pv = meanet.predict(m, s, va["X"], cols)
        rows.append({"feat_set": name, "n_feat": len(cols), "val_wrmse": h["best"],
                     "val_rmse": float(np.sqrt(np.mean((pv - va["d"]) ** 2))),
                     "baseline_wrmse": base_rmse, "gain_pct": 100 * (1 - h["best"] / base_rmse),
                     "n_epoch": h["n_epoch"]})
        print(f"  {name}: val_wrmse={h['best']:.4f} (baseline {base_rmse:.4f})")
    feat_df = pd.DataFrame(rows)
    feat_df.to_csv(os.path.join(OUT, "feat_ablation.csv"), index=False)
    best_feat = feat_df.sort_values("val_wrmse").iloc[0]["feat_set"]
    best_cols = FEAT_SETS[best_feat]
    print(f"best feature set: {best_feat}")

    # ② 结构消融（最优特征）
    print("-- structure ablation --")
    rows = []
    for name, (w, dep) in STRUCTS.items():
        m, s, h, _ = meanet.train_meanet(**base, feat_cols=best_cols, width=w, depth=dep)
        rows.append({"struct": name, "width": w, "depth": dep, "val_wrmse": h["best"],
                     "n_epoch": h["n_epoch"]})
        print(f"  {name}: val_wrmse={h['best']:.4f}")
    struct_df = pd.DataFrame(rows)
    struct_df.to_csv(os.path.join(OUT, "struct_ablation.csv"), index=False)
    best_struct = struct_df.sort_values("val_wrmse").iloc[0]["struct"]
    bw, bd = STRUCTS[best_struct]
    print(f"best struct: {best_struct}")

    # ③ final 模型（最优配置，全 train 池训练，val 早停）
    model, stdr, hist, fcols = meanet.train_meanet(
        **base, feat_cols=best_cols, width=bw, depth=bd, verbose=True)
    meanet.save_model(os.path.join(WDIR, "meanet_final"), model, stdr, fcols,
                      extra={"best_feat": best_feat, "best_struct": best_struct,
                             "val_wrmse": hist["best"], "sigma_floor": hist["floor"],
                             "date": time.strftime("%Y-%m-%d")})
    print(f"final model saved (val_wrmse={hist['best']:.4f})")

    # ④ held-out（calib 场景）评估
    print("-- held-out evaluation (calib scenes) --")
    pred_cal = meanet.predict(model, stdr, ds_calib["X"], fcols)
    e = ds_calib["d"] - pred_cal
    ho_rows = []
    rmse_d = float(np.sqrt(np.mean(ds_calib["d"] ** 2)))
    rmse_e = float(np.sqrt(np.mean(e ** 2)))
    for bi in range(6):
        m = ds_calib["band"] == bi
        if m.sum() < 50:
            continue
        ho_rows.append({"band": uobs.ZB_NAMES[bi], "n": int(m.sum()),
                        "rmse_d_baseline": float(np.sqrt(np.mean(ds_calib["d"][m] ** 2))),
                        "rmse_e": float(np.sqrt(np.mean(e[m] ** 2))),
                        "var_reduction_pct": 100 * (1 - np.var(e[m]) / np.var(ds_calib["d"][m]))})
    ho_df = pd.DataFrame(ho_rows)
    ho_df.to_csv(os.path.join(OUT, "heldout_eval.csv"), index=False)
    print(f"held-out RMSE: baseline(d)={rmse_d:.4f} -> residual(e)={rmse_e:.4f} "
          f"({100*(1-rmse_e**2/rmse_d**2):.1f}% var reduced)")
    diag = cond_diagnostics(e, ds_calib)
    diag.to_csv(os.path.join(OUT, "cond_diag.csv"), index=False)
    if diag["flag"].any():
        print("  [diag] conditional correlation flagged:",
              diag[diag["flag"]]["feature"].tolist())

    # ⑤ u_m 评定（in-dist：final 模型在 calib 场景）
    print("-- u_m random-effects assessment --")
    um_df, sw = random_effects_um(e, ds_calib["u"], ds_calib["scene_id"],
                                  ds_calib["band"], ds_calib["season"], ds_calib["prof_id"])
    um_df.to_csv(os.path.join(OUT, "um_indist.csv"), index=False)
    sw["um_fill"] = None
    print(um_df[["band", "season", "n_scenes", "b2_point", "b2_upper", "um"]]
          .to_string(index=False))

    # ⑥ OOD 加强评定：LOSO / LORO（重训 + 对应 held-out 子集重评）
    print("-- OOD: leave-one-season-out / leave-one-region-out --")
    ood_rows = []
    sbi = scenelib.scene_by_id(scenes)
    for mode in ("season", "region"):
        vals = [0, 1, 2, 3] if mode == "season" else [0, 1]
        for v in vals:
            tr_sids = [int(s["scene_id"]) for s in scenes
                       if s["role"] == "train" and s[mode] != v]
            ca_sids = [int(s["scene_id"]) for s in scenes
                       if s["role"] == "calib" and s[mode] == v]
            if len(tr_sids) < 10 or len(ca_sids) < 2:
                print(f"  {mode}={v}: insufficient scenes (train={len(tr_sids)}, "
                      f"calib={len(ca_sids)}), skip")
                continue
            dtr = anchor_dataset(argo, res, X, prof_scene, prof_of_obs,
                                 scene_ids=tr_sids)
            dca = anchor_dataset(argo, res, X, prof_scene, prof_of_obs,
                                 scene_ids=ca_sids)
            ttr, tva, _ = split_train_val(dtr, scenes, seed=C.SEED)
            mo, so, _, _ = meanet.train_meanet(
                Xtr=ttr["X"], dtr=ttr["d"], utr=ttr["u"], Xva=tva["X"],
                dva=tva["d"], uva=tva["u"], feat_cols=best_cols, width=bw,
                depth=bd, seed=C.SEED)
            pc = meanet.predict(mo, so, dca["X"], best_cols)
            ec = dca["d"] - pc
            cells, _ = random_effects_um(ec, dca["u"], dca["scene_id"], dca["band"],
                                         dca["season"], dca["prof_id"], tag=f"{mode}{v}")
            for _, r in cells.iterrows():
                ood_rows.append({"mode": mode, "held": int(v), "band": int(r["band"]),
                                 "season": int(r["season"]), "n_scenes": r["n_scenes"],
                                 "um_ood": r["um"],
                                 "um_indist": float(um_df[(um_df["band"] == r["band"]) &
                                                           (um_df["season"] == r["season"])]["um"].iloc[0]),
                                 "rmse_e_ood": float(np.sqrt(np.mean(ec[dca["band"] == r["band"]] ** 2)))})
            print(f"  {mode}={v}: train_scenes={len(tr_sids)}, held_calib={len(ca_sids)}, "
                  f"mean um_ood={np.mean([r['um_ood'] for r in ood_rows[-24:]]) if ood_rows else 0:.3f}")
    ood_df = pd.DataFrame(ood_rows)
    ood_df.to_csv(os.path.join(OUT, "um_ood.csv"), index=False)

    # ⑦ σ² 组内存档（EXP-4 交叉验证）+ val 场景清单存档（EXP-4 复用同协议）
    with open(os.path.join(OUT, "val_scene_ids.json"), "w", encoding="utf-8") as f:
        json.dump(sorted(int(v) for v in val_sids), f)
    sw.to_csv(os.path.join(OUT, "sigma2_within.csv"), index=False)

    # 汇总
    summary = {
        "exp": "exp3", "date": time.strftime("%Y-%m-%d %H:%M"),
        "n_anchor_train": int(len(ds_train["d"])),
        "n_anchor_calib": int(len(ds_calib["d"])),
        "val_scene_split": {"n_val_scenes": len(val_sids)},
        "baseline_val_wrmse": base_rmse,
        "feat_ablation": feat_df.to_dict("records"),
        "struct_ablation": struct_df.to_dict("records"),
        "best_feat": best_feat, "best_struct": best_struct,
        "heldout": {"rmse_baseline": rmse_d, "rmse_residual": rmse_e,
                    "var_reduction_pct": 100 * (1 - rmse_e ** 2 / rmse_d ** 2),
                    "by_band": ho_df.to_dict("records")},
        "cond_diag_flagged": diag[diag["flag"]]["feature"].tolist(),
        "um_indist_mean": float(um_df["um"].mean()),
        "um_indist_by_band": {str(k): float(v) for k, v in
                              um_df.groupby("band")["um"].mean().round(4).items()},
        "um_ood_summary": ({f"{m}_held{h}": float(v) for (m, h), v in
                            ood_df.groupby(["mode", "held"])["um_ood"].mean()
                            .round(4).items()} if len(ood_df) else {}),
        "um_floor_ms": UM_FLOOR,
        "uobs_source": uobs.load_uobs()[2],
        "elapsed_s": time.time() - t0,
    }
    with open(os.path.join(OUT, "exp3_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("feat_ablation", "struct_ablation")},
                     ensure_ascii=False, indent=2, default=float)[:1500])
    print("EXP-3 DONE", time.time() - t0)


if __name__ == "__main__":
    main()
