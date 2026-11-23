# -*- coding: utf-8 -*-
"""EXP-6 独立验收：实测场景（WP3，framework §4-EXP6）。
① test 场景 × 重复划分（seeds 0..4）系统验收：GP 在留出 B 点直接评估（D15）
② 指标：经验覆盖率（总体+分场景+分深度带）、区间宽度（均值+分位数）、
   Winkler score、RMSE/MAE；验证点按 站×深度带 聚合计权 + 点级口径并列
③ 场景级 bootstrap（1000 次）→ 指标 95% CI
④ 覆盖率 vs 二项容差带（J2）
⑤ CTD 72 条纯外部验证（锚定场景优先；同一指标体系，独立报告）
产出：results/exp6/*.csv + exp6_summary.json
"""
import sys, os, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from exp import config as C
from exp.lib import io_data, physics, uobs, features, meanet, scenes as scenelib
from exp.lib.pipeline import (SceneData, load_stack, coverage_tolerance,
                              km_dist, season_from_t64)
from exp.lib.uc import UcComposer

OUT = C.result_dir("exp6")
RNG = np.random.default_rng(C.SEED)
EPOCH = np.datetime64("2000-01-01T00:00:00").astype(np.int64)
N_BOOT = 1000


def run_scene(SD, uc, sid, seed):
    """单场景单 seed 的 B 点明细（含 u_c 分解与 k 区间）。"""
    A, B = SD.ab(sid, seed)
    dsA, dsB = SD.build(sid, A), SD.build(sid, B)
    g = SD.fit_gp(dsA)
    pred, var = g.predict(dsB["X4"], dsB["mean"])
    z, season = dsB["z"], dsB["season"]
    u_extra = uc.extra_std(z, season)
    u_c = np.sqrt(var + u_extra ** 2)
    dh, _ = SD.dist_to_inputs(dsB["X4"], dsA["X4"])
    k = uc.k_at(z, season, dh)
    err = pred - dsB["d"]
    return pd.DataFrame({
        "sid": sid, "seed": seed, "prof": dsB["prof"], "band": dsB["band"],
        "z": z, "season": season, "err": err, "u_c": u_c, "u_r": np.sqrt(var),
        "u_extra": u_extra, "k": k, "covered": np.abs(err) <= k * u_c,
        "width": 2 * k * u_c, "err2": err ** 2,
        "abserr": np.abs(err), "sparse_km": dh})


def weighted_metrics(df, label=""):
    """站×深度带聚合计权 + 点级口径（Winkler 向量化）。"""
    df = df.copy()
    half = df["k"] * df["u_c"]
    w = 2 * half
    df["wink"] = np.where(df["err"] < -half, w + 2 / 0.05 * (-half - df["err"]),
                          np.where(df["err"] > half,
                                   w + 2 / 0.05 * (df["err"] - half), w))
    grp = df.groupby(["prof", "band"])
    return {"label": label, "n": len(df), "n_groups": int(grp.ngroups),
            "cover_weighted": float(grp["covered"].mean().mean()),
            "rmse_weighted": float(np.sqrt(grp["err2"].mean().mean())),
            "winkler_weighted": float(grp["wink"].mean().mean()),
            "cover_pointwise": float(df["covered"].mean()),
            "rmse_pointwise": float(np.sqrt(df["err2"].mean())),
            "mae_pointwise": float(df["abserr"].mean()),
            "width_mean": float(df["width"].mean()),
            "width_p50": float(df["width"].quantile(0.5)),
            "width_p95": float(df["width"].quantile(0.95))}


def scene_bootstrap(df, n_boot=N_BOOT):
    """场景级 bootstrap：覆盖率/RMSE 的 95% CI（点级口径，聚类稳健）。"""
    sids = df["sid"].unique()
    grouped = {s: df[df["sid"] == s][["covered", "err2"]].values for s in sids}
    n_pts = {s: len(grouped[s]) for s in sids}
    covs, rmses = [], []
    for _ in range(n_boot):
        pick = RNG.choice(sids, len(sids), replace=True)
        cov = np.concatenate([grouped[s][:, 0] for s in pick])
        e2 = np.concatenate([grouped[s][:, 1] for s in pick])
        covs.append(cov.mean())
        rmses.append(np.sqrt(e2.mean()))
    return {"cover_ci95": [float(np.quantile(covs, 0.025)),
                           float(np.quantile(covs, 0.975))],
            "rmse_ci95": [float(np.quantile(rmses, 0.025)),
                          float(np.quantile(rmses, 0.975))]}


def _predict_ctd_station(SD, uc, g, dsA, ctd, j, sid):
    """GP 拟合完成后对单个 CTD 站的预测明细 DataFrame。"""
    sl = ctd.slice(j)
    zz = physics.z_from_p(ctd.p[sl], ctd.lat[j])
    tt = np.full(len(zz), float(ctd.t64[j] - EPOCH))
    Xq = np.column_stack([np.full(len(zz), ctd.lon[j]),
                          np.full(len(zz), ctd.lat[j]), zz, tt])
    # query_features 只给残差域均值 m_θ；绝对声速须再加背景声速 c_b
    # （d = c_obs − c_b 为残差域，原写法 pred−c_true 差 ~1500 m/s，日志【问题26】）
    Xf, _ = features.make_features(SD.bg, Xq[:, 3].astype(np.int64), zz,
                                   Xq[:, 1], Xq[:, 0])
    mq = meanet.predict(SD.model, SD.stdr, Xf, SD.fcols)
    cb_q = Xf[:, features.FEAT_NAMES.index("cb")]
    pred_res, var = g.predict(Xq, mq)
    pred = cb_q + pred_res
    c_true = physics.sound_speed(ctd.T[sl].astype(float), ctd.S[sl].astype(float),
                                 ctd.p[sl].astype(float), C.SOUND_SPEED_EQ)
    season = int(season_from_t64(np.array([ctd.t64[j]]))[0])
    u_extra = uc.extra_std(zz, np.full(len(zz), season))
    u_c = np.sqrt(var + u_extra ** 2)
    dh, _ = SD.dist_to_inputs(Xq, dsA["X4"])
    k = uc.k_at(zz, np.full(len(zz), season), dh)
    err = pred - c_true
    z_ok = zz <= C.Z_MAX
    # prof 列与 test 场景口径对齐（weighted_metrics 按 站×深度带 聚合；
    # CTD 无剖面索引，以站号 j 充当分组键——缺此列触发 KeyError: 'prof'）
    return pd.DataFrame({
        "station": j, "prof": j, "sid": sid, "z": zz, "band": uobs.band_index(zz),
        "season": season, "err": err, "u_c": u_c, "k": k,
        "covered": (np.abs(err) <= k * u_c) & z_ok,
        "width": 2 * k * u_c, "err2": err ** 2, "abserr": np.abs(err),
        "sparse_km": dh})


def ctd_validation(SD, uc, argo):
    """CTD 72 条纯外部验证。
    主口径：锚定场景内（±15 d、空间块内）的 CTD 站，A 输入重构 -> CTD 点预测；
    扩展口径：其余站以时空邻域剖面（±15 d/±250 km，至多 12 条）在线模式。"""
    ctd = io_data.load_ctd()
    scenes = SD.sbi
    with open(os.path.join(C.RESULTS, "exp2", "scenes_meta.json"), encoding="utf-8") as f:
        anchored = [int(x) for x in json.load(f)["anchored_scene_ids"]]

    # 主口径：按锚定场景分组 CTD 站（每场景一次 GP 拟合）
    rows = []
    used_stations = set()
    for sid in anchored:
        s = scenes[int(sid)]
        mem_t = argo.t64[np.array(s["member_idx"])]
        t0, t1 = mem_t.min(), mem_t.max()
        stations = []
        for j in range(ctd.n_prof):
            if j in used_stations:
                continue
            in_space = (s["lon0"] <= ctd.lon[j] <= s["lon1"]
                        and s["lat0"] <= ctd.lat[j] <= s["lat1"])
            in_time = t0 - 86400 * 15 <= ctd.t64[j] <= t1 + 86400 * 15
            if in_space and in_time:
                stations.append(j)
        if not stations:
            continue
        A, _ = SD.ab(sid, seed=0)
        dsA = SD.build(sid, A)
        g = SD.fit_gp(dsA)
        for j in stations:
            used_stations.add(j)
            rows.append(_predict_ctd_station(SD, uc, g, dsA, ctd, j, sid))
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    # 扩展口径：时空邻域输入（在线模式）
    ext_rows = []
    for j in range(ctd.n_prof):
        if j in used_stations:
            continue
        dt_days = np.abs(argo.t64 - ctd.t64[j]) / 86400.0
        near = np.where(dt_days <= 15)[0]
        if len(near) < 5:
            continue
        dist = km_dist(argo.lat[near], argo.lon[near], ctd.lat[j], ctd.lon[j])
        # dist 须与 near 同步筛选——原写法只筛 near 不筛 dist，
        # argsort(dist) 返回筛选前索引，越界（日志【问题25】）
        keep = dist <= 250
        near, dist = near[keep], dist[keep]
        if len(near) < 5:
            continue
        near = near[np.argsort(dist)[:12]]
        dsA = SD.build_from_profiles(near)
        g = SD.fit_gp(dsA)
        ext_rows.append(_predict_ctd_station(SD, uc, g, dsA, ctd, j, -1))
    ext_df = pd.concat(ext_rows, ignore_index=True) if ext_rows else pd.DataFrame()
    return df, ext_df


def main():
    t0 = time.time()
    print("== EXP-6: acceptance on test scenes + CTD external validation ==")
    stack = load_stack()
    SD = SceneData(stack)
    scenes = stack["scenes"]
    uc = UcComposer()
    argo = stack["argo"]
    test_sids = scenelib.scene_ids_by_role(scenes, "test")
    print(f"test scenes: {len(test_sids)}")

    # ①-② test 场景 × 5 seeds（明细已落盘则直接复用，避免重跑 ~18 万点；
    # 删除 test_B_detail.csv 即强制重算）
    detail_path = os.path.join(OUT, "test_B_detail.csv")
    if os.path.exists(detail_path):
        all_df = pd.read_csv(detail_path)
        print(f"-- reused cached test detail: {len(all_df)} points --")
    else:
        print("-- running test scenes x 5 seeds --")
        dfs = []
        for sid in test_sids:
            for seed in range(5):
                dfs.append(run_scene(SD, uc, sid, seed))
        all_df = pd.concat(dfs, ignore_index=True)
        all_df.to_csv(detail_path, index=False)
        print(f"  B points total: {len(all_df)}")

    main_m = weighted_metrics(all_df, "all_test")
    tol_lo, tol_hi = coverage_tolerance(len(all_df))
    main_m["coverage_tolerance_band"] = [tol_lo, tol_hi]
    by_scene = [weighted_metrics(all_df[all_df["sid"] == s], str(s))
                for s in all_df["sid"].unique()]
    by_band = [weighted_metrics(all_df[all_df["band"] == b], f"band{b}")
               for b in sorted(all_df["band"].unique())]
    by_season = [weighted_metrics(all_df[all_df["season"] == s], f"season{s}")
                 for s in sorted(all_df["season"].unique())]
    pd.DataFrame([main_m]).to_csv(os.path.join(OUT, "main_metrics.csv"), index=False)
    pd.DataFrame(by_scene).to_csv(os.path.join(OUT, "metrics_by_scene.csv"), index=False)
    pd.DataFrame(by_band).to_csv(os.path.join(OUT, "metrics_by_band.csv"), index=False)
    pd.DataFrame(by_season).to_csv(os.path.join(OUT, "metrics_by_season.csv"), index=False)
    print(f"  cover(w)={main_m['cover_weighted']:.3f} "
          f"[tol {tol_lo:.3f},{tol_hi:.3f}], rmse(w)={main_m['rmse_weighted']:.3f}, "
          f"width={main_m['width_mean']:.2f}, winkler={main_m['winkler_weighted']:.2f}")

    # ③ bootstrap
    print("-- scene bootstrap --")
    boot = scene_bootstrap(all_df)
    # ④ 分场景覆盖偏离检查（系统性偏离 = 超容差带 + 2% 松弛）
    scen_dev = [r for r in by_scene
                if not (tol_lo - 0.02 <= r["cover_weighted"] <= tol_hi + 0.02)]
    print(f"  bootstrap: cover CI={boot['cover_ci95']}, rmse CI={boot['rmse_ci95']}")
    print(f"  scenes outside tolerance(+2% slack): {len(scen_dev)}/{len(by_scene)}")

    # ⑤ CTD 外部验证
    print("-- CTD 72-station external validation --")
    ctd_df, ctd_ext = ctd_validation(SD, uc, argo)
    ctd_m = weighted_metrics(ctd_df, "ctd_anchored") if len(ctd_df) else None
    ctd_ext_m = weighted_metrics(ctd_ext, "ctd_adhoc") if len(ctd_ext) else None
    if ctd_m:
        n = len(ctd_df)
        ctd_m["coverage_tolerance_band"] = list(coverage_tolerance(n))
        ctd_df.to_csv(os.path.join(OUT, "ctd_validation_detail.csv"), index=False)
        pd.DataFrame([ctd_m]).to_csv(os.path.join(OUT, "ctd_metrics.csv"), index=False)
        print(f"  anchored CTD: n={n}, stations={ctd_df['station'].nunique()}, "
              f"cover(w)={ctd_m['cover_weighted']:.3f}, rmse={ctd_m['rmse_pointwise']:.3f}")
    if ctd_ext_m:
        ctd_ext_m["coverage_tolerance_band"] = list(coverage_tolerance(len(ctd_ext)))
        ctd_ext.to_csv(os.path.join(OUT, "ctd_adhoc_detail.csv"), index=False)
        pd.DataFrame([ctd_ext_m]).to_csv(
            os.path.join(OUT, "ctd_adhoc_metrics.csv"), index=False)
        print(f"  adhoc CTD: n={len(ctd_ext)}, stations={ctd_ext['station'].nunique()}, "
              f"cover(w)={ctd_ext_m['cover_weighted']:.3f}")

    summary = {
        "exp": "exp6", "date": time.strftime("%Y-%m-%d %H:%M"),
        "n_test_scenes": len(test_sids), "n_seeds": 5,
        "n_B_points": int(len(all_df)),
        "main": main_m, "bootstrap": boot,
        "n_scenes_outside_tolerance": len(scen_dev),
        "scenes_outside_tolerance_ids": [r["label"] for r in scen_dev],
        "by_band": by_band, "by_season": by_season,
        "ctd_anchored": ctd_m, "ctd_adhoc": ctd_ext_m,
        "elapsed_s": time.time() - t0,
    }
    with open(os.path.join(OUT, "exp6_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    print("EXP-6 DONE", time.time() - t0)


if __name__ == "__main__":
    main()
