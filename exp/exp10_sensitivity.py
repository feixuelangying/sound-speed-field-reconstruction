# -*- coding: utf-8 -*-
"""EXP-10 灵敏度与稳健性（WP5，framework §4-EXP10）。
① 声速方程对照：chen-millero / mackenzie / gsw 三方程链路重评
   （背景端=方程差网格场插值修正；观测端=T/S/p 直接换算；m_θ 冻结、
   特征 cb/dcb_dz 列同步修正——"冻结 m_θ 的链路重评"口径，注记于报告）
② 输出深度网格对照：跃层加密网格 vs 均匀网格（u_c 宽度 + 网格插值覆盖损失）
③ 随机种子敏感性 + 重复划分收敛性：seeds 0..9 覆盖率 running-mean 收敛
④ 跨季节/海区子集稳健性分列
⑤ 报告模板实例化：单场景逐点结果 + 来源分解 + 支撑声明 JSON
产出：results/exp10/*.csv|json + exp10_summary.json
"""
import sys, os, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import netCDF4
from scipy.ndimage import distance_transform_edt

from exp import config as C
from exp.lib import physics, uobs, scenes as scenelib, features, anchors
from exp.lib.pipeline import SceneData, load_stack, km_dist
from exp.lib.uc import UcComposer
from exp.lib import banded
from exp.lib.interp import pchip_eval

OUT = C.result_dir("exp10")
RNG = np.random.default_rng(C.SEED)
EPOCH = np.datetime64("2000-01-01T00:00:00").astype(np.int64)
N_EQ_STEPS = 200       # 方程差场构建的时间子采样步数
MAX_SCENES = 20        # 对照实验的场景数
N_SEEDS_CONV = 10      # 收敛性检验种子数


# ------------------------------------------------ 方程差场 ----
class EqDelta:
    """c^eq − c^cm 的网格场（时间子采样均值 + 时间 std）与任意点插值。"""

    def __init__(self, eq, bg, force=False):
        self.eq = eq
        path = os.path.join(OUT, f"eq_delta_{eq}.npz")
        if os.path.exists(path) and not force:
            z = np.load(path)
        else:
            z = self._build(path, bg)
        self.dmean = z["dmean"]           # (8, ny, nx)
        self.dstd = z["dstd"]
        self.depths, self.lat, self.lon = z["depths"], z["lat"], z["lon"]

    def _build(self, path, bg):
        t0 = time.time()
        ds = netCDF4.Dataset(C.HYCOM_NC)
        ds.set_auto_mask(True)
        depths = np.asarray(ds.variables["depth"][:], dtype=float)
        lat = np.asarray(ds.variables["lat"][:], dtype=float)
        lon = np.asarray(ds.variables["lon"][:], dtype=float)
        tv, salt = ds.variables["water_temp"], ds.variables["salinity"]
        nt, nz, ny, nx = tv.shape
        P_grid = np.stack([np.broadcast_to(
            physics.p_from_z(np.full(ny, d), lat)[:, None], (ny, nx)) for d in depths])
        idx = np.unique(np.linspace(5, nt - 6, N_EQ_STEPS).astype(int))
        # 掩膜与读取口径对齐 io_data.build_hycom_cs_cache（日志【问题3/4/5】）：
        # HYCOM mask 随时间变化且温盐不一致 -> 块内温盐联合掩膜并集为固定海陆场，
        # EDT 前景=缺测点。
        # 内存修正（日志【问题37】）：原实现按 chunk 时间块整块 astype("float64")
        # 并 filled，单块 (4077,8,84,63) 峰值 ~22 GB（float64 T+S + filled 副本），
        # 进程被系统静默终止（无 traceback，exit 1）。现按 chunk 边界整块**读取**
        # （保留"一次解压"的效率），但**逐时间步**转 float64 并填充，峰值降至
        # 两个 float32 块 ≈ 5.5 GB。
        chunks = tv.chunking()
        tc = int(chunks[0]) if chunks and chunks[0] != "contiguous" else nt
        acc = np.zeros((nz, ny, nx))
        acc2 = np.zeros((nz, ny, nx))
        n_done = 0
        for c0 in range(0, nt, tc):
            c1 = min(c0 + tc, nt)
            sel = idx[(idx >= c0) & (idx < c1)]
            if len(sel) == 0:
                continue
            Tb = tv[c0:c1]
            Sb = salt[c0:c1]
            mU = (np.ma.getmaskarray(Tb) | np.ma.getmaskarray(Sb)).any(axis=0)
            fill = {}
            for k in range(nz):
                m_k = mU[k]
                if not m_k.any():
                    continue
                assert not m_k.all(), f"eq_delta {self.eq}: layer {k} fully masked"
                _, (iy, ix) = distance_transform_edt(m_k, return_indices=True)
                fill[k] = (iy[m_k], ix[m_k])
            for it in sel:
                j = int(it) - c0
                T = np.ma.filled(Tb[j].astype("float64"), np.nan)
                S = np.ma.filled(Sb[j].astype("float64"), np.nan)
                for k, (sy, sx) in fill.items():
                    m_k = mU[k]
                    T[k][m_k] = T[k][sy, sx]
                    S[k][m_k] = S[k][sy, sx]
                assert np.isfinite(T).all() and np.isfinite(S).all(), \
                    "non-finite after land fill"
                dj = (physics.sound_speed(T, S, P_grid, eq=self.eq)
                      - physics.sound_speed(T, S, P_grid, eq=C.SOUND_SPEED_EQ))
                acc += dj
                acc2 += dj ** 2
            del Tb, Sb
            n_done += len(sel)
            print(f"    eq_delta {self.eq}: block[{c0},{c1}) cum {n_done}/{len(idx)}"
                  f" ({time.time()-t0:.0f}s)", flush=True)
        ds.close()
        dmean = acc / n_done
        dstd = np.sqrt(np.maximum(acc2 / n_done - dmean ** 2, 0))
        np.savez_compressed(path, dmean=dmean, dstd=dstd, depths=depths, lat=lat, lon=lon)
        return np.load(path)

    def at(self, z, lat, lon, want="val"):
        """水平双线性 + PCHIP 深度。want: 'val' | 'gradz'。"""
        D = self.dmean
        ix = np.clip(np.searchsorted(self.lon, lon, side="right") - 1, 0, len(self.lon) - 2)
        iy = np.clip(np.searchsorted(self.lat, lat, side="right") - 1, 0, len(self.lat) - 2)
        fx = np.clip((lon - self.lon[ix]) / (self.lon[ix + 1] - self.lon[ix]), 0, 1)
        fy = np.clip((lat - self.lat[iy]) / (self.lat[iy + 1] - self.lat[iy]), 0, 1)
        V = (D[:, iy, ix] * (1 - fx) * (1 - fy) + D[:, iy, ix + 1] * fx * (1 - fy)
             + D[:, iy + 1, ix] * (1 - fx) * fy
             + D[:, iy + 1, ix + 1] * fx * fy).T          # (N,8)
        if want == "gradz":
            _, dv = pchip_eval(self.depths, V, z, return_deriv=True)
            return dv
        return pchip_eval(self.depths, V, z)


def eval_eq(SD, uc, sid, seed, eq, eqd, band_kerns=None):
    """三方程对照的单场景评估：重建 d^eq 与修正特征，GP 验收（seed 单次）。"""
    A, B = SD.ab(sid, seed)
    dsA0, dsB0 = SD.build(sid, A), SD.build(sid, B)

    def shifted(ds0):
        X4 = ds0["X4"]
        z, lat, lon = X4[:, 2], X4[:, 1], X4[:, 0]
        d_bg = eqd.at(z, lat, lon)
        dz_bg = eqd.at(z, lat, lon, want="gradz")
        # 观测端：锚点 T/S/p。锚点全局索引须按 ds0["prof"] 的出现顺序重建
        # （build 用 ab() 的 permutation 顺序；np.unique 排序会静默错位）
        prof_seq = pd.unique(ds0["prof"])
        idx = anchors.anchor_indices(SD.argo, prof_seq)
        assert np.allclose(SD.res["z"][idx], z), "anchor rebuild mismatch"
        T = SD.argo.T[idx].astype(float)
        S = SD.argo.S[idx].astype(float)
        p = SD.argo.p[idx].astype(float)
        c_eq = physics.sound_speed(T, S, p, eq=eq)
        c_cm = physics.sound_speed(T, S, p, eq=C.SOUND_SPEED_EQ)
        d_obs_delta = c_eq - c_cm
        d_eq = ds0["d"] + d_obs_delta - d_bg
        # 特征修正（cb 列 5、dcb_dz 列 6；grad_h 列 7 保留 cm 口径——δ 水平梯度可忽略）
        feat = ds0["feat"].copy()
        feat[:, 5] = feat[:, 5] + d_bg
        feat[:, 6] = feat[:, 6] + dz_bg
        mean = mean_from_feat(SD, feat)
        return dict(ds0, d=d_eq, mean=mean, feat=feat)

    dsA, dsB = shifted(dsA0), shifted(dsB0)
    pred, var = banded.predict_main(SD, dsA, dsB["X4"], band_kerns,
                                    mean_q=dsB["mean"])
    z, season = dsB["z"], dsB["season"]
    u_ex = uc.extra_std(z, np.full(len(z), season))
    u_c = np.sqrt(var + u_ex ** 2)
    dh, _ = SD.dist_to_inputs(dsB["X4"], dsA["X4"])
    k = uc.k_at(z, np.full(len(z), season), dh)
    err = pred - dsB["d"]
    return pd.DataFrame({
        "sid": sid, "seed": seed, "eq": eq, "band": dsB["band"], "z": z,
        "err": err, "u_c": u_c, "k": k,
        "covered": np.abs(err) <= k * u_c, "width": 2 * k * u_c,
        "err2": err ** 2})


def mean_from_feat(SD, feat):
    from exp.lib import meanet
    return meanet.predict(SD.model, SD.stdr, feat, SD.fcols)


def run_scene_main(SD, uc, sid, seed, band_kerns=None):
    """单场景单 seed 的 B 点明细（主方案口径：banded block GP + 冻结 k）。
    与 exp6_acceptance.run_scene 同输出列，但预测走 banded.predict_main，
    以与 EXP-6d/7/8 冻结的主方案保持一致。"""
    A, B = SD.ab(sid, seed)
    dsA, dsB = SD.build(sid, A), SD.build(sid, B)
    pred, var = banded.predict_main(SD, dsA, dsB["X4"], band_kerns,
                                    mean_q=dsB["mean"])
    z, season = dsB["z"], dsB["season"]
    u_extra = uc.extra_std(z, np.full(len(z), season))
    u_c = np.sqrt(var + u_extra ** 2)
    dh, _ = SD.dist_to_inputs(dsB["X4"], dsA["X4"])
    k = uc.k_at(z, np.full(len(z), season), dh)
    err = pred - dsB["d"]
    return pd.DataFrame({
        "sid": sid, "seed": seed, "prof": dsB["prof"], "band": dsB["band"],
        "z": z, "season": season, "err": err, "u_c": u_c, "u_r": np.sqrt(var),
        "u_extra": u_extra, "k": k, "covered": np.abs(err) <= k * u_c,
        "width": 2 * k * u_c, "err2": err ** 2,
        "abserr": np.abs(err), "sparse_km": dh})


# ------------------------------------------------ 主流程 ----
def main():
    t0 = time.time()
    print("== EXP-10: sensitivity & robustness ==")
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
    print(f"test scenes: {len(test_sids)}")

    # ---------- ① 声速方程对照 ----------
    print("-- (1) sound speed equation comparison --")
    eq_dfs = []
    for eq in ("chen-millero", "mackenzie", "gsw"):
        if eq == C.SOUND_SPEED_EQ:
            eqd = None
        else:
            eqd = EqDelta(eq, SD.bg)
            print(f"  {eq}: delta mean={eqd.dmean.mean():.4f} m/s, "
                  f"std={eqd.dstd.mean():.4f} (time), range="
                  f"[{eqd.dmean.min():.3f},{eqd.dmean.max():.3f}]")
        for sid in test_sids:
            if eqd is None:
                df = eval_eq(SD, uc, sid, 0, eq, _NullDelta())
            else:
                df = eval_eq(SD, uc, sid, 0, eq, eqd)
            eq_dfs.append(df)
        print(f"  eq={eq} done ({time.time()-t0:.0f}s)")
    eq_df = pd.concat(eq_dfs, ignore_index=True)
    eq_df.to_csv(os.path.join(OUT, "eq_compare_detail.csv"), index=False)
    eq_m = eq_df.groupby("eq").apply(lambda x: pd.Series({
        "rmse": float(np.sqrt(x["err2"].mean())),
        "cover": float(x["covered"].mean()),
        "width_mean": float(x["width"].mean()),
        "n": len(x)})).reset_index()
    eq_m.to_csv(os.path.join(OUT, "eq_compare.csv"), index=False)
    print(eq_m.round(4).to_string(index=False))

    # ---------- ② 输出网格对照 ----------
    print("-- (2) output grid comparison --")
    grid_uniform = np.arange(0, 2001, 100.0)
    grid_rows = []
    for sid in test_sids[:3]:
        s = SD.sbi[sid]
        A, _ = SD.ab(sid, 0)
        dsA = SD.build(sid, A)
        mem_t = SD.argo.t64[np.array(s["member_idx"])]
        t_mid = float(np.mean([mem_t.min(), mem_t.max()]) - EPOCH)
        clat = 0.5 * (s["lat0"] + s["lat1"])
        clon = 0.5 * (s["lon0"] + s["lon1"])
        season = s["season"]
        for gname, zg in (("refined", C.OUT_Z_GRID), ("uniform100", grid_uniform)):
            Xq = np.column_stack([np.full(len(zg), clon), np.full(len(zg), clat),
                                  zg, np.full(len(zg), t_mid)])
            pred, var = banded.predict_main(SD, dsA, Xq, band_kerns)
            u_ex = uc.extra_std(zg, np.full(len(zg), season))
            u_c = np.sqrt(var + u_ex ** 2)
            k = uc.k_at(zg, np.full(len(zg), season),
                        np.full(len(zg), 50.0))
            grid_rows.append({"sid": sid, "grid": gname, "n": len(zg),
                              "width_mean": float(np.mean(2 * k * u_c)),
                              "width_thermocline": float(
                                  np.mean(2 * k * u_c[(zg >= 60) & (zg <= 250)])),
                              "width_deep": float(
                                  np.mean(2 * k * u_c[zg >= 800]))})
    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(os.path.join(OUT, "grid_compare.csv"), index=False)
    print(grid_df.groupby("grid")[["width_mean", "width_thermocline", "width_deep"]]
          .mean().round(3).to_string())
    # 网格插值覆盖损失：refined 网格 u_c 插到 B 点深度 vs 直接 B 点 u_c
    loss_rows = []
    for sid in test_sids[:3]:
        A, B = SD.ab(sid, 0)
        dsA, dsB = SD.build(sid, A), SD.build(sid, B)
        pred, var = banded.predict_main(SD, dsA, dsB["X4"], band_kerns,
                                        mean_q=dsB["mean"])
        z, season = dsB["z"], dsB["season"]
        u_ex = uc.extra_std(z, np.full(len(z), season))
        u_c_direct = np.sqrt(var + u_ex ** 2)
        s = SD.sbi[sid]
        mem_t = SD.argo.t64[np.array(s["member_idx"])]
        t_mid = float(np.mean([mem_t.min(), mem_t.max()]) - EPOCH)
        clat = 0.5 * (s["lat0"] + s["lat1"])
        clon = 0.5 * (s["lon0"] + s["lon1"])
        Xg = np.column_stack([np.full(len(C.OUT_Z_GRID), clon),
                              np.full(len(C.OUT_Z_GRID), clat),
                              C.OUT_Z_GRID, np.full(len(C.OUT_Z_GRID), t_mid)])
        _, var_g = banded.predict_main(SD, dsA, Xg, band_kerns)
        u_ex_g = uc.extra_std(C.OUT_Z_GRID, np.full(len(C.OUT_Z_GRID), season))
        u_c_grid = np.sqrt(var_g + u_ex_g ** 2)
        u_c_interp = np.interp(z, C.OUT_Z_GRID, u_c_grid)
        err = pred - dsB["d"]
        k = uc.k_at(z, np.full(len(z), season),
                    np.full(len(z), 50.0))
        cov_direct = float(np.mean(np.abs(err) <= k * u_c_direct))
        cov_interp = float(np.mean(np.abs(err) <= k * u_c_interp))
        loss_rows.append({"sid": sid, "cov_direct": cov_direct,
                          "cov_grid_interp": cov_interp,
                          "u_c_gap_mean": float(np.mean(np.abs(u_c_interp - u_c_direct)))})
    loss_df = pd.DataFrame(loss_rows)
    loss_df.to_csv(os.path.join(OUT, "grid_interp_coverage_loss.csv"), index=False)
    print(loss_df.round(4).to_string(index=False))

    # ---------- ③ 种子敏感性 + 收敛性 ----------
    print("-- (3) seed sensitivity & split convergence --")
    conv_rows = []
    for sid in test_sids:
        for seed in range(N_SEEDS_CONV):
            df = run_scene_main(SD, uc, sid, seed, band_kerns)
            df["seed10"] = seed
            conv_rows.append(df)
    conv_df = pd.concat(conv_rows, ignore_index=True)
    conv_df.to_csv(os.path.join(OUT, "seed_convergence_detail.csv"), index=False)
    conv_m = []
    for s_max in range(1, N_SEEDS_CONV + 1):
        sub = conv_df[conv_df["seed10"] < s_max]
        by_scene = sub.groupby(["sid", "seed"])["covered"].mean()
        conv_m.append({"n_seeds": s_max, "cover_mean": float(by_scene.groupby("sid").mean().mean()),
                       "cover_std_across_seeds": float(
                           sub.groupby("seed")["covered"].mean().std())})
    conv_mdf = pd.DataFrame(conv_m)
    conv_mdf.to_csv(os.path.join(OUT, "seed_convergence.csv"), index=False)
    print(conv_mdf.round(4).to_string(index=False))
    # ④ 季节/海区分列（seed 0 明细）
    s0 = conv_df[conv_df["seed10"] == 0]
    rob_rows = []
    for season in sorted(s0["season"].unique()):
        m = s0["season"] == season
        rob_rows.append({"subset": f"season{season}", "n": int(m.sum()),
                         "rmse": float(np.sqrt(s0["err2"][m].mean())),
                         "cover": float(s0["covered"][m].mean()),
                         "width": float(s0["width"][m].mean())})
    reg_of = {sid: SD.sbi[sid]["region"] for sid in test_sids}
    s0_reg = s0.assign(region=s0["sid"].map(reg_of))
    for region in sorted(s0_reg["region"].unique()):
        m = s0_reg["region"] == region
        rob_rows.append({"subset": f"region{region}", "n": int(m.sum()),
                         "rmse": float(np.sqrt(s0_reg["err2"][m].mean())),
                         "cover": float(s0_reg["covered"][m].mean()),
                         "width": float(s0_reg["width"][m].mean())})
    rob_df = pd.DataFrame(rob_rows)
    rob_df.to_csv(os.path.join(OUT, "robust_by_subset.csv"), index=False)
    print(rob_df.round(4).to_string(index=False))

    # ---------- ⑤ 报告模板实例化 ----------
    print("-- (5) report template instantiation --")
    sid = test_sids[0]
    s = SD.sbi[sid]
    A, _ = SD.ab(sid, 0)
    dsA = SD.build(sid, A)
    mem_t = SD.argo.t64[np.array(s["member_idx"])]
    t_mid = float(np.mean([mem_t.min(), mem_t.max()]) - EPOCH)
    clat = 0.5 * (s["lat0"] + s["lat1"])
    clon = 0.5 * (s["lon0"] + s["lon1"])
    zg = C.OUT_Z_GRID
    Xq = np.column_stack([np.full(len(zg), clon), np.full(len(zg), clat),
                          zg, np.full(len(zg), t_mid)])
    pred, var = banded.predict_main(SD, dsA, Xq, band_kerns)
    season = s["season"]
    season_arr = np.full(len(zg), season)
    um = uc.um_at(zg, season_arr)
    ueq = uc.ueq_at(zg)
    urep = uc.urep_at(zg)
    ur = np.sqrt(var)
    u_c = np.sqrt(var + um ** 2 + ueq ** 2 + urep ** 2)
    k = np.broadcast_to(uc.k_at(zg, season_arr, np.full(len(zg), 50.0)), zg.shape)
    with open(os.path.join(C.RESULTS, "exp2", "scenes_meta.json"), encoding="utf-8") as f:
        smeta = json.load(f)
    report = {
        "report_meta": {
            "scene_id": int(sid), "role": s["role"], "season": int(season),
            "spatial_block": [s["lon0"], s["lon1"], s["lat0"], s["lat1"]],
            "time_window_rel2000_s": [float(mem_t.min() - EPOCH), float(mem_t.max() - EPOCH)],
            "n_input_profiles": len(A),
            "method": "NN-residual GP (two-stage, offline-calibrated)",
            "sound_speed_equation": C.SOUND_SPEED_EQ,
            "depth_range_claim": f"0-{C.Z_MAX:.0f} m (D8)",
            "support_claim": "point support observation -> gridded query; "
                             "u_rep covers voxel-support mismatch (Type B)",
            "grid": "refined thermocline grid (35 levels, 0-2000 m)",
            "split_hash": smeta.get("split_hash"),
            "date": time.strftime("%Y-%m-%d"),
        },
        "points": [
            {"lon": float(clon), "lat": float(clat), "z": float(zz),
             "t_rel2000_s": t_mid, "s_hat": float(pp),
             "u_c": float(uu), "u_r": float(ur[i]), "u_m": float(um[i]),
             "u_eq": float(ueq[i]), "u_rep": float(urep[i]), "k": float(k[i]),
             "interval_95": [float(pp - k[i] * uu), float(pp + k[i] * uu)]}
            for i, (zz, pp, uu) in enumerate(zip(zg, pred, u_c))],
    }
    with open(os.path.join(OUT, "report_template_example.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1,
                  default=lambda o: o.item() if hasattr(o, "item") else float(o))
    print(f"  report template: {len(zg)} points, scene {sid}")

    # ---------- 汇总 ----------
    summary = {
        "exp": "exp10", "date": time.strftime("%Y-%m-%d %H:%M"),
        "n_test_scenes": len(test_sids),
        "eq_compare": eq_m.to_dict("records"),
        "eq_note": "frozen m_theta; cb/dcb_dz features shifted by grid eq-delta; "
                   "grad_h kept at cm-basis (delta horizontal gradient negligible)",
        "grid_compare": grid_df.groupby("grid")[
            ["width_mean", "width_thermocline", "width_deep"]].mean().round(4)
            .reset_index().to_dict("records"),
        "grid_interp_coverage_loss": loss_df.to_dict("records"),
        "seed_convergence": conv_mdf.to_dict("records"),
        "robust_by_subset": rob_df.to_dict("records"),
        "report_template_scene": int(sid),
        "elapsed_s": time.time() - t0,
    }
    with open(os.path.join(OUT, "exp10_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    print("EXP-10 DONE", time.time() - t0)


class _NullDelta:
    """cm 方程自身的零平移占位。"""
    def at(self, z, lat, lon, want="val"):
        z = np.atleast_1d(np.asarray(z, dtype=float))
        return np.zeros(len(z))


if __name__ == "__main__":
    main()
