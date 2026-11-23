# -*- coding: utf-8 -*-
"""EXP-7 独立验收：仿真场景行为检验（WP3/WP4，framework §4-EXP7）。
定位：可控行为检验（真值=背景+注入扰动，非最终覆盖率验收；R12）。
① 稀疏度扫描：输入剖面数 N_in ∈ {2..24}，RMSE/宽度响应（J5 平缓退化，
   极稀端退化为背景场行为）
② 异步度扫描：观测时间散布 T_span ∈ {0..60} d
③ 簇集度扫描：k 簇 ∈ {1..N}（均匀 vs 成簇）
④ 覆盖率补充：注入与背景独立的扰动场（合成随机场 / 独立时段交叉），
   完整管线（m_θ + GP + u_c + k）覆盖率统计
产出：results/exp7/*.csv + exp7_summary.json
"""
import sys, os, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from exp import config as C
from exp.lib import physics, uobs, scenes as scenelib, anchors
from exp.lib import gp as gplib
from exp.lib import banded
from exp.lib.pipeline import SceneData, load_stack, season_from_t64
from exp.lib.uc import UcComposer
from exp.lib.interp import BgField

OUT = C.result_dir("exp7")
EPOCH = np.datetime64("2000-01-01T00:00:00").astype(np.int64)
RNG = np.random.default_rng(C.SEED)
N_REP = 5          # 每配置重复（随机放置）
N_QUERY = 400      # 每场景验证查询点数
DAY_S = 86400.0


# ------------------------------------------------ 合成观测基础设施 ----
def synth_u_obs_by_band():
    """声速域合成观测噪声分带表（uT/uS 规格在名义剖面 Jacobian 下传播）。"""
    uT, uS, _ = uobs.load_uobs()
    zc = uobs.BAND_CENTERS
    T_nom = 24.0 * np.exp(-zc / 300.0) + 4.0
    S_nom = np.full(len(zc), 34.6)
    p_nom = physics.p_from_z(zc, lat=20.0)
    dT, dS, _ = physics.sound_speed_jacobian(T_nom, S_nom, p_nom)
    return np.sqrt((dT * uT) ** 2 + (dS * uS) ** 2)


class GaussianRF2D:
    """指数相关二维高斯随机场（谱方法），场景水平范围内的双线性取值。"""

    def __init__(self, lat0, lat1, lon0, lon1, corr_km, sigma, rng):
        self.lat0, self.lat1, self.lon0, self.lon1 = lat0, lat1, lon0, lon1
        n = 128
        # 物理网格 (km)，纬度参考 20° 换算
        kx = 111.32 * np.cos(np.radians(20.0))
        self.dx = (lon1 - lon0) * kx / n
        self.dy = (lat1 - lat0) * 110.57 / n
        k = np.fft.fftfreq(n, d=1.0) * 2 * np.pi
        KX = np.fft.fftfreq(n, d=self.dx) * 2 * np.pi
        KY = np.fft.fftfreq(n, d=self.dy) * 2 * np.pi
        KX, KY = np.meshgrid(KX, KY)
        S = 1.0 / (1.0 + (np.sqrt(KX ** 2 + KY ** 2) * corr_km) ** 2)  # 类 Matérn 谱
        S[0, 0] = 0.0
        w = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
        f = np.fft.ifft2(np.sqrt(S) * w).real
        f -= f.mean()
        f /= f.std()
        self.field = f * sigma
        self.n = n

    def at(self, lat, lon):
        kx = 111.32 * np.cos(np.radians(20.0))
        fx = np.clip((np.asarray(lon) - self.lon0) * kx / self.dx, 0, self.n - 1.001)
        fy = np.clip((np.asarray(lat) - self.lat0) * 110.57 / self.dy, 0, self.n - 1.001)
        ix, iy = fx.astype(int), fy.astype(int)
        tx, ty = fx - ix, fy - iy
        F = self.field
        return (F[iy, ix] * (1 - tx) * (1 - ty) + F[iy, ix + 1] * tx * (1 - ty)
                + F[iy + 1, ix] * (1 - tx) * ty + F[iy + 1, ix + 1] * tx * ty)


def make_delta(kind, scene, bg, rng, sigma_ms=0.8, corr_km=80.0, dt_shift_d=183.0):
    """扰动场工厂。返回 f(z, t_sec, lat, lon) -> δ (m/s)。
    kind: 'none' | 'rf' (合成随机场) | 'xtime' (独立时段交叉)。"""
    if kind == "none":
        return lambda z, t, lat, lon: np.zeros(len(np.atleast_1d(z)))
    if kind == "rf":
        rf = GaussianRF2D(scene["lat0"], scene["lat1"], scene["lon0"], scene["lon1"],
                          corr_km, sigma_ms, rng)
        # 深度结构：表层/跃层强，深层衰减；时间线性缓变 ±25%
        t_lo, t_hi = scene["_t_lo"], scene["_t_hi"]

        def delta(z, t, lat, lon):
            z = np.atleast_1d(np.asarray(z, dtype=float))
            t = np.atleast_1d(np.asarray(t, dtype=float))
            lat = np.atleast_1d(np.asarray(lat, dtype=float))
            lon = np.atleast_1d(np.asarray(lon, dtype=float))
            fz = np.exp(-z / 400.0)
            base = rf.at(lat, lon)
            ph = 0.75 + 0.5 * np.clip((t - t_lo) / max(t_hi - t_lo, 1), 0, 1)
            return base * fz * ph
        return delta
    # xtime: δ = c_b(t+ΔT) − c_b(t)
    t_shift = dt_shift_d * DAY_S

    def delta(z, t, lat, lon):
        c0 = bg.cs_at(t, z, lat, lon)
        t2 = np.clip(np.asarray(t, dtype=float) + t_shift,
                     (bg.t_hours[0] + 1) * 3600.0, (bg.t_hours[-1] - 1) * 3600.0)
        c1 = bg.cs_at(t2, z, lat, lon)
        return c1 - c0
    return delta


def synth_scene(SD, scene, n_in, t_span_d, n_clusters, delta, rng, u_obs_band):
    """生成一个仿真场景：合成输入剖面 + 验证查询点（含真值）。
    返回 ds_in(dict for GPFit)、Xq、c_true、q_season。"""
    lon0, lon1 = scene["lon0"], scene["lon1"]
    lat0, lat1 = scene["lat0"], scene["lat1"]
    t_lo, t_hi = scene["_t_lo"], scene["_t_hi"]
    # ---- 剖面位置：簇集度控制 ----
    n_cl = max(1, min(n_clusters, n_in))
    cl_lon = rng.uniform(lon0, lon1, n_cl)
    cl_lat = rng.uniform(lat0, lat1, n_cl)
    nper = np.diff(np.linspace(0, n_in, n_cl + 1).astype(int))
    lon_p, lat_p = [], []
    for k in range(n_cl):
        cnt = nper[k]
        if cnt == 0:
            continue
        if n_cl >= n_in:   # 均匀布放
            lon_p += rng.uniform(lon0, lon1, cnt).tolist()
            lat_p += rng.uniform(lat0, lat1, cnt).tolist()
        else:
            lon_p += (cl_lon[k] + rng.normal(0, 0.35, cnt)).tolist()
            lat_p += (cl_lat[k] + rng.normal(0, 0.35, cnt)).tolist()
    lon_p = np.clip(lon_p, lon0 + 0.05, lon1 - 0.05)
    lat_p = np.clip(lat_p, lat0 + 0.05, lat1 - 0.05)
    # ---- 时间：异步度控制 ----
    if t_span_d <= 0:
        t_p = np.full(n_in, 0.5 * (t_lo + t_hi))
    else:
        t_p = rng.uniform(t_lo, t_hi, n_in) if t_span_d >= (t_hi - t_lo) / DAY_S \
            else 0.5 * (t_lo + t_hi) + rng.uniform(-0.5, 0.5, n_in) * t_span_d * DAY_S
    t_p = np.clip(t_p, t_lo, t_hi)
    # ---- 合成观测（锚点深度集）----
    Z = anchors.ANCHOR_Z
    Z = Z[Z <= C.Z_MAX]
    rows_X, rows_d, rows_u = [], [], []
    for i in range(n_in):
        z = Z
        t = np.full(len(z), t_p[i])
        cb = SD.bg.cs_at(t, z, np.full(len(z), lat_p[i]), np.full(len(z), lon_p[i]))
        dl = delta(z, t, np.full(len(z), lat_p[i]), np.full(len(z), lon_p[i]))
        bi = np.clip(uobs.band_index(z), 0, 5)
        eps = rng.normal(0, 1, len(z)) * u_obs_band[bi]
        o = cb + dl + eps
        rows_X.append(np.column_stack([np.full(len(z), lon_p[i]),
                                       np.full(len(z), lat_p[i]), z, t]))
        rows_d.append(o - cb)
        rows_u.append(u_obs_band[bi])
    X4 = np.concatenate(rows_X)
    d = np.concatenate(rows_d)
    u = np.concatenate(rows_u)
    mean = SD.query_features(X4)
    # ---- 验证查询点：时空均匀 ----
    q_lon = rng.uniform(lon0, lon1, N_QUERY)
    q_lat = rng.uniform(lat0, lat1, N_QUERY)
    q_t = rng.uniform(t_lo, t_hi, N_QUERY)
    q_z = np.exp(rng.uniform(np.log(10), np.log(1900), N_QUERY))  # 对数均匀
    # 真值取**残差域**（= 扰动场 δ 本身：合成观测残差 d = δ + ε，无噪声真值即 δ），
    # 与 GP 的残差域预测同域。若误用绝对声速 c_b+δ 作真值，err 恒差 ~1520 m/s、
    # 覆盖率归零（与 EXP-6【问题26】同型错误）。
    c_true = delta(q_z, q_t, q_lat, q_lon)
    Xq = np.column_stack([q_lon, q_lat, q_z, q_t])
    # q_t 为相对 2000-01-01 秒，+EPOCH 转绝对 datetime64[s]（直接传相对秒会被
    # 误读为 ~1989 年，季节全错）
    q_season = int(np.bincount(season_from_t64(
        (q_t + EPOCH).astype(np.int64))).argmax())
    ds = {"X4": X4, "d": d, "u": u, "mean": mean, "z": X4[:, 2],
          "band": uobs.band_index(X4[:, 2]), "season": scene["season"], "sid": -1}
    return ds, Xq, c_true, q_season


def eval_sim(SD, uc, ds, Xq, c_true, q_season, band_kerns=None, use_full_uc=True):
    """GP 拟合 + 验证指标。band_kerns 非空 -> 分带块 GP（主方案）；None -> 单 GP（对照）。
    use_full_uc: u_c = sqrt(u_r²+u_extra²)（行为扫描可 False）。"""
    mq = SD.query_features(Xq)
    if band_kerns:
        bq = uobs.band_index(Xq[:, 2])
        fallback = SD.kern_for(ds["season"])
        pred, var = banded.predict_banded(ds, Xq, mq, bq, band_kerns, fallback)
    else:
        g = SD.fit_gp(ds)
        pred, var = g.predict(Xq, mq)
    err = pred - c_true
    u_r = np.sqrt(var)
    dh, _ = SceneData.dist_to_inputs(Xq, ds["X4"])
    season_arr = np.full(len(Xq), q_season)
    if use_full_uc:
        u_extra = uc.extra_std(Xq[:, 2], season_arr)
        u_c = np.sqrt(var + u_extra ** 2)
        k = uc.k_at(Xq[:, 2], season_arr, dh)
    else:
        u_c = u_r
        k = 1.96
    near = dh <= np.quantile(dh, 0.25)
    return {"rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
            "cover": float(np.mean(np.abs(err) <= k * u_c)),
            "width_mean": float(np.mean(2 * k * u_c)),
            "u_r_mean": float(u_r.mean()),
            "u_r_near": float(u_r[near].mean()),
            "u_r_far": float(u_r[~near].mean())}


def pick_scenes(stack, n_per_season=1):
    """test 场景中每季节挑 n 个（n_profiles 适中）。"""
    scenes = stack["scenes"]
    SD = SceneData(stack)
    out = []
    for season in range(4):
        cand = [s for s in scenes if s["role"] == "test" and s["season"] == season]
        cand.sort(key=lambda s: abs(s["n_profiles"] - 20))
        out += cand[:n_per_season]
    for s in out:
        mem_t = SD.argo.t64[np.array(s["member_idx"])]
        s["_t_lo"] = float(mem_t.min())
        s["_t_hi"] = float(mem_t.max())
    return out


def main():
    t0 = time.time()
    print("== EXP-7: synthetic-scenario behavior checks ==")
    stack = load_stack()
    SD = SceneData(stack)
    uc = UcComposer(k_json=C.MAIN_K_JSON)
    band_kerns = banded.load_band_kerns()
    bg = stack["bg"]
    print(f"main k = {float(uc.k_at()):.4f} (banded GP, EXP-6d frozen); "
          f"band kerns: {sorted(band_kerns.keys()) if band_kerns else None}")
    scenes = pick_scenes(stack)
    print(f"scenes for simulation: {[s['scene_id'] for s in scenes]}")
    u_obs_band = synth_u_obs_by_band()
    print(f"synth u_obs by band (m/s): {np.round(u_obs_band, 4).tolist()}")

    # ---------- ① 稀疏度扫描 ----------
    print("-- (1) sparsity scan --")
    rows = []
    for scene in scenes:
        for n_in in (2, 4, 6, 8, 12, 16, 24):
            for rep in range(N_REP):
                rng = np.random.default_rng(7000 + rep)
                delta = make_delta("xtime", scene, bg, rng)
                ds, Xq, ct, qs = synth_scene(SD, scene, n_in, 15.0, n_in,
                                              delta, rng, u_obs_band)
                for tag, bk in (("banded", band_kerns), ("single", None)):
                    m = eval_sim(SD, uc, ds, Xq, ct, qs, band_kerns=bk, use_full_uc=True)
                    rows.append({"scan": "sparse", "model": tag, "sid": scene["scene_id"],
                                 "season": scene["season"], "n_in": n_in, "rep": rep,
                                 "t_span_d": 15.0, "n_clusters": n_in, "delta": "xtime", **m})
        print(f"  scene {scene['scene_id']} done ({time.time()-t0:.0f}s)")
    sparse_df = pd.DataFrame(rows)
    sparse_df.to_csv(os.path.join(OUT, "sparse_scan.csv"), index=False)
    agg = sparse_df.groupby(["model", "n_in"])[["rmse", "cover", "width_mean", "u_r_mean"]].mean()
    print(agg.round(3).to_string())

    # ---------- ② 异步度扫描（n_in=12，簇=均匀） ----------
    print("-- (2) asynchrony scan --")
    rows = []
    for scene in scenes:
        for span in (0.0, 3.0, 7.0, 15.0, 30.0, 60.0):
            for rep in range(N_REP):
                rng = np.random.default_rng(7100 + rep)
                delta = make_delta("xtime", scene, bg, rng)
                ds, Xq, ct, qs = synth_scene(SD, scene, 12, span, 12,
                                              delta, rng, u_obs_band)
                m = eval_sim(SD, uc, ds, Xq, ct, qs, band_kerns=band_kerns)
                rows.append({"scan": "async", "model": "banded", "sid": scene["scene_id"],
                             "season": scene["season"], "n_in": 12, "rep": rep,
                             "t_span_d": span, "n_clusters": 12, "delta": "xtime", **m})
        print(f"  scene {scene['scene_id']} done ({time.time()-t0:.0f}s)")
    async_df = pd.DataFrame(rows)
    async_df.to_csv(os.path.join(OUT, "async_scan.csv"), index=False)
    print(async_df.groupby("t_span_d")[["rmse", "cover", "width_mean"]].mean().round(3).to_string())

    # ---------- ③ 簇集度扫描（n_in=12） ----------
    print("-- (3) clustering scan --")
    rows = []
    for scene in scenes:
        for n_cl in (1, 2, 3, 6, 12):
            for rep in range(N_REP):
                rng = np.random.default_rng(7200 + rep)
                delta = make_delta("xtime", scene, bg, rng)
                ds, Xq, ct, qs = synth_scene(SD, scene, 12, 15.0, n_cl,
                                              delta, rng, u_obs_band)
                m = eval_sim(SD, uc, ds, Xq, ct, qs, band_kerns=band_kerns)
                rows.append({"scan": "cluster", "model": "banded", "sid": scene["scene_id"],
                             "season": scene["season"], "n_in": 12, "rep": rep,
                             "t_span_d": 15.0, "n_clusters": n_cl, "delta": "xtime", **m})
        print(f"  scene {scene['scene_id']} done ({time.time()-t0:.0f}s)")
    clust_df = pd.DataFrame(rows)
    clust_df.to_csv(os.path.join(OUT, "cluster_scan.csv"), index=False)
    print(clust_df.groupby("n_clusters")[["rmse", "cover", "width_mean"]].mean().round(3).to_string())

    # ---------- ④ 扰动注入覆盖率补充（两种 δ × 稀疏度） ----------
    print("-- (4) perturbation-injection coverage --")
    rows = []
    for scene in scenes:
        for kind in ("rf", "xtime"):
            for n_in in (6, 12, 20):
                for rep in range(N_REP):
                    rng = np.random.default_rng(7300 + rep)
                    delta = make_delta(kind, scene, bg, rng)
                    ds, Xq, ct, qs = synth_scene(SD, scene, n_in, 15.0, n_in,
                                                  delta, rng, u_obs_band)
                    m = eval_sim(SD, uc, ds, Xq, ct, qs, band_kerns=band_kerns)
                    rows.append({"scan": "perturb", "model": "banded", "sid": scene["scene_id"],
                                 "season": scene["season"], "n_in": n_in, "rep": rep,
                                 "t_span_d": 15.0, "n_clusters": n_in,
                                 "delta": kind, **m})
        print(f"  scene {scene['scene_id']} done ({time.time()-t0:.0f}s)")
    pert_df = pd.DataFrame(rows)
    pert_df.to_csv(os.path.join(OUT, "perturb_coverage.csv"), index=False)
    print(pert_df.groupby(["delta", "n_in"])[["rmse", "cover", "width_mean"]]
          .mean().round(3).to_string())

    # ---------- 汇总 ----------
    def mono_check(df, x, y):
        g = df.groupby(x)[y].mean()
        spearman = float(pd.Series(g.values).corr(pd.Series(g.index), method="spearman"))
        return {"x": x, "y": y, "spearman": spearman,
                "curve": {str(k): round(float(v), 4) for k, v in g.items()}}

    summary = {
        "exp": "exp7", "date": time.strftime("%Y-%m-%d %H:%M"),
        "n_scenes": len(scenes),
        "scene_ids": [s["scene_id"] for s in scenes],
        "n_rep": N_REP, "n_query": N_QUERY,
        "synth_u_obs_band_ms": u_obs_band.tolist(),
        "main_k": float(uc.k_at()),
        "sparse_rmse_monotonic_banded": mono_check(sparse_df[sparse_df["model"] == "banded"],
                                                   "n_in", "rmse"),
        "sparse_rmse_monotonic_single": mono_check(sparse_df[sparse_df["model"] == "single"],
                                                   "n_in", "rmse"),
        "sparse_width_monotonic_banded": mono_check(sparse_df[sparse_df["model"] == "banded"],
                                                    "n_in", "width_mean"),
        "sparse_scarcest_rmse_vs_bg": {
            "rmse_n2": float(sparse_df[(sparse_df["model"] == "banded")
                                       & (sparse_df["n_in"] == 2)]["rmse"].mean()),
            "rmse_n24": float(sparse_df[(sparse_df["model"] == "banded")
                                        & (sparse_df["n_in"] == 24)]["rmse"].mean())},
        "async": async_df.groupby("t_span_d")[["rmse", "cover", "width_mean"]]
            .mean().round(4).reset_index().to_dict("records"),
        "cluster": clust_df.groupby("n_clusters")[["rmse", "cover", "width_mean"]]
            .mean().round(4).reset_index().to_dict("records"),
        "perturb_coverage": pert_df.groupby(["delta", "n_in"])[["rmse", "cover", "width_mean"]]
            .mean().round(4).reset_index().to_dict("records"),
        "elapsed_s": time.time() - t0,
    }
    with open(os.path.join(OUT, "exp7_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    print("EXP-7 DONE", time.time() - t0)


if __name__ == "__main__":
    main()
