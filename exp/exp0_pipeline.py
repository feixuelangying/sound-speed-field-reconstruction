# -*- coding: utf-8 -*-
"""EXP-0 数据管线与测量基座（WP0）。
产出：hycom_cs 缓存 / argo/ctd QC 缓存 / 声速方程选型对比 / 单元测试 / 残差指示 demo。
结果：results/exp0/exp0_summary.json + 残差统计 csv
"""
import sys, os, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from exp import config as C
from exp.lib import physics, io_data
from exp.lib.interp import BgField, pchip_eval

RNG = np.random.default_rng(C.SEED)
OUT = C.result_dir("exp0")


def eq_comparison(bg: BgField, argo):
    """三声速方程对比：观测分布域 + 背景网格抽样。"""
    n = 20000
    sel = RNG.choice(argo.n_obs, min(n, argo.n_obs), replace=False)
    T, S, p = argo.T[sel], argo.S[sel], argo.p[sel]
    cm = physics.sound_speed(T, S, p, "chen-millero")
    gz = physics.sound_speed(T, S, p, "gsw")
    mk = physics.sound_speed(T, S, p, "mackenzie")
    d1 = gz - cm; d2 = mk - cm
    return {
        "n": int(len(sel)),
        "gsw_minus_cm": {"mean": float(d1.mean()), "std": float(d1.std()),
                         "absmax": float(np.abs(d1).max())},
        "mackenzie_minus_cm": {"mean": float(d2.mean()), "std": float(d2.std()),
                               "absmax": float(np.abs(d2).max())},
    }


def unit_tests(bg: BgField, argo):
    res = {}
    # 1) PCHIP vs scipy（随机 8 节点剖面）
    from scipy.interpolate import PchipInterpolator
    nodes = np.array([0, 10, 30, 60, 125, 350, 800, 2000.])
    errs = []
    for _ in range(200):
        xq = RNG.uniform(0, 2000, size=5)
        V = RNG.uniform(1470, 1550, size=(5, 8))
        mine = pchip_eval(nodes, V, xq)
        for j in range(5):
            ref = PchipInterpolator(nodes, V[j])(xq[j])
            errs.append(abs(mine[j] - ref))
    res["pchip_vs_scipy_maxerr"] = float(np.max(errs))

    # 2) z<->p 往返
    z = RNG.uniform(0, 2000, 1000); lat = RNG.uniform(10, 30, 1000)
    p = physics.p_from_z(z, lat)
    z2 = physics.z_from_p(p, lat)
    res["zp_roundtrip_maxerr_m"] = float(np.abs(z2 - z).max())

    # 3) cs_at 交叉核验：3 个真实观测点，独立路径（netCDF 直读 + scipy PCHIP）
    ds_path = C.HYCOM_NC
    import netCDF4
    ds = netCDF4.Dataset(ds_path); ds.set_auto_mask(True)
    t_hours_ax = np.asarray(ds.variables["time"][:], dtype=float)
    lat_ax = np.asarray(ds.variables["lat"][:]); lon_ax = np.asarray(ds.variables["lon"][:])
    dep_ax = np.asarray(ds.variables["depth"][:], dtype=float)
    i_prof = RNG.choice(argo.n_prof, 3, replace=False)
    errs_cb = []
    for ip in i_prof:
        sl = argo.slice(int(ip))
        k = sl.start + int(argo.counts[ip] // 2)
        t_h = (argo.t64[ip] - np.datetime64("2000-01-01T00:00:00").astype(np.int64)) / 3600.0
        zq = float(physics.z_from_p(argo.p[k], argo.lat[ip]))
        la, lo = float(argo.lat[ip]), float(argo.lon[ip])
        got = float(bg.cs_at(np.array([t_h * 3600.0]), np.array([zq]),
                             np.array([la]), np.array([lo]))[0])
        # 独立实现
        it = np.searchsorted(t_hours_ax, t_h, side="right") - 1
        it = np.clip(it, 0, len(t_hours_ax) - 2)
        wt = (t_h - t_hours_ax[it]) / (t_hours_ax[it + 1] - t_hours_ax[it])
        ix = np.clip(np.searchsorted(lon_ax, lo) - 1, 0, len(lon_ax) - 2)
        iy = np.clip(np.searchsorted(lat_ax, la) - 1, 0, len(lat_ax) - 2)
        fx = (lo - lon_ax[ix]) / (lon_ax[ix + 1] - lon_ax[ix])
        fy = (la - lat_ax[iy]) / (lat_ax[iy + 1] - lat_ax[iy])
        prof8 = []
        for itp in (it, it + 1):
            Ta = ds.variables["water_temp"][itp]; Sa = ds.variables["salinity"][itp]
            v8 = np.zeros(8)
            for kk in range(8):
                t_ = Ta[kk]; s_ = Sa[kk]
                tv = (t_[iy, ix] * (1 - fx) * (1 - fy) + t_[iy, ix + 1] * fx * (1 - fy)
                      + t_[iy + 1, ix] * (1 - fx) * fy + t_[iy + 1, ix + 1] * fx * fy)
                sv = (s_[iy, ix] * (1 - fx) * (1 - fy) + s_[iy, ix + 1] * fx * (1 - fy)
                      + s_[iy + 1, ix] * (1 - fx) * fy + s_[iy + 1, ix + 1] * fx * fy)
                tv = float(np.ma.filled(tv, np.nan)); sv = float(np.ma.filled(sv, np.nan))
                v8[kk] = physics.sound_speed(tv, sv,
                                             physics.p_from_z(dep_ax[kk], la))
            prof8.append(v8)
        v8 = (1 - wt) * prof8[0] + wt * prof8[1]
        if not np.isfinite(v8).all():
            continue  # 近岸 mask 命中，跳过该点
        ref = float(PchipInterpolator(dep_ax, v8)(zq))
        errs_cb.append(abs(got - ref))
    ds.close()
    res["cs_at_crosscheck_maxerr_ms"] = float(np.max(errs_cb)) if errs_cb else None

    # 4) 批量插值无 NaN
    sel = RNG.choice(argo.n_prof, 200, replace=False)
    t_s, z_s, la_s, lo_s = [], [], [], []
    for ip in sel:
        sl = argo.slice(int(ip))
        zz = physics.z_from_p(argo.p[sl], argo.lat[ip])
        tt = np.full(len(zz), (argo.t64[ip] - np.datetime64("2000-01-01T00:00:00").astype(np.int64)), dtype=float)
        t_s.append(tt); z_s.append(zz)
        la_s.append(np.full(len(zz), argo.lat[ip])); lo_s.append(np.full(len(zz), argo.lon[ip]))
    t_s = np.concatenate(t_s); z_s = np.concatenate(z_s)
    la_s = np.concatenate(la_s); lo_s = np.concatenate(lo_s)
    cb = bg.cs_at(t_s, z_s, la_s, lo_s)
    res["batch_interp_nan_count"] = int(np.isnan(cb).sum())
    res["batch_cb_range"] = [float(cb.min()), float(cb.max())]
    return res, (t_s, z_s, la_s, lo_s, cb)


def main():
    t_start = time.time()
    summary = {"exp": "exp0", "date": time.strftime("%Y-%m-%d %H:%M")}
    print("== EXP-0: build caches ==")
    io_data.build_hycom_cs_cache(force=False)
    io_data.build_argo_cache(force=False, n_workers=16)
    io_data.build_ctd_cache(force=False)

    bg = BgField(C.CACHE)
    argo = io_data.load_argo()
    ctd = io_data.load_ctd()
    print(f"profiles: argo={argo.n_prof} ({argo.n_obs} obs), ctd={ctd.n_prof} ({ctd.n_obs} obs)")

    # QC 统计
    st = np.load(os.path.join(C.CACHE, "argo_clean.npz"))["qc_stats"]
    summary["argo_qc"] = {
        "n_profiles_total_csv": 19423,
        "n_profiles_kept": int(st.shape[0]),
        "n_obs_raw": int(st[:, 0].sum()),
        "n_obs_after_flag_range": int(st[:, 1].sum()),
        "n_obs_after_spike": int(st[:, 2].sum()),
        "n_obs_final": int(st[:, 3].sum()),
        "loss_rate_final": float(1 - st[:, 3].sum() / st[:, 0].sum()),
    }
    ctd_st = {"n_stations": ctd.n_prof, "n_obs": ctd.n_obs}
    summary["ctd_qc"] = ctd_st

    # 声速方程对比（D12 选型依据）
    print("== sound speed equation comparison ==")
    summary["eq_comparison"] = eq_comparison(bg, argo)

    # 单元测试
    print("== unit tests ==")
    ut, subset = unit_tests(bg, argo)
    # 残差指示 demo（同 200 剖面子集）
    t_s, z_s, la_s, lo_s, cb = subset
    sel_prof = RNG.choice(argo.n_prof, 200, replace=False)
    c_obs_all, z_all, cb_all, band_all = [], [], [], []
    zbands = [(0, 30), (30, 60), (60, 125), (125, 350), (350, 800), (800, 2000)]
    for ip in sel_prof:
        sl = argo.slice(int(ip))
        c_obs = physics.sound_speed(argo.T[sl], argo.S[sl], argo.p[sl], "chen-millero")
        zz = physics.z_from_p(argo.p[sl], argo.lat[ip])
        tt = np.full(len(zz), float(argo.t64[ip] -
                       np.datetime64("2000-01-01T00:00:00").astype(np.int64)))
        cbb = bg.cs_at(tt, zz, np.full(len(zz), argo.lat[ip]), np.full(len(zz), argo.lon[ip]))
        c_obs_all.append(c_obs); z_all.append(zz); cb_all.append(cbb)
    c_obs_all = np.concatenate(c_obs_all); z_all = np.concatenate(z_all); cb_all = np.concatenate(cb_all)
    d = c_obs_all - cb_all
    band_all = np.zeros(len(d), dtype=int)
    for bi, (z0, z1) in enumerate(zbands):
        band_all[(z_all >= z0) & (z_all < z1)] = bi
    import pandas as pd
    stat_rows = []
    for bi, (z0, z1) in enumerate(zbands):
        m = band_all == bi
        if m.sum() > 0:
            stat_rows.append({"band": f"{z0}-{z1}m", "n": int(m.sum()),
                              "d_mean": float(d[m].mean()), "d_std": float(d[m].std()),
                              "d_absmax": float(np.abs(d[m]).max()),
                              "cb_mean": float(cb_all[m].mean())})
    pd.DataFrame(stat_rows).to_csv(os.path.join(OUT, "residual_demo_bands.csv"), index=False)
    summary["unit_tests"] = ut
    summary["residual_demo"] = {
        "n_profiles": 200, "n_points": int(len(d)),
        "d_mean": float(d.mean()), "d_std": float(d.std()),
        "d_p05_p95": [float(np.percentile(d, 5)), float(np.percentile(d, 95))],
    }
    summary["elapsed_s"] = time.time() - t_start
    with open(os.path.join(OUT, "exp0_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:2000])
    print("EXP-0 DONE")


if __name__ == "__main__":
    main()
