# -*- coding: utf-8 -*-
"""EXP-1 输入量不确定度评定（WP1）。
① u_obs：profiler_type 分型（Type B 规格 + Argo-CTD 时空匹配交叉比对 Type A，双窗口敏感性）
② u_rep：点→体元支撑代表性（背景水平梯度 × 网格尺度，Type B，分带）
③ u_b：HYCOM-观测残差统计（分深度带×季节；层间 PCHIP 形状误差检验；相关长度粗估）
④ u_eq：三方程差分抵消数值检验 + Jacobian LPU
⑤ CTD 双重用途循环性检查（留半交叉）
⑥ 全量残差对缓存（供 EXP-2/3 复用）：argo_residuals.npz
结果：results/exp1/*.csv + exp1_summary.json
"""
import sys, os, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from exp import config as C
from exp.lib import physics, io_data
from exp.lib.interp import BgField

RNG = np.random.default_rng(C.SEED)
OUT = C.result_dir("exp1")
EPOCH = np.datetime64("2000-01-01T00:00:00").astype(np.int64)

ZBANDS = [(0, 30), (30, 60), (60, 125), (125, 350), (350, 800), (800, 2000)]
ZB_NAMES = [f"{a}-{b}" for a, b in ZBANDS]


def band_index(z):
    b = np.full(len(z), -1, dtype=int)
    for bi, (z0, z1) in enumerate(ZBANDS):
        b[(z >= z0) & (z < z1)] = bi
    return b


def season_index(t64):
    month = pd.to_datetime(t64.astype("datetime64[s]")).month.values
    s = np.zeros(len(month), dtype=int)
    s[np.isin(month, [1, 2, 12])] = 0   # 冬
    s[np.isin(month, [3, 4, 5])] = 1    # 春
    s[np.isin(month, [6, 7, 8])] = 2    # 夏
    s[np.isin(month, [9, 10, 11])] = 3  # 秋
    return s


def km_dist(lat1, lon1, lat2, lon2):
    R = 6371.0
    la1, lo1, la2, lo2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dla = la2 - la1; dlo = lo2 - lo1
    a = np.sin(dla / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlo / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------- u_obs ----
def match_argo_ctd(argo, ctd, win_h, win_km):
    """时空匹配：每 CTD 站找窗口内最近 Argo 剖面。返回匹配对索引列表。"""
    pairs = []
    for j in range(ctd.n_prof):
        dt_h = np.abs(argo.t64 - ctd.t64[j]) / 3600.0
        near = np.where(dt_h <= win_h)[0]
        if len(near) == 0:
            continue
        dist = km_dist(argo.lat[near], argo.lon[near], ctd.lat[j], ctd.lon[j])
        ok = near[dist <= win_km]
        if len(ok) == 0:
            continue
        i_best = ok[np.argmin(km_dist(argo.lat[ok], argo.lon[ok], ctd.lat[j], ctd.lon[j]))]
        pairs.append((int(i_best), int(j), float(km_dist(argo.lat[i_best], argo.lon[i_best],
                                                          ctd.lat[j], ctd.lon[j])),
                      float(abs(argo.t64[i_best] - ctd.t64[j]) / 3600.0)))
    return pairs


def profile_diff_stats(argo, ctd, i, j):
    """Argo 剖面 i vs CTD 站 j 的分带温/盐差（Argo - CTD，层对齐线性插值）。"""
    sa, sb = argo.slice(i), ctd.slice(j)
    pa, Ta, Sa_ = argo.p[sa], argo.T[sa], argo.S[sa]
    pc, Tc, Sc = ctd.p[sb], ctd.T[sb], ctd.S[sb]
    out = []
    for bi, (z0, z1) in enumerate(ZBANDS):
        p0, p1 = z0, z1  # 压力近似深度（带边界处差 <1%）
        ma = (pa >= p0) & (pa < p1)
        mc = (pc >= p0) & (pc < p1)
        if ma.sum() < 3 or mc.sum() < 3:
            continue
        # CTD 站通常采样密（2 dbar）；Argo 采样较稀 -> 把 CTD 插到 Argo 压力层
        Tc_i = np.interp(pa[ma], pc[mc], Tc[mc])
        Sc_i = np.interp(pa[ma], pc[mc], Sc[mc])
        dT = Ta[ma] - Tc_i
        dS = Sa_[ma] - Sc_i
        out.append((bi, len(dT), np.median(dT), np.median(dS),
                    1.4826 * np.median(np.abs(dT - np.median(dT))),
                    1.4826 * np.median(np.abs(dS - np.median(dS)))))
    return out


def uobs_assess(argo, ctd):
    rows = []
    pair_sets = {}
    # 定标候选两档 + 宽窗佐证档。佐证档（72h/50km）时空变异混入差值，
    # 仅作量级一致性检查、不进定标表（日志【问题11·计划外发现】：
    # CTD72 与 Argo 时空交叠过少，24h/50km 仅 1 对，Type A 不可行）
    for win_h, win_km, tag in [(12, 25, "w12h25km"), (24, 50, "w24h50km"),
                               (72, 50, "w72h50km_corrob")]:
        pairs = match_argo_ctd(argo, ctd, win_h, win_km)
        pair_sets[tag] = pairs
        for i, j, dist, dth in pairs:
            for bi, n, medT, medS, madT, madS in profile_diff_stats(argo, ctd, i, j):
                rows.append({"win": tag, "band": ZB_NAMES[bi], "n": n,
                             "d_T_med": medT, "d_T_mad": madT,
                             "d_S_med": medS, "d_S_mad": madS,
                             "dist_km": dist, "dt_h": dth,
                             "ptype": int(argo.ptype[i])})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "uobs_match_pairs_raw.csv"), index=False)
    # 汇总（按窗口×带）：p68 口径（时空变异正偏，MAD 会高估 -> 用分位对）
    summ = []
    for win in df["win"].unique():
        sub = df[df["win"] == win]
        for band in ZB_NAMES:
            sb = sub[sub["band"] == band]
            if len(sb) < 5:
                continue
            # 逐对中值差的总体分布（对内 MAD 已剔观测随机项 -> 剩余 = 系统差+时空变异）
            summ.append({
                "win": win, "band": band, "n_pairs": len(sb),
                "dT_med_overall": float(sb["d_T_med"].median()),
                "dT_p68": float(np.percentile(np.abs(sb["d_T_med"]), 68)),
                "dT_p95": float(np.percentile(np.abs(sb["d_T_med"]), 95)),
                "dS_med_overall": float(sb["d_S_med"].median()),
                "dS_p68": float(np.percentile(np.abs(sb["d_S_med"]), 68)),
                "dS_p95": float(np.percentile(np.abs(sb["d_S_med"]), 95)),
                "within_mad_T_median": float(sb["d_T_mad"].median()),
                "within_mad_S_median": float(sb["d_S_mad"].median()),
            })
    sdf = pd.DataFrame(summ)
    sdf.to_csv(os.path.join(OUT, "uobs_match_summary.csv"), index=False)
    # 分型统计（宽窗）
    type_rows = []
    sub = df[df["win"] == "w24h50km"]
    for pt, g in sub.groupby("ptype"):
        type_rows.append({"profiler_type": int(pt), "n_pairs": len(g),
                          "dT_med": float(g["d_T_med"].median()),
                          "dT_p68": float(np.percentile(np.abs(g["d_T_med"]), 68)),
                          "dS_med": float(g["d_S_med"].median()),
                          "dS_p68": float(np.percentile(np.abs(g["d_S_med"]), 68))})
    pd.DataFrame(type_rows).to_csv(os.path.join(OUT, "uobs_by_type.csv"), index=False)
    # 循环性检查：宽窗佐证档（对子最多）留半交叉——检验佐证统计对 CTD 子集
    # 划分的稳健性；定标主体已转 Type B，此检查降级为佐证稳健性记录
    pairs_w = pair_sets["w72h50km_corrob"]
    ctd_ids = sorted({j for _, j, _, _ in pairs_w})
    half = set(ctd_ids[::2])
    halves = {}
    for name, sel in (("A", half), ("B", set(ctd_ids) - half)):
        rec = []
        for i, j, dist, dth in pairs_w:
            if j not in sel:
                continue
            for bi, n, medT, medS, madT, madS in profile_diff_stats(argo, ctd, i, j):
                rec.append((bi, medT, medS))
        arr = np.array(rec, dtype=float).reshape(-1, 3)  # 空半集 -> (0,3)，安全跳过
        h = {}
        for bi in range(len(ZBANDS)):
            m = arr[:, 0] == bi
            if m.sum() >= 3:
                h[ZB_NAMES[bi]] = {"dT_med": float(np.median(arr[m, 1])),
                                   "dS_med": float(np.median(arr[m, 2]))}
        halves[name] = h
    return sdf, halves, {t: len(p) for t, p in pair_sets.items()}


def ctd_spec_assess(argo):
    """CTD 船基规格（Type B）-> 声速域分带表。验收真值自身的误差预算：
    SBE 911plus 类常规航次规格 u_T=0.002°C, u_S=0.003, u_p=3 dbar，
    Jacobian 传播到声速域（框架 §EXP-1①"CTD 按船基规格评定"，原缺实现）。"""
    uT, uS, up = 0.002, 0.003, 3.0
    sel = RNG.choice(argo.n_obs, 20000, replace=False)
    T, S_, p = (argo.T[sel].astype(float), argo.S[sel].astype(float),
                argo.p[sel].astype(float))
    dcdT, dcdS, dcdp = physics.sound_speed_jacobian(T, S_, p)
    u = np.sqrt((dcdT * uT) ** 2 + (dcdS * uS) ** 2 + (dcdp * up) ** 2)
    z = physics.z_from_p(p, np.full(len(p), 20.0))
    b = band_index(z)
    rows = [{"band": ZB_NAMES[bi], "n": int((b == bi).sum()),
             "u_ctd_rms_ms": float(np.sqrt((u[b == bi] ** 2).mean())),
             "u_ctd_p95_ms": float(np.percentile(u[b == bi], 95))}
            for bi in range(len(ZBANDS)) if (b == bi).sum() > 0]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "uctd_spec_by_band.csv"), index=False)
    return df


# ---------------------------------------------------------------- u_rep ----
def urep_assess(bg: BgField):
    """点->体元支撑代表性：0.5*格距*|水平梯度| 的分带 RMS（背景场统计）。"""
    dkm_x = 0.48 * 111.32 * np.cos(np.radians(20.0))
    dkm_y = 0.24 * 110.57
    rows = []
    n_sample_t = 200
    its = RNG.choice(bg.shape[0], n_sample_t, replace=False)
    g2_by_band = {bi: [] for bi in range(len(ZBANDS))}
    for it in its:
        snap = bg.snapshot(int(it))    # (8,84,63)
        gx = np.gradient(snap, axis=2) / dkm_x   # m/s per km
        gy = np.gradient(snap, axis=1) / dkm_y
        for bi, (z0, z1) in enumerate(ZBANDS):
            layers = [k for k, d in enumerate(bg.depths) if z0 <= d < z1] or [min(range(8), key=lambda k: min(abs(bg.depths[k]-z0), abs(bg.depths[k]-z1)))]
            g2_by_band[bi].append(0.25 * (dkm_x ** 2 * gx[layers] ** 2 + dkm_y ** 2 * gy[layers] ** 2).mean(axis=(0,)))
    out = []
    for bi, (z0, z1) in enumerate(ZBANDS):
        allg = np.concatenate(g2_by_band[bi])
        out.append({"band": ZB_NAMES[bi],
                    "u_rep_rms_ms": float(np.sqrt(allg.mean())),
                    "u_rep_p95_ms": float(np.sqrt(np.percentile(allg, 95)))})
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(OUT, "urep_by_band.csv"), index=False)
    return df


# ---------------------------------------------------------------- u_b ----
def build_residual_pairs(argo, bg, eq="chen-millero"):
    """全量残差对：d = c_obs - c_b（带分带/季节标签）。缓存 npz。"""
    cache = os.path.join(C.CACHE, "argo_residuals.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z
    t0 = time.time()
    n = argo.n_obs
    d = np.empty(n, dtype=np.float32)
    cb_all = np.empty(n, dtype=np.float32)
    z_all = np.empty(n, dtype=np.float32)
    # 按剖面批处理（每批 500 条）
    B = 500
    for s in range(0, argo.n_prof, B):
        e = min(s + B, argo.n_prof)
        t_s, z_s, la_s, lo_s = [], [], [], []
        for ip in range(s, e):
            sl = argo.slice(ip)
            zz = physics.z_from_p(argo.p[sl], argo.lat[ip]).astype(np.float64)
            tt = np.full(len(zz), float(argo.t64[ip] - EPOCH))
            t_s.append(tt); z_s.append(zz)
            la_s.append(np.full(len(zz), argo.lat[ip])); lo_s.append(np.full(len(zz), argo.lon[ip]))
        t_s = np.concatenate(t_s); z_s = np.concatenate(z_s)
        la_s = np.concatenate(la_s); lo_s = np.concatenate(lo_s)
        cb = bg.cs_at(t_s, z_s, la_s, lo_s)
        i0 = argo.starts[s]; i1 = argo.starts[e - 1] + argo.counts[e - 1]
        co = physics.sound_speed(argo.T[i0:i1].astype(float), argo.S[i0:i1].astype(float),
                                 argo.p[i0:i1].astype(float), eq)
        d[i0:i1] = (co - cb).astype(np.float32)
        cb_all[i0:i1] = cb.astype(np.float32)
        z_all[i0:i1] = z_s.astype(np.float32)
        if s % 4000 == 0:
            print(f"  residuals {s}/{argo.n_prof} ({time.time()-t0:.0f}s)", flush=True)
    band = band_index(z_all.astype(float))
    t64_exp = np.repeat(argo.t64, argo.counts)
    seas = season_index(t64_exp)
    np.savez_compressed(cache, d=d, cb=cb_all, z=z_all, band=band, season=seas,
                        ptype=np.repeat(argo.ptype, argo.counts))
    print(f"  residual pairs done: {n} points, {time.time()-t0:.0f}s")
    return np.load(cache)


def ub_assess(res, argo):
    """u_b：分带×季节 偏差/方差 + 层间 PCHIP 形状误差 + 相关长度粗估。"""
    d, z, band, seas = res["d"].astype(float), res["z"].astype(float), res["band"], res["season"]
    rows = []
    for bi in range(len(ZBANDS)):
        for s in range(4):
            m = (band == bi) & (seas == s)
            if m.sum() < 100:
                continue
            rows.append({"band": ZB_NAMES[bi], "season": ["winter", "spring", "summer", "autumn"][s],
                         "n": int(m.sum()), "bias": float(d[m].mean()), "std": float(d[m].std()),
                         "std_debias": float((d[m] - d[m].mean()).std()),
                         "p95_abs": float(np.percentile(np.abs(d[m]), 95))})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "ub_by_band_season.csv"), index=False)
    # 注：水平/时间相关长度估计改在 EXP-2 场景库内做（同块内剖面对，统计口径更贴 GP 超参）
    return df


def pchip_shape_error(argo, bg):
    """层间 PCHIP 形状误差：观测剖面在 HYCOM 节点深度的值 -> PCHIP 重建 vs 观测层间真值。"""
    nodes = bg.depths.copy()
    errs_by_band = {bi: [] for bi in range(len(ZBANDS))}
    sel = RNG.choice(argo.n_prof, 400, replace=False)
    from exp.lib.interp import pchip_eval
    for ip in sel:
        sl = argo.slice(int(ip))
        p, T, S_ = argo.p[sl].astype(float), argo.T[sl].astype(float), argo.S[sl].astype(float)
        if len(p) < 20:
            continue
        c = physics.sound_speed(T, S_, p)
        # 节点深度处观测值（压力近似深度；线性插值）
        cn = np.interp(nodes, p, c)
        V = np.tile(cn, (len(p), 1))
        c_hat = pchip_eval(nodes, V, p)
        e = c_hat - c
        b = band_index(p)
        for bi in range(len(ZBANDS)):
            m = b == bi
            if m.sum() > 0:
                errs_by_band[bi].append(np.sqrt(np.mean(e[m] ** 2)))
    rows = [{"band": ZB_NAMES[bi],
             "shape_rmse_ms": float(np.mean(v)) if v else None,
             "n_prof": len(v)} for bi, v in errs_by_band.items()]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "pchip_shape_error.csv"), index=False)
    return df


# ---------------------------------------------------------------- u_eq ----
def ueq_assess(argo, bg, res):
    """u_eq：三方程差分抵消检验 + Jacobian LPU + 压深换算敏感性。
    状态域统一取观测端样本（与背景 T/S 域同海区重叠；背景端方程差场由 EXP-10
    EqDelta 网格场覆盖，此处不做 NC 逐点抽样——8 万次单点读取会触发 HDF5
    chunk 重复解压，日志【问题5】）。"""
    n = 20000
    sel = RNG.choice(argo.n_obs, n, replace=False)
    T, S_, p = argo.T[sel].astype(float), argo.S[sel].astype(float), argo.p[sel].astype(float)
    # 三方程绝对值（观测端）
    c_cm = physics.sound_speed(T, S_, p, "chen-millero")
    c_gsw = physics.sound_speed(T, S_, p, "gsw")
    c_mk = physics.sound_speed(T, S_, p, "mackenzie")
    abs_gsw = c_gsw - c_cm
    abs_mk = c_mk - c_cm
    # 差分差异（二阶）：方程差异随声速的变化率 × 观测-背景差分幅度 p68
    dc_p68 = float(np.percentile(np.abs(res["d"].astype(float)), 68))
    slope_gsw = np.polyfit(c_cm, c_gsw - c_cm, 1)[0]
    slope_mk = np.polyfit(c_cm, c_mk - c_cm, 1)[0]
    u_eq_gsw = abs(slope_gsw) * dc_p68
    u_eq_mk = abs(slope_mk) * dc_p68
    # Jacobian（观测样本，分带）
    dcdT, dcdS, dcdp = physics.sound_speed_jacobian(T, S_, p)
    z_s = physics.z_from_p(p, np.full(n, 20.0))
    b = band_index(z_s)
    jac_rows = []
    for bi in range(len(ZBANDS)):
        m = b == bi
        if m.sum() < 10:
            continue
        jac_rows.append({"band": ZB_NAMES[bi], "n": int(m.sum()),
                         "dcdT_mean": float(dcdT[m].mean()), "dcdS_mean": float(dcdS[m].mean()),
                         "dcdp_mean": float(dcdp[m].mean())})
    # 压深换算敏感性：p ±0.5%（覆盖深度节点 -> 压力的换算误差量级）
    c_p_up = physics.sound_speed(T, S_, p * 1.005, "chen-millero")
    c_p_dn = physics.sound_speed(T, S_, p * 0.995, "chen-millero")
    u_p = float(np.abs(c_p_up - c_p_dn).max() / 2)
    out = {
        "abs_eq_diff": {
            "gsw_minus_cm_mean": float(abs_gsw.mean()), "gsw_minus_cm_std": float(abs_gsw.std()),
            "mk_minus_cm_mean": float(abs_mk.mean()), "mk_minus_cm_std": float(abs_mk.std())},
        "differential_second_order": {
            "slope_gsw": float(slope_gsw), "slope_mk": float(slope_mk),
            "dc_p68_ms": dc_p68,
            "u_eq_gsw_ms": u_eq_gsw, "u_eq_mk_ms": u_eq_mk},
        "z_to_p_sensitivity_0.5pct_max_ms": u_p,
        "jacobian_by_band": jac_rows,
    }
    return out


def main():
    t_start = time.time()
    print("== EXP-1: input uncertainty ==")
    bg = BgField(C.CACHE)
    argo = io_data.load_argo()
    ctd = io_data.load_ctd()

    # ① u_obs（交叉比对 + 分型 + 循环性 + CTD 规格）
    sdf, halves, n_pairs = uobs_assess(argo, ctd)
    ctd_spec = ctd_spec_assess(argo)
    uobs_decision = (
        "Type A Argo-CTD cross-comparison infeasible: n_pairs "
        f"{{'w12h25km': {n_pairs['w12h25km']}, 'w24h50km': {n_pairs['w24h50km']}, "
        f"'w72h50km_corrob': {n_pairs['w72h50km_corrob']}}} "
        "(threshold: >=5 pairs per band for calibration). u_obs primary = "
        "Type B instrument spec (lib/uobs.py SPEC table, surface band doubled); "
        "wide-window corroboration archived with caveat: 72h/50km differences "
        "conflate instrument error with spatiotemporal variability")
    # ② u_rep
    urep = urep_assess(bg)
    # 全量残差对
    res = build_residual_pairs(argo, bg)
    # ③ u_b + 形状误差
    ub = ub_assess(res, argo)
    pse = pchip_shape_error(argo, bg)
    # ④ u_eq
    ueq = ueq_assess(argo, bg, res)

    summary = {
        "exp": "exp1", "date": time.strftime("%Y-%m-%d %H:%M"),
        "uobs_decision": uobs_decision,
        "uobs_n_pairs": n_pairs,
        "uobs_match_summary": sdf.to_dict("records"),
        "uobs_ctd_half_crosscheck": halves,
        "ctd_spec_by_band": ctd_spec.to_dict("records"),
        "urep": urep.to_dict("records"),
        "ub_head": ub.to_dict("records")[:12],
        "pchip_shape_error": pse.to_dict("records"),
        "ueq": ueq,
        "n_residual_pairs": int(len(res["d"])),
        "elapsed_s": time.time() - t_start,
    }
    with open(os.path.join(OUT, "exp1_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    print("EXP-1 DONE", time.time() - t_start)


if __name__ == "__main__":
    main()
