# -*- coding: utf-8 -*-
"""EXP-8 基线对照（WP4，framework §4-EXP8）。
七类基线 × 同一防泄漏协议 × 同一验证点集（test 场景 B 点，seeds 0-2）：
  B1 hycom   : HYCOM 直接输出（点估计=c_b；区间=u_b 分带统计，GUM 正态）
  B2 okrig   : 克里金/目标分析（3D 核、零均值、同方差噪声；克里金方差）
  B3 eof     : EOF 回归（train 残差剖面 EOF 基 + 场景内 IDW 系数插值）
  B4 enoi    : EnOI（HYCOM 时间异常集合 + 距离局地化分析）
  B5 dl      : DL 逐点集成基线（φ(q)+kNN 上下文特征，5 成员 deep ensemble + conformal）
  B6 const-u : 主方法点估计 + 常数不确定度（train 残差 RMS）
  B7 gum     : 主方法点估计 + GUM 简化合成 sqrt(u_b²+u_obs²/n_in)
  ours       : 完整两阶段 NN 残差 GP + u_c + k
配对显著性：站×深度带聚合 err² 的配对 Wilcoxon 检验（ours vs 各基线）。
产出：results/exp8/*.csv + exp8_summary.json
"""
import sys, os, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from exp import config as C
from exp.lib import physics, uobs, anchors, scenes as scenelib, meanet
from exp.lib import gp as gplib
from exp.lib.pipeline import SceneData, load_stack, km_dist
from exp.lib.interp import pchip_eval
from exp.lib.uc import UcComposer
from exp.lib import banded
from sklearn.neighbors import NearestNeighbors
from scipy.stats import wilcoxon

OUT = C.result_dir("exp8")
RNG = np.random.default_rng(C.SEED)
EPOCH = np.datetime64("2000-01-01T00:00:00").astype(np.int64)
SEEDS = (0, 1, 2)
MAX_SCENES = 40          # test 场景上限（超出按季节分层子采样）
N_ENS = 150              # EnOI 集合成员数
K_EOF = 8                # EOF 模态数
DL_ENSEMBLE = 5
K_CTX = 8                # DL 上下文近邻数


# ------------------------------------------------ B1 hycom ----
def ub_by_band(stack):
    """u_b 分带统计：train 场景观测残差 d 的 RMS。"""
    scenes = stack["scenes"]
    argo = stack["argo"]
    res = stack["res"]
    prof_of_obs = np.repeat(np.arange(argo.n_prof), argo.counts)
    from exp.lib.scenes import prof_scene_index
    ps = prof_scene_index(scenes, argo.n_prof)
    train_mask = np.isin(ps[prof_of_obs],
                         scenelib.scene_ids_by_role(scenes, "train"))
    z, d = res["z"], res["d"]
    bi = uobs.band_index(z)
    ub = np.zeros(6)
    for b in range(6):
        m = train_mask & (bi == b)
        ub[b] = np.sqrt(np.mean(d[m] ** 2)) if m.sum() > 50 else np.sqrt(np.mean(d ** 2))
    return ub


# ------------------------------------------------ B2 okrig ----
class OKrig:
    """克里金/目标分析：3D（时间维置零）核、零均值、同方差噪声。
    超参在 train 子集上 ML 拟合一次（全局共享、无季节分型——与传统目标分析
    的经验变差图口径对应）。"""

    def __init__(self, SD, train_sids, rng):
        ds = []
        for sid in rng.choice(train_sids, min(12, len(train_sids)), replace=False):
            A, _ = SD.ab(int(sid), seed=0)
            d = SD.build(int(sid), A)
            X4 = d["X4"].copy()
            X4[:, 3] = 0.0
            u = np.full(len(d["u"]), float(np.sqrt(np.mean(d["u"] ** 2))))
            ds.append((X4, d["d"], u, np.zeros(len(d["d"]))))
        init = {"log_s2": np.log(np.var(np.concatenate([x[1] for x in ds]))),
                "log_lh": np.log(150.0), "log_lz": np.log(120.0), "log_lt": np.log(1e6)}
        kern, lmls, _ = gplib.fit_hyperparams(ds, form="prod", init=init, n_iter=120)
        self.kern = kern
        self.mean_lml = float(np.mean(lmls))

    def fit(self, dsA):
        X4 = dsA["X4"].copy()
        X4[:, 3] = 0.0
        u = np.full(len(dsA["u"]), float(np.sqrt(np.mean(dsA["u"] ** 2))))
        return gplib.GPFit(X4, dsA["d"], u, np.zeros(len(dsA["d"])), self.kern)

    def predict(self, g, Xq):
        Xq3 = Xq.copy()
        Xq3[:, 3] = 0.0
        pred, var = g.predict(Xq3, np.zeros(len(Xq)))
        return pred, var


# ------------------------------------------------ B3 eof ----
class EOFModel:
    """EOF 回归：train 剖面锚点残差 SVD；场景内 A 剖面系数的时空 IDW 插值。"""

    def __init__(self, SD, scenes):
        argo = SD.argo
        res_z = np.asarray(SD.res["z"])
        res_d = np.asarray(SD.res["d"])
        train_sids = set(scenelib.scene_ids_by_role(scenes, "train"))
        z_grid = anchors.ANCHOR_Z[anchors.ANCHOR_Z <= C.Z_MAX]
        rows, keep_prof = [], []
        for s in scenes:
            if s["scene_id"] not in train_sids:
                continue
            for pi in s["member_idx"]:
                sl = argo.slice(int(pi))
                z, d = res_z[sl], res_d[sl]
                ok = (z >= 0) & (z <= C.Z_MAX)
                z, d = z[ok], d[ok]
                if len(z) < 15:
                    continue
                rows.append(np.interp(z_grid, np.sort(z), d[np.argsort(z)]))
                keep_prof.append(int(pi))
        M = np.array(rows)
        self.mean_prof = M.mean(axis=0)
        Mc = M - self.mean_prof
        _, _, Vt = np.linalg.svd(Mc, full_matrices=False)
        self.eof = Vt[:K_EOF]
        self.z_grid = z_grid
        self.keep_prof = np.array(keep_prof, dtype=int)
        rec = self.mean_prof + (Mc @ self.eof.T) @ self.eof
        err = rec - M
        self.ub = np.array([np.sqrt(np.mean(err[:, (z_grid >= a) & (z_grid < b)] ** 2))
                            for a, b in uobs.ZBANDS])
        self.n_train = len(keep_prof)

    def predict(self, SD, dsA, Xq):
        """A 剖面 EOF 系数 -> 时空 IDW -> 查询点重构（向量化）。"""
        prof_ids = np.unique(dsA["prof"])
        coefs, pos = [], []
        for pi in prof_ids:
            m = dsA["prof"] == pi
            z, d = dsA["z"][m], dsA["d"][m]
            o = np.argsort(z)
            dv = np.interp(self.z_grid, z[o], d[o])
            coefs.append((dv - self.mean_prof) @ self.eof.T)
            pos.append([SD.argo.lat[pi], SD.argo.lon[pi],
                        SD.argo.t64[pi] - EPOCH])   # 与 X4[:,3] 同基准（相对 2000）
        coefs = np.array(coefs)                      # (nA, K)
        pos = np.array(pos)                          # (nA, 3)
        dh = km_dist(Xq[:, 1][:, None], Xq[:, 0][:, None],
                     pos[None, :, 0], pos[None, :, 1])
        dt = np.abs(Xq[:, 3][:, None] - pos[None, :, 2]) / 86400.0
        w = 1.0 / np.sqrt((dh + 20.0) ** 2 + (dt * 30.0 + 5.0) ** 2)   # (Nq, nA)
        a_hat = (w @ coefs) / w.sum(1, keepdims=True)                   # (Nq, K)
        pred = np.interp(Xq[:, 2], self.z_grid, self.mean_prof)
        for k in range(K_EOF):
            pred = pred + a_hat[:, k] * np.interp(Xq[:, 2], self.z_grid, self.eof[k])
        bi = np.clip(uobs.band_index(Xq[:, 2]), 0, 5)
        return pred, self.ub[bi]


# ------------------------------------------------ B4 enoi ----
class EnOI:
    """EnOI：HYCOM 时间异常静态集合 + 高斯距离局地化，全向量化。"""

    def __init__(self, SD, n_ens=N_ENS, loc_km=300.0):
        bg = SD.bg
        nt = bg.shape[0]
        idx = np.linspace(5, nt - 6, n_ens).astype(int)
        sub = np.stack([bg.snapshot(int(i)) for i in idx])     # (E,8,ny,nx)
        self.anom = sub - sub.mean(axis=0)
        self.bg, self.E, self.loc_km = bg, n_ens, loc_km

    def _anom_at_batch(self, z, lat, lon):
        """集合异常在任意 (z,lat,lon) 的插值。返回 (E, N)。"""
        lon_ax, lat_ax = self.bg.lon, self.bg.lat
        ix = np.clip(np.searchsorted(lon_ax, lon, side="right") - 1, 0, len(lon_ax) - 2)
        iy = np.clip(np.searchsorted(lat_ax, lat, side="right") - 1, 0, len(lat_ax) - 2)
        fx = np.clip((lon - lon_ax[ix]) / (lon_ax[ix + 1] - lon_ax[ix]), 0, 1)
        fy = np.clip((lat - lat_ax[iy]) / (lat_ax[iy + 1] - lat_ax[iy]), 0, 1)
        A = self.anom                                       # (E,8,ny,nx)
        V = (A[:, :, iy, ix] * (1 - fx) * (1 - fy)
             + A[:, :, iy, ix + 1] * fx * (1 - fy)
             + A[:, :, iy + 1, ix] * (1 - fx) * fy
             + A[:, :, iy + 1, ix + 1] * fx * fy)           # (E,8,N)
        N = len(z)
        Vr = V.transpose(0, 2, 1).reshape(self.E * N, 8)
        vals = pchip_eval(self.bg.depths, Vr, np.tile(z, self.E))
        return vals.reshape(self.E, N)

    def analyze_predict(self, dsA, Xq):
        """EnOI 分析：返回 (增量 pred_inc, 分析方差 var_a) 在查询点。
        inc = K d，K = (S_qᵀS/(E-1) ⊙ L_qo) A⁻¹；var = var_ens − K (SᵀS/(E-1) ⊙ L_oo)ᵀ…"""
        X4, d, u = dsA["X4"], dsA["d"], dsA["u"]
        S = self._anom_at_batch(X4[:, 2], X4[:, 1], X4[:, 0])    # (E,Nobs)
        E, Nobs = S.shape
        # 观测-观测局地化
        dhoo = km_dist(X4[:, 1][:, None], X4[:, 0][:, None],
                       X4[None, :, 1], X4[None, :, 0])
        Loo = np.exp(-0.5 * (dhoo / self.loc_km) ** 2)
        # 状态维样本协方差：S 为 (E, Nobs)，须用 SᵀS（Nobs×Nobs），
        # 写成 SSᵀ 会得到集合维 (E×E) 并触发广播错误（日志【问题33】）
        B = S.T @ S / (E - 1)
        # 数值正则：SᵀS 秩 ≤ E-1=149，而 Nobs 可达 740；仅加 diag(u²)（~4e-4）
        # 时 A 近奇异，解 w 爆炸（实测 |err| 达 100 m/s，日志【问题35】）。
        # 加状态方差 1% 的 ridge 保证可解（EnOI 标准做法：局地化 + 正则）。
        ridge = 1e-2 * float(np.mean(np.diag(B)))
        A = B * Loo + np.diag(u ** 2 + ridge)
        w = np.linalg.solve(A, d)                                 # (Nobs,)
        Sq = self._anom_at_batch(Xq[:, 2], Xq[:, 1], Xq[:, 0])    # (E,Nq)
        dhqo = km_dist(Xq[:, 1][:, None], Xq[:, 0][:, None],
                       X4[None, :, 1], X4[None, :, 0])
        Lqo = np.exp(-0.5 * (dhqo / self.loc_km) ** 2)            # (Nq,Nobs)
        C = (Sq.T @ S / (E - 1)) * Lqo                            # (Nq,Nobs)
        inc = C @ w
        var_ens = (Sq ** 2).sum(0) / (E - 1)
        Ainv_C = np.linalg.solve(A, C.T)                          # (Nobs,Nq)
        var_a = var_ens - np.einsum("ij,ji->i", C, Ainv_C)
        return inc, np.maximum(var_a, 0.0)


# ------------------------------------------------ B5 dl ----
def make_ctx_features(qX4, ctx_X4, ctx_d, ctx_u, k=K_CTX, q_prof=None, ctx_prof=None):
    """查询点 kNN 上下文特征。
    q_prof/ctx_prof 给定时排除同剖面近邻（训练防泄漏）；推理（B 点）不传。"""
    from exp.lib.pipeline import km_dist as _km
    n = len(qX4)
    kk = min(k + (8 if q_prof is not None else 0), len(ctx_X4))
    nn = NearestNeighbors(n_neighbors=kk).fit(ctx_X4[:, :2])
    dd, ii = nn.kneighbors(qX4[:, :2])
    if q_prof is not None:
        same = ctx_prof[ii] == q_prof[:, None]
        # 从近邻列表中剔除同剖面，取前 k
        order = np.argsort(same, axis=1, kind="stable")   # False(0) 排前
        ii_sorted = np.take_along_axis(ii, order, axis=1)
        ii = ii_sorted[:, :min(k, ii_sorted.shape[1])]
    k2 = ii.shape[1]
    dh = _km(qX4[:, 1][:, None].repeat(k2, 1).ravel(),
             qX4[:, 0][:, None].repeat(k2, 1).ravel(),
             ctx_X4[ii, 1].ravel(), ctx_X4[ii, 0].ravel()).reshape(n, k2)
    dt = np.abs(qX4[:, 3][:, None] - ctx_X4[ii, 3]) / 86400.0
    dz = np.abs(qX4[:, 2][:, None] - ctx_X4[ii, 2])
    wgt = 1.0 / np.sqrt(dh ** 2 + (dz * 0.02) ** 2 + 1.0)
    wd = ctx_d[ii]
    wd_mean = (wgt * wd).sum(1) / wgt.sum(1)
    var_eff = (wgt * (wd - wd_mean[:, None]) ** 2).sum(1) / wgt.sum(1)
    return np.column_stack([wd_mean, np.sqrt(var_eff), dh.min(1), dt.mean(1),
                            dh.mean(1), np.full(n, len(ctx_X4)),
                            np.full(n, float(np.mean(ctx_u)))]).astype(np.float32)


def train_dl_ensemble(SD, scenes, rng):
    """5 成员 deep ensemble：φ(q) + 上下文特征 -> d。成员差异=种子+场景 bootstrap。

    训练口径与推理口径严格一致：场景内 A(2/3)/B(1/3) 划分，**查询点取 B、
    上下文集取 A**。早期版本用「全场景自查询 + 排除同剖面」，使训练时查询点与
    上下文点几乎重合（dh.min≈0），而推理时 B 点与 A 点空间分离，特征分布错位
    导致外推崩坏（实测 val 1.26 -> 推理 rmse 5~11，日志【问题34】）。"""
    train_sids = scenelib.scene_ids_by_role(scenes, "train")
    with open(os.path.join(C.RESULTS, "exp3", "val_scene_ids.json"), encoding="utf-8") as f:
        val_sids = json.load(f)
    tr_sids = [s for s in train_sids if s not in set(val_sids)]

    def ab_samples(sid, seed=0):
        A, B = SD.ab(int(sid), seed)
        dsA, dsB = SD.build(int(sid), A), SD.build(int(sid), B)
        Cf = make_ctx_features(dsB["X4"], dsA["X4"], dsA["d"], dsA["u"])
        return np.column_stack([dsB["feat"], Cf]), dsB["d"], dsB["u"]

    members, stdrs, hist_all = [], [], []
    for mi in range(DL_ENSEMBLE):
        boot = rng.choice(tr_sids, int(0.9 * len(tr_sids)), replace=True)
        Xs, ys, us = [], [], []
        for sid in boot:
            X, y, u = ab_samples(sid)
            Xs.append(X); ys.append(y); us.append(u)
        Xtr = np.concatenate(Xs); ytr = np.concatenate(ys); utr = np.concatenate(us)
        Xv, yv, uv = [], [], []
        for sid in val_sids[:8]:
            X, y, u = ab_samples(sid)
            Xv.append(X); yv.append(y); uv.append(u)
        Xva = np.concatenate(Xv); yva = np.concatenate(yv); uva = np.concatenate(uv)
        model, stdr, hist, fcols = meanet.train_meanet(
            Xtr, ytr, utr, Xva, yva, uva, width=128, depth=4, seed=100 + mi,
            max_epoch=60, patience=6)
        members.append(model); stdrs.append(stdr); hist_all.append(hist["best"])
        print(f"    dl member {mi}: val_wrmse={hist['best']:.4f}")
    return {"members": members, "stdrs": stdrs, "val_hist": hist_all}


def dl_predict(SD, dle, Xq, dsA):
    """DL 基线预测：φ(q)=查询点原始 8 维特征 + A 集上下文（B 剖面不在 A 集，无需排除）。
    注意：须与 train_dl_ensemble 的特征口径一致（feat 8 列 + ctx 7 列 = 15），
    不能用 query_features 的标量预测值代替 feat（否则列数错位）。"""
    from exp.lib import features
    phi, _ = features.make_features(SD.bg, Xq[:, 3].astype(np.int64),
                                    Xq[:, 2], Xq[:, 1], Xq[:, 0])
    Cf = make_ctx_features(Xq, dsA["X4"], dsA["d"], dsA["u"])
    X = np.column_stack([phi, Cf])
    preds = np.stack([meanet.predict(m, s, X) for m, s in
                      zip(dle["members"], dle["stdrs"])])
    return preds.mean(0), preds.std(0)


def calib_dl_k(SD, dle, n_scenes=8):
    """DL 基线共形 k：calib 场景 A->B 上 s=|err|/u 的 95% 分位。"""
    scenes = SD.sbi
    calib_sids = [sid for sid, s in scenes.items() if s["role"] == "calib"][:n_scenes]
    s_all = []
    for sid in calib_sids:
        A, B = SD.ab(sid, 0)
        dsA, dsB = SD.build(sid, A), SD.build(sid, B)
        pd_, sd_ = dl_predict(SD, dle, dsB["X4"], dsA)
        u = np.sqrt(sd_ ** 2 + dsB["u"] ** 2)
        s_all.append(np.abs(pd_ - dsB["d"]) / np.maximum(u, 1e-6))
    s = np.concatenate(s_all)
    r = int(np.ceil((len(s) + 1) * 0.95))
    return float(np.sort(s)[min(r, len(s)) - 1])


# ------------------------------------------------ 主流程 ----
def main():
    t0 = time.time()
    print("== EXP-8: baselines comparison ==")
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

    print("-- building baselines --")
    train_sids = scenelib.scene_ids_by_role(scenes, "train")
    UB = ub_by_band(stack)
    print(f"  u_b by band: {np.round(UB, 3).tolist()}")
    ok = OKrig(SD, train_sids, RNG)
    print(f"  okrig kern: {ok.kern.to_dict()}")
    eofm = EOFModel(SD, scenes)
    print(f"  eof: {eofm.n_train} train profiles, K={K_EOF}")
    enoi = EnOI(SD)
    print(f"  enoi: E={enoi.E}")
    print("-- training DL ensemble --")
    dle = train_dl_ensemble(SD, scenes, RNG)
    dl_k = calib_dl_k(SD, dle)
    print(f"  dl conformal k: {dl_k:.3f}")
    const_u = float(np.sqrt(np.mean(np.concatenate(
        [SD.build(int(s))["d"] ** 2
         for s in RNG.choice(train_sids, 15, replace=False)]))))

    print("-- evaluation loop --")
    rows = []
    for sid in test_sids:
        for seed in SEEDS:
            A, B = SD.ab(sid, seed)
            dsA, dsB = SD.build(sid, A), SD.build(sid, B)
            Xq, y_true, zq = dsB["X4"], dsB["d"], dsB["z"]
            band = np.clip(uobs.band_index(zq), 0, 5)
            season = dsB["season"]
            dh, _ = SD.dist_to_inputs(Xq, dsA["X4"])
            n_in = len(np.unique(dsA["prof"]))

            def emit(tag, pred, u, k=1.96):
                err = pred - y_true
                rows.append(pd.DataFrame({
                    "sid": sid, "seed": seed, "method": tag, "band": band,
                    "prof": dsB["prof"],
                    "err": err, "u": u, "k": k,
                    "covered": np.abs(err) <= k * u, "err2": err ** 2,
                    "width": 2 * k * u}))

            # ours（完整管线：banded block GP + 共形 k，EXP-6d 冻结）
            pred, var = banded.predict_main(SD, dsA, Xq, band_kerns, mean_q=dsB["mean"])
            u_ex = uc.extra_std(zq, np.full(len(zq), season))
            u_c = np.sqrt(var + u_ex ** 2)
            k_ours = uc.k_at(zq, np.full(len(zq), season), dh)
            emit("ours", pred, u_c, k_ours)
            # B1 hycom（残差域口径：HYCOM 直接输出 = 背景本身，残差预测恒为 0）
            cb = SD.bg.cs_at(Xq[:, 3], zq, Xq[:, 1], Xq[:, 0])
            emit("B1_hycom", np.zeros(len(Xq)), UB[band])
            # B2 okrig
            gk = ok.fit(dsA)
            pk, vk = ok.predict(gk, Xq)
            emit("B2_okrig", pk, np.sqrt(vk + dsB["u"] ** 2))
            # B3 eof（predict 返回残差域重构，勿再加背景 c_b）
            pe, ue = eofm.predict(SD, dsA, Xq)
            emit("B3_eof", pe, np.sqrt(ue ** 2 + dsB["u"] ** 2))
            # B4 enoi（inc 为相对背景的分析增量，即残差域估计，勿再加背景 c_b）
            inc, vq = enoi.analyze_predict(dsA, Xq)
            emit("B4_enoi", inc, np.sqrt(vq + UB[band] ** 2))
            # B5 dl
            pdl, sdl = dl_predict(SD, dle, Xq, dsA)
            emit("B5_dl", pdl, np.sqrt(sdl ** 2 + dsB["u"] ** 2), dl_k)
            # B6 const-u / B7 gum（主方法点估计 + 简化区间口径）
            emit("B6_constu", pred, np.full(len(Xq), const_u))
            emit("B7_gum", pred, np.sqrt(UB[band] ** 2 + np.mean(dsA["u"] ** 2) / n_in))
        print(f"  scene {sid} done ({time.time() - t0:.0f}s)")

    df = pd.concat(rows, ignore_index=True)
    df.to_csv(os.path.join(OUT, "all_methods_detail.csv"), index=False)

    # ---- 汇总（站×带聚合计权） ----
    grp = df.groupby(["method", "sid", "seed", "prof", "band"])
    agg = grp.agg(cov=("covered", "mean"), e2=("err2", "mean")).reset_index()
    m2 = agg.groupby(["method", "sid", "seed"]).agg(cov=("cov", "mean"),
                                                    mse=("e2", "mean")).reset_index()
    m3 = m2.groupby("method").agg(cov=("cov", "mean"),
                                  rmse=("mse", lambda x: float(np.sqrt(x.mean()))),
                                  n_eval=("cov", "size")).reset_index()
    width = df.groupby("method")["width"].mean().rename("width_mean")
    half = df["k"] * df["u"]
    df["wink"] = np.where(df["err"] < -half, df["width"] + 2 / 0.05 * (-half - df["err"]),
                          np.where(df["err"] > half,
                                   df["width"] + 2 / 0.05 * (df["err"] - half),
                                   df["width"]))
    wink = df.groupby("method")["wink"].mean().rename("winkler")
    m3 = m3.merge(width.reset_index(), on="method").merge(wink.reset_index(),
                                                          on="method")
    m3.to_csv(os.path.join(OUT, "baseline_metrics.csv"), index=False)
    print(m3.round(3).to_string(index=False))

    # ---- 配对显著性 ----
    print("-- paired tests --")
    pair_rows = []
    base_e2 = df[df["method"] == "ours"].groupby(
        ["sid", "seed", "prof", "band"])["err2"].mean()
    for meth in m3["method"]:
        if meth == "ours":
            continue
        other = df[df["method"] == meth].groupby(
            ["sid", "seed", "prof", "band"])["err2"].mean()
        j = pd.concat([base_e2.rename("ours"), other.rename("other")], axis=1).dropna()
        d = j["ours"] - j["other"]
        if len(d) > 20 and (d != 0).any():
            stat, p = wilcoxon(d, alternative="less")
        else:
            p = np.nan
        pair_rows.append({"method": meth, "n_pairs": len(d),
                          "mean_err2_diff": float(d.mean()),
                          "wilcoxon_p_ours_smaller": float(p) if p == p else None})
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(os.path.join(OUT, "paired_tests.csv"), index=False)
    print(pair_df.to_string(index=False))

    summary = {
        "exp": "exp8", "date": time.strftime("%Y-%m-%d %H:%M"),
        "n_test_scenes": len(test_sids), "seeds": list(SEEDS),
        "ub_by_band": UB.tolist(), "const_u": const_u, "dl_k": dl_k,
        "okrig_kern": ok.kern.to_dict(), "okrig_mean_lml": ok.mean_lml,
        "eof_n_train_profiles": eofm.n_train, "eof_K": K_EOF,
        "enoi_E": enoi.E, "enoi_loc_km": enoi.loc_km,
        "dl_val_hist": dle["val_hist"],
        "metrics": m3.to_dict("records"),
        "paired": pair_df.to_dict("records"),
        "elapsed_s": time.time() - t0,
    }
    with open(os.path.join(OUT, "exp8_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    print("EXP-8 DONE", time.time() - t0)


if __name__ == "__main__":
    main()
