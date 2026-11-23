# -*- coding: utf-8 -*-
"""阶段 II 在线测量流程组件（EXP-4/5/6/7/8/9 共用）：
SceneData（场景锚点/全点数据集构建）、ab_split（A/B 划分）、
GP 端到端重构（背景 + m_theta + GP 后验）、B 点指标计算。"""
import os
import numpy as np
import pandas as pd

from .. import config as C
from . import anchors, uobs, meanet, scenes as scenelib
from . import gp as gplib

EPOCH = np.datetime64("2000-01-01T00:00:00").astype(np.int64)


def load_stack():
    """加载阶段 I 冻结产物：模型、特征、场景库。返回 dict。"""
    argo = None
    from . import io_data
    argo = io_data.load_argo()
    # np.load(npz) 返回懒加载 NpzFile：每次 res["z"] 都会重新解压整个数组。
    # 下游存在按剖面循环索引 res 的用法（如 EXP-8 的 EOFModel），懒加载会
    # 退化为 O(n_prof) 次全数组解压（实测 ~10 min 无进展）。此处一次性
    # 物化为 dict，语义不变、消除重复解压（日志【问题30】）。
    with np.load(os.path.join(C.CACHE, "argo_residuals.npz")) as _z:
        res = {k: _z[k] for k in _z.files}
    from .interp import BgField
    from . import features
    bg = BgField(C.CACHE)
    Xfeat = features.build_obs_features(argo, bg)["X"]
    model, stdr, fcols, mcfg = meanet.load_model(
        os.path.join(C.WEIGHTS, "exp3", "meanet_final"))
    scenes, meta = scenelib.load_scenes()
    hp = None
    hp_path = os.path.join(C.RESULTS, os.environ.get("HP_DIR", "exp4b"),
                           "hp_table.json")
    if os.path.exists(hp_path):
        import json
        try:
            with open(hp_path, encoding="utf-8") as f:
                hp = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"hp_table.json 无效（{hp_path}）：{e}。"
                f"若为 EXP-4 中途崩溃残留（流式写入未收尾），"
                f"请删除该文件后重跑 EXP-4。") from e
    return {"argo": argo, "res": res, "bg": bg, "Xfeat": Xfeat,
            "model": model, "stdr": stdr, "fcols": fcols, "mcfg": mcfg,
            "scenes": scenes, "smeta": meta, "hp": hp}


def km_dist(lat1, lon1, lat2, lon2):
    """球面大圆距离 (km)，numpy 向量化。"""
    R = 6371.0
    la1, lo1, la2, lo2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dla, dlo = la2 - la1, lo2 - lo1
    a = np.sin(dla / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlo / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def season_from_t64(t64):
    """月份 -> 季节索引（12/1/2 冬=0，3-5 春=1，6-8 夏=2，9-11 秋=3）。"""
    month = pd.Series(pd.to_datetime(np.asarray(t64).astype("datetime64[s]"))).dt.month.values
    s = np.zeros(len(month), dtype=int)
    s[np.isin(month, [12, 1, 2])] = 0
    s[np.isin(month, [3, 4, 5])] = 1
    s[np.isin(month, [6, 7, 8])] = 2
    s[np.isin(month, [9, 10, 11])] = 3
    return s


class SceneData:
    """场景级数据访问（锚点降采样或全点）。"""

    def __init__(self, stack):
        self.argo = stack["argo"]
        self.res = stack["res"]
        self.Xfeat = stack["Xfeat"]
        self.model, self.stdr, self.fcols = (stack["model"], stack["stdr"],
                                             stack["fcols"])
        self.bg = stack["bg"]
        self.mcfg = stack["mcfg"]
        self.sbi = scenelib.scene_by_id(stack["scenes"])
        self._prof_of_obs = np.repeat(np.arange(self.argo.n_prof),
                                      self.argo.counts)
        self.hp = stack["hp"]

    def kern_for(self, season, best_key=None):
        """按 EXP-4 冻结超参表取核（季节分型或全局）。"""
        if best_key is None:
            best_key = self.hp["_meta"]["best_key"]
        form, gran = best_key.split("_")
        g = self.hp[best_key]["groups"]
        kd = g[str(season)] if gran == "season" else g["all"]
        return gplib.kernel_from_dict(kd)

    def build(self, sid, prof_subset=None, full=False):
        """构建场景数据集 dict。prof_subset: 用作输入的剖面索引（A 集）；
        None = 全部成员。full=True 用全观测点（对照变体）。"""
        s = self.sbi[int(sid)]
        prof = np.array(s["member_idx"]) if prof_subset is None \
            else np.asarray(prof_subset)
        if full or len(prof) == 0:
            # 全观测点：slice(p) 返回 slice 对象（非索引数组），须展开为 arange
            # 再拼接——原写法 np.concatenate([slice,...]) 触发
            # "zero-dimensional arrays cannot be concatenated"（日志【问题22】）
            if len(prof):
                parts = []
                for p in prof:
                    sl = self.argo.slice(int(p))
                    parts.append(np.arange(sl.start, sl.stop))
                idx = np.concatenate(parts)
            else:
                idx = np.array([], dtype=int)
        else:
            idx = anchors.anchor_indices(self.argo, prof)
        z = self.res["z"][idx].astype(float)
        t = (self.argo.t64[self._prof_of_obs[idx]] - EPOCH).astype(float)
        lon = self.argo.lon[self._prof_of_obs[idx]].astype(float)
        lat = self.argo.lat[self._prof_of_obs[idx]].astype(float)
        X4 = np.column_stack([lon, lat, z, t])
        mean = meanet.predict(self.model, self.stdr, self.Xfeat[idx], self.fcols)
        d = self.res["d"][idx].astype(float)
        T = self.argo.T[idx].astype(float)
        S = self.argo.S[idx].astype(float)
        p = self.argo.p[idx].astype(float)
        u = uobs.u_obs_c(T, S, p, z)
        return {"sid": int(sid), "X4": X4, "d": d, "u": u, "mean": mean, "z": z,
                "band": uobs.band_index(z), "season": s["season"],
                "prof": self._prof_of_obs[idx], "feat": self.Xfeat[idx]}

    def ab(self, sid, seed=0):
        """场景内 A(输入 2/3)/B(留出 1/3) 划分（EXP-2 规则）。"""
        mem = np.array(self.sbi[int(sid)]["member_idx"])
        r = np.random.default_rng(1000 + seed)
        perm = r.permutation(len(mem))
        n_in = max(8, int(round(len(mem) * 2 / 3)))
        return mem[perm[:n_in]], mem[perm[n_in:]]

    def build_from_profiles(self, prof_idx):
        """非场景输入集（EXP-6 CTD 扩展验证 / EXP-7 仿真输入）：
        以给定剖面索引构建 GP 输入数据集，season 由剖面时间推导。"""
        prof = np.asarray(prof_idx, dtype=int)
        idx = anchors.anchor_indices(self.argo, prof)
        z = self.res["z"][idx].astype(float)
        t = (self.argo.t64[self._prof_of_obs[idx]] - EPOCH).astype(float)
        lon = self.argo.lon[self._prof_of_obs[idx]].astype(float)
        lat = self.argo.lat[self._prof_of_obs[idx]].astype(float)
        X4 = np.column_stack([lon, lat, z, t])
        mean = meanet.predict(self.model, self.stdr, self.Xfeat[idx], self.fcols)
        T = self.argo.T[idx].astype(float)
        S = self.argo.S[idx].astype(float)
        p = self.argo.p[idx].astype(float)
        u = uobs.u_obs_c(T, S, p, z)
        seasons = season_from_t64(self.argo.t64[prof])
        season = int(np.bincount(seasons).argmax())
        return {"sid": -1, "X4": X4, "d": self.res["d"][idx].astype(float), "u": u,
                "mean": mean, "z": z, "band": uobs.band_index(z), "season": season,
                "prof": self._prof_of_obs[idx], "feat": self.Xfeat[idx]}

    def query_features(self, X4):
        """任意查询点 (lon,lat,z,t_rel_sec) 的均值网络预测（在线口径）。"""
        from . import features
        Xf, _ = features.make_features(self.bg, X4[:, 3].astype(np.int64),
                                       X4[:, 2], X4[:, 1], X4[:, 0])
        return meanet.predict(self.model, self.stdr, Xf, self.fcols)

    def fit_gp(self, dsA, kern=None):
        """单场景 GP 拟合（阶段 II 伪代码 7-9 行）。"""
        if kern is None:
            kern = self.kern_for(dsA["season"])
        return gplib.GPFit(dsA["X4"], dsA["d"], dsA["u"], dsA["mean"], kern)

    @staticmethod
    def dist_to_inputs(Xq, X4_in, batch=2000):
        """查询点到输入点集的最小水平距离(km)与最小时间距离(day)。"""
        kmx = 111.32 * np.cos(np.radians(np.mean(Xq[:, 1])))
        dh = np.empty(len(Xq))
        for s0 in range(0, len(Xq), batch):
            blk = Xq[s0:s0 + batch]
            dd = np.sqrt(((blk[:, None, 0] - X4_in[None, :, 0]) * kmx) ** 2
                         + ((blk[:, None, 1] - X4_in[None, :, 1]) * 110.57) ** 2)
            dh[s0:s0 + batch] = dd.min(axis=1)
        dt = np.abs(Xq[:, 3][:, None] - X4_in[:, 3][None, :]).min(axis=1) / 86400.0
        return dh, dt


def eval_on_B(pred, var, dsB, u_extra=None, k=1.96, tag=""):
    """B 点验收指标。pred/var: GP 预测；u_extra: 外加不确定度分量（u_m/u_eq/u_rep
    合成后的逐点值，EXP-5+ 传入；None 则纯 GP 口径）。返回指标 dict。"""
    err = pred - dsB["d"]
    if u_extra is None:
        sig = np.sqrt(var + dsB["u"] ** 2)
    else:
        sig = np.sqrt(var + u_extra ** 2)
    cov = float(np.mean(np.abs(err) <= k * sig))
    nll = float(np.mean(0.5 * (err ** 2 / sig ** 2 + np.log(2 * np.pi * sig ** 2))))
    # Winkler score（95% 名义）：条件以 err 与半宽比较（原 err<lo 把误差与
    # 声速绝对值 ~1500 m/s 比较，恒真——日志【问题19】）
    alpha = 0.05
    half = k * sig
    wink = np.where(err < -half, 2 * half + 2 / alpha * (-half - err),
                    np.where(err > half, 2 * half + 2 / alpha * (err - half),
                             2 * half))
    return {"tag": tag, "n": len(err), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))), "nll": nll,
            "cover95": cov, "mean_interval_width": float(np.mean(2 * k * sig)),
            "winkler": float(np.mean(wink)), "mean_sig": float(np.mean(sig))}


def coverage_tolerance(n, alpha=0.05, p=0.95):
    """二项置信带（J2 容差带）：真实覆盖率 p 的名义 95% 覆盖经验值置信区间。"""
    from scipy.stats import binom
    lo = binom.ppf(alpha / 2, n, p) / n
    hi = binom.ppf(1 - alpha / 2, n, p) / n
    return float(lo), float(hi)
