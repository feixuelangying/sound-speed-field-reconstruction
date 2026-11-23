# -*- coding: utf-8 -*-
"""残差 GP：4D 各向异性时空-深度核（D5），逐点异方差噪声固定为评定值 u_obs²（D6，
噪声不拟合——计量关键）。torch 实现（float64 + GPU）。

坐标约定（输入 X 的 4 列）：lon(deg), lat(deg), z(m), t(相对2000-01-01秒)。
核内部换算：水平 km（纬度参考 cos）、深度 m、时间 day。

核形式候选（EXP-4 定标）：
  prod : σ² · M52(Δh/ℓ_h) · M32(Δz/ℓ_z) · SE(Δt/ℓ_t)        [主方案]
  add  : σ²_h M52(Δh/ℓ_h) + σ²_z M32(Δz/ℓ_z) + σ²_t SE(Δt/ℓ_t) + σ²_c
  se   : σ² · SE(Δh/ℓ_h) · SE(Δz/ℓ_z) · SE(Δt/ℓ_t)           [光滑对照]
"""
import numpy as np
import torch

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float64
KM_LON = 111.32
KM_LAT = 110.57
SEC_DAY = 86400.0

FORM_PARAMS = {
    "prod": ["log_s2", "log_lh", "log_lz", "log_lt"],
    "add": ["log_s2h", "log_s2z", "log_s2t", "log_s2c", "log_lh", "log_lz", "log_lt"],
    "se": ["log_s2", "log_lh", "log_lz", "log_lt"],
}


def _m52(r):
    s = np.sqrt(5.0)
    return (1 + s * r + 5 * r * r / 3) * torch.exp(-s * r)


def _m32(r):
    s = np.sqrt(3.0)
    return (1 + s * r) * torch.exp(-s * r)


def _se(r2):
    return torch.exp(-r2 / 2)


class Kernel4D:
    """可微 4D 核。params: dict（键见 FORM_PARAMS，值为原始数值，内部取 log）。"""

    def __init__(self, form, params, lat_ref=20.0):
        assert form in FORM_PARAMS
        self.form = form
        self.lat_ref = float(lat_ref)
        # 保留 tensor 原样（fit_hyperparams 可微路径）；数值用 float（to_dict/json 兼容）
        self.raw = {k: (v if torch.is_tensor(v) else float(v)) for k, v in params.items()}

    def _theta(self):
        out = {}
        for k in FORM_PARAMS[self.form]:
            v = self.raw[k]
            # tensor 输入时用 .to() 保持 autograd 图（torch.tensor() 会复制切断梯度）
            out[k] = v.to(device=DEV, dtype=DT) if torch.is_tensor(v) \
                else torch.tensor(v, dtype=DT, device=DEV)
        return out

    def _prep(self, A, B):
        """(N,4),(M,4) torch -> 水平/深度/时间差（物理单位）。
        +1e-12 防对角元 dh=0 处 sqrt 反向 0/0 -> NaN 梯度（日志【问题7】）。"""
        A = torch.as_tensor(A, dtype=DT, device=DEV).reshape(-1, 4)
        B = torch.as_tensor(B, dtype=DT, device=DEV).reshape(-1, 4)
        cx = KM_LON * np.cos(np.radians(self.lat_ref))
        dlon = (A[:, 0, None] - B[None, :, 0]) * cx
        dlat = (A[:, 1, None] - B[None, :, 1]) * KM_LAT
        dh = torch.sqrt(dlon ** 2 + dlat ** 2 + 1e-12)
        dz = (A[:, 2, None] - B[None, :, 2]).abs()
        dt = ((A[:, 3, None] - B[None, :, 3]) / SEC_DAY).abs()
        return dh, dz, dt

    def __call__(self, A, B):
        return self.k_from_dh(*self._prep(A, B))

    def k_from_dh(self, dh, dz, dt):
        """可微核评估（dh/dz/dt 为预计算差值 tensor，见 _prep）。"""
        p = self._theta()
        if self.form == "prod":
            return torch.exp(p["log_s2"]) * _m52(dh / torch.exp(p["log_lh"])) \
                * _m32(dz / torch.exp(p["log_lz"])) * _se((dt / torch.exp(p["log_lt"])) ** 2)
        if self.form == "se":
            return torch.exp(p["log_s2"]) * _se((dh / torch.exp(p["log_lh"])) ** 2) \
                * _se((dz / torch.exp(p["log_lz"])) ** 2) \
                * _se((dt / torch.exp(p["log_lt"])) ** 2)
        # add
        return (torch.exp(p["log_s2h"]) * _m52(dh / torch.exp(p["log_lh"]))
                + torch.exp(p["log_s2z"]) * _m32(dz / torch.exp(p["log_lz"]))
                + torch.exp(p["log_s2t"]) * _se((dt / torch.exp(p["log_lt"])) ** 2)
                + torch.exp(p["log_s2c"]))

    def diag_var(self):
        """核对角线值（先验方差 σ²；add 为各分量和）。"""
        p = self._theta()
        if self.form == "prod":
            return float(torch.exp(p["log_s2"]))
        if self.form == "se":
            return float(torch.exp(p["log_s2"]))
        return float(torch.exp(p["log_s2h"]) + torch.exp(p["log_s2z"])
                     + torch.exp(p["log_s2t"]) + torch.exp(p["log_s2c"]))

    def to_dict(self):
        return {"form": self.form, "lat_ref": self.lat_ref, **self.raw}


def kernel_from_dict(d):
    return Kernel4D(d["form"], {k: v for k, v in d.items()
                                if k not in ("form", "lat_ref")}, d.get("lat_ref", 20.0))


class GPFit:
    """单场景 GP 后验（阶段 II 伪代码 9-12 行）。噪声 diag(u²) 固定。"""

    def __init__(self, X, d, u, mean, kern: Kernel4D, jitter=1e-8):
        self.X = torch.as_tensor(X, dtype=DT, device=DEV).reshape(-1, 4)
        self.kern = kern
        y = torch.as_tensor(np.asarray(d) - np.asarray(mean), dtype=DT, device=DEV)
        K = kern(self.X, self.X)
        u2 = torch.as_tensor(np.asarray(u) ** 2, dtype=DT, device=DEV)
        K = K + torch.diag(u2) + jitter * torch.eye(len(y), dtype=DT, device=DEV)
        self.L = torch.linalg.cholesky(K)
        self.alpha = torch.cholesky_solve(y[:, None], self.L).squeeze(-1)
        n = len(y)
        self.log_ml = float(-0.5 * (y @ self.alpha + 2 * torch.log(torch.diag(self.L)).sum()
                                    + n * np.log(2 * np.pi)))

    @torch.no_grad()
    def predict(self, Xq, mean_q, batch=20000):
        """返回 (pred_mean, var) numpy。pred = m + K*α；var = σ² - K*ᵀK⁻¹K*。"""
        Xq = torch.as_tensor(np.asarray(Xq), dtype=DT, device=DEV).reshape(-1, 4)
        mq = torch.as_tensor(np.asarray(mean_q), dtype=DT, device=DEV)
        s2 = self.kern.diag_var()
        out_m = np.empty(len(Xq))
        out_v = np.empty(len(Xq))
        for s in range(0, len(Xq), batch):
            k = self.kern(Xq[s:s + batch], self.X)                # (B,N)
            out_m[s:s + batch] = (mq[s:s + batch] + k @ self.alpha).cpu().numpy()
            v = torch.linalg.solve_triangular(self.L, k.T, upper=False)
            out_v[s:s + batch] = s2 - (v ** 2).sum(dim=0).cpu().numpy()
        return out_m, np.maximum(out_v, 0.0)


def fit_hyperparams(datasets, form="prod", n_iter=120, lr=0.08, init=None,
                    use_nugget=False, seed=0, verbose=False, patience=12,
                    s2_mode="ml"):
    """type-II ML 分型估计：多场景联合（共享超参的 -ΣlogML 最小化）。
    datasets: [(X, d, u, mean)]；init: dict 覆盖默认初值。
    返回 (Kernel4D, 每场景 logML 列表, 优化轨迹)。use_nugget=True 时拟合
    标量 nugget（乘 σ²，仅 EXP-4 诊断变体）。

    s2_mode（EXP-11 引入）：σ² 的估计口径。
      "ml"    ：σ² 自由优化。u_obs² 极小时似然可无上界（M52 负相关核 + 近零
                噪声 → GP 过拟合），σ² 被推向不合理大值——EXP-11 诊断实测
                prod_global σ²=10.12 而训练数据方差仅 2.48，-ΣlogML 随 σ² 单调
                改善至 σ²>40 仍未见峰，逐场景单独 ML 亦系统性给出 3~8 倍。
      "moment"：σ² 固定为跨场景池化残差方差（矩估计锚点），仅优化 ℓ。锚点
                语义见框架 §3.3.4⑥（σ̂²_GP ≈ σ̂²_随机效应 + û_m²）。仅对
                prod/se（单 σ² 参数）生效，add 的 4 分量结构回退 "ml"。
    性能（日志【问题17】）：差值矩阵 (dh,dz,dt) 预计算一次（no_grad 常量），
    每场景 Kernel4D 循环外构造一次（raw 的 view 随 Adam in-place 更新），
    早停 patience=12（rel 改善 <1e-4）。"""
    torch.manual_seed(seed)
    d_all = np.concatenate([np.asarray(ds[1]) - np.asarray(ds[3]) for ds in datasets])
    d_var = float(np.var(d_all)) + 1e-6
    s2_fixed = d_var if (s2_mode == "moment" and form in ("prod", "se")) else None
    # 初值均为 log 域：ℓ_h=75 km、ℓ_z=80 m、ℓ_t=20 d（原 75.0/80.0 误用物理值，
    # exp(75)≈4e32 使优化不可收敛，日志【问题8】）
    if form == "add":
        base = {"log_s2h": np.log(d_var * 0.5), "log_s2z": np.log(d_var * 0.3),
                "log_s2t": np.log(d_var * 0.15), "log_s2c": np.log(d_var * 0.05),
                "log_lh": np.log(75.0), "log_lz": np.log(80.0), "log_lt": np.log(20.0)}
    else:
        base = {"log_s2": np.log(d_var), "log_lh": np.log(75.0),
                "log_lz": np.log(80.0), "log_lt": np.log(20.0)}
    if init:
        base.update(init)
    if use_nugget:
        # nugget 为绝对方差（非乘 σ²），初值取 u_obs² 中位数——量级应与观测噪声
        # 相当；缺此初值则 keys 含 "log_nug" 而 base 无该键 -> KeyError（日志【问题21】）
        u2_med = float(np.median(np.concatenate(
            [np.asarray(ds[2]) ** 2 for ds in datasets])))
        base["log_nug"] = np.log(max(u2_med, 1e-6))
    lat_ref = float(np.mean([np.asarray(ds[0])[:, 1].mean() for ds in datasets]))
    keys = list(FORM_PARAMS[form]) + (["log_nug"] if use_nugget else [])
    if s2_fixed is not None:
        keys = [k for k in keys if k != "log_s2"]      # σ² 固定，不进优化变量
    raw = torch.tensor([base[k] for k in keys], dtype=DT, device=DEV, requires_grad=True)
    data = []
    with torch.no_grad():
        for X, d, u, m in datasets:
            Xt = torch.as_tensor(np.asarray(X), dtype=DT, device=DEV).reshape(-1, 4)
            y = torch.as_tensor(np.asarray(d) - np.asarray(m), dtype=DT, device=DEV)
            u2 = torch.as_tensor(np.asarray(u) ** 2, dtype=DT, device=DEV)
            cx = KM_LON * np.cos(np.radians(lat_ref))
            dlon = (Xt[:, 0, None] - Xt[None, :, 0]) * cx
            dlat = (Xt[:, 1, None] - Xt[None, :, 1]) * KM_LAT
            dh = torch.sqrt(dlon ** 2 + dlat ** 2 + 1e-12)
            dz = (Xt[:, 2, None] - Xt[None, :, 2]).abs()
            dt = ((Xt[:, 3, None] - Xt[None, :, 3]) / SEC_DAY).abs()
            eye = torch.eye(len(y), dtype=DT, device=DEV)
            data.append((dh, dz, dt, y, u2, eye))
    opt = torch.optim.Adam([raw], lr=lr)
    # 循环外构造：raw[i] 是优化变量的 view，Adam in-place 更新后自动生效；
    # s2_fixed 时 log_s2 以常量 tensor 注入（无梯度，等价于固定 σ²）
    fixed = ({"log_s2": torch.tensor(float(np.log(s2_fixed)), dtype=DT, device=DEV)}
             if s2_fixed is not None else {})
    kern = Kernel4D(form, {**fixed, **{kk: raw[i] for i, kk in enumerate(keys)
                                       if kk != "log_nug"}}, lat_ref)
    nug_i = keys.index("log_nug") if use_nugget else None

    def loss():
        tot = 0.0
        for dh, dz, dt, y, u2, eye in data:
            K = kern.k_from_dh(dh, dz, dt) + torch.diag(u2)
            if use_nugget:
                K = K + torch.exp(raw[nug_i]) * eye
            K = K + 1e-8 * eye
            L = torch.linalg.cholesky(K)
            alpha = torch.cholesky_solve(y[:, None], L).squeeze(-1)
            tot = tot + 0.5 * (y @ alpha + 2 * torch.log(torch.diag(L)).sum()
                               + len(y) * np.log(2 * np.pi))
        return tot

    trace = []
    best, bad = np.inf, 0
    for it in range(n_iter):
        opt.zero_grad()
        l = loss()
        l.backward()
        opt.step()
        lv = float(l)
        trace.append(lv)
        if lv < best * (1 - 1e-4):
            best, bad = lv, 0
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and it % 40 == 0:
            print(f"    hp iter {it}: -ΣlogML={lv:.1f}", flush=True)
    final_vals = raw.detach().cpu().numpy()
    fin = {kk: float(v) for kk, v in zip(keys, final_vals) if kk != "log_nug"}
    if s2_fixed is not None:
        fin["log_s2"] = float(np.log(s2_fixed))
    kern = Kernel4D(form, fin, lat_ref)
    lmls = []
    with torch.no_grad():
        for dh, dz, dt, y, u2, eye in data:
            K = kern.k_from_dh(dh, dz, dt) + torch.diag(u2)
            if use_nugget:                      # 与 loss 口径一致（否则 lmls 与目标不符）
                K = K + torch.exp(raw[nug_i]) * eye
            K = K + 1e-8 * eye
            L = torch.linalg.cholesky(K)
            alpha = torch.cholesky_solve(y[:, None], L).squeeze(-1)
            lmls.append(float(-0.5 * (y @ alpha + 2 * torch.log(torch.diag(L)).sum()
                                      + len(y) * np.log(2 * np.pi))))
    extra = {}
    if use_nugget:
        extra["log_nug"] = float(raw[nug_i].detach())
    return kern, lmls, {"trace": trace, "n_iter": len(trace), **extra}
