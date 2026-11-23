# -*- coding: utf-8 -*-
"""均值网络特征工程：phi(q) 全部由背景场导出、任意查询点可计算（D4/D13）。
特征缓存 results/cache/obs_features.npz（全量观测点）。
"""
import os
import time
import numpy as np

from .. import config as C
from . import physics
from .interp import BgField

EPOCH = np.datetime64("2000-01-01T00:00:00").astype(np.int64)

FEAT_NAMES = ["x", "y", "zlog", "sin_doy", "cos_doy", "cb", "dcb_dz", "grad_h"]


def day_of_year_angle(t_rel_sec):
    """t_rel_sec: 相对 2000-01-01 的秒数 -> 周年相位角（2019-01-01 为相位零点）。"""
    t_unix = np.asarray(t_rel_sec, dtype=np.int64) + EPOCH
    doy = (t_unix - np.datetime64("2019-01-01T00:00:00").astype(np.int64)) / 86400.0
    return 2 * np.pi * (doy % 365.25) / 365.25


def make_features(bg: BgField, t_sec, z, lat, lon, z_mode="log"):
    """返回 X (N, 8) 与列名。t_sec: 相对 2000-01-01 秒。"""
    ang = day_of_year_angle(np.asarray(t_sec, dtype=np.int64))
    cb = bg.cs_at(t_sec, z, lat, lon)
    dcb = bg.gradz_at(t_sec, z, lat, lon)
    gx, gy = bg.gradh_at(t_sec, z, lat, lon)
    # gx,gy 为 (N,8)（8 层各自的水平梯度）；深度插值：取 PCHIP 加权——简化用对应深度的线性权重
    # 更准确：把 8 层梯度也做 PCHIP。此处用按深度线性插值（梯度平滑，误差可忽略）：
    nodes = bg.depths
    zi = np.clip(np.searchsorted(nodes, z) - 1, 0, 6)
    fz = (z - nodes[zi]) / (nodes[zi + 1] - nodes[zi])
    rows = np.arange(len(z))
    gxm = gx[rows, zi] * (1 - fz) + gx[rows, zi + 1] * fz
    gym = gy[rows, zi] * (1 - fz) + gy[rows, zi + 1] * fz
    grad_h = np.sqrt(gxm ** 2 + gym ** 2)
    if z_mode == "log":
        zz = np.log1p(z)
    else:
        zz = z.copy()
    X = np.column_stack([
        (lon - 110.0) / 30.0, (lat - 10.0) / 20.0, zz / 8.0,
        np.sin(ang), np.cos(ang), cb, dcb, grad_h])
    return X, FEAT_NAMES


def build_obs_features(argo, bg, force=False):
    """全量观测点特征缓存。"""
    cache = os.path.join(C.CACHE, "obs_features.npz")
    if os.path.exists(cache) and not force:
        return np.load(cache)
    t0 = time.time()
    n = argo.n_obs
    lat_exp = np.repeat(argo.lat, argo.counts)
    lon_exp = np.repeat(argo.lon, argo.counts)
    t_exp = np.repeat(argo.t64, argo.counts).astype(np.int64) - EPOCH
    z = argo.obs_z()
    B = 400000
    X = np.empty((n, len(FEAT_NAMES)), dtype=np.float32)
    for s in range(0, n, B):
        e = min(s + B, n)
        X[s:e], _ = make_features(bg, t_exp[s:e], z[s:e].astype(float),
                                  lat_exp[s:e], lon_exp[s:e])
        if s % (B * 4) == 0:
            print(f"  features {s}/{n} ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(cache, X=X, names=np.array(FEAT_NAMES))
    print(f"  obs features done: {n} x {len(FEAT_NAMES)}, {time.time()-t0:.0f}s")
    return np.load(cache)
