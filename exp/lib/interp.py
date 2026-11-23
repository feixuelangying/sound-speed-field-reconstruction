# -*- coding: utf-8 -*-
"""4D 时空插值（时间线性 / 水平双线性 / 深度向量化 PCHIP）。
实现决策（记入实验日志）：背景场先做时空（水平×时间）双线性得到目标位置的 8 层剖面，
再对该剖面做 PCHIP 深度连续化——与 framework §3.2 中 c_b 的分层定义一致，
且保证 ∂c_b/∂z 特征与值出自同一插值剖面。"""
import numpy as np

from . import physics


def _pchip_edge(h0, h1, m0, m1):
    """scipy 兼容的 PCHIP 端点斜率（三点单侧估计 + 保形限制）。"""
    d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    mask = np.sign(d) != np.sign(m0)
    mask2 = (np.sign(m0) != np.sign(m1)) & (np.abs(d) > 3.0 * np.abs(m0))
    mmm = (~mask) & mask2
    d = np.where(mask, 0.0, d)
    d = np.where(mmm, 3.0 * m0, d)
    return d


def pchip_eval(x, V, xq, return_deriv=False):
    """向量化 PCHIP。
    x: (K,) 节点（全体查询共用）；V: (N,K) 每行一组节点值；xq: (N,) 查询点。
    返回 (N,) 插值值（return_deriv=True 时再返回解析导数 (N,)）。
    xq 超出 [x0, xK-1] 时取端点值（导数取端点斜率）。"""
    x = np.asarray(x, dtype=float)
    V = np.asarray(V, dtype=float)
    xq = np.asarray(xq, dtype=float)
    N, K = V.shape
    h = np.diff(x)                              # (K-1,)
    delta = np.diff(V, axis=1) / h              # (N,K-1)
    # 内部节点斜率（调和平均，保形）。delta=0（相邻层值相等，float32 下深层常见）
    # 时 denom=inf -> m=0，结果正确，仅产生除零警告 -> errstate 屏蔽（日志【问题13】）
    hL, hR = h[:-1], h[1:]                      # (K-2,)
    w1 = 2 * hR + hL
    w2 = hR + 2 * hL
    same = delta[:, :-1] * delta[:, 1:] > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = w1 / delta[:, :-1] + w2 / delta[:, 1:]
    m_int = np.where(same, (w1 + w2) / denom, 0.0)   # (N,K-2)
    m = np.empty((N, K))
    m[:, 1:K - 1] = m_int
    m[:, 0] = _pchip_edge(h[0], h[1], delta[:, 0], delta[:, 1])
    m[:, K - 1] = _pchip_edge(h[-1], h[-2], delta[:, -1], delta[:, -2])
    # 区间定位
    inside = (xq >= x[0]) & (xq <= x[-1])
    xqc = np.clip(xq, x[0], x[-1])
    idx = np.clip(np.searchsorted(x, xqc, side="right") - 1, 0, K - 2)
    hh = h[idx]
    s = (xqc - x[idx]) / hh
    # Hermite 基
    s2, s3 = s * s, s * s * s
    H00 = 2 * s3 - 3 * s2 + 1
    H10 = s3 - 2 * s2 + s
    H01 = -2 * s3 + 3 * s2
    H11 = s3 - s2
    rows = np.arange(N)
    v = (H00 * V[rows, idx] + H10 * hh * m[rows, idx]
         + H01 * V[rows, idx + 1] + H11 * hh * m[rows, idx + 1])
    if not return_deriv:
        return v
    dH00 = 6 * s2 - 6 * s
    dH10 = 3 * s2 - 4 * s + 1
    dH01 = -6 * s2 + 6 * s
    dH11 = 3 * s2 - 2 * s
    dv = (dH00 * V[rows, idx] + dH10 * hh * m[rows, idx]
          + dH01 * V[rows, idx + 1] + dH11 * hh * m[rows, idx + 1]) / hh
    # 端点外：取端点值/斜率
    v = np.where(inside, v, np.where(xq < x[0], V[:, 0], V[:, K - 1]))
    dv = np.where(inside, dv, np.where(xq < x[0], m[:, 0], m[:, K - 1]))
    return v, dv


class BgField:
    """HYCOM 背景声速场（memmap 后端），提供任意 (t,z,lat,lon) 的插值与梯度。"""

    def __init__(self, cache_dir):
        import os
        meta = np.load(os.path.join(cache_dir, "hycom_meta.npz"))
        self.t_hours = meta["t_hours"]
        self.depths = meta["depths"]
        self.lat = meta["lat"]
        self.lon = meta["lon"]
        nt, nz, ny, nx = meta["shape"]
        self.shape = (int(nt), int(nz), int(ny), int(nx))
        self.cs = np.memmap(os.path.join(cache_dir, "hycom_cs.dat"), dtype=np.float32,
                            mode="r", shape=self.shape)

    def snapshot(self, it):
        return np.asarray(self.cs[it])

    def _time_locate(self, t_sec):
        """t_sec: 相对 2000-01-01 的秒数 -> (i0, i1, wt, in_range)"""
        t_h = np.asarray(t_sec, dtype=float) / 3600.0
        nt = self.shape[0]
        i0 = np.clip(np.searchsorted(self.t_hours, t_h, side="right") - 1, 0, nt - 2)
        i1 = i0 + 1
        span = self.t_hours[i1] - self.t_hours[i0]
        wt = np.clip((t_h - self.t_hours[i0]) / span, 0.0, 1.0)
        return i0, i1, wt

    def _interp3(self, it, z, lat, lon, want="val"):
        """单时间步：水平双线性(得8层剖面) -> PCHIP 深度插值。
        want='val' | 'gradz' | 'gradh'（gradh 返回 (dc/dx, dc/dy) m/s per km）"""
        snap = self.snapshot(it)                       # (8,84,63)
        lon_ax, lat_ax = self.lon, self.lat
        ix = np.clip(np.searchsorted(lon_ax, lon, side="right") - 1, 0, len(lon_ax) - 2)
        iy = np.clip(np.searchsorted(lat_ax, lat, side="right") - 1, 0, len(lat_ax) - 2)
        fx = np.clip((lon - lon_ax[ix]) / (lon_ax[ix + 1] - lon_ax[ix]), 0.0, 1.0)
        fy = np.clip((lat - lat_ax[iy]) / (lat_ax[iy + 1] - lat_ax[iy]), 0.0, 1.0)
        # 8 层剖面 = 四格点双线性
        V = (snap[:, iy, ix] * (1 - fx) * (1 - fy) + snap[:, iy, ix + 1] * fx * (1 - fy)
             + snap[:, iy + 1, ix] * (1 - fx) * fy + snap[:, iy + 1, ix + 1] * fx * fy)  # (8,N)
        V = V.T                                          # (N,8)
        if want == "gradh":
            # 沿网格方向的前向差分（对每个查询点所在的格点单元）
            dx_deg = lon_ax[ix + 1] - lon_ax[ix]
            dy_deg = lat_ax[iy + 1] - lat_ax[iy]
            km_per_deg_x = 111.32 * np.cos(np.radians(lat))
            km_per_deg_y = 110.57
            dVx = ((snap[:, iy, ix + 1] - snap[:, iy, ix]) * (1 - fy)
                   + (snap[:, iy + 1, ix + 1] - snap[:, iy + 1, ix]) * fy).T   # (N,8)
            dVy = ((snap[:, iy + 1, ix] - snap[:, iy, ix]) * (1 - fx)
                   + (snap[:, iy + 1, ix + 1] - snap[:, iy, ix + 1]) * fx).T
            gx = dVx / (dx_deg * km_per_deg_x)[:, None]  # m/s per km，(N,8)
            gy = dVy / (dy_deg * km_per_deg_y)[:, None]
            return gx, gy
        if want == "gradz":
            _, dv = pchip_eval(self.depths, V, z, return_deriv=True)
            return dv
        return pchip_eval(self.depths, V, z)

    def _blend(self, t_sec, z, lat, lon, want):
        i0, i1, wt = self._time_locate(t_sec)
        z, lat, lon = map(lambda a: np.atleast_1d(np.asarray(a, dtype=float)), (z, lat, lon))
        wt = np.asarray(wt, dtype=float)
        if want == "gradh":
            groups = np.unique(i0)
            gx = gy = None
            for it in groups:
                sel = i0 == it
                a_x, a_y = self._interp3(int(it), z[sel], lat[sel], lon[sel], "gradh")
                b_x, b_y = self._interp3(int(it) + 1, z[sel], lat[sel], lon[sel], "gradh")
                w = wt[sel][:, None]
                ax_ = (1 - w) * a_x + w * b_x
                ay_ = (1 - w) * a_y + w * b_y
                if gx is None:
                    gx = np.empty((len(z), ax_.shape[1]), dtype=float)
                    gy = np.empty_like(gx)
                gx[sel] = ax_
                gy[sel] = ay_
            return gx, gy
        # val / gradz：按时间步分组处理再时间混合
        res = np.empty(len(z), dtype=float)
        for it in np.unique(i0):
            sel = i0 == it
            a = self._interp3(int(it), z[sel], lat[sel], lon[sel], want)
            b = self._interp3(int(it) + 1, z[sel], lat[sel], lon[sel], want)
            res[sel] = (1 - wt[sel]) * a + wt[sel] * b
        return res

    # ---- 公共接口 ----
    def cs_at(self, t_sec, z, lat, lon):
        return self._blend(t_sec, z, lat, lon, "val")

    def gradz_at(self, t_sec, z, lat, lon):
        return self._blend(t_sec, z, lat, lon, "gradz")

    def gradh_at(self, t_sec, z, lat, lon):
        return self._blend(t_sec, z, lat, lon, "gradh")
