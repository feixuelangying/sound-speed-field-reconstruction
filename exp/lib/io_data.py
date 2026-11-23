# -*- coding: utf-8 -*-
"""数据 IO：HYCOM 声速 memmap 缓存、Argo 剖面并行 QC 读取、CTD 主表读取。
缓存文件（results/cache/）：
  hycom_cs.dat     float32 (16306,8,84,63) 背景声速场
  hycom_meta.npz   t_hours/depths/lat/lon/shape/eq
  argo_clean.npz   剖面表 + 观测扁平数组（QC 后）
  ctd_clean.npz    同构（72 站）
"""
import os
import time
import numpy as np
import pandas as pd
import netCDF4
from scipy.ndimage import distance_transform_edt

from .. import config
from . import physics


# ---------------------------------------------------------------- HYCOM ----
def build_hycom_cs_cache(force=False, eq=None):
    eq = eq or config.SOUND_SPEED_EQ
    cs_path = os.path.join(config.CACHE, "hycom_cs.dat")
    meta_path = os.path.join(config.CACHE, "hycom_meta.npz")
    if (not force) and os.path.exists(cs_path) and os.path.exists(meta_path):
        return
    t0 = time.time()
    ds = netCDF4.Dataset(config.HYCOM_NC)
    ds.set_auto_mask(True)
    depths = np.asarray(ds.variables["depth"][:], dtype=float)
    lat = np.asarray(ds.variables["lat"][:], dtype=float)
    lon = np.asarray(ds.variables["lon"][:], dtype=float)
    t_hours = np.asarray(ds.variables["time"][:], dtype=float)
    nt, nz, ny, nx = (len(t_hours), len(depths), len(lat), len(lon))
    # 深度层 -> 压力场 (8,84,63)（纬度依赖）
    P_grid = np.stack([np.broadcast_to(physics.p_from_z(np.full(ny, d), lat)[:, None], (ny, nx))
                       for d in depths])
    # 陆地/缺测填充：温盐联合掩膜，块内并集为固定海陆场（日志【问题5】——
    # HYCOM mask 随时间变化且温盐掩膜不一致；填充源 = 块内温盐均从未缺测的格点，
    # 被填格点均为至少一变量永远缺测、本就无法计算声速的点，零有效数据损失）。
    tv = ds.variables["water_temp"]
    salt = ds.variables["salinity"]
    # HDF5 chunk 时间跨度 4077 步：逐时间步读取会令每步重复解压全部触及 chunk
    # （~700MB/步）；按 chunk 边界整块读取使每 chunk 只解压一次（日志【问题3】）。
    chunks = tv.chunking()
    tc = int(chunks[0]) if chunks and chunks[0] != "contiguous" else nt
    mm = np.memmap(cs_path, dtype=np.float32, mode="w+", shape=(nt, nz, ny, nx))
    for c0 in range(0, nt, tc):
        c1 = min(c0 + tc, nt)
        Tb = tv[c0:c1]
        Sb = salt[c0:c1]
        mU = (np.ma.getmaskarray(Tb) | np.ma.getmaskarray(Sb)).any(axis=0)  # (8,ny,nx)
        T = np.ma.filled(Tb.astype("float64"), np.nan)    # (nb,8,ny,nx)
        S = np.ma.filled(Sb.astype("float64"), np.nan)
        del Tb, Sb
        for k in range(nz):
            m_k = mU[k]
            if not m_k.any():
                continue
            assert not m_k.all(), f"layer {k} fully masked in block [{c0},{c1})"
            _, (iy, ix) = distance_transform_edt(m_k, return_indices=True)
            sy, sx = iy[m_k], ix[m_k]
            T[:, k][:, m_k] = T[:, k][:, sy, sx]
            S[:, k][:, m_k] = S[:, k][:, sy, sx]
        assert np.isfinite(T).all() and np.isfinite(S).all(), "non-finite after land fill"
        c = physics.sound_speed(T, S, P_grid[None], eq=eq)
        mm[c0:c1] = c.astype(np.float32)
        del T, S, c
        print(f"  hycom->sound speed {c1}/{nt}  ({time.time()-t0:.0f}s)", flush=True)
    mm.flush()
    ds.close()
    np.savez_compressed(meta_path, t_hours=t_hours, depths=depths, lat=lat, lon=lon,
                        shape=np.array([nt, nz, ny, nx]), eq=eq)
    print(f"  hycom cs cache done: {nt} steps, eq={eq}, {time.time()-t0:.0f}s")


# ---------------------------------------------------------------- Argo ----
def _read_one_argo(args):
    """worker：读单剖面 nc -> QC -> (meta, p, T, S)。在子进程中执行。"""
    from .. import config as C
    from . import physics as P
    idx, path, t64, lat, lon, ptype, wmo = args
    try:
        ds = netCDF4.Dataset(path)
        ds.set_auto_mask(True)
        n_raw = int(ds.variables["PRES"][0].shape[0])

        def _get_q(name):
            v = ds.variables[name]
            try:
                return np.array(list(v[0].tobytes().decode("ascii", "ignore")))
            except Exception:
                return np.array(["4"] * v.shape[1])

        adj_avail = False
        if "PRES_ADJUSTED" in ds.variables:
            pa = np.ma.filled(ds.variables["PRES_ADJUSTED"][0].astype("float64"), np.nan)
            if np.isfinite(pa).sum() > 10:
                adj_avail = True
        if adj_avail:
            p = pa
            T = np.ma.filled(ds.variables["TEMP_ADJUSTED"][0].astype("float64"), np.nan)
            S = np.ma.filled(ds.variables["PSAL_ADJUSTED"][0].astype("float64"), np.nan)
            qP = _get_q("PRES_ADJUSTED_QC"); qT = _get_q("TEMP_ADJUSTED_QC"); qS = _get_q("PSAL_ADJUSTED_QC")
        else:
            p = np.ma.filled(ds.variables["PRES"][0].astype("float64"), np.nan)
            T = np.ma.filled(ds.variables["TEMP"][0].astype("float64"), np.nan)
            S = np.ma.filled(ds.variables["PSAL"][0].astype("float64"), np.nan)
            qP = _get_q("PRES_QC"); qT = _get_q("TEMP_QC"); qS = _get_q("PSAL_QC")
        ds.close()
        good = np.array([q in C.QC_GOOD for q in qP]) & np.array([q in C.QC_GOOD for q in qT]) \
            & np.array([q in C.QC_GOOD for q in qS])
        good &= np.isfinite(p) & np.isfinite(T) & np.isfinite(S)
        good &= (p >= 0) & (T > C.T_RANGE[0]) & (T < C.T_RANGE[1]) \
            & (S > C.S_RANGE[0]) & (S < C.S_RANGE[1])
        p, T, S = p[good], T[good], S[good]
        n_flag = len(p)
        if len(p) < 5:
            return None
        # 压力排序去重
        order = np.argsort(p, kind="stable")
        p, T, S = p[order], T[order], S[order]
        keep = np.concatenate([[True], np.diff(p) > 0.1])
        p, T, S = p[keep], T[keep], S[keep]
        if len(p) < 5:
            return None
        # med3 尖峰检验（T/S 分别）
        from scipy.ndimage import median_filter
        for arr, floor in ((T, C.SPIKE_ABS_T), (S, C.SPIKE_ABS_S)):
            if len(arr) >= 5:
                med3 = median_filter(arr, size=3, mode="nearest")
                r = np.abs(arr - med3)
                mad = np.median(r)
                if mad < 1e-8:
                    continue
                arr[(r > C.SPIKE_FACTOR * 1.4826 * mad) & (r > floor)] = np.nan
        ok = np.isfinite(T) & np.isfinite(S)
        p, T, S = p[ok], T[ok], S[ok]
        n_spike = len(p)
        if len(p) < 5:
            return None
        # 限深（z <= Z_MAX）
        z = P.z_from_p(p, lat)
        ok = z <= C.Z_MAX
        p, T, S = p[ok], T[ok], S[ok]
        if len(p) < 5:
            return None
        return (idx, int(wmo), int(t64), float(lat), float(lon), int(ptype),
                p.astype(np.float32), T.astype(np.float32), S.astype(np.float32),
                (n_raw, n_flag, n_spike, len(p)))
    except Exception:
        return None


def build_argo_cache(force=False, n_workers=16):
    out = os.path.join(config.CACHE, "argo_clean.npz")
    if os.path.exists(out) and not force:
        return
    t0 = time.time()
    df = pd.read_csv(config.ARGO_CSV)
    df["t64"] = pd.to_datetime(df["date"]).values.astype("datetime64[s]").astype(np.int64)
    # 文件名从 CSV file 列提取 basename
    tasks = []
    for i, row in df.iterrows():
        base = str(row["file"]).replace("\\", "/").rsplit("/", 1)[-1]
        path = os.path.join(config.ARGO_DIR, base)
        if os.path.exists(path):
            tasks.append((i, path, int(row["t64"]), row["latitude"], row["longitude"],
                          int(row["profiler_type"]), int(row["wmo"])))
    print(f"  argo files found: {len(tasks)}/{len(df)}")
    results = []
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for k, r in enumerate(ex.map(_read_one_argo, tasks, chunksize=64)):
            if r is not None:
                results.append(r)
            if k % 4000 == 0:
                print(f"  argo qc {k}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)
    results.sort(key=lambda r: r[0])
    # 合并为扁平数组
    pid = np.array([r[0] for r in results], dtype=np.int64)
    wmo = np.array([r[1] for r in results], dtype=np.int64)
    t64 = np.array([r[2] for r in results], dtype=np.int64)
    lat = np.array([r[3] for r in results], dtype=np.float64)
    lon = np.array([r[4] for r in results], dtype=np.float64)
    ptype = np.array([r[5] for r in results], dtype=np.int32)
    counts = np.array([len(r[6]) for r in results], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    P_all = np.concatenate([r[6] for r in results])
    T_all = np.concatenate([r[7] for r in results])
    S_all = np.concatenate([r[8] for r in results])
    stats = np.array([r[9] for r in results], dtype=np.int64)  # (n_prof,4): raw/flag/spike/final
    np.savez_compressed(out, pid=pid, wmo=wmo, t64=t64, lat=lat, lon=lon,
                        ptype=ptype, counts=counts, starts=starts, qc_stats=stats,
                        p=P_all, T=T_all, S=S_all)
    print(f"  argo cache done: {len(results)} profiles, {len(P_all)} obs, "
          f"{time.time()-t0:.0f}s")


# ---------------------------------------------------------------- CTD ----
def build_ctd_cache(force=False):
    out = os.path.join(config.CACHE, "ctd_clean.npz")
    if os.path.exists(out) and not force:
        return
    ds = netCDF4.Dataset(config.CTD_NC)
    ds.set_auto_mask(True)
    p = np.ma.filled(ds.variables["pressure"][:].astype("float64"), np.nan)      # (72,6130)
    T = np.ma.filled(ds.variables["temperature"][:].astype("float64"), np.nan)
    S = np.ma.filled(ds.variables["salinity"][:].astype("float64"), np.nan)
    lat = np.asarray(ds.variables["latitude"][:], dtype=float)
    lon = np.asarray(ds.variables["longitude"][:], dtype=float)
    date = ds.variables["date"][:]
    stnnbr = ds.variables["stnnbr"][:]
    cruise = ds.variables["cruise"][:]
    ds.close()
    # date -> t64
    if date.dtype.kind in "SU" or date.ndim == 1 and date.dtype.kind == "O":
        try:
            dt = pd.to_datetime(np.array([str(x) for x in date]))
            t64 = dt.values.astype("datetime64[s]").astype(np.int64)
        except Exception:
            raise RuntimeError("CTD date parse failed: " + str(date[:3]))
    else:
        t64 = None
    recs = []
    for i in range(len(lat)):
        pi, Ti, Si = p[i], T[i], S[i]
        ok = np.isfinite(pi) & np.isfinite(Ti) & np.isfinite(Si)
        ok &= (pi >= 0) & (Ti > config.T_RANGE[0]) & (Ti < config.T_RANGE[1]) \
            & (Si > config.S_RANGE[0]) & (Si < config.S_RANGE[1])
        pi, Ti, Si = pi[ok], Ti[ok], Si[ok]
        order = np.argsort(pi, kind="stable")
        pi, Ti, Si = pi[order], Ti[order], Si[order]
        keep = np.concatenate([[True], np.diff(pi) > 0.1])
        pi, Ti, Si = pi[keep], Ti[keep], Si[keep]
        z = physics.z_from_p(pi, lat[i])
        ok = z <= config.Z_MAX
        pi, Ti, Si = pi[ok], Ti[ok], Si[ok]
        if len(pi) >= 5:
            recs.append((i, str(stnnbr[i]) if not hasattr(stnnbr[i], 'item') else int(stnnbr[i]),
                         str(cruise[i]) if not hasattr(cruise[i], 'item') else str(cruise[i]),
                         int(t64[i]), float(lat[i]), float(lon[i]),
                         pi.astype(np.float32), Ti.astype(np.float32), Si.astype(np.float32)))
    counts = np.array([len(r[6]) for r in recs], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    np.savez_compressed(out,
                        pid=np.array([r[0] for r in recs], dtype=np.int64),
                        stnnbr=np.array([r[1] for r in recs]),
                        cruise=np.array([r[2] for r in recs]),
                        t64=np.array([r[3] for r in recs], dtype=np.int64),
                        lat=np.array([r[4] for r in recs]), lon=np.array([r[5] for r in recs]),
                        counts=counts, starts=starts,
                        p=np.concatenate([r[6] for r in recs]),
                        T=np.concatenate([r[7] for r in recs]),
                        S=np.concatenate([r[8] for r in recs]))
    print(f"  ctd cache done: {len(recs)} stations, {int(counts.sum())} obs")


# ---------------------------------------------------------------- 载入 ----
class ProfileSet:
    """剖面集合（Argo 或 CTD），扁平存储 + 剖面索引。"""

    def __init__(self, npz):
        self.pid = npz["pid"]; self.t64 = npz["t64"]
        self.lat = npz["lat"]; self.lon = npz["lon"]
        self.counts = npz["counts"]; self.starts = npz["starts"]
        self.p = npz["p"]; self.T = npz["T"]; self.S = npz["S"]
        if "wmo" in npz:
            self.wmo = npz["wmo"]; self.ptype = npz["ptype"]
        self.n_prof = len(self.pid)
        self.n_obs = len(self.p)
        self._z = None

    def obs_z(self, lat_vec=None):
        """观测深度（缓存）。lat_vec: 每观测点的纬度（默认用剖面纬度展开）。"""
        if self._z is None:
            from . import physics
            lat_exp = np.repeat(self.lat, self.counts) if lat_vec is None else lat_vec
            self._z = physics.z_from_p(self.p, lat_exp)
        return self._z

    def slice(self, i):
        s, c = self.starts[i], self.counts[i]
        return slice(s, s + c)


def load_argo():
    return ProfileSet(np.load(os.path.join(config.CACHE, "argo_clean.npz")))


def load_ctd():
    return ProfileSet(np.load(os.path.join(config.CACHE, "ctd_clean.npz")))
