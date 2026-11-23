# -*- coding: utf-8 -*-
"""u_c 合成与共形因子查询（framework §3.2 合成律 / §3.3.1 校准；EXP-5 定标，
EXP-6/7/8/9 冻结使用）。

u_c = sqrt(u_r² + u_m² + u_eq² + u_rep²)
- u_r：GP 后验方差内生（调用方传入）
- u_m：EXP-3 分格随机效应查表（band×season -> PCHIP 深度连续化）
- u_eq：EXP-1 差分二阶项 + 压深换算敏感性（保守取 max）
- u_rep：EXP-1 支撑代表性分带表
- k：EXP-5 共形校准定标（全局或分层），冻结于 results/exp5/k_global.json
"""
import json
import os
import numpy as np
import pandas as pd

from .. import config as C
from .interp import pchip_eval
from .uobs import ZBANDS, BAND_CENTERS, band_index


class UcComposer:
    def __init__(self, um_csv=None, urep_csv=None, k_json=None):
        self.um = pd.read_csv(um_csv or os.path.join(C.RESULTS, "exp3", "um_indist.csv"))
        self.urep = pd.read_csv(urep_csv or os.path.join(C.RESULTS, "exp1", "urep_by_band.csv"))
        self._load_ueq()
        self._load_um_table()
        self.urep_by_band = self.urep["u_rep_rms_ms"].values.astype(float)
        self.kcfg = None
        kpath = k_json or os.path.join(C.RESULTS, "exp5", "k_global.json")
        if os.path.exists(kpath):
            with open(kpath, encoding="utf-8") as f:
                self.kcfg = json.load(f)

    # ---- u_eq：EXP-1 汇总（保守 max 口径）----
    def _load_ueq(self):
        p = os.path.join(C.RESULTS, "exp1", "exp1_summary.json")
        self.ueq_val = 0.02  # fallback 下限
        self.ueq_detail = {}
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                s = json.load(f)
            d = s.get("ueq", {}).get("differential_second_order", {})
            zp = s.get("ueq", {}).get("z_to_p_sensitivity_0.5pct_max_ms", 0.0)
            vals = [abs(float(d.get("u_eq_gsw_ms", 0))), abs(float(d.get("u_eq_mk_ms", 0))),
                    abs(float(zp)), 0.02]
            self.ueq_val = max(vals)
            self.ueq_detail = {"gsw": d.get("u_eq_gsw_ms"), "mackenzie": d.get("u_eq_mk_ms"),
                               "z_to_p": zp, "adopted": self.ueq_val}

    # ---- u_m：分格值 -> PCHIP 深度连续化 ----
    def _load_um_table(self):
        self.um_table = np.full((4, 6), np.nan)
        for _, r in self.um.iterrows():
            self.um_table[int(r["season"]), int(r["band"])] = float(r["um"])
        # 空格用带内（跨季节）中位数填充
        for bi in range(6):
            col = self.um_table[:, bi]
            bad = ~np.isfinite(col)
            if bad.any() and (~bad).any():
                col[bad] = np.nanmedian(col)

    def um_at(self, z, season):
        season = np.broadcast_to(np.asarray(season, dtype=int), np.atleast_1d(z).shape)
        z = np.atleast_1d(np.asarray(z, dtype=float))
        out = np.empty(len(z))
        for se in range(4):
            m = season == se
            if not m.any():
                continue
            V = np.tile(self.um_table[se], (int(m.sum()), 1))
            out[m] = pchip_eval(BAND_CENTERS, V, np.clip(z[m], 5, 1900))
        return out

    def urep_at(self, z):
        z = np.atleast_1d(np.asarray(z, dtype=float))
        bi = np.clip(band_index(np.clip(z, 1, 1999)), 0, 5)
        return self.urep_by_band[bi]

    def ueq_at(self, z=None):
        if z is None:
            return self.ueq_val
        return np.full(np.atleast_1d(z).shape, self.ueq_val, dtype=float)

    def extra_std(self, z, season):
        """外加分量 sqrt(u_m²+u_eq²+u_rep²)（u_r 由 GP 提供）。"""
        return np.sqrt(self.um_at(z, season) ** 2 + self.ueq_at(z) ** 2
                       + self.urep_at(z) ** 2)

    def compose(self, ur, z, season):
        """返回 (u_c, 分解 dict)：各分量进入 u_c² 的值。"""
        ur = np.atleast_1d(np.asarray(ur, dtype=float))
        um, ueq, urep = self.um_at(z, season), self.ueq_at(z), self.urep_at(z)
        u2 = ur ** 2 + um ** 2 + ueq ** 2 + urep ** 2
        return np.sqrt(u2), {"ur": ur, "um": um, "ueq": ueq, "urep": urep}

    # ---- k：共形因子 ----
    def k_at(self, z=None, season=None, sparse_km=None):
        """共形因子 k。分层方案按 EXP-5 冻结的 kcfg['scheme'] 查表。
        k_table 键为格子 id 字符串（"0".."5" 等，EXP-5 k_by_scheme 产出），
        缺格回退 k_global。"""
        if self.kcfg is None:
            return 1.96
        scheme = self.kcfg.get("scheme", "global")
        kg = float(self.kcfg["k_global"])
        if scheme == "global" or z is None:
            return kg
        table = self.kcfg["k_table"]
        if scheme == "band":
            z = np.atleast_1d(np.asarray(z, dtype=float))
            bi = np.clip(band_index(np.clip(z, 1, 1999)), 0, 5)
            k_b = np.array([table.get(str(b), kg) for b in range(6)])
            return k_b[bi]
        if scheme == "season":
            season = np.asarray(season, dtype=int)
            k_s = np.array([table.get(str(s), kg) for s in range(4)])
            return np.broadcast_to(k_s, np.atleast_1d(season).shape)
        if scheme == "sparse":
            sp = np.digitize(np.asarray(sparse_km, dtype=float), [50.0, 200.0])
            k_sp = np.array([table.get(str(i), kg) for i in range(3)])
            return np.broadcast_to(k_sp, np.atleast_1d(sp).shape)
        return kg
