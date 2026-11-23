# -*- coding: utf-8 -*-
"""u_obs 查表与声速域换算（EXP-1 评定结果 -> EXP-3/4/5 使用）。

口径决策（2026-09-27 定稿，日志【问题11·计划外发现】）：
- **Type B 仪器规格为主体**：Argo-CTD 时空交叠过少（24h/50km 仅 1 对，
  72h/50km 8 对），Type A 交叉比对定标不可行；u_obs 采用规格表
  （SPEC_U_T/U_S，分带，表层保守加倍——跃层梯度大、传感器时滞效应），
  经 Jacobian 传播至声速域。
- 不分型：Argo 型间差异主要在系统项（由均值网络 m_theta 学习、残余进
  u_m/GP 核），规格值（随机项）各型接近；
- 若后续 EXP-1 产出有效 within-MAD 表（每带 >=5 对），仍按
  max(within-MAD, 规格) 采用——此路径保留但当前数据下不触发。
"""

import os
import numpy as np
import pandas as pd

from .. import config as C
from . import physics

ZBANDS = [(0, 30), (30, 60), (60, 125), (125, 350), (350, 800), (800, 2000)]
ZB_NAMES = [f"{a}-{b}" for a, b in ZBANDS]
BAND_CENTERS = np.array([(a + b) / 2 for a, b in ZBANDS])

# Type B 规格（分带，°C / psu；表层保守加倍）
SPEC_U_T = np.array([0.010, 0.007, 0.005, 0.005, 0.005, 0.005])
SPEC_U_S = np.array([0.010, 0.007, 0.005, 0.005, 0.005, 0.005])


def load_uobs():
    """返回 (uT_by_band, uS_by_band, source)。主体 Type B 规格；
    EXP-1 within-MAD 表每带 >=5 对时按 max(within-MAD, 规格) 采纳。"""
    uT = SPEC_U_T.copy()
    uS = SPEC_U_S.copy()
    src = "typeB_spec_primary"
    path = os.path.join(C.RESULTS, "exp1", "uobs_match_summary.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        sub = df[df["win"] == "w24h50km"]
        if len(sub):
            wt, ws = [], []
            for bi, name in enumerate(ZB_NAMES):
                r = sub[sub["band"] == name]
                if len(r):
                    wt.append(float(r["within_mad_T_median"].iloc[0]))
                    ws.append(float(r["within_mad_S_median"].iloc[0]))
                else:
                    wt.append(SPEC_U_T[bi])
                    ws.append(SPEC_U_S[bi])
            uT = np.maximum(np.array(wt), SPEC_U_T)
            uS = np.maximum(np.array(ws), SPEC_U_S)
            src = "exp1_within_mad_with_spec_floor"
    return uT, uS, src


def band_index(z):
    b = np.full(len(z), -1, dtype=int)
    for bi, (z0, z1) in enumerate(ZBANDS):
        b[(z >= z0) & (z < z1)] = bi
    return b


def u_obs_c(T, S, p, z, lat=None):
    """逐点声速域观测不确定度 u_obs (m/s)。T/S/p: 同长数组；z: 深度(m)。"""
    uT_by_band, uS_by_band, _ = load_uobs()
    z = np.asarray(z, dtype=float)
    T = np.asarray(T, dtype=float)
    S = np.asarray(S, dtype=float)
    p = np.asarray(p, dtype=float)
    bi = np.clip(band_index(z), 0, 5)
    uT = uT_by_band[bi]
    uS = uS_by_band[bi]
    dcdT, dcdS, _ = physics.sound_speed_jacobian(T, S, p)
    return np.sqrt((dcdT * uT) ** 2 + (dcdS * uS) ** 2)
