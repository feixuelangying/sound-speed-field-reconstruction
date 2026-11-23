# -*- coding: utf-8 -*-
"""全局配置：路径、冻结决策、常量。所有实验脚本 import 此模块。"""
import os
from pathlib import Path
import numpy as np

ROOT = str(Path(__file__).resolve().parents[1])
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
CACHE = os.path.join(RESULTS, "cache")
LOGS = os.path.join(ROOT, "logs")
WEIGHTS = os.path.join(ROOT, "weights")

# 主方案冻结共形因子（EXP-6d：banded block GP + val+calib 场景 bootstrap q95）
# 下游 EXP-7/8/9/10 一律以 UcComposer(k_json=MAIN_K_JSON) 取 k
MAIN_K_JSON = os.path.join(RESULTS, "frozen", "k_main.json")

HYCOM_NC = os.path.join(DATA, "hycom_TS.nc")
ARGO_CSV = os.path.join(DATA, "Argo", "argo_hits_in_box.csv")
ARGO_DIR = os.path.join(DATA, "Argo", "profiles")
CTD_NC = os.path.join(DATA, "CTD", "master_ctd_box.nc")

# ---- 冻结决策（D12 等，EXP-0 定标后确认）----
SEED = 42
SOUND_SPEED_EQ = "chen-millero"   # 主方程：Chen-Millero (UNESCO, seawater.svel)；对照: mackenzie, gsw
HYCOM_EPOCH = np.datetime64("2000-01-01T00:00:00")  # HYCOM time: hours since this epoch
Z_MAX = 2000.0                    # 限深声明（D8）：超出 HYCOM 最深层 2000 m 的观测截断

# ---- Argo QC（EXP-0）----
QC_GOOD = ("1", "2", "5", "8")    # Argo QC flag: 1 good, 2 probably good, 5 changed, 8 interpolated
T_RANGE = (-2.5, 33.0)            # degC
S_RANGE = (24.0, 42.0)            # psu（下限放宽至 24，容纳近岸低盐水）
SPIKE_FACTOR = 6.0                # med3 尖峰检验倍数（|x-med3| > factor * local_MAD 且绝对差 > abs_floor）
SPIKE_ABS_T = 0.30                # degC，绝对门限（避免深层 MAD~0 时误杀）
SPIKE_ABS_S = 0.15                # psu

# ---- 输出深度网格（跃层加密，D8；EXP-4 复核定标）----
OUT_Z_GRID = np.array([
    0, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 175, 200,
    225, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1100,
    1200, 1400, 1600, 1800, 2000], dtype=float)

# ---- CTD 规格（Type B，船基 SBE911 量级；EXP-1 确认）----
CTD_U_T = 0.002   # degC
CTD_U_S = 0.003   # psu

# ---- Argo 规格（Type B 初值，按 profiler_type 分型调整；EXP-1 定标）----
ARGO_U_T = 0.005   # degC
ARGO_U_S = 0.005   # psu（深层；表层加倍由交叉比定标）

def result_dir(exp_id):
    d = os.path.join(RESULTS, exp_id)
    os.makedirs(d, exist_ok=True)
    return d
