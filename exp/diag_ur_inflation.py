# -*- coding: utf-8 -*-
"""诊断：u_r 高估的方差来源分解（σ²_gp 是否被"场景间差异"污染）。

框架 §3.2 正交性设计：组间方差 -> u_m（均值函数误差），组内方差 -> σ²（GP 先验）。
若 EXP-4 超参拟合用跨场景合并方差（未按场景中心化），σ² 会吸收场景间差异 ->
u_r 重复包含 u_m 份额 -> u_c 保守约 1.6-1.8 倍（EXP-5 候选发现 C3）。

本脚本实测 pooled / within / between 三方分解，与 σ²_gp=10.122 及
EXP-3 sigma2_within 对比；并模拟"若 σ² 取 within 值"时 B 点的 u_r/覆盖/宽度。
"""
import sys, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from exp import config as C
from exp.lib import scenes as scenelib
from exp.lib.pipeline import SceneData, load_stack
from exp.lib import gp as gplib

stack = load_stack()
SD = SceneData(stack)
scenes = stack["scenes"]
train_sids = scenelib.scene_ids_by_role(scenes, "train")
with open(os.path.join(C.RESULTS, "exp3", "val_scene_ids.json"), encoding="utf-8") as f:
    val_sids = set(json.load(f))
tr_only = [s for s in train_sids if s not in val_sids]

rng = np.random.default_rng(101)
sids = [int(s) for s in rng.choice(tr_only, 16, replace=False)]

ys, bands, sidscol = [], [], []
for sid in sids:
    A, _ = SD.ab(sid, 0)
    dsA = SD.build(sid, A)
    y = dsA["d"] - dsA["mean"]
    ys.append(y)
    bands.append(dsA["band"])
    sidscol.append(np.full(len(y), sid))
Y = np.concatenate(ys)
B = np.concatenate(bands)
S = np.concatenate(sidscol)
print("A 集总点数 %d，场景数 %d，每场景点数中位 %d"
      % (len(Y), len(sids), int(np.median([len(y) for y in ys]))))
print("y = d - m_theta：mean=%.3f  std=%.3f" % (Y.mean(), Y.std(ddof=1)))


def decomp(Y, key):
    df = pd.DataFrame({"y": Y, "g": key})
    g = df.groupby("g")["y"]
    n = g.size().values.astype(float)
    m = g.mean().values
    v = g.var(ddof=1).values
    pooled = float(np.var(Y, ddof=1))
    within = float((n * np.nan_to_num(v)).sum() / n.sum())
    gm = float((n * m).sum() / n.sum())
    between = float((n * (m - gm) ** 2).sum() / n.sum())
    return pooled, within, between


print("\n== ① 按场景分组（全部深度）==")
p, w, b = decomp(Y, S)
print("pooled=%.3f  within=%.3f  between=%.3f  (within+between=%.3f)" % (p, w, b, w + b))
print("EXP-4 sigma2_gp(prod_global) = 10.122")
print("  sigma2_gp / within  = %.2f" % (10.122 / w))
print("  sigma2_gp / pooled  = %.2f" % (10.122 / p))

print("\n== ② 分带：按场景分组 ==")
rows = []
for bi in range(6):
    m = B == bi
    if m.sum() < 20:
        continue
    p2, w2, b2 = decomp(Y[m], S[m])
    rows.append({"band": bi, "n": int(m.sum()), "pooled": round(p2, 3),
                 "within": round(w2, 3), "between": round(b2, 3),
                 "between_share": round(b2 / p2, 3) if p2 > 0 else np.nan})
print(pd.DataFrame(rows).to_string(index=False))

print("\n== ③ 分带内、再按 场景×季节 与 band_kerns σ² 对比 ==")
bk_path = os.path.join(C.RESULTS, "exp4", "band_kerns.json")
if os.path.exists(bk_path):
    with open(bk_path, encoding="utf-8") as f:
        bk = json.load(f)
    rows2 = []
    for bi in range(6):
        m = B == bi
        if m.sum() < 20:
            continue
        _, w2, _ = decomp(Y[m], S[m])
        kd = bk.get(str(bi))
        s2b = float(np.exp(kd["log_s2"])) if kd else np.nan
        rows2.append({"band": bi, "within": round(w2, 3),
                      "banded_sigma2": round(s2b, 3) if np.isfinite(s2b) else np.nan,
                      "ratio": round(s2b / w2, 2) if w2 > 0 and np.isfinite(s2b) else np.nan})
    print(pd.DataFrame(rows2).to_string(index=False))

print("\n== ④ EXP-3 sigma2_within（随机效应组内口径）==")
sw = pd.read_csv(os.path.join(C.RESULTS, "exp3", "sigma2_within.csv"))
print("columns:", list(sw.columns))
print(sw.to_string(index=False))

print("\n== ⑤ 反事实模拟：σ² 缩放到 within 后，B 点 u_r / 覆盖 / 宽度 ==")
test_sids = scenelib.scene_ids_by_role(scenes, "test")[:6]
kern_g = SD.kern_for(0)  # 全局核（prod_global）
raw = dict(kern_g.raw)
s2_now = float(np.exp(raw["log_s2"]))
rows3 = []
for sid in test_sids:
    A, Bp = SD.ab(int(sid), 0)
    dsA, dsB = SD.build(int(sid), A), SD.build(int(sid), Bp)
    # 当前 σ²
    g0 = gplib.GPFit(dsA["X4"], dsA["d"], dsA["u"], dsA["mean"], kern_g)
    p0, v0 = g0.predict(dsB["X4"], dsB["mean"])
    err = p0 - dsB["d"]
    # σ² -> 场景内中心化方差（按 A 集场景内中心化估）
    yA = dsA["d"] - dsA["mean"]
    wA = float(np.var(yA, ddof=1))
    scale = wA / s2_now
    raw2 = dict(raw)
    raw2["log_s2"] = float(np.log(s2_now * scale))
    kern2 = gplib.Kernel4D("prod", {k: v for k, v in raw2.items()},
                           lat_ref=kern_g.lat_ref)
    g1 = gplib.GPFit(dsA["X4"], dsA["d"], dsA["u"], dsA["mean"], kern2)
    p1, v1 = g1.predict(dsB["X4"], dsB["mean"])
    for tag, vv, pp in (("cur", v0, p0), ("shrunk", v1, p1)):
        ur = np.sqrt(vv)
        err2 = pp - dsB["d"]
        rows3.append({"sid": int(sid), "cfg": tag, "ur_mean": float(ur.mean()),
                      "rmse": float(np.sqrt(np.mean(err2 ** 2))),
                      "err_var": float(np.var(err2, ddof=1)),
                      "cov95_gp": float(np.mean(np.abs(err2) <= 1.96 * ur))})
d3 = pd.DataFrame(rows3)
print(d3.round(3).to_string(index=False))
print("\n汇总：")
print(d3.groupby("cfg")[["ur_mean", "rmse", "err_var", "cov95_gp"]].mean().round(3).to_string())
print("\n场景内中心化方差 wA(中位)=%.3f  vs  sigma2_gp=%.3f  ->  scale=%.3f"
      % (np.median([float(np.var(SD.build(int(s), SD.ab(int(s), 0)[0])["d"]
                                    - SD.build(int(s), SD.ab(int(s), 0)[0])["mean"],
                                    ddof=1)) for s in test_sids]), s2_now,
         np.median([float(np.var(SD.build(int(s), SD.ab(int(s), 0)[0])["d"]
                                 - SD.build(int(s), SD.ab(int(s), 0)[0])["mean"],
                                 ddof=1)) for s in test_sids]) / s2_now))
