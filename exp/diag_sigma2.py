# -*- coding: utf-8 -*-
"""诊断 σ²_gp 与数据方差严重不匹配的根因。

现象：prod_global 拟合 σ²=10.122（ℓ_h≈14km/ℓ_z≈19m/ℓ_t≈4d 短程核），
而训练数据 y=d-m_θ 的样本方差仅约 2.5。短程核下 K≈diag(σ²+u²)，ML 解
应满足 σ²≈var(y)，故 4 倍差距在数学上可疑。

本脚本：
① 复现 EXP-4 的 HP 抽样，打印各场景 y 的方差/极值；
② 用冻结核与"σ² 改回数据方差"的核分别计算 -ΣlogML，判定 ML 是否真偏好大 σ²；
③ 逐场景单独 ML 拟合，看 σ² 是否仍偏大。
"""
import sys, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

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
print("tr_only n =", len(tr_only))

HP_RNG = np.random.default_rng(C.SEED + 101)
sids = list(HP_RNG.choice(tr_only, min(12, len(tr_only)), replace=False))
print("EXP-4 prod_global sids:", sids)

ds = [SD.build(s) for s in sids]
print("\n== ① 各场景（SD.build 全锚点口径）==")
for x in ds:
    y = x["d"] - x["mean"]
    print("  sid=%3d n=%5d var=%9.3f mean=%7.3f min=%8.2f max=%8.2f"
          % (x["sid"], len(y), np.var(y, ddof=1), y.mean(), y.min(), y.max()))
allv = np.concatenate([x["d"] - x["mean"] for x in ds])
print("pooled: n=%d var=%.3f mean=%.3f std=%.3f"
      % (len(allv), np.var(allv, ddof=1), allv.mean(), allv.std(ddof=1)))
q = np.percentile(np.abs(allv), [50, 90, 99, 99.9, 100])
print("  |y| 分位: p50=%.2f p90=%.2f p99=%.2f p99.9=%.2f max=%.2f" % tuple(q))

kern = SD.kern_for(0)
r = kern.raw
print("\nfitted prod_global: sigma2=%.3f  lh=%.2fkm  lz=%.2fm  lt=%.2fd"
      % (kern.diag_var(), np.exp(r["log_lh"]), np.exp(r["log_lz"]), np.exp(r["log_lt"])))


def total_logml(kern, ds, tag=""):
    tot = 0.0
    for x in ds:
        g = gplib.GPFit(x["X4"], x["d"], x["u"], x["mean"], kern)
        tot += g.log_ml
    print("  logML(%s) = %.1f" % (tag, tot))
    return tot


print("\n== ② 冻结核 vs σ² 改值后的 -ΣlogML 对比 ==")
base = total_logml(kern, ds, "fitted s2=%.2f" % kern.diag_var())
for s2 in (np.var(allv, ddof=1), 2.5, 5.0, 8.0, 20.0, 40.0):
    raw2 = dict(r)
    raw2["log_s2"] = float(np.log(s2))
    k2 = gplib.Kernel4D("prod", raw2, lat_ref=kern.lat_ref)
    total_logml(k2, ds, "s2=%.3f" % s2)

print("\n== ③ 逐场景单独 ML（form=prod, 初值取默认）==")
for x in ds[:6]:
    one = [(x["X4"], x["d"], x["u"], x["mean"])]
    k1, lmls, info = gplib.fit_hyperparams(one, form="prod")
    rr = k1.raw
    print("  sid=%3d n=%5d  var(y)=%7.3f -> s2=%8.3f lh=%6.2fkm lz=%6.2fm lt=%5.2fd"
          % (x["sid"], len(x["d"]), np.var(x["d"] - x["mean"], ddof=1),
             k1.diag_var(), np.exp(rr["log_lh"]), np.exp(rr["log_lz"]),
             np.exp(rr["log_lt"])))
