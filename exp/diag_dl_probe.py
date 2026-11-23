# -*- coding: utf-8 -*-
"""诊断 EXP-8 DL 基线：隔离「加权 vs 非加权」与「val vs test 场景」两个因素。
只读，不改结果文件。"""
import sys, json, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from exp import config as C
from exp.lib import scenes as scenelib, meanet
from exp.lib.pipeline import SceneData, load_stack
import exp.exp8_baselines as E8

stack = load_stack()
SD = SceneData(stack)
scenes = stack["scenes"]
rng = np.random.default_rng(C.SEED)

# 只训 1 个成员以省时
E8.DL_ENSEMBLE = 1
dle = E8.train_dl_ensemble(SD, scenes, rng)
m, s = dle["members"][0], dle["stdrs"][0]

with open(os.path.join(C.RESULTS, "exp3", "val_scene_ids.json"), encoding="utf-8") as f:
    val_sids = json.load(f)


def wrmsse(pred, y, u):
    w = 1.0 / (u ** 2 + 0.02 ** 2)
    w = w / w.mean()
    return float(np.sqrt(np.mean(w * (pred - y) ** 2)))


print("\n--- (a) val 场景：训练同口径（全场景 ctx，排除同剖面）---")
rows = []
for sid in val_sids[:8]:
    dA = SD.build(int(sid))
    Cf = E8.make_ctx_features(dA["X4"], dA["X4"], dA["d"], dA["u"],
                              q_prof=dA["prof"], ctx_prof=dA["prof"])
    X = np.column_stack([dA["feat"], Cf])
    p = meanet.predict(m, s, X)
    y = dA["d"]
    rows.append({"sid": sid, "n": len(y),
                 "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
                 "wrmse": wrmsse(p, y, dA["u"])})
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n--- (b) val 场景：推理口径（A->B 留出，ctx=dsA）---")
rows = []
for sid in val_sids[:8]:
    A, B = SD.ab(int(sid), 0)
    dsA, dsB = SD.build(int(sid), A), SD.build(int(sid), B)
    p, sd_ = E8.dl_predict(SD, dle, dsB["X4"], dsA)
    y = dsB["d"]
    rows.append({"sid": sid, "n": len(y),
                 "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
                 "wrmse": wrmsse(p, y, dsB["u"]),
                 "mtheta_rmse": float(np.sqrt(np.mean((dsB["mean"] - y) ** 2)))})
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n--- (c) test 场景：推理口径 ---")
rows = []
for sid in scenelib.scene_ids_by_role(scenes, "test")[:8]:
    A, B = SD.ab(int(sid), 0)
    dsA, dsB = SD.build(int(sid), A), SD.build(int(sid), B)
    p, sd_ = E8.dl_predict(SD, dle, dsB["X4"], dsA)
    y = dsB["d"]
    rows.append({"sid": sid, "n": len(y),
                 "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
                 "wrmse": wrmsse(p, y, dsB["u"]),
                 "mtheta_rmse": float(np.sqrt(np.mean((dsB["mean"] - y) ** 2)))})
print(pd.DataFrame(rows).round(3).to_string(index=False))
