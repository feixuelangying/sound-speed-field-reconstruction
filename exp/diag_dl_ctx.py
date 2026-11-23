# -*- coding: utf-8 -*-
"""诊断 EXP-8 DL 基线（B5_dl）与 EnOI（B4_enoi）误差异常偏大的根因。
只读，不改任何结果文件。输出到 stdout。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from exp import config as C
from exp.lib import scenes as scenelib, uobs
from exp.lib.pipeline import SceneData, load_stack

stack = load_stack()
SD = SceneData(stack)
scenes = stack["scenes"]
test_sids = scenelib.scene_ids_by_role(scenes, "test")[:6]
print("probe scenes:", test_sids)

rows = []
for sid in test_sids:
    for seed in (0, 1):
        A, B = SD.ab(sid, seed)
        dsA, dsB = SD.build(sid, A), SD.build(sid, B)
        y = dsB["d"]
        # (a) 仅均值网络 m_theta
        m = dsB["mean"]
        # (b) kNN 上下文残差估计（训练/推理同口径）
        from exp.exp8_baselines import make_ctx_features
        Cf = make_ctx_features(dsB["X4"], dsA["X4"], dsA["d"], dsA["u"])
        wd = Cf[:, 0]
        rows.append({"sid": sid, "seed": seed,
                     "rmse_mtheta": float(np.sqrt(np.mean((m - y) ** 2))),
                     "rmse_ctx_knn": float(np.sqrt(np.mean((wd - y) ** 2))),
                     "ctx_range": float(wd.max() - wd.min()),
                     "ctx_absmax": float(np.abs(wd).max()),
                     "y_absmax": float(np.abs(y).max()),
                     "nA": len(dsA["d"]), "nB": len(y)})
df = pd.DataFrame(rows)
print(df.round(3).to_string(index=False))
print("\nmean rmse_mtheta=%.3f  rmse_ctx_knn=%.3f" %
      (df.rmse_mtheta.mean(), df.rmse_ctx_knn.mean()))
