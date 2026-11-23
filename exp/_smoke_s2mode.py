# -*- coding: utf-8 -*-
"""冒烟：验证 fit_hyperparams(s2_mode='moment') 的 σ² 锚定行为。"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from exp.lib import scenes as scenelib, gp as gplib
from exp.lib.pipeline import SceneData, load_stack

stack = load_stack()
SD = SceneData(stack)
tr = scenelib.scene_ids_by_role(stack["scenes"], "train")
ds = [SD.build(int(s)) for s in tr[:4]]
X = [(x["X4"], x["d"], x["u"], x["mean"]) for x in ds]
yv = np.concatenate([x["d"] - x["mean"] for x in ds])
print("n_scenes=%d  n_pts=%d  data var(pooled)=%.4f" % (len(ds), len(yv), np.var(yv, ddof=1)))

for mode in ("ml", "moment"):
    k, lmls, info = gplib.fit_hyperparams(X, form="prod", s2_mode=mode, n_iter=60)
    r = k.raw
    print("%7s -> s2=%9.4f lh=%7.2fkm lz=%7.2fm lt=%6.2fd  mean_logML=%.1f"
          % (mode, np.exp(r["log_s2"]), np.exp(r["log_lh"]), np.exp(r["log_lz"]),
             np.exp(r["log_lt"]), np.mean(lmls)))

# add 形式在 moment 下应回退 ml（σ² 分量结构不适用）
ka, _, _ = gplib.fit_hyperparams(X, form="add", s2_mode="moment", n_iter=30)
print("add/moment -> s2h=%.3f s2z=%.3f (应等同 ml 路径)"
      % (np.exp(ka.raw["log_s2h"]), np.exp(ka.raw["log_s2z"])))
print("SMOKE OK")
