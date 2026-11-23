# -*- coding: utf-8 -*-
"""EXP-3 summary 重建：训练/OOD 全部完成后 json.dump 因 groupby tuple 键崩溃
（问题16，已修复主脚本），csv/权重产物完整。本脚本从已存 csv 重建
exp3_summary.json，不重跑训练。数值来源：results/exp3/*.csv + 运行日志
logs/exp3_run.txt（anchor 规模、val 场景数）。"""
import sys, os, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from exp import config as C
from exp.lib import uobs

OUT = C.result_dir("exp3")

feat_df = pd.read_csv(os.path.join(OUT, "feat_ablation.csv"))
struct_df = pd.read_csv(os.path.join(OUT, "struct_ablation.csv"))
ho_df = pd.read_csv(os.path.join(OUT, "heldout_eval.csv"))
diag = pd.read_csv(os.path.join(OUT, "cond_diag.csv"))
um_df = pd.read_csv(os.path.join(OUT, "um_indist.csv"))
ood_df = pd.read_csv(os.path.join(OUT, "um_ood.csv"))
with open(os.path.join(OUT, "val_scene_ids.json"), encoding="utf-8") as f:
    val_sids = json.load(f)

# held-out 总体指标（分带平方加权重建；6 带全、Σn=57580 与锚点集一致，无遗漏）
n = ho_df["n"].values
rmse_d = float(np.sqrt(np.sum(n * ho_df["rmse_d_baseline"] ** 2) / n.sum()))
rmse_e = float(np.sqrt(np.sum(n * ho_df["rmse_e"] ** 2) / n.sum()))

summary = {
    "exp": "exp3", "date": time.strftime("%Y-%m-%d %H:%M"),
    "rebuilt_from_csv": True,
    "n_anchor_train": 244740,      # logs/exp3_run.txt
    "n_anchor_calib": 57580,       # logs/exp3_run.txt
    "val_scene_split": {"n_val_scenes": len(val_sids)},
    "baseline_val_wrmse": float(feat_df["baseline_wrmse"].iloc[0]),
    "feat_ablation": feat_df.to_dict("records"),
    "struct_ablation": struct_df.to_dict("records"),
    "best_feat": "F3_full", "best_struct": "S128x6",
    "heldout": {"rmse_baseline": rmse_d, "rmse_residual": rmse_e,
                "var_reduction_pct": 100 * (1 - rmse_e ** 2 / rmse_d ** 2),
                "by_band": ho_df.to_dict("records")},
    "cond_diag_flagged": diag[diag["flag"]]["feature"].tolist(),
    "um_indist_mean": float(um_df["um"].mean()),
    "um_indist_by_band": {str(k): float(v) for k, v in
                          um_df.groupby("band")["um"].mean().round(4).items()},
    "um_ood_summary": {f"{m}_held{int(h)}": float(v) for (m, h), v in
                       ood_df.groupby(["mode", "held"])["um_ood"].mean()
                       .round(4).items()},
    "um_ood_vs_indist_ratio": {f"{m}_held{int(h)}": float(v)
                               for (m, h), v in (ood_df.groupby(["mode", "held"])
                                                 .apply(lambda g: g["um_ood"].mean()
                                                        / g["um_indist"].mean(),
                                                        include_groups=False)
                                                 .round(3)).items()},
    "um_floor_ms": 0.05,
    "uobs_source": uobs.load_uobs()[2],
    "elapsed_s": None,
}
with open(os.path.join(OUT, "exp3_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
print(json.dumps({k: v for k, v in summary.items()
                  if k not in ("feat_ablation", "struct_ablation", "heldout")},
                 ensure_ascii=False, indent=2, default=float))
print("OK")
