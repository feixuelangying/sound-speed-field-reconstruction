# -*- coding: utf-8 -*-
"""banded block GP（framework D14 / R6 降级路径，EXP-6 后升为候选主方案）：
按深度带独立拟合超参，使 GP 后验方差 u_r 获得深度结构。

动机（EXP-6 发现）：单 GP 用平稳核（σ² 为常数），u_r 几乎不随深度变化
（各带 2.49–2.73 m/s，range/mean 0.09），而实际 rmse 从深层 0.82 到主跃层 2.09
变化 2.55 倍 → 分带覆盖率失衡（主跃层欠覆盖 0.887、深层过覆盖 0.996），
总体 0.936 低于 J2 容差下界 0.949。

EXP-6b 验证：分带块 GP 使 u_r 的 range/mean 升至 1.31，各带所需共形 k 跨度由
[0.53,1.79] 收窄至 [1.06,1.51]——深度结构问题被修复。
"""
import os
import json
import numpy as np

from .. import config as C
from . import gp as gplib
from .uobs import band_index

NBAND = 6
HP_DIR = os.environ.get("HP_DIR", "exp4b")    # EXP-11 起主口径 = exp4b（σ² 矩估计锚定）
KERN_PATH = os.path.join(C.RESULTS, HP_DIR, "band_kerns.json")


def fit_band_kerns(SD, tr_sids, form="prod", hp_subset=12, seed=101, verbose=True,
                   s2_mode="ml"):
    """train-only 场景按深度带拟合 NBAND 套超参（跨场景共享同带超参）。"""
    rng = np.random.default_rng(seed)
    out = {}
    for bi in range(NBAND):
        ds_list = []
        for s in rng.permutation(tr_sids):
            x = SD.build(int(s))
            m = x["band"] == bi
            if m.sum() >= 15:
                ds_list.append((x["X4"][m], x["d"][m], x["u"][m], x["mean"][m]))
            if len(ds_list) >= hp_subset:
                break
        if len(ds_list) < 3:
            out[bi] = None
            if verbose:
                print(f"  band {bi}: 样本不足，跳过", flush=True)
            continue
        kern, _, _ = gplib.fit_hyperparams(ds_list, form=form, s2_mode=s2_mode)
        out[bi] = kern
        if verbose:
            r = kern.raw
            print(f"  band {bi}: n_scenes={len(ds_list):2d} "
                  f"s2={np.exp(r['log_s2']):8.3f} "
                  f"lh={np.exp(r['log_lh']):7.2f}km lz={np.exp(r['log_lz']):7.2f}m "
                  f"lt={np.exp(r['log_lt']):6.2f}d", flush=True)
    return out


def save_band_kerns(bk, path=None):
    path = path or KERN_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({str(k): (v.to_dict() if v is not None else None)
                   for k, v in bk.items()}, f, ensure_ascii=False, indent=1)
    return path


def load_band_kerns(path=None):
    """返回 {band: Kernel4D|None}；文件不存在返回 None。"""
    path = path or KERN_PATH
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return {int(k): (gplib.kernel_from_dict(v) if v is not None else None)
            for k, v in d.items()}


def predict_banded(dsA, Xq, mean_q, band_q, band_kerns, fallback_kern):
    """分带块 GP 预测，返回 (pred, var) 与 Xq 同序。

    规则：带内核缺失 → 用 fallback_kern（全点单 GP 核）；
    A 集在该带无点 → 先验（pred = 均值网络输出, var = σ²_带）。
    band_q 中的 -1（z≥Z_MAX 越界）按最近带 5 处理（clip）。
    """
    pred = np.empty(len(Xq))
    var = np.empty(len(Xq))
    bq = np.clip(np.asarray(band_q, dtype=int), 0, NBAND - 1)
    for bi in range(NBAND):
        mb = bq == bi
        if not mb.any():
            continue
        kern = band_kerns.get(bi) or fallback_kern
        ma = dsA["band"] == bi
        if ma.sum() == 0:
            pred[mb] = mean_q[mb]
            var[mb] = kern.diag_var()
        else:
            gb = gplib.GPFit(dsA["X4"][ma], dsA["d"][ma], dsA["u"][ma],
                             dsA["mean"][ma], kern)
            pp, vv = gb.predict(Xq[mb], mean_q[mb])
            pred[mb], var[mb] = pp, vv
    return pred, var


def predict_main(SD, ds, Xq, band_kerns=None, mean_q=None):
    """下游实验（EXP-8/9/10）统一的主方案预测入口。

    band_kerns 非空 -> 分带块 GP（主方案）；None -> 单 GP（对照口径）。
    mean_q 缺省时按 Xq 现场取均值网络输出。返回 (pred, var)。
    """
    mq = SD.query_features(Xq) if mean_q is None else mean_q
    if band_kerns:
        bq = band_index(Xq[:, 2])
        return predict_banded(ds, Xq, mq, bq, band_kerns, SD.kern_for(ds["season"]))
    g = SD.fit_gp(ds)
    return g.predict(Xq, mq)
