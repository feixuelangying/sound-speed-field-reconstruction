# -*- coding: utf-8 -*-
"""剖面垂向锚点降采样（framework §3.3.3：跃层密、深水疏，20-30 点/剖面）。
EXP-3 训练点构造与 EXP-4 在线 GP 输入共用同一锚点集，保证训练/推断一致。"""
import numpy as np

# 跃层加密锚点深度（m）：表层 5m 间隔 -> 跃层段 25m -> 深水 300m+
ANCHOR_Z = np.array([
    0, 5, 10, 15, 20, 30, 40, 50, 60, 75, 90, 110, 130, 160, 200,
    250, 300, 350, 420, 500, 600, 750, 900, 1100, 1300, 1600, 2000], dtype=float)


def anchor_pick(z_prof):
    """单剖面：为每个锚点取最近观测深度，返回去重后的剖面内索引（升序）。"""
    z_prof = np.asarray(z_prof, dtype=float)
    if len(z_prof) == 0:
        return np.array([], dtype=int)
    idx = np.abs(z_prof[:, None] - ANCHOR_Z[None, :]).argmin(axis=0)
    return np.unique(idx)


def anchor_indices(ps, prof_ids):
    """批量：ProfileSet + 剖面索引数组 -> 全局观测点索引（拼接展平）。"""
    out = []
    for ip in prof_ids:
        sl = ps.slice(int(ip))
        z = ps.obs_z()[sl]
        out.append(sl.start + anchor_pick(z))
    return np.concatenate(out) if out else np.array([], dtype=int)
