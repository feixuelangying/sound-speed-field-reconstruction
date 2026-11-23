# -*- coding: utf-8 -*-
"""场景库加载工具（EXP-2 产出 -> EXP-3/4/5/6 使用）。"""
import json
import os
import numpy as np
import pandas as pd

from .. import config as C


def load_scenes():
    """返回 (scenes 列表含 role/region/member_idx, meta dict)。
    scenes_full.json 的每元素含 scene_id/season/region/role/member_idx/lon0/..."""
    with open(os.path.join(C.RESULTS, "exp2", "scenes_full.json"), encoding="utf-8") as f:
        scenes = json.load(f)
    df = pd.read_csv(os.path.join(C.RESULTS, "exp2", "scenes.csv"))
    role_map = dict(zip(df["scene_id"].astype(int), df["role"]))
    for s in scenes:
        s["role"] = role_map[int(s["scene_id"])]
    with open(os.path.join(C.RESULTS, "exp2", "scenes_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return scenes, meta


def prof_scene_index(scenes, n_prof):
    """剖面 -> 场景归属（-1 = 无效场景，不参与训练/评估）。"""
    ps = np.full(n_prof, -1, dtype=np.int64)
    for s in scenes:
        for pi in s["member_idx"]:
            ps[pi] = int(s["scene_id"])
    return ps


def scene_ids_by_role(scenes, role):
    return [int(s["scene_id"]) for s in scenes if s["role"] == role]


def scene_by_id(scenes):
    return {int(s["scene_id"]): s for s in scenes}
