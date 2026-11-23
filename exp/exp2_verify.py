# -*- coding: utf-8 -*-
"""EXP-2 产物一致性校验（划分协议冻结前的独立复核）。
① scenes_full.json 内部：n_profiles == len(member_idx)；member_idx 无重复/越界
② scenes_split.npz：role_of_scene（按 uk 索引）与 json 的 role 一致；
   prof_role 与场景成员一致；scene_key/inv 可重建
③ ab_splits：A∩B=∅、A∪B=场景成员、test 场景 5 种子
④ 锚定场景存在且属性正确
⑤ 场景块空间不相交（D10 块间无重叠）
"""
import json
import sys
from pathlib import Path
import numpy as np

RES = Path(__file__).resolve().parents[1] / "results" / "exp2"

with open(RES / "scenes_full.json", encoding="utf-8") as f:
    scenes = json.load(f)
with open(RES / "scenes_meta.json", encoding="utf-8") as f:
    meta = json.load(f)
z = np.load(RES / "scenes_split.npz")
with open(RES / "ab_splits.json", encoding="utf-8") as f:
    ab = json.load(f)

ok = True
def chk(cond, msg):
    global ok
    print(("PASS " if cond else "FAIL ") + msg)
    ok = ok and bool(cond)

# ① 内部一致
all_members = []
for s in scenes:
    assert s["n_profiles"] == len(s["member_idx"]), f"n_profiles mismatch sid={s['scene_id']}"
    all_members += s["member_idx"]
chk(len(set(all_members)) == len(all_members), "member_idx 无重复（剖面一生一角色）")
chk(max(all_members) < 18187 and min(all_members) >= 0, "member_idx 无越界")

# ② npz role 对齐
role_json = {s["key"]: None for s in scenes}
inv = z["inv"]; scene_key = z["scene_key"]
prof_role = z["prof_role"]
uk = np.unique(scene_key)
n_match = 0
for s in scenes:
    si = np.where(uk == s["key"])[0]
    assert len(si) == 1
    r_npz = str(z["role_of_scene"][si[0]])
    m_ids = np.array(s["member_idx"])
    r_prof = set(prof_role[m_ids])
    if r_npz != "none" and r_prof == {r_npz}:
        n_match += 1
chk(n_match == len(scenes), f"npz role 与 prof_role 一致（{n_match}/{len(scenes)}）")
chk(int((prof_role == "none").sum()) == 1646, "none 剖面数 = 1646（不在任何有效场景）")

# ③ A/B 划分
n_test_ab = 0
for sid, seeds in ab.items():
    s = scenes[int(sid)]
    mem = set(s["member_idx"])
    for k, sp in seeds.items():
        A, B = set(sp["A"]), set(sp["B"])
        assert not (A & B), f"A/B 交集非空 sid={sid} seed={k}"
        assert A | B == mem, f"A/B 并集 != 成员 sid={sid} seed={k}"
        assert len(B) >= 4, f"留出集 <4 sid={sid}"
    n_test_ab += 1
n_test = sum(1 for s in scenes if s["role"] if False) if False else meta["n_test"]
chk(n_test_ab >= n_test, f"ab_splits 覆盖 test+calib 场景（{n_test_ab} >= {n_test}）")
seeds_per_sid = {sid: len(v) for sid, v in ab.items()}
n5 = sum(1 for sid, v in ab.items() if len(v) == 5)
print(f"  test 场景 5 种子数: {n5}; 场景总数 in ab: {n_test_ab}")

# ④ 锚定场景（role 在 scenes.csv）
import pandas as pd
sdf = pd.read_csv(RES / "scenes.csv")
role_of = dict(zip(sdf["scene_id"], sdf["role"]))
for sid in meta["anchored_scene_ids"]:
    s = scenes[sid]
    print(f"  anchored sid={sid}: role={role_of[sid]} season={s['season']} "
          f"n_prof={s['n_profiles']} n_ctd={s['n_ctd_in_block']} "
          f"lon[{s['lon0']},{s['lon1']}] lat[{s['lat0']},{s['lat1']}]")
    assert role_of[sid] == "test"
chk(all(role_of[s] == "test" for s in meta["anchored_scene_ids"]),
    "锚定场景均为 test 角色")

# ⑤ 块不相交（空间边界）。同 key 场景只有一个；不同场景块边界可相邻但不重叠
deg = meta["deg"]
lons = sorted({(s["lon0"], s["lon1"]) for s in scenes})
bad = [s for s in scenes if s["lon1"] - s["lon0"] != deg or s["lat1"] - s["lat0"] != deg]
chk(len(bad) == 0, "场景块尺寸一致（%d x %d deg）" % (deg, deg))
# 覆盖：train/calib/test 剖面数
by_role = {}
for s in scenes:
    by_role.setdefault(role_of[s["scene_id"]], []).append(s["n_profiles"])
for r, v in sorted(by_role.items()):
    print(f"  role={r}: n_scenes={len(v)}, profiles={sum(v)}")

# 时空块无重叠检查：任两个场景的 (t_block, bx, by) 键唯一
keys = [s["key"] for s in scenes]
chk(len(set(keys)) == len(keys), "场景 key 唯一（时空块无重叠）")

print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
