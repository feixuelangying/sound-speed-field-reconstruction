# -*- coding: utf-8 -*-
"""EXP-2 场景库与防泄漏划分协议（WP0/WP3，D10/D16）。
① 观测密度普查：月 × 空间块（3°网格）统计 Argo/CTD 数
② 场景模板定标：时段宽度 × 空间块尺寸扫描（可行域：场景剖面数中位数落于 10-40）
③ 三重划分：场景（月块×空间块）按 季节×海区 分层分配 train/calib/test
④ 角色分配：每条 Argo 剖面一生一个角色；CTD 独立
⑤ 测试/calib 场景内 A(输入)/B(留出) 划分 × 多随机种子
⑥ 泄漏自检：同 WMO 跨角色检测、时空邻接检测
⑦ 划分协议冻结（哈希）
产出：results/exp2/census_*.csv, scenes.csv, split.json, scenes_meta.json
"""
import sys, os, time, json, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from exp import config as C
from exp.lib import io_data

RNG = np.random.default_rng(C.SEED)
OUT = C.result_dir("exp2")


def _jd(o):
    """numpy 标量 -> Python 原生（json.dump default；Windows 下 astype(int) 为
    int32、np.unique 运算产物为 int64，均不可直接序列化，日志【问题12】）。"""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not serializable: {type(o)}")


def census(argo, ctd):
    """月 × 3°×3° 块的 Argo/CTD 密度普查。"""
    def census_one(t64, lat, lon, name):
        # pd.to_datetime(ndarray) 返回 DatetimeIndex：直接取 .year/.month（无 .dt，日志【问题10】）
        dt = pd.DatetimeIndex(t64.astype("datetime64[s]"))
        ym = dt.year.values * 100 + dt.month.values
        bx = np.floor((lon - 110.0) / 3.0).astype(int)
        by = np.floor((lat - 10.0) / 3.0).astype(int)
        df = pd.DataFrame({"ym": ym, "bx": bx, "by": by})
        g = df.groupby(["ym", "bx", "by"]).size().reset_index(name=name)
        return g
    ga = census_one(argo.t64, argo.lat, argo.lon, "argo")
    gc = census_one(ctd.t64, ctd.lat, ctd.lon, "ctd")
    m = ga.merge(gc, on=["ym", "bx", "by"], how="outer").fillna(0)
    m["total"] = m["argo"] + m["ctd"]
    m = m.sort_values("total", ascending=False)
    m.to_csv(os.path.join(OUT, "census_3deg_month.csv"), index=False)
    # 空间聚合版（无时间维度，供锚定场景选择）
    sp = m.groupby(["bx", "by"]).agg(argo=("argo", "sum"), ctd=("ctd", "sum")).reset_index()
    sp["total"] = sp["argo"] + sp["ctd"]
    sp = sp.sort_values("total", ascending=False)
    sp.to_csv(os.path.join(OUT, "census_3deg_space.csv"), index=False)
    return m, sp


def calibrate_scene_template(argo):
    """扫描（时段宽度月数, 空间块尺寸°）-> 有效场景数与剖面数分布。
    有效场景：QC 后剖面数 >= 12（支撑 8 输入 + 4 留出的下限）。"""
    rows = []
    dt = pd.DatetimeIndex(argo.t64.astype("datetime64[s]"))
    for months in (1, 2):
        for deg in (4, 5, 6, 8, 10):
            t_block = (dt.year.values * 12 + dt.month.values - 1) // months
            bx = np.floor((argo.lon - 110.0) / deg).astype(int)
            by = np.floor((argo.lat - 10.0) / deg).astype(int)
            key = t_block * 10000 + bx * 100 + by
            uk, cnt = np.unique(key, return_counts=True)
            valid = cnt[cnt >= 12]
            med = float(np.median(cnt)) if len(cnt) else 0
            # 剖面数落在 12-40 的场景占比
            in_range = ((cnt >= 12) & (cnt <= 40)).sum()
            rows.append({"months": months, "deg": deg, "n_scene_cells": len(uk),
                         "n_valid_scenes": int((cnt >= 12).sum()),
                         "n_scenes_12_40": int(in_range),
                         "profiles_per_scene_median": med,
                         "profiles_per_scene_p90": float(np.percentile(cnt, 90)) if len(cnt) else 0,
                         "n_profiles_in_valid": int(valid.sum())})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "scene_template_scan.csv"), index=False)
    return df


def build_scenes(argo, ctd, months, deg, min_profiles=12, split_ratio=(0.60, 0.15, 0.25),
                 n_seeds=5):
    """构建场景库 + 三重划分 + 角色分配 + A/B 划分 + 泄漏自检。"""
    dt = pd.DatetimeIndex(argo.t64.astype("datetime64[s]"))
    t_block = (dt.year.values * 12 + dt.month.values - 1) // months
    bx = np.floor((argo.lon - 110.0) / deg).astype(int)
    by = np.floor((argo.lat - 10.0) / deg).astype(int)
    scene_key = t_block * 10000 + bx * 100 + by
    uk, inv = np.unique(scene_key, return_inverse=True)
    counts = np.bincount(inv)
    valid_scene = counts >= min_profiles
    # 场景属性
    scenes = []
    for si, key in enumerate(uk):
        if not valid_scene[si]:
            continue
        tb = key // 10000; bxx = (key // 100) % 100; byy = key % 100
        m_ids = np.where(inv == si)[0]
        months_list = sorted(set(dt.month.values[m_ids]))
        mmode = int(np.bincount(months_list, minlength=13)[1:13].argmax()) + 1
        season = 0 if mmode in (12, 1, 2) else 1 if mmode in (3, 4, 5) \
            else 2 if mmode in (6, 7, 8) else 3  # 冬0 春1 夏2 秋3（原 mmode//3 会把 12 月映射为 4，日志【问题6】）
        scenes.append({
            "scene_id": len(scenes), "key": int(key), "t_block": int(tb),
            "uk_idx": int(si),  # unique-key 数组索引（scenes_split.npz 对齐用）
            "bx": int(bxx), "by": int(byy),
            "lon0": int(110 + bxx * deg), "lon1": int(110 + (bxx + 1) * deg),
            "lat0": int(10 + byy * deg), "lat1": int(10 + (byy + 1) * deg),
            "n_profiles": int(counts[si]),
            "season": season,
            "member_idx": m_ids.tolist(),
        })
    # 分层分配 train/calib/test：按 季节 × 海区（bx 中位二分）分层，比例 60/15/25
    for s in scenes:
        s["region"] = 0 if s["bx"] < np.median([x["bx"] for x in scenes]) else 1
    strata = {}
    for s in scenes:
        strata.setdefault((s["season"], s["region"]), []).append(s)
    role_of_scene = {}
    for k, lst in strata.items():
        idx = np.arange(len(lst))
        order = RNG.permutation(idx)
        n = len(lst)
        n_tr = int(round(split_ratio[0] * n)); n_ca = int(round(split_ratio[1] * n))
        for rank, ii in enumerate(order):
            s = lst[ii]
            role_of_scene[s["scene_id"]] = "train" if rank < n_tr else \
                ("calib" if rank < n_tr + n_ca else "test")
    # 剖面角色
    prof_role = np.full(argo.n_prof, "none", dtype=object)
    for s in scenes:
        for pi in s["member_idx"]:
            prof_role[pi] = role_of_scene[s["scene_id"]]
    # 泄漏自检：同 WMO 跨角色
    wmo = argo.wmo
    cross = 0; cross_pairs = []
    for w in np.unique(wmo):
        roles = set(prof_role[wmo == w])
        roles.discard("none")
        if len(roles) > 1:
            cross += 1
            cross_pairs.append((int(w), sorted(roles)))
    # A/B 划分（test 与 calib 场景）：种子 0..n_seeds-1，输入 2/3、留出 1/3
    ab_splits = {}
    for s in scenes:
        sid = s["scene_id"]
        if role_of_scene[sid] not in ("test", "calib"):
            continue
        ab_splits[sid] = {}
        mem = np.array(s["member_idx"])
        for seed in range(n_seeds if role_of_scene[sid] == "test" else 1):
            r = np.random.default_rng(1000 + seed)
            perm = r.permutation(len(mem))
            n_in = max(8, int(round(len(mem) * 2 / 3)))
            ab_splits[sid][str(seed)] = {
                "A": mem[perm[:n_in]].tolist(),
                "B": mem[perm[n_in:]].tolist()}
    # 时空邻接检测：test 场景边界与 train 场景边界的最小间隔（同 t_block 相邻才算）
    adjacency = []
    test_scenes = [s for s in scenes if role_of_scene[s["scene_id"]] == "test"]
    train_scenes = [s for s in scenes if role_of_scene[s["scene_id"]] == "train"]
    for ts in test_scenes:
        for trs in train_scenes:
            if abs(ts["t_block"] - trs["t_block"]) > 1:
                continue
            dx = max(ts["lon0"] - trs["lon1"], trs["lon0"] - ts["lon1"], 0)
            dy = max(ts["lat0"] - trs["lat1"], trs["lat0"] - ts["lat1"], 0)
            if dx < 0.5 and dy < 0.5:  # 块几乎相接（边界距离 0）
                adjacency.append({"test": ts["scene_id"], "train": trs["scene_id"],
                                  "dx_deg": dx, "dy_deg": dy,
                                  "dt_block": abs(ts["t_block"] - trs["t_block"])})
    # 汇总
    scene_df = pd.DataFrame([{k: v for k, v in s.items() if k != "member_idx"}
                             for s in scenes])
    scene_df["role"] = [role_of_scene[s["scene_id"]] for s in scenes]
    scene_df.to_csv(os.path.join(OUT, "scenes.csv"), index=False)
    # 锚定场景：test 角色中联合密度最高（含 CTD 邻近）+ 季节分层代表
    dt_ctd = pd.DatetimeIndex(ctd.t64.astype("datetime64[s]"))
    ctd_bx = np.floor((ctd.lon - 110.0) / deg).astype(int)
    ctd_by = np.floor((ctd.lat - 10.0) / deg).astype(int)
    ctd_tb = (dt_ctd.year.values * 12 + dt_ctd.month.values - 1) // months
    n_ctd_by_scene = {}
    for tb, bxx, byy in zip(ctd_tb, ctd_bx, ctd_by):
        key = tb * 10000 + bxx * 100 + byy
        n_ctd_by_scene[int(key)] = n_ctd_by_scene.get(int(key), 0) + 1
    for s in scenes:
        s["n_ctd_in_block"] = n_ctd_by_scene.get(int(s["key"]), 0)
    # 锚定：test 场景按 (n_ctd_in_block, n_profiles) 排序取 top；每季节一个
    anchored = []
    test_list = [s for s in scenes if role_of_scene[s["scene_id"]] == "test"]
    for season in range(4):
        cand = [s for s in test_list if s["season"] == season and s["n_ctd_in_block"] > 0]
        if not cand:
            cand = [s for s in test_list if s["season"] == season]
        if not cand:
            continue
        cand.sort(key=lambda s: (s["n_ctd_in_block"], s["n_profiles"]), reverse=True)
        anchored.append(cand[0]["scene_id"])
    # 划分哈希冻结
    h = hashlib.sha256()
    h.update(prof_role.tobytes())
    h.update(json.dumps(ab_splits, sort_keys=True).encode())
    h.update(json.dumps([[s["key"], role_of_scene[s["scene_id"]]] for s in scenes],
                        sort_keys=True).encode())
    split_hash = h.hexdigest()[:16]
    meta = {
        "months": months, "deg": deg, "min_profiles": min_profiles,
        "n_scenes": len(scenes),
        "n_train": sum(1 for s in scenes if role_of_scene[s["scene_id"]] == "train"),
        "n_calib": sum(1 for s in scenes if role_of_scene[s["scene_id"]] == "calib"),
        "n_test": sum(1 for s in scenes if role_of_scene[s["scene_id"]] == "test"),
        "profiles_by_role": {r: int((prof_role == r).sum())
                             for r in ("train", "calib", "test", "none")},
        "wmo_cross_role_count": cross, "wmo_cross_examples": cross_pairs[:10],
        "test_train_adjacent_pairs": len(adjacency),
        "adjacency_examples": adjacency[:10],
        "anchored_scene_ids": anchored,
        "n_seeds_ab": n_seeds,
        "split_hash": split_hash,
    }
    with open(os.path.join(OUT, "scenes_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=_jd)
    # 完整划分存 npz（供后续实验）。role 数组按 uk 全单元索引对齐
    # （原实现误把 uk 索引当 scene_id 查 dict，有效场景数 < uk 数时错位，日志【问题12】）
    role_arr = np.full(len(uk), "none", dtype="U8")
    for s in scenes:
        role_arr[s["uk_idx"]] = role_of_scene[s["scene_id"]]
    np.savez_compressed(
        os.path.join(OUT, "scenes_split.npz"),
        scene_key=scene_key, inv=inv, prof_role=prof_role.astype("U8"),
        role_of_scene=role_arr, split_hash=split_hash)
    with open(os.path.join(OUT, "scenes_full.json"), "w", encoding="utf-8") as f:
        json.dump(scenes, f, ensure_ascii=False, indent=2, default=_jd)
    with open(os.path.join(OUT, "ab_splits.json"), "w", encoding="utf-8") as f:
        json.dump(ab_splits, f, ensure_ascii=False, default=_jd)
    return scene_df, meta, scenes, ab_splits


def main():
    t0 = time.time()
    print("== EXP-2: scene library & leakage-proof split ==")
    argo = io_data.load_argo()
    ctd = io_data.load_ctd()
    m, sp = census(argo, ctd)
    print("census top-5 3deg blocks:")
    print(sp.head(5).to_string())
    scan = calibrate_scene_template(argo)
    print(scan.to_string())
    # 模板选择规则：剖面数中位数最接近 18（十余条输入+留出）且有效场景数最多的可行组合
    scan["score"] = -np.abs(scan["profiles_per_scene_median"] - 18) * 1.0 \
        - np.abs(scan["n_scenes_12_40"] - scan["n_scenes_12_40"].max()) * 0.001
    best = scan.sort_values("score", ascending=False).iloc[0]
    months, deg = int(best["months"]), int(best["deg"])
    print(f"chosen template: months={months}, deg={deg}")
    scene_df, meta, scenes, ab = build_scenes(argo, ctd, months, deg)
    meta["elapsed_s"] = time.time() - t0
    meta["template_scan"] = scan.to_dict("records")
    with open(os.path.join(OUT, "scenes_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=_jd)
    print(json.dumps({k: v for k, v in meta.items() if k != "template_scan"},
                     ensure_ascii=False, indent=2))
    print("EXP-2 DONE", time.time() - t0)


if __name__ == "__main__":
    main()
