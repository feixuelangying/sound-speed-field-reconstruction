# -*- coding: utf-8 -*-
"""EXP-11：σ² 矩估计锚定重拟合（修复 type-II ML 的 σ² 发散 → u_r 高估）。

诊断结论（exp/diag_sigma2.py，logs/diag_sigma2.txt）：
  prod_global 拟合 σ²=10.122，而训练残差数据方差仅 2.484；-ΣlogML 随 σ² 单调
  改善（σ²=2.48→−88814，8→−44347，10.12→−40310，20→−33928，40→−32611）——
  似然无上界。逐场景单独 ML 亦系统性给出 3~8 倍（5.55~11.78 vs var 1.17~2.84）。
  机制：u_obs² 近零（median 3.5e-4）+ M52 短程负相关核 → GP 可无限"吸收"数据
  振荡以提高似然（过拟合），σ² 无约束发散。
  后果：u_r² 高估约 2.8 倍 → u_c 保守 1.6~1.8 倍 → 共形 k=1.3485（远离 1.96）。
  注：场景间差异（between）仅占 pooled 方差 6%，非主因（原假设被否）。

修复：σ² 用矩估计锚定（= 跨场景池化残差方差，框架 §3.3.4⑥ 语义
  σ̂²_GP ≈ σ̂²_随机效应 + û_m²），仅用 ML 优化 ℓ。对应 gp.fit_hyperparams
  的 s2_mode="moment"。

本脚本：
① 复现 EXP-4 的 HP 抽样协议，对 prod/se × global/season 四配置重拟合（moment）
② val 场景核选型（同 EXP-4 协议），确认 best_key
③ 分带核重拟合（moment，协议同 exp6c 的 bl.fit_band_kerns）
④ v1/v2 核参数对比
产出：results/exp4b/{hp_table.json, band_kerns.json, kernel_selection.csv,
      v1_v2_kernel_compare.csv, exp11_summary.json}
"""
import sys, os, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from exp import config as C
from exp.lib import scenes as scenelib
from exp.lib import gp as gplib
from exp.lib import banded as bl
from exp.lib.pipeline import SceneData, load_stack, eval_on_B

OUT = C.result_dir("exp4b")
RNG = np.random.default_rng(C.SEED)
HP_SUBSET = 12
S2_MODE = os.environ.get("S2_MODE", "moment")
# add 核在 moment 模式下回退 ml（4 分量 σ² 结构不适用），结果与 v1 相同 →
# 直接复用 v1 条目，避免无谓重算（省约 1/3 时间）
REUSE_V1_ADD = True


def _jd(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


def main():
    t0 = time.time()
    print(f"== EXP-11: sigma2 moment-anchored refit (s2_mode={S2_MODE}) ==")
    stack = load_stack()
    SD = SceneData(stack)
    scenes = stack["scenes"]
    train_sids = scenelib.scene_ids_by_role(scenes, "train")
    with open(os.path.join(C.RESULTS, "exp3", "val_scene_ids.json"), encoding="utf-8") as f:
        val_sids = json.load(f)
    tr_only = [s for s in train_sids if s not in set(val_sids)]
    print(f"scenes: train={len(tr_only)}, val={len(val_sids)}")

    # 场景数据缓存（EXP-4 每次 SD.build 重复构建 → 本脚本复用，显著提速）
    _cache = {}

    def build(s):
        s = int(s)
        if s not in _cache:
            _cache[s] = SD.build(s)
        return _cache[s]

    with open(os.path.join(C.RESULTS, "exp4", "hp_table.json"), encoding="utf-8") as f:
        v1 = json.load(f)

    HP_RNG = np.random.default_rng(C.SEED + 101)   # 与 EXP-4 同序抽样
    hp_store = {}
    print("-- refitting (moment-anchored sigma2, ML for lengths) --")
    for form in ("prod", "add", "se"):
        for gran in ("global", "season"):
            key = f"{form}_{gran}"
            if REUSE_V1_ADD and form == "add":
                hp_store[key] = {"kern": v1[key]["kern"], "groups": v1[key]["groups"],
                                 "mean_lml": v1[key]["mean_lml"],
                                 "n_scenes": v1[key]["n_scenes"]}
                print(f"  {key}: reused from v1 (add 回退 ml)", flush=True)
                continue
            if gran == "global":
                sids = list(HP_RNG.choice(tr_only, min(HP_SUBSET, len(tr_only)),
                                          replace=False))
                ds = [build(s) for s in sids]
                kern, lmls, info = gplib.fit_hyperparams(
                    [(x["X4"], x["d"], x["u"], x["mean"]) for x in ds],
                    form=form, s2_mode=S2_MODE)
                hp_store[key] = {"kern": kern.to_dict(), "mean_lml": float(np.mean(lmls)),
                                 "n_scenes": len(ds), "groups": {"all": kern.to_dict()}}
            else:
                groups, lml_all = {}, []
                for season in range(4):
                    ss = [s for s in tr_only if SD.sbi[s]["season"] == season]
                    if len(ss) < 2:
                        continue
                    ss = list(HP_RNG.choice(ss, min(HP_SUBSET, len(ss)), replace=False))
                    ds = [build(s) for s in ss]
                    kern, lmls, info = gplib.fit_hyperparams(
                        [(x["X4"], x["d"], x["u"], x["mean"]) for x in ds],
                        form=form, s2_mode=S2_MODE)
                    groups[str(season)] = kern.to_dict()
                    lml_all += lmls
                hp_store[key] = {"kern": groups.get("0", {}),
                                 "mean_lml": float(np.mean(lml_all)),
                                 "n_scenes": len(lml_all), "groups": groups}
            kd = hp_store[key]["kern"]
            if "log_s2" in kd:
                print(f"  {key}: mean_lml={hp_store[key]['mean_lml']:.1f} "
                      f"s2={np.exp(kd['log_s2']):.3f} lh={np.exp(kd['log_lh']):.2f}km "
                      f"lz={np.exp(kd['log_lz']):.2f}m lt={np.exp(kd['log_lt']):.2f}d",
                      flush=True)
            else:
                print(f"  {key}: mean_lml={hp_store[key]['mean_lml']:.1f} (add)", flush=True)

    # ---------- ② val 核选型（协议同 EXP-4） ----------
    print("-- kernel & granularity selection on val scenes --")
    val_sel = []
    for season in range(4):
        ss = [s for s in val_sids if SD.sbi[s]["season"] == season]
        val_sel += list(RNG.choice(ss, min(3, len(ss)), replace=False)) if ss else []
    kern_rows = []
    for key, hp in hp_store.items():
        form, gran = key.split("_")
        metrics = []
        for sid in val_sel:
            A, B = SD.ab(sid, seed=0)
            dsA, dsB = SD.build(sid, A), SD.build(sid, B)
            kd = hp["groups"].get(str(dsA["season"])) if gran == "season" \
                else hp["groups"]["all"]
            kern = gplib.kernel_from_dict(kd)
            g = gplib.GPFit(dsA["X4"], dsA["d"], dsA["u"], dsA["mean"], kern)
            pred, var = g.predict(dsB["X4"], dsB["mean"])
            metrics.append(eval_on_B(pred, var, dsB, tag=key))
        dfm = pd.DataFrame(metrics)
        kern_rows.append({"config": key, "n_scenes_eval": len(val_sel),
                          **{c: float(dfm[c].mean()) for c in
                             ("rmse", "mae", "nll", "cover95", "mean_sig")}})
        print(f"  {key}: rmse={kern_rows[-1]['rmse']:.4f} nll={kern_rows[-1]['nll']:.3f} "
              f"cov95={kern_rows[-1]['cover95']:.3f} mean_sig={kern_rows[-1]['mean_sig']:.3f}")
    kern_df = pd.DataFrame(kern_rows)
    kern_df.to_csv(os.path.join(OUT, "kernel_selection.csv"), index=False)
    best_key = kern_df.sort_values(["nll", "rmse"]).iloc[0]["config"]
    print(f"best config (v2): {best_key}")

    hp_out = {k: {"kern": v["kern"], "groups": v["groups"], "mean_lml": v["mean_lml"],
                  "n_scenes": v["n_scenes"]} for k, v in hp_store.items()}
    hp_out["_meta"] = {"best_key": best_key, "n_hp_subset": HP_SUBSET,
                       "val_sids": [int(s) for s in val_sel],
                       "s2_mode": S2_MODE, "date": time.strftime("%Y-%m-%d"),
                       "note": "σ² 矩估计锚定（池化残差方差），仅 ℓ 由 ML 估计"}
    with open(os.path.join(OUT, "hp_table.json"), "w", encoding="utf-8") as f:
        json.dump(hp_out, f, ensure_ascii=False, indent=1, default=_jd)
    print("saved hp_table.json ->", OUT)

    # ---------- ③ 分带核（协议同 exp6c：bl.fit_band_kerns + train_sids + seed=101） ----------
    print("-- refitting per-band kernels (moment) --")
    bk = bl.fit_band_kerns(SD, train_sids, s2_mode=S2_MODE)
    bl.save_band_kerns(bk, os.path.join(OUT, "band_kerns.json"))
    print("saved band_kerns.json ->", OUT)

    # ---------- ④ v1/v2 核参数对比 ----------
    cmp_rows = []
    for key in hp_store:
        k2 = hp_store[key]["kern"]
        k1 = v1.get(key, {}).get("kern", {})
        for dim in ("log_s2", "log_lh", "log_lz", "log_lt"):
            if dim in k2 and dim in k1:
                cmp_rows.append({"config": key, "param": dim,
                                 "v1": float(np.exp(k1[dim])),
                                 "v2": float(np.exp(k2[dim])),
                                 "ratio_v2_v1": float(np.exp(k2[dim]) / np.exp(k1[dim]))})
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(os.path.join(OUT, "v1_v2_kernel_compare.csv"), index=False)
    print("\n== v1/v2 核参数对比 ==")
    print(cmp_df.round(3).to_string(index=False))

    # 分带对比
    with open(os.path.join(C.RESULTS, "exp4", "band_kerns.json"), encoding="utf-8") as f:
        bk1 = json.load(f)
    brows = []
    for bi in range(6):
        k2, k1 = bk.get(bi), bk1.get(str(bi))
        if k1 is None:
            continue
        row = {"band": bi, "v1_s2": float(np.exp(k1["log_s2"]))}
        if k2 is not None:
            row["v2_s2"] = float(np.exp(k2.raw["log_s2"]))
            row["ratio_s2"] = row["v2_s2"] / row["v1_s2"]
            row["v1_lz"] = float(np.exp(k1["log_lz"]))
            row["v2_lz"] = float(np.exp(k2.raw["log_lz"]))
            row["v1_lh"] = float(np.exp(k1["log_lh"]))
            row["v2_lh"] = float(np.exp(k2.raw["log_lh"]))
        brows.append(row)
    bdf = pd.DataFrame(brows)
    bdf.to_csv(os.path.join(OUT, "v1_v2_band_kernels.csv"), index=False)
    print("\n== v1/v2 分带核对比 ==")
    print(bdf.round(3).to_string(index=False))

    summary = {"exp": "exp11", "date": time.strftime("%Y-%m-%d %H:%M"),
               "s2_mode": S2_MODE, "best_key_v2": best_key,
               "best_key_v1": v1.get("_meta", {}).get("best_key"),
               "kernel_selection_v2": kern_df.to_dict("records"),
               "v1_v2_kernel": cmp_rows,
               "v1_v2_band": brows,
               "elapsed_s": time.time() - t0}
    with open(os.path.join(OUT, "exp11_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_jd)
    print("\nEXP-11 DONE", time.time() - t0)


if __name__ == "__main__":
    main()
