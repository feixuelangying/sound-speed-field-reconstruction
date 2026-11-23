# -*- coding: utf-8 -*-
"""EXP-4 GP 层设计与在线重构（WP2，framework §3.3.2/§4-EXP4）。
① 超参分型估计（全局 vs 季节分型；train 场景联合 type-II ML，噪声固定 u_obs²）
② 核选型（prod / add / se × 分型粒度；val 场景 A→B 评估 RMSE/NLL/覆盖）
③ 五对照变体：分带块 GP / nugget 诊断 / 在线微调 vs 冻结 / 全点 vs 锚点降采样 /
   σ² 双路交叉验证（GP sill vs EXP-3 随机效应组内方差 + u_m²）
④ 在线端到端流程 + 计时（J6）
⑤ 后验方差行为检验（远离观测 u_r 增大）
产出：results/exp4/*.csv|json
"""
import sys, os, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from exp import config as C
from exp.lib import io_data, features, uobs, meanet, scenes as scenelib
from exp.lib import gp as gplib
from exp.lib.pipeline import SceneData, load_stack, eval_on_B

OUT = C.result_dir("exp4")
EPOCH = np.datetime64("2000-01-01T00:00:00").astype(np.int64)
RNG = np.random.default_rng(C.SEED)
HP_SUBSET = 12       # 每分型组参与超参估计的最大场景数
VAL_SUBSET = 12      # 核选型/变体评估的 val 场景数（每季节最多 3 个）
HP_CACHE = os.path.join(OUT, "_hp_cache.json")   # 超参拟合增量缓存（防崩溃重跑）


def _jd(o):
    """json 兜底：numpy 标量 -> 原生（日志【问题12/20】）。"""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


def load_hp_cache():
    """加载已完成的超参拟合缓存；文件不完整或缺失返回 {}。"""
    if not os.path.exists(HP_CACHE):
        return {}
    try:
        with open(HP_CACHE, encoding="utf-8") as f:
            d = json.load(f)
        return d.get("store", {})
    except Exception:
        return {}


def save_hp_cache(store, complete=False):
    """原子写入超参缓存（临时文件 + replace）。"""
    tmp = HP_CACHE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"_complete": complete, "store": store}, f,
                  ensure_ascii=False, indent=1, default=_jd)
    os.replace(tmp, HP_CACHE)


def ab_split(SD, sid, seed=0):
    return SD.ab(sid, seed)


def eval_gp_on_B(g, dsB, tag):
    """GPFit -> B 点指标（GP 自身口径：var+u_B²）。"""
    pred, var = g.predict(dsB["X4"], dsB["mean"])
    return eval_on_B(pred, var, dsB, tag=tag)


# ------------------------------------------------------------ 主流程 ----
def main():
    t0 = time.time()
    print("== EXP-4: GP layer design & online reconstruction ==")
    stack = load_stack()
    SD = SceneData(stack)
    bg = stack["bg"]
    model, stdr, fcols = stack["model"], stack["stdr"], stack["fcols"]
    scenes = stack["scenes"]

    train_sids = scenelib.scene_ids_by_role(scenes, "train")
    # 复用 EXP-3 的场景级 val 划分（同协议防泄漏：val ⊂ train 角色）
    with open(os.path.join(C.RESULTS, "exp3", "val_scene_ids.json"), encoding="utf-8") as f:
        val_sids = json.load(f)
    tr_only = [s for s in train_sids if s not in set(val_sids)]
    print(f"scenes: train={len(tr_only)}, val={len(val_sids)}")

    # ---------- ① 超参估计（train-only 场景，全局 & 季节分型） ----------
    print("-- hyperparameter fitting (global & per-season) --")
    # 独立 RNG：超参阶段的场景抽样与后续 val_sel/variant 解耦，
    # 使缓存命中与否不改变下游抽样序列（可复现性）
    HP_RNG = np.random.default_rng(C.SEED + 101)
    hp_store = load_hp_cache()
    if hp_store:
        print(f"  loaded {len(hp_store)} cached hp configs from {HP_CACHE}")
    for form in ("prod", "add", "se"):
        for gran in ("global", "season"):
            key = f"{form}_{gran}"
            if key in hp_store:
                print(f"  {key}: cached (mean_lml={hp_store[key]['mean_lml']:.1f})")
                continue
            if gran == "global":
                sids = list(HP_RNG.choice(tr_only, min(HP_SUBSET, len(tr_only)),
                                          replace=False))
                ds = [SD.build(s) for s in sids]
                kern, lmls, info = gplib.fit_hyperparams(
                    [(x["X4"], x["d"], x["u"], x["mean"]) for x in ds], form=form)
                hp_store[key] = {"kern": kern.to_dict(), "mean_lml": float(np.mean(lmls)),
                                 "n_scenes": len(ds), "groups": {"all": kern.to_dict()}}
            else:
                groups = {}
                lml_all = []
                for season in range(4):
                    ss = [s for s in tr_only if SD.sbi[s]["season"] == season]
                    if len(ss) < 2:
                        continue
                    ss = list(HP_RNG.choice(ss, min(HP_SUBSET, len(ss)), replace=False))
                    ds = [SD.build(s) for s in ss]
                    kern, lmls, info = gplib.fit_hyperparams(
                        [(x["X4"], x["d"], x["u"], x["mean"]) for x in ds], form=form)
                    groups[str(season)] = kern.to_dict()
                    lml_all += lmls
                hp_store[key] = {"kern": groups.get("0", {}), "mean_lml": float(np.mean(lml_all)),
                                 "n_scenes": len(lml_all), "groups": groups}
            save_hp_cache(hp_store)          # 每个配置完成后立即落盘
            print(f"  {key}: mean_lml={hp_store[key]['mean_lml']:.1f}")

    # ---------- ② 核选型 + 分型粒度（val 场景 A→B） ----------
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
            A, B = ab_split(SD, sid, seed=0)
            dsA, dsB = SD.build(sid, A), SD.build(sid, B)
            kd = hp["groups"].get(str(dsA["season"])) if gran == "season" \
                else hp["groups"]["all"]
            kern = gplib.kernel_from_dict(kd)
            g = gplib.GPFit(dsA["X4"], dsA["d"], dsA["u"], dsA["mean"], kern)
            metrics.append(eval_gp_on_B(g, dsB, key))
        dfm = pd.DataFrame(metrics)
        kern_rows.append({"config": key, "n_scenes_eval": len(val_sel),
                          **{c: float(dfm[c].mean()) for c in
                             ("rmse", "mae", "nll", "cover95", "mean_sig")}})
        print(f"  {key}: rmse={kern_rows[-1]['rmse']:.4f} nll={kern_rows[-1]['nll']:.3f} "
              f"cov95={kern_rows[-1]['cover95']:.3f}")
    kern_df = pd.DataFrame(kern_rows)
    kern_df.to_csv(os.path.join(OUT, "kernel_selection.csv"), index=False)
    best_key = kern_df.sort_values(["nll", "rmse"]).iloc[0]["config"]
    best_form, best_gran = best_key.split("_")
    best_hp = hp_store[best_key]
    print(f"best config: {best_key}")
    # 保存超参表（冻结）——须保留 "groups" 外层，供 EXP-5~10 经
    # pipeline.kern_for 以 hp[key]["groups"] 读取；原写法只落 v["groups"]，
    # 致 kern_for KeyError: 'groups'（日志【问题23】）
    hp_out = {k: {"kern": v["kern"], "groups": v["groups"],
                  "mean_lml": v["mean_lml"], "n_scenes": v["n_scenes"]}
              for k, v in hp_store.items()}
    hp_out["_meta"] = {"best_key": best_key, "n_hp_subset": HP_SUBSET,
                       "val_sids": [int(s) for s in val_sel],
                       "date": time.strftime("%Y-%m-%d")}
    with open(os.path.join(OUT, "hp_table.json"), "w", encoding="utf-8") as f:
        json.dump(hp_out, f, ensure_ascii=False, indent=1, default=_jd)

    def get_kern(season):
        g = best_hp["groups"]
        return gplib.kernel_from_dict(g[str(season)] if best_gran == "season" else g["all"])

    # ---------- ③ 五对照变体 ----------
    print("-- variant A: banded block GP vs single GP --")
    # 分带超参（train-only 场景按带）
    band_kerns = {}
    for bi in range(6):
        ds_list, sids_used = [], []
        for s in RNG.permutation(tr_only):
            x = SD.build(s)
            m = x["band"] == bi
            if m.sum() >= 15:
                ds_list.append((x["X4"][m], x["d"][m], x["u"][m], x["mean"][m]))
                sids_used.append(s)
            if len(ds_list) >= HP_SUBSET:
                break
        if len(ds_list) < 3:
            band_kerns[bi] = None
            continue
        kern, _, _ = gplib.fit_hyperparams(ds_list, form=best_form)
        band_kerns[bi] = kern
    rows_a = []
    for sid in val_sel:
        A, B = ab_split(SD, sid, seed=0)
        dsA, dsB = SD.build(sid, A), SD.build(sid, B)
        # 单 GP（主方案）
        g = gplib.GPFit(dsA["X4"], dsA["d"], dsA["u"], dsA["mean"],
                        get_kern(dsA["season"]))
        rows_a.append({**eval_gp_on_B(g, dsB, "single"), "sid": sid})
        # 分带块 GP
        preds, vars_ = np.zeros(len(dsB["d"])), np.zeros(len(dsB["d"]))
        for bi in range(6):
            if band_kerns[bi] is None:
                continue
            ma = dsA["band"] == bi
            mb = dsB["band"] == bi
            if ma.sum() < 5 or not mb.any():
                continue
            gb = gplib.GPFit(dsA["X4"][ma], dsA["d"][ma], dsA["u"][ma],
                             dsA["mean"][ma], band_kerns[bi])
            p, v = gb.predict(dsB["X4"][mb], dsB["mean"][mb])
            preds[mb], vars_[mb] = p, v
        ok = vars_ > 0
        sig2 = vars_[ok] + dsB["u"][ok] ** 2
        err = preds[ok] - dsB["d"][ok]
        rows_a.append({"tag": "banded", "sid": sid, "n": int(ok.sum()),
                       "rmse": float(np.sqrt(np.mean(err ** 2))),
                       "mae": float(np.mean(np.abs(err))),
                       "nll": float(np.mean(0.5 * (err ** 2 / sig2
                                                   + np.log(2 * np.pi * sig2)))),
                       "cover95": float(np.mean(np.abs(err) <= 1.96 * np.sqrt(sig2))),
                       "mean_sig": float(np.mean(np.sqrt(sig2)))})
    dfA = pd.DataFrame(rows_a)
    dfA_agg = dfA.groupby("tag")[["rmse", "mae", "nll", "cover95", "mean_sig"]].mean()
    dfA_agg.to_csv(os.path.join(OUT, "variant_banded.csv"))
    dfA.to_csv(os.path.join(OUT, "variant_banded_by_scene.csv"), index=False)
    print(dfA_agg.to_string())

    print("-- variant B: nugget diagnostic --")
    sids = list(RNG.choice(tr_only, min(HP_SUBSET, len(tr_only)), replace=False))
    ds = [SD.build(s) for s in sids]
    kern_n, _, info_n = gplib.fit_hyperparams(
        [(x["X4"], x["d"], x["u"], x["mean"]) for x in ds], form=best_form,
        use_nugget=True)
    u2_med = float(np.median(np.concatenate([x["u"] for x in ds]) ** 2))
    s2 = kern_n.diag_var()
    nug = float(np.exp(info_n["log_nug"])) if "log_nug" in info_n else 0.0
    nug_diag = {"fitted_nugget_var": nug, "median_u_obs_sq": u2_med,
                "ratio_nugget_over_uobs2": nug / u2_med if u2_med > 0 else None,
                "interpretation": "ratio>1: u_obs 评定不足或存在未建模小尺度噪声; "
                                  "ratio<1: u_obs 偏保守"}
    with open(os.path.join(OUT, "variant_nugget.json"), "w", encoding="utf-8") as f:
        json.dump(nug_diag, f, ensure_ascii=False, indent=1, default=_jd)
    print(f"  {nug_diag}")

    print("-- variant C: online refit vs frozen --")
    rows_c = []
    for sid in val_sel:
        A, B = ab_split(SD, sid, seed=0)
        dsA, dsB = SD.build(sid, A), SD.build(sid, B)
        kern0 = get_kern(dsA["season"])
        g0 = gplib.GPFit(dsA["X4"], dsA["d"], dsA["u"], dsA["mean"], kern0)
        rows_c.append({**eval_gp_on_B(g0, dsB, "frozen"), "sid": sid})
        kern1, _, _ = gplib.fit_hyperparams(
            [(dsA["X4"], dsA["d"], dsA["u"], dsA["mean"])], form=best_form,
            init={k: v for k, v in kern0.raw.items()}, n_iter=40, lr=0.05)
        g1 = gplib.GPFit(dsA["X4"], dsA["d"], dsA["u"], dsA["mean"], kern1)
        rows_c.append({**eval_gp_on_B(g1, dsB, "online_refit"), "sid": sid})
    dfC = pd.DataFrame(rows_c)
    dfC.groupby("tag")[["rmse", "nll", "cover95"]].mean().to_csv(
        os.path.join(OUT, "variant_online_refit.csv"))
    dfC.to_csv(os.path.join(OUT, "variant_online_refit_by_scene.csv"), index=False)
    print(dfC.groupby("tag")[["rmse", "nll", "cover95"]].mean().to_string())

    print("-- variant D: full-point GP vs anchor downsample --")
    rows_d = []
    for sid in val_sel[:6]:
        A, B = ab_split(SD, sid, seed=0)
        dsA = SD.build(sid, A)
        dsB = SD.build(sid, B)
        g0 = gplib.GPFit(dsA["X4"], dsA["d"], dsA["u"], dsA["mean"],
                         get_kern(dsA["season"]))
        rows_d.append({**eval_gp_on_B(g0, dsB, "anchor"), "sid": sid,
                       "n_fit": len(dsA["d"])})
        dsAf = SD.build(sid, A, full=True)
        gf = gplib.GPFit(dsAf["X4"], dsAf["d"], dsAf["u"], dsAf["mean"],
                         get_kern(dsAf["season"]))
        rows_d.append({**eval_gp_on_B(gf, dsB, "full"), "sid": sid,
                       "n_fit": len(dsAf["d"])})
    dfD = pd.DataFrame(rows_d)
    dfD.groupby("tag")[["rmse", "nll", "cover95", "mean_sig"]].mean().to_csv(
        os.path.join(OUT, "variant_fullpoint.csv"))
    dfD.to_csv(os.path.join(OUT, "variant_fullpoint_by_scene.csv"), index=False)
    print(dfD.groupby("tag")[["rmse", "nll", "cover95"]].mean().to_string())

    print("-- variant E: sigma2 dual-path cross-check --")
    um = pd.read_csv(os.path.join(C.RESULTS, "exp3", "um_indist.csv"))
    rows_e = []
    for season in range(4):
        g = best_hp["groups"].get(str(season)) or best_hp["groups"]["all"]
        kern = gplib.kernel_from_dict(g)
        s2_gp = kern.diag_var() if best_form != "add" else None
        if s2_gp is None:
            p = kern.raw
            s2_gp = np.exp(p["log_s2h"]) + np.exp(p["log_s2z"]) \
                    + np.exp(p["log_s2t"]) + np.exp(p["log_s2c"])
        sub = um[um["season"] == season]
        s2_re = float(((sub["sigma2_within"] + sub["um"] ** 2) * sub["n_pts"]).sum()
                      / sub["n_pts"].sum()) if len(sub) else np.nan
        rows_e.append({"season": season, "sigma2_gp_marginal": float(s2_gp),
                       "sigma2_RE_plus_um2": s2_re,
                       "ratio_gp_over_RE": float(s2_gp / s2_re) if s2_re and s2_re > 0 else None})
    dfE = pd.DataFrame(rows_e)
    dfE.to_csv(os.path.join(OUT, "sigma2_dualpath.csv"), index=False)
    print(dfE.to_string(index=False))

    # ---------- ④ 方差行为检验 + ⑤ 端到端计时 ----------
    print("-- variance behavior & timing --")
    beh_rows = []
    timing = []
    for sid in val_sel[:4]:
        s = SD.sbi[int(sid)]
        A, _ = ab_split(SD, sid, seed=0)
        dsA = SD.build(sid, A)
        # 输出网格：场景范围 0.5° × OUT_Z_GRID × 场景中时
        lons = np.arange(s["lon0"] + 0.25, s["lon1"], 0.5)
        lats = np.arange(s["lat0"] + 0.25, s["lat1"], 0.5)
        t_mid = float(dsA["X4"][:, 3].mean())
        GL, GLAT, GZ, GT = np.meshgrid(lons, lats, C.OUT_Z_GRID, [t_mid], indexing="ij")
        Xq = np.column_stack([GL.ravel(), GLAT.ravel(), GZ.ravel(), GT.ravel()])
        # 场景背景值 + m_theta
        t_q = Xq[:, 3]
        Xq_feat, _ = features.make_features(bg, t_q.astype(np.int64), Xq[:, 2],
                                            Xq[:, 1], Xq[:, 0])
        mq = meanet.predict(model, stdr, Xq_feat, fcols)
        t_start = time.time()
        g = gplib.GPFit(dsA["X4"], dsA["d"], dsA["u"], dsA["mean"],
                        get_kern(dsA["season"]))
        t_fit = time.time() - t_start
        t_start = time.time()
        rq, var_q = g.predict(Xq, mq)
        t_pred = time.time() - t_start
        timing.append({"sid": sid, "n_fit": len(dsA["d"]), "n_out": len(Xq),
                       "t_fit_s": t_fit, "t_pred_s": t_pred,
                       "t_total_s": t_fit + t_pred})
        # 行为：到最近 A 点的水平距离 / 时间距离（分块计算）
        kmx = 111.32 * np.cos(np.radians(Xq[:, 1].mean()))
        dh = np.empty(len(Xq))
        for s0 in range(0, len(Xq), 2000):
            blk = Xq[s0:s0 + 2000]
            dd = np.sqrt(((blk[:, None, 0] - dsA["X4"][None, :, 0]) * kmx) ** 2
                         + ((blk[:, None, 1] - dsA["X4"][None, :, 1]) * 110.57) ** 2)
            dh[s0:s0 + 2000] = dd.min(axis=1)
        dt_days = np.abs(Xq[:, 3][:, None] - dsA["X4"][:, 3][None, :]).min(axis=1) / 86400.0
        bins_d = np.array([0, 10, 25, 50, 100, 200, 400, 1e9])
        for k in range(7):
            m = (dh >= bins_d[k]) & (dh < bins_d[k + 1])
            if m.sum() > 10:
                beh_rows.append({"sid": sid, "kind": "dist_h_km",
                                 "bin": f"{bins_d[k]:.0f}-{bins_d[k+1]:.0f}",
                                 "n": int(m.sum()), "ur_mean": float(np.sqrt(var_q[m]).mean())})
        bins_t = np.array([0, 0.5, 1, 2, 5, 10, 20, 1e9])
        for k in range(7):
            m = (dt_days >= bins_t[k]) & (dt_days < bins_t[k + 1])
            if m.sum() > 10:
                beh_rows.append({"sid": sid, "kind": "dist_t_day",
                                 "bin": f"{bins_t[k]:.0f}-{bins_t[k+1]:.0f}",
                                 "n": int(m.sum()), "ur_mean": float(np.sqrt(var_q[m]).mean())})
    pd.DataFrame(beh_rows).to_csv(os.path.join(OUT, "var_behavior.csv"), index=False)
    dfT = pd.DataFrame(timing)
    dfT.to_csv(os.path.join(OUT, "timing.csv"), index=False)
    print(dfT.to_string(index=False))
    print(pd.DataFrame(beh_rows).groupby(["kind", "bin"])["ur_mean"].mean().to_string())

    # 汇总
    summary = {
        "exp": "exp4", "date": time.strftime("%Y-%m-%d %H:%M"),
        "kernel_selection": kern_df.to_dict("records"),
        "best_config": best_key,
        "hp_table_file": Path(OUT, "hp_table.json").relative_to(C.ROOT).as_posix(),
        "variant_banded": dfA_agg.reset_index().to_dict("records"),
        "variant_nugget": nug_diag,
        "variant_online": dfC.groupby("tag")[["rmse", "nll", "cover95"]]
            .mean().reset_index().to_dict("records"),
        "variant_fullpoint": dfD.groupby("tag")[["rmse", "nll", "cover95"]]
            .mean().reset_index().to_dict("records"),
        "sigma2_dualpath": dfE.to_dict("records"),
        "timing": dfT.to_dict("records"),
        "j6_ok": bool((dfT["t_total_s"] < 300).all()) if len(dfT) else None,
        "elapsed_s": time.time() - t0,
    }
    with open(os.path.join(OUT, "exp4_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_jd)
    print("EXP-4 DONE", time.time() - t0)


if __name__ == "__main__":
    main()
