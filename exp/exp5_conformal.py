# -*- coding: utf-8 -*-
"""EXP-5 不确定度合成与共形校准（WP3，framework §4-EXP5）。
① u_c 合成（u_r 内生 + u_m/u_eq/u_rep 外加，lib/uc.py）在 B 点逐点实现
② split-conformal：s_j=|err|/u_c；全局 k 与分层 k（band/season/sparse 候选维度）
   方案选择在 val 场景 leave-one-scene-out 条件覆盖（防自评乐观偏差）
③ 最终 k 在 calib 场景定标（冻结 k_global.json）
④ 校准集规模敏感性（calib 子采样 25/50/75/100% × 5 重复）
⑤ 合成层一致性自检（分量占比、观测邻近点 u_r 收缩行为）
⑥ k 诊断联动（k>1 / k<1 -> 预算诊断）
产出：results/exp5/k_global.json + *.csv + exp5_summary.json
"""
import sys, os, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from exp import config as C
from exp.lib import scenes as scenelib
from exp.lib.pipeline import SceneData, load_stack
from exp.lib.uc import UcComposer

OUT = C.result_dir("exp5")
RNG = np.random.default_rng(C.SEED)
MIN_CELL = 200      # 分层格子最低样本量（不足回退全局 k）
IMPROVE_THR = 0.30  # 分层方案相对全局的条件覆盖偏差改善阈值（低于则回退全局）


def conformal_quantile(s, q=0.95):
    n = len(s)
    r = int(np.ceil((n + 1) * q))
    r = min(max(r, 1), n)
    return float(np.sort(s)[r - 1])


def scene_sj(SD, uc, sid, seed=0):
    """场景 A->B 的 s_j 明细（含分层维度标签）。"""
    A, B = SD.ab(sid, seed)
    dsA, dsB = SD.build(sid, A), SD.build(sid, B)
    g = SD.fit_gp(dsA)
    pred, var = g.predict(dsB["X4"], dsB["mean"])
    z, season = dsB["z"], dsB["season"]
    u_extra = uc.extra_std(z, season)
    u_c = np.sqrt(var + u_extra ** 2)
    err = pred - dsB["d"]
    dh, _ = SD.dist_to_inputs(dsB["X4"], dsA["X4"])
    band = np.clip((z < 0) * 0 + np.searchsorted([30, 60, 125, 350, 800, 2000], z), 0, 5)
    return pd.DataFrame({
        "sid": sid, "s": np.abs(err) / u_c, "u_c": u_c, "u_r": np.sqrt(var),
        "u_extra": u_extra, "err": err, "band": band, "season": season,
        "sparse_km": dh, "z": z})


def k_by_scheme(df, scheme, k_global_ref):
    """按方案计算 k（格子样本不足回退全局）。返回逐点 k 数组。"""
    if scheme == "global":
        k = conformal_quantile(df["s"].values)
        return np.full(len(df), k), {"global": k}
    keys, table = None, {}
    if scheme == "band":
        keys = df["band"].values
    elif scheme == "season":
        keys = df["season"].values
    elif scheme == "sparse":
        keys = np.digitize(df["sparse_km"].values, [50.0, 200.0])
    k_arr = np.full(len(df), k_global_ref)
    for kk in np.unique(keys):
        m = keys == kk
        if m.sum() >= MIN_CELL:
            table[str(int(kk))] = conformal_quantile(df["s"].values[m])
            k_arr[m] = table[str(int(kk))]
        else:
            table[str(int(kk))] = k_global_ref
    return k_arr, table


def cond_cov_dev(df, k_arr, scheme):
    """方案维度的条件覆盖偏差 RMS（cell 覆盖率 - 0.95）。"""
    cover = np.abs(df["err"].values) <= k_arr * df["u_c"].values
    if scheme == "global":
        return float(np.sqrt((cover.mean() - 0.95) ** 2))
    if scheme == "band":
        keys = df["band"].values
    elif scheme == "season":
        keys = df["season"].values
    else:
        keys = np.digitize(df["sparse_km"].values, [50.0, 200.0])
    devs = []
    for kk in np.unique(keys):
        m = keys == kk
        if m.sum() >= 50:
            devs.append(cover[m].mean() - 0.95)
    return float(np.sqrt(np.mean(np.square(devs)))) if devs else 0.0


def main():
    t0 = time.time()
    print("== EXP-5: u_c composition & conformal calibration ==")
    stack = load_stack()
    SD = SceneData(stack)
    scenes = stack["scenes"]
    uc = UcComposer()
    with open(os.path.join(C.RESULTS, "exp3", "val_scene_ids.json"), encoding="utf-8") as f:
        val_sids = json.load(f)
    calib_sids = scenelib.scene_ids_by_role(scenes, "calib")
    print(f"scenes: val={len(val_sids)}, calib={len(calib_sids)}")

    # ① 场景 s_j 明细（val 与 calib，seed 0）
    print("-- conformal scores on val & calib scenes --")
    dfs = []
    for tag, sids in (("val", val_sids), ("calib", calib_sids)):
        for sid in sids:
            df = scene_sj(SD, uc, sid, seed=0)
            df["role"] = tag
            dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    all_df.to_csv(os.path.join(OUT, "conformal_scores_detail.csv"), index=False)
    val_df = all_df[all_df["role"] == "val"]
    calib_df = all_df[all_df["role"] == "calib"]
    print(f"scores: val n={len(val_df)} ({val_df['sid'].nunique()} scenes), "
          f"calib n={len(calib_df)} ({calib_df['sid'].nunique()} scenes)")

    # ② 分层方案选择（val 场景 LOSO）
    print("-- stratified k vs global k (val, leave-one-scene-out) --")
    loso_rows = []
    for scheme in ("global", "band", "season", "sparse"):
        devs = []
        for sid in val_df["sid"].unique():
            df_in = val_df[val_df["sid"] != sid]
            df_out = val_df[val_df["sid"] == sid]
            if len(df_in) < 100:
                continue
            k_ref = conformal_quantile(df_in["s"].values)
            k_arr, _ = k_by_scheme(df_in, scheme, k_ref)
            # k_arr 为定标集逐点 k；应用等价格子规则到外样本
            if scheme == "global":
                k_out = np.full(len(df_out), k_ref)
            else:
                _, table = k_by_scheme(df_in, scheme, k_ref)
                if scheme == "band":
                    kk = df_out["band"].values.astype(int).astype(str)
                elif scheme == "season":
                    kk = df_out["season"].values.astype(int).astype(str)
                else:
                    kk = np.digitize(df_out["sparse_km"].values,
                                     [50.0, 200.0]).astype(int).astype(str)
                k_out = np.array([table.get(x, k_ref) for x in kk])
            devs.append(cond_cov_dev(df_out, k_out, scheme))
        loso_rows.append({"scheme": scheme, "cond_cov_dev_rms": float(np.mean(devs)),
                          "n_loso": len(devs)})
        print(f"  {scheme}: dev_rms={loso_rows[-1]['cond_cov_dev_rms']:.4f}")
    loso_df = pd.DataFrame(loso_rows)
    loso_df.to_csv(os.path.join(OUT, "k_scheme_loso.csv"), index=False)
    dev_g = loso_df[loso_df["scheme"] == "global"]["cond_cov_dev_rms"].iloc[0]
    best_row = loso_df[loso_df["scheme"] != "global"].sort_values(
        "cond_cov_dev_rms").iloc[0]
    improve = 1 - best_row["cond_cov_dev_rms"] / dev_g if dev_g > 0 else 0
    scheme = best_row["scheme"] if improve >= IMPROVE_THR else "global"
    print(f"chosen scheme: {scheme} (best stratified improve={improve:.0%}, "
          f"threshold {IMPROVE_THR:.0%})")

    # ③ 最终 k 定标（calib 全体，seed 0 split）
    print("-- final k calibration on calib scenes --")
    k_global = conformal_quantile(calib_df["s"].values)
    k_arr, k_table = k_by_scheme(calib_df, scheme, k_global)
    kcfg = {"scheme": scheme, "k_global": k_global, "k_table": k_table,
            "n_calib_pts": int(len(calib_df)),
            "n_calib_scenes": int(calib_df["sid"].nunique()),
            "s_quantiles": {str(q): float(np.quantile(calib_df["s"].values, q))
                            for q in (0.5, 0.75, 0.9, 0.95, 0.99)},
            "improve_of_best_stratified": float(improve),
            "date": time.strftime("%Y-%m-%d")}
    with open(os.path.join(OUT, "k_global.json"), "w", encoding="utf-8") as f:
        json.dump(kcfg, f, ensure_ascii=False, indent=1)
    print(f"  k_global={k_global:.3f}, scheme={scheme}, table={k_table}")

    # ④ 规模敏感性
    print("-- calibration size sensitivity --")
    size_rows = []
    csids = calib_df["sid"].unique()
    for frac in (0.25, 0.50, 0.75, 1.00):
        ks = []
        for rep in range(5):
            n = max(1, int(round(len(csids) * frac)))
            sel = RNG.choice(csids, n, replace=False)
            sub = calib_df[calib_df["sid"].isin(sel)]
            ks.append(conformal_quantile(sub["s"].values))
        size_rows.append({"frac_scenes": frac, "n_scenes": int(round(len(csids) * frac)),
                          "k_mean": float(np.mean(ks)), "k_std": float(np.std(ks)),
                          "k_min": float(np.min(ks)), "k_max": float(np.max(ks))})
    size_df = pd.DataFrame(size_rows)
    size_df.to_csv(os.path.join(OUT, "calib_size_sensitivity.csv"), index=False)
    print(size_df.to_string(index=False))

    # ⑤ 合成层一致性自检
    print("-- u_c decomposition self-check --")
    dec_rows = []
    cb = calib_df.copy()
    dec_rows.append({
        "check": "component_share_of_uc2",
        "ur_share": float((cb["u_r"] ** 2).sum() / (cb["u_c"] ** 2).sum()),
        "uextra_share": float((cb["u_extra"] ** 2).sum() / (cb["u_c"] ** 2).sum()),
        "note": "u_r 为 GP 内生；u_extra=sqrt(um²+ueq²+urep²)"})
    q10 = cb["sparse_km"].quantile(0.1)
    near = cb[cb["sparse_km"] <= q10]
    dec_rows.append({
        "check": "near_obs_ur_share",
        "ur_share": float((near["u_r"] ** 2).sum() / (near["u_c"] ** 2).sum()),
        "uextra_share": float((near["u_extra"] ** 2).sum() / (near["u_c"] ** 2).sum()),
        "note": f"距输入最近 10% B 点（dh<={q10:.0f}km）：u_r 占比应显著低于全体"})
    dec_df = pd.DataFrame(dec_rows)
    dec_df.to_csv(os.path.join(OUT, "uc_decomposition.csv"), index=False)
    print(dec_df.to_string(index=False))
    # 分带 u_c 分解（供报告）
    band_rows = []
    for bi in range(6):
        m = cb["band"] == bi
        if m.sum() < 20:
            continue
        band_rows.append({"band": bi, "n": int(m.sum()),
                          "u_c_mean": float(cb["u_c"][m].mean()),
                          "u_r_mean": float(cb["u_r"][m].mean()),
                          "u_extra_mean": float(cb["u_extra"][m].mean())})
    pd.DataFrame(band_rows).to_csv(os.path.join(OUT, "uc_by_band.csv"), index=False)

    # ⑥ k 诊断联动。基准 K0=1.96：u_c 是标准合成不确定度（GUM 的 u_c），
    # 其 95% 正态分位为 1.96，共形 k 显著偏离才提示预算问题。原判据以 k≈1
    # 为基准（隐含把 u_c 当扩展不确定度），与 uc.py 的定义不符，会把"偏保守"
    # 误报为"评定不足"——日志【问题24】。
    K0 = 1.96
    cons = float(calib_df["err"].std() / calib_df["u_c"].mean())
    if k_global > K0 * 1.12:
        diag = ("k>2.2: u_c 偏小（评定不足）——复查 u_m（OOD 档）、u_rep 支撑尺度、"
                "u_obs 分型；或残差存在 GP 核未覆盖的小尺度结构")
    elif k_global < K0 * 0.87:
        diag = ("k<1.7: u_c 偏保守（区间过宽）——复查 GP σ²/sill 是否偏大、"
                "核相关长度 ℓ 是否偏小、u_m 卡方上界口径")
    else:
        diag = ("k∈[1.7,2.2]: 与正态 95% 分位 1.96 一致，预算自洽，"
                "共形因子仅作小幅修正")
    diag += (f"｜u_c 保守度 std(err)/mean(u_c)={cons:.3f}"
             "（≈1 自洽；<1 偏保守；>1 偏乐观）")
    print(f"  k diagnosis: {diag}")

    summary = {
        "exp": "exp5", "date": time.strftime("%Y-%m-%d %H:%M"),
        "k_reference_normal95": K0,
        "uc_conservatism": {
            "std_err_over_mean_uc_calib": cons,
            "mean_uc_calib": float(calib_df["u_c"].mean()),
            "std_err_calib": float(calib_df["err"].std()),
            "note": "u_c 为标准合成不确定度时该比值应≈1；<1 表示 u_c 偏保守"},
        "k_scheme_loso": loso_df.to_dict("records"),
        "chosen_scheme": scheme, "improve_threshold": IMPROVE_THR,
        "k_global": k_global, "k_table": k_table,
        "s_quantiles": kcfg["s_quantiles"],
        "calib_size": size_df.to_dict("records"),
        "uc_decomposition": dec_df.to_dict("records"),
        "uc_by_band": band_rows,
        "k_diagnosis": diag,
        "ueq_detail": uc.ueq_detail,
        "elapsed_s": time.time() - t0,
    }
    with open(os.path.join(OUT, "exp5_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    print("EXP-5 DONE", time.time() - t0)


if __name__ == "__main__":
    main()
