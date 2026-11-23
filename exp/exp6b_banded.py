# -*- coding: utf-8 -*-
"""EXP-6b 验证：banded block GP 能否修复 u_r 的深度结构（J2 覆盖率）。

背景（EXP-6 发现）：test 场景加权覆盖率 0.936 < 容差带下界 0.949；根因是
GP 后验方差 u_r 几乎不随深度变化（2.50–2.74 m/s），而实际 rmse 从 0.81（深层
band5）到 2.12（跃层 band3）变化 2.6 倍。u_c 87.5% 由 u_r 贡献，故 u_c 也近乎
无深度结构 → 深层过覆盖 0.996、跃层欠覆盖 0.886。
机制：平稳核假设 σ² 为常数，无法表达"跃层误差大、深层误差小"。

方案：按深度带独立拟合 GP 超参（banded block GP，EXP-4 变体 A 的机制），
使每个带拥有自己的 σ²/ℓ，u_r 从而获得深度结构。

对照：single GP（主方案）vs banded GP，在 test 场景（seeds 0..N_SEEDS-1）上
以冻结 k=1.211 评估总体与分带覆盖率。二者取共同可用点子集，保证可比。

产出：results/exp6b/banded_vs_single.csv|_by_band.csv|_detail.csv + exp6b_summary.json
"""
import sys, os, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from exp import config as C
from exp.lib import scenes as scenelib
from exp.lib import gp as gplib
from exp.lib.pipeline import SceneData, load_stack, coverage_tolerance
from exp.lib.uc import UcComposer

OUT = C.result_dir("exp6b")
RNG = np.random.default_rng(C.SEED)
HP_SUBSET = 12      # 每带参与超参估计的最大场景数（与 EXP-4 一致）
N_SEEDS = 3         # test 场景重复划分种子数（0..N_SEEDS-1）


def fit_band_kerns(SD, tr_sids, form="prod"):
    """train-only 场景按深度带拟合 6 套超参（跨场景共享同带超参）。"""
    band_kerns = {}
    for bi in range(6):
        ds_list = []
        for s in RNG.permutation(tr_sids):
            x = SD.build(int(s))
            m = x["band"] == bi
            if m.sum() >= 15:
                ds_list.append((x["X4"][m], x["d"][m], x["u"][m], x["mean"][m]))
            if len(ds_list) >= HP_SUBSET:
                break
        if len(ds_list) < 3:
            band_kerns[bi] = None
            print(f"  band {bi}: 样本不足，跳过", flush=True)
            continue
        kern, _, _ = gplib.fit_hyperparams(ds_list, form=form)
        band_kerns[bi] = kern
        r = kern.raw
        print(f"  band {bi}: n_scenes={len(ds_list):2d} s2={np.exp(r['log_s2']):8.3f} "
              f"lh={np.exp(r['log_lh']):7.2f}km lz={np.exp(r['log_lz']):7.2f}m "
              f"lt={np.exp(r['log_lt']):6.2f}d", flush=True)
    return band_kerns


def scene_pair(SD, uc, sid, seed, band_kerns):
    """单场景单 seed：返回 single / banded 在共同可用点子集上的逐点明细。"""
    A, B = SD.ab(sid, seed)
    dsA, dsB = SD.build(sid, A), SD.build(sid, B)
    z, season = dsB["z"], dsB["season"]
    u_extra = uc.extra_std(z, season)
    dh, _ = SD.dist_to_inputs(dsB["X4"], dsA["X4"])
    # scheme=global 时 k_at 返回标量，须广播为逐点数组（否则 k[ok] 下标报错）
    k = np.broadcast_to(np.asarray(uc.k_at(z, season, dh), dtype=float),
                        z.shape).copy()

    g = SD.fit_gp(dsA)
    p_s, v_s = g.predict(dsB["X4"], dsB["mean"])

    p_b = np.full(len(dsB["d"]), np.nan)
    v_b = np.full(len(dsB["d"]), np.nan)
    for bi in range(6):
        if band_kerns.get(bi) is None:
            continue
        ma, mb = dsA["band"] == bi, dsB["band"] == bi
        if ma.sum() < 5 or not mb.any():
            continue
        gb = gplib.GPFit(dsA["X4"][ma], dsA["d"][ma], dsA["u"][ma],
                         dsA["mean"][ma], band_kerns[bi])
        pp, vv = gb.predict(dsB["X4"][mb], dsB["mean"][mb])
        p_b[mb], v_b[mb] = pp, vv

    ok = np.isfinite(p_b) & (v_b > 0) & (v_s > 0)
    if ok.sum() == 0:
        return None
    rows = []
    for tag, pred, var in (("single", p_s, v_s), ("banded", p_b, v_b)):
        err = pred[ok] - dsB["d"][ok]
        u_c = np.sqrt(var[ok] + u_extra[ok] ** 2)
        rows.append(pd.DataFrame({
            "tag": tag, "sid": sid, "seed": seed,
            "band": dsB["band"][ok], "z": dsB["z"][ok],
            "err": err, "u_c": u_c, "u_r": np.sqrt(var[ok]),
            "covered": np.abs(err) <= k[ok] * u_c}))
    return pd.concat(rows, ignore_index=True)


def main():
    t0 = time.time()
    print("== EXP-6b: banded block GP vs single GP (coverage fix test) ==")
    stack = load_stack()
    SD = SceneData(stack)
    scenes = stack["scenes"]
    uc = UcComposer()
    tr_sids = scenelib.scene_ids_by_role(scenes, "train")
    test_sids = scenelib.scene_ids_by_role(scenes, "test")
    print(f"train scenes={len(tr_sids)}, test scenes={len(test_sids)}")

    print("-- fitting per-band hyperparameters (train-only) --")
    band_kerns = fit_band_kerns(SD, tr_sids)
    n_ok_band = sum(v is not None for v in band_kerns.values())
    print(f"  usable bands: {n_ok_band}/6")

    print(f"-- evaluating test scenes x {N_SEEDS} seeds --")
    dfs = []
    for i, sid in enumerate(test_sids):
        for seed in range(N_SEEDS):
            df = scene_pair(SD, uc, int(sid), seed, band_kerns)
            if df is not None:
                dfs.append(df)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(test_sids)} scenes ({time.time()-t0:.0f}s)", flush=True)
    all_df = pd.concat(dfs, ignore_index=True)
    all_df.to_csv(os.path.join(OUT, "banded_vs_single_detail.csv"), index=False)

    def agg(df):
        return {"n": len(df),
                "cover": float(df["covered"].mean()),
                "rmse": float(np.sqrt((df["err"] ** 2).mean())),
                "mae": float(df["err"].abs().mean()),
                "u_c_mean": float(df["u_c"].mean()),
                "u_r_mean": float(df["u_r"].mean()),
                "width_mean": float((2 * df["u_c"]).mean())}

    rows = []
    for tag in ("single", "banded"):
        d = all_df[all_df["tag"] == tag]
        rows.append({"tag": tag, **agg(d)})
    main_df = pd.DataFrame(rows)
    tol_lo, tol_hi = coverage_tolerance(int(main_df["n"].iloc[0]))
    main_df["tol_lo"], main_df["tol_hi"] = tol_lo, tol_hi
    main_df["pass_J2"] = (main_df["cover"] >= tol_lo) & (main_df["cover"] <= tol_hi)
    main_df.to_csv(os.path.join(OUT, "banded_vs_single.csv"), index=False)
    print(main_df.to_string(index=False))

    brows = []
    for tag in ("single", "banded"):
        d = all_df[all_df["tag"] == tag]
        for b, g in d.groupby("band"):
            brows.append({"tag": tag, "band": int(b), "z_med": float(g["z"].median()),
                          **agg(g)})
    band_df = pd.DataFrame(brows)
    band_df.to_csv(os.path.join(OUT, "banded_vs_single_by_band.csv"), index=False)
    print(band_df.to_string(index=False))

    # u_r 的深度结构强度：各带 u_r 的极差/均值
    struct = {}
    for tag in ("single", "banded"):
        d = all_df[all_df["tag"] == tag]
        ur = d.groupby("band")["u_r"].mean()
        struct[tag] = {"u_r_by_band": {int(k): float(v) for k, v in ur.items()},
                       "u_r_range_over_mean": float((ur.max() - ur.min()) / ur.mean())}
    print("u_r depth structure:", json.dumps(struct, ensure_ascii=False))

    summary = {
        "exp": "exp6b", "date": time.strftime("%Y-%m-%d %H:%M"),
        "n_test_scenes": len(test_sids), "n_seeds": N_SEEDS,
        "bands_usable": int(n_ok_band),
        "k_frozen": (float(uc.kcfg["k_global"]) if uc.kcfg else None),
        "main": main_df.to_dict("records"),
        "by_band": band_df.to_dict("records"),
        "ur_depth_structure": struct,
        "elapsed_s": time.time() - t0,
    }
    with open(os.path.join(OUT, "exp6b_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    print("EXP-6b DONE", time.time() - t0)


if __name__ == "__main__":
    main()
