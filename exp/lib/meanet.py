# -*- coding: utf-8 -*-
"""均值网络 m_theta：轻量 MLP（SiLU + 残差 + LayerNorm，D13），异方差加权 MSE（D6）。
提供 torch 训练器与保存/加载（EXP-3 训练，EXP-4/9 冻结推理）。"""
import json
import os
import numpy as np
import torch
import torch.nn as nn

from .. import config as C

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.fc1(self.act(self.ln(x)))
        h = self.fc2(self.act(h))
        return x + h


class MeanNet(nn.Module):
    """in_dim -> width 投影 -> depth 个残差块 -> 线性输出（m/s）。"""

    def __init__(self, in_dim, width=128, depth=4):
        super().__init__()
        self.proj = nn.Linear(in_dim, width)
        self.blocks = nn.Sequential(*[ResBlock(width) for _ in range(depth)])
        self.head = nn.Linear(width, 1)
        self.in_dim, self.width, self.depth = in_dim, width, depth

    def forward(self, x):
        return self.head(self.blocks(self.proj(x))).squeeze(-1)


class Standardizer:
    """特征标准化（train 统计），保存供在线推理。"""

    def __init__(self, X):
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-8

    def apply(self, X):
        return (np.asarray(X, dtype=np.float32) - self.mu) / self.sd

    def to_dict(self):
        return {"mu": self.mu.tolist(), "sd": self.sd.tolist()}

    @staticmethod
    def from_dict(d):
        s = Standardizer.__new__(Standardizer)
        s.mu = np.array(d["mu"], dtype=np.float32)
        s.sd = np.array(d["sd"], dtype=np.float32)
        return s


def train_meanet(Xtr, dtr, utr, Xva, dva, uva, width=128, depth=4,
                 feat_cols=None, max_epoch=80, patience=8, lr=1e-3, wd=1e-4,
                 batch=16384, seed=None, verbose=False):
    """异方差加权 MSE 训练。返回 (model, stdr, history, feat_cols)。
    feat_cols: 特征列索引（消融用）；utr/uva: 逐点 u_obs(m/s)。"""
    if seed is not None:
        torch.manual_seed(seed)
    if feat_cols is None:
        feat_cols = list(range(Xtr.shape[1]))
    Xtr = np.asarray(Xtr)[:, feat_cols]
    Xva = np.asarray(Xva)[:, feat_cols]
    stdr = Standardizer(Xtr)
    Xtr_t = torch.tensor(stdr.apply(Xtr), device=DEV)
    Xva_t = torch.tensor(stdr.apply(Xva), device=DEV)
    dtr_t = torch.tensor(np.asarray(dtr, dtype=np.float32), device=DEV)
    dva_t = torch.tensor(np.asarray(dva, dtype=np.float32), device=DEV)
    # w = 1/(u^2 + floor^2)，floor 防零除 + 数值稳健（§3.3.1 sigma_floor）
    floor = max(0.02, float(np.percentile(utr, 5)))
    wtr = 1.0 / (np.asarray(utr, dtype=np.float64) ** 2 + floor ** 2)
    wtr = torch.tensor((wtr / wtr.mean()).astype(np.float32), device=DEV)
    wva = 1.0 / (np.asarray(uva, dtype=np.float64) ** 2 + floor ** 2)
    wva = torch.tensor((wva / wva.mean()).astype(np.float32), device=DEV)

    model = MeanNet(len(feat_cols), width, depth).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epoch)
    n = len(Xtr_t)
    hist = []
    best_val, best_state, bad = np.inf, None, 0
    for ep in range(max_epoch):
        model.train()
        perm = torch.randperm(n, device=DEV)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            pred = model(Xtr_t[idx])
            loss = (wtr[idx] * (pred - dtr_t[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xva_t)
            vloss = float((wva * (pv - dva_t) ** 2).mean() ** 0.5)
        hist.append(vloss)
        if vloss < best_val - 1e-5:
            best_val, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and ep % 10 == 0:
            print(f"    ep{ep} val_wrmse={vloss:.4f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, stdr, {"val_wrmse": hist, "best": best_val, "floor": floor,
                         "n_epoch": len(hist)}, feat_cols


@torch.no_grad()
def predict(model, stdr, X, feat_cols=None, batch=200000):
    """冻结网络批量前向（EXP-4 在线推理入口）。X: (N,8) 原始特征。"""
    model.eval()
    if feat_cols is None:
        feat_cols = list(range(X.shape[1]))
    Xn = stdr.apply(np.asarray(X)[:, feat_cols])
    out = np.empty(len(Xn), dtype=np.float32)
    for s in range(0, len(Xn), batch):
        xt = torch.tensor(Xn[s:s + batch], device=DEV)
        out[s:s + batch] = model(xt).cpu().numpy()
    return out


def save_model(path_base, model, stdr, feat_cols, extra=None):
    """保存：path_base.pt（权重）+ path_base.json（配置）。"""
    torch.save(model.state_dict(), path_base + ".pt")
    cfg = {"in_dim": model.in_dim, "width": model.width, "depth": model.depth,
           "feat_cols": list(feat_cols), "std": stdr.to_dict()}
    if extra:
        cfg.update(extra)
    with open(path_base + ".json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)


def load_model(path_base):
    """加载冻结模型（EXP-4/9 用）。返回 (model, stdr, feat_cols, cfg)。"""
    with open(path_base + ".json", encoding="utf-8") as f:
        cfg = json.load(f)
    model = MeanNet(cfg["in_dim"], cfg["width"], cfg["depth"])
    model.load_state_dict(torch.load(path_base + ".pt", map_location="cpu"))
    model.to(DEV).eval()
    stdr = Standardizer.from_dict(cfg["std"])
    return model, stdr, cfg["feat_cols"], cfg
