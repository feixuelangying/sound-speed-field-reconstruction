# -*- coding: utf-8 -*-
"""一次性抢救工具：从崩溃残留的 hp_table.json 提取已完成的超参 groups，
生成 _hp_cache.json 供 exp4_gp.py 重跑时命中缓存。

背景：EXP-4 首跑完成 15 次超参拟合与核选型后，在写 hp_table.json 的
_meta.val_sids 时崩于 numpy int32 序列化（日志【问题20】）。json.dump 为
流式写入，groups 部分（6 配置 × 各季节分组）已完整落盘，仅缺 _meta 与收尾。
本脚本截断残留文件并解析，配以日志中记录的 mean_lml，重建缓存，避免重做
约 12 分钟的超参拟合。

保留说明：本文件为过程性抢救工具，跑完即完成使命；保留以忠实记录修复过程。
"""
import json
import os
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp import config as C

OUT = C.result_dir("exp4")
SRC = os.path.join(OUT, "hp_table.json")
DST = os.path.join(OUT, "_hp_cache.json")

# 首跑日志（logs/exp4_run.txt）记录的各配置 mean_lml
MEAN_LML = {
    "prod_global": -1197.5, "prod_season": -1844.0,
    "add_global": -19370.8, "add_season": -39760.1,
    "se_global": -1614.1, "se_season": -1619.6,
}


def main():
    with open(SRC, encoding="utf-8") as f:
        txt = f.read()
    i = txt.index('"_meta"')
    head = txt[:i].rstrip()
    if head.endswith(","):
        head = head[:-1].rstrip()
    groups_out = json.loads(head + "\n}")
    print("recovered configs:", list(groups_out.keys()))
    for k, g in groups_out.items():
        print(f"  {k}: groups={list(g.keys())}")

    store = {}
    for key, groups in groups_out.items():
        g0 = groups.get("all") or groups.get("0") or {}
        store[key] = {"kern": g0, "mean_lml": MEAN_LML.get(key, 0.0),
                      "n_scenes": len(groups), "groups": groups}

    with open(DST, "w", encoding="utf-8") as f:
        json.dump({"_complete": True, "store": store}, f,
                  ensure_ascii=False, indent=1)
    print("wrote", DST)


if __name__ == "__main__":
    main()
