#!/usr/bin/env python3
"""量化产物自检：布局、shard 完整性、FA3 层数、fa_v scale 形状。

用法:
    python3 verify_quant.py <量化输出目录> [--mode v1|c8] [--layers 88]

退出码 0 = 通过；非 0 = 有问题（会打印原因）。
"""
import argparse
import glob
import json
import os
import sys

FULL_ATTN_LAYERS = [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49,
                    53, 57, 61, 65, 69, 73, 77, 81, 85, 86, 87]
REQUIRED_FILES = [
    "config.json", "quant_model_description.json",
    "quant_model_weights.safetensors.index.json",
]


def fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--mode", choices=["v1", "c8"], default=None)
    ap.add_argument("--layers", type=int, default=88)
    args = ap.parse_args()
    d = args.out_dir

    for name in REQUIRED_FILES:
        if not os.path.exists(os.path.join(d, name)):
            fail(f"缺少 {name}")

    with open(os.path.join(d, "quant_model_description.json")) as f:
        desc = json.load(f)

    # shard 是否齐全
    n_shard = len(glob.glob(os.path.join(d, "quant_model_weights-*-of-*.safetensors")))
    if n_shard == 0:
        fail("没有权重 shard")

    with open(os.path.join(d, "quant_model_weights.safetensors.index.json")) as f:
        idx = json.load(f)
    weight_map = idx.get("weight_map", {})
    if not weight_map:
        fail("index.json 的 weight_map 为空")

    suffix = sorted({v for v in weight_map.values()})
    missing = [s for s in suffix if not os.path.exists(os.path.join(d, s))]
    if missing:
        fail(f"index 引用了不存在的 shard: {missing[:3]}")

    # 线性量化
    q_keys = [k for k in desc if k.endswith(".weight_scale")]
    if not q_keys:
        fail("没有任何 *.weight_scale，线性量化没生效")

    # MoE 融合布局（vLLM-ascend 期望 routed_experts）
    routed = [k for k in weight_map if "routed_experts" in k]
    if not routed:
        fail("没有 routed_experts.* —— MoE 专家融合没写出，vLLM 无法加载")

    print(f"shards        : {n_shard}")
    print(f"desc keys     : {len(desc)}")
    print(f"weight_scale  : {len(q_keys)}")
    print(f"routed_experts: {len(routed)}")

    fa_scale = [k for k in desc if k.endswith(".self_attn.fa_v.scale")]
    fa_layers = sorted({int(k.split(".")[2]) for k in fa_scale})
    # 冒烟跑（IQUEST_MAX_LAYERS=n）只覆盖前 n 层
    expected = [l for l in FULL_ATTN_LAYERS if l < args.layers]
    if args.mode == "c8":
        if fa_layers != expected:
            fail(f"C8 的 fa_v.scale 层不对: 期望 {len(expected)} 层全注意力，"
                 f"实际 {len(fa_layers)} 层，差异 {set(fa_layers) ^ set(expected)}")
        for k in fa_scale:
            if desc[k] not in ("FAQuant", "FLOAT"):
                fail(f"{k} 的 desc 类型异常: {desc[k]}")
        qtypes = [desc[k] for k in desc if k.endswith(".self_attn.quant_type")]
        if len(qtypes) != len(expected):
            fail(f"逐层 quant_type 数量不对: {len(qtypes)}")
        print(f"fa_v.scale    : {len(fa_scale)} 层（全注意力）OK")
    else:
        if fa_scale:
            print(f"[WARN] 出现 {len(fa_scale)} 层 fa_v.scale，v1 配方不应该有")

    print("[PASS] 产物布局检查通过")


if __name__ == "__main__":
    main()
