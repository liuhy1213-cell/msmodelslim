#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""校验 HiFloat8 per-tensor 全 C8 导出件。

用法（容器内，PYTHONPATH 指向 repo）：
    python script/quantize/verify_hif8_pkg.py <out_dir> [--layers 88]

检查：
  1. quant_model_description.json 中每层 self_attn.quant_type == HIF8_PER_TENSOR
  2. 每层 fa_q/fa_k/fa_v.scale 存在、float32、元素数为 1
  3. 统计 scale 分布；与 C8-MXFP 基线（可选）对比层覆盖
  4. 线性层 tensor 数量与 model_quant_type
"""

import argparse
import json
import os
import sys
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--layers", type=int, default=0, help="期望层数（0=不检查）")
    ap.add_argument("--baseline", default="", help="可选：C8-MXFP 导出目录，用于对比 tensor 集合")
    args = ap.parse_args()

    desc_path = os.path.join(args.out_dir, "quant_model_description.json")
    with open(desc_path, "r", encoding="utf-8") as f:
        desc = json.load(f)

    qt = {k: v for k, v in desc.items() if k.endswith(".self_attn.quant_type")}
    print(f"[desc] keys={len(desc)} layers_with_quant_type={len(qt)}")
    print(f"[desc] quant_type counter={dict(Counter(qt.values()))}")
    if args.layers and len(qt) != args.layers:
        print(f"[FAIL] expected {args.layers} layers, got {len(qt)}")
        return 1
    bad = [k for k, v in qt.items() if v != "HIF8_PER_TENSOR"]
    if bad:
        print(f"[FAIL] non-HIF8_PER_TENSOR layers: {bad[:5]}")
        return 1

    import torch
    from safetensors import safe_open

    idx_path = os.path.join(args.out_dir, "quant_model_weights.safetensors.index.json")
    with open(idx_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    files = sorted(set(weight_map.values()))
    handles = {f: safe_open(os.path.join(args.out_dir, f), framework="pt") for f in files}

    def get(name):
        return handles[weight_map[name]].get_tensor(name)

    scales = {"q": [], "k": [], "v": []}
    missing = []
    for k in qt:
        parent = k[: -len(".quant_type")]
        for br in ("q", "k", "v"):
            name = f"{parent}.fa_{br}.scale"
            if name not in weight_map:
                missing.append(name)
                continue
            t = get(name)
            if t.dtype != torch.float32 or t.numel() != 1:
                print(f"[FAIL] {name}: dtype={t.dtype} numel={t.numel()}")
                return 1
            scales[br].append(float(t.flatten()[0]))
    if missing:
        print(f"[FAIL] missing scales: {missing[:5]} (total {len(missing)})")
        return 1

    for br, vals in scales.items():
        vals_t = torch.tensor(vals)
        print(
            f"[scale] fa_{br}: n={len(vals)} min={vals_t.min():.6g} med={vals_t.median():.6g} "
            f"max={vals_t.max():.6g} zero={(vals_t == 0).sum().item()}"
        )

    print(f"[desc] model_quant_type={desc.get('model_quant_type')} fa_quant_type={desc.get('fa_quant_type')}")
    print(f"[pkg] tensors={len(weight_map)} files={len(files)}")

    if args.baseline:
        bpath = os.path.join(args.baseline, "quant_model_weights.safetensors.index.json")
        if os.path.exists(bpath):
            with open(bpath, "r", encoding="utf-8") as f:
                bmap = json.load(f)["weight_map"]
            fa_new = {k for k in weight_map if ".fa_" in k}
            fa_old = {k for k in bmap if ".fa_" in k}
            print(f"[base] fa tensors new={len(fa_new)} old={len(fa_old)}")
            print(f"[base] only-new={len(fa_new - fa_old)} only-old={len(fa_old - fa_new)}")
        else:
            print(f"[base] no baseline index at {bpath}")

    print("[OK] hif8 package verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
