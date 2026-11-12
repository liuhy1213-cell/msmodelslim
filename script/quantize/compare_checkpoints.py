"""逐张量比较两个 W8A8-MXFP8 checkpoint 的量化 scale。

用途：两版量化权重精度不同时，先用它定位差异出在哪个张量上。
线性层的 weight_scale 一样、只有 fa_v.scale 不同，就说明差异全在 KV 量化。

用法（在有 safetensors 的环境里跑，通常是推理容器内）：
    python3 compare_checkpoints.py <ckpt_A> <ckpt_B>
    python3 compare_checkpoints.py <ckpt_A> <ckpt_B> --layers 0,45,87

输出：每个张量的 shape、dtype、逐元素完全相同的比例、相对误差；
以及 fa_v.scale 的 E8M0 指数差直方图（0 = 相同，-1 = 对方是我们的 2 倍）。
"""
import argparse
import json
import os

import torch
from safetensors import safe_open

PROJS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


def index(d):
    path = os.path.join(d, "quant_model_weights.safetensors.index.json")
    with open(path) as f:
        return json.load(f)["weight_map"]


def load(d, wm, key):
    with safe_open(os.path.join(d, wm[key]), framework="pt") as f:
        return f.get_tensor(key)


def to_float(t):
    """E8M0 的 uint8 字节 -> float；其它直接转 float32。"""
    if t.dtype == torch.uint8:
        return torch.pow(2.0, t.to(torch.float32) - 127.0)
    return t.to(torch.float32)


def compare(a_dir, a_i, b_dir, b_i, key):
    a = load(a_dir, a_i, key)
    b = load(b_dir, b_i, key)
    if to_float(a).numel() != to_float(b).numel():
        return f"{key}: 形状不同 A={tuple(a.shape)}{a.dtype} B={tuple(b.shape)}{b.dtype}"
    fa = to_float(a).flatten()
    fb = to_float(b).flatten()
    rel = (fa - fb).abs() / fa.abs().clamp_min(1e-12)
    same = int((fa == fb).sum().item())
    rel = rel[torch.isfinite(rel)]
    return (
        f"{key}: shape={tuple(a.shape)} {a.dtype}->{b.dtype} | "
        f"完全相同 {same}/{fa.numel()} ({100.0 * same / fa.numel():5.1f}%) | "
        f"max_rel={rel.max().item():.3e} mean_rel={rel.mean().item():.3e}"
    )


def fav_layers(wm):
    return sorted(
        int(k.split(".")[2]) for k in wm
        if k.startswith("model.layers.") and k.endswith(".self_attn.fa_v.scale")
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="checkpoint A 目录")
    ap.add_argument("b", help="checkpoint B 目录（差值按 B - A 算）")
    ap.add_argument("--layers", default="0,1,2,3,5,13,25,45,65,87",
                    help="线性层抽样，逗号分隔")
    args = ap.parse_args()

    a_i, b_i = index(args.a), index(args.b)

    print("=== weight_scale ===")
    for L in [int(x) for x in args.layers.split(",")]:
        for p in PROJS:
            key = f"model.layers.{L}.{p}.weight_scale"
            if key in a_i and key in b_i:
                print(compare(args.a, a_i, args.b, b_i, key))

    print("\n=== fa_v.scale ===")
    for L in fav_layers(a_i):
        key = f"model.layers.{L}.self_attn.fa_v.scale"
        if key in b_i:
            print(compare(args.a, a_i, args.b, b_i, key))

    print("\n=== fa_v.scale E8M0 指数差（B - A，只看 2 的幂差）===")
    print(f"{'layer':>5} {'相同':>7} {'-2':>6} {'-1':>6} {'+1':>6} {'+2':>6}")
    for L in fav_layers(a_i):
        key = f"model.layers.{L}.self_attn.fa_v.scale"
        if key not in b_i:
            continue
        a = load(args.a, a_i, key).to(torch.int32).flatten()
        b = load(args.b, b_i, key).to(torch.int32).flatten()
        if a.numel() != b.numel():
            continue
        d = b - a
        cnt = {x: int((d == x).sum().item()) for x in (-2, -1, 0, 1, 2)}
        print(f"{L:>5} {cnt[0]:>7} {cnt[-2]:>6} {cnt[-1]:>6} {cnt[1]:>6} {cnt[2]:>6}")


if __name__ == "__main__":
    main()
