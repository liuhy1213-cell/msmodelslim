#!/usr/bin/env python3
"""IQuest-Q1 标定模型自检：结构 / 掩码 / 数值安定性（不需要真实权重）。

判据（扰动位置 0 的输入，只有 q0/k0/v0 变化）：
  sliding_attention, window=2 -> 只有第 0/1 行变，第 2..T-1 行逐位不变；
  full_attention              -> 所有行都变；
  T=1 不得 NaN（mask + sink 的边界）；
  真实 config.json 的层型统计必须是 25 full / 63 sliding。

用法：IQUEST_MODEL_DIR=/path/to/IQuest-Q1 python iquest_q1_selftest.py
"""
import json
import os
import sys
import types

import torch

MODEL_DIR = os.environ.get("IQUEST_MODEL_DIR", "/mnt/sfs_turbo/models/IQuest-Q1")
# script/quantize/xxx.py -> 仓库根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from msmodelslim.model.iquest_q1.iquest_q1_model import (  # noqa: E402
    IQuestQ1Attention,
    IQuestQ1DecoderLayer,
    layer_types,
)


def fake_config(num_layers=8, window=2, first=("sliding_attention",)):
    c = types.SimpleNamespace()
    c.hidden_size = 64
    c.num_attention_heads = 4
    c.num_key_value_heads = 2
    c.head_dim = 16
    c.rms_norm_eps = 1e-6
    c.rotary_dim = 8
    c.rope_theta = 1e6
    c.swa_rope_theta = 1e4
    c.sliding_window = window
    c.num_hidden_layers = num_layers
    c.use_hybrid_layers = True
    c.first_layers_types = list(first)
    c.hybrid_layers_types_block = ["full_attention"]
    c.num_hybrid_layers_block = num_layers - len(first) - 1
    c.last_layers_types = ["full_attention"]
    c.softmax_scale = None
    c.enable_sink_attention = True
    c.no_rope_layers = []
    c.shared_kv_num_layers = 0
    # decoder layer 用到的 MLP 字段
    c.num_experts = 4
    c.num_experts_per_tok = 2
    c.intermediate_size = 16
    c.dense_intermediate_size = 32
    c.hidden_act = "silu"
    c.mlp_only_layers = [0]
    c.attn_out_scale = 1.0
    c.ffn_out_scale = 0.5
    c.first_layer_attn_out_scale = 1.0
    c.first_layer_ffn_out_scale = 1.0
    return c


def changed_rows(y0, y1, tol=1e-6):
    d = (y0 - y1).abs().amax(dim=-1)[0]
    return [i for i, v in enumerate(d.tolist()) if v > tol]


def check_mask(cfg, seq=8, label=""):
    torch.manual_seed(0)
    attn = IQuestQ1Attention(cfg, 0).eval()
    with torch.no_grad():
        for p in attn.parameters():
            p.normal_(0.0, 0.02)
        x = torch.randn(1, seq, cfg.hidden_size)
        pos = torch.arange(seq)
        y0 = attn(pos, x)
        x2 = x.clone()
        x2[:, 0, :] += 3.0
        y2 = attn(pos, x2)
        y1 = attn(torch.zeros(1, dtype=torch.long), x[:, :1, :])
    assert torch.isfinite(y0).all(), f"{label}: baseline NaN/Inf"
    assert torch.isfinite(y1).all(), f"{label}: T=1 NaN/Inf"
    return changed_rows(y0, y2)


def main():
    fails = []

    cfg = fake_config(window=2, first=("sliding_attention",))
    got = check_mask(cfg, label="sliding")
    print("sliding window=2 -> changed rows", got, "(expect [0, 1])")
    if got != [0, 1]:
        fails.append(f"sliding mask wrong: {got}")

    cfg = fake_config(window=2, first=("full_attention",))
    got = check_mask(cfg, label="full")
    print("full               -> changed rows", got, "(expect all)")
    if got != list(range(8)):
        fails.append(f"full mask wrong: {got}")

    cfg = fake_config(window=4, first=("sliding_attention",))
    got = check_mask(cfg, label="sliding4")
    print("sliding window=4 -> changed rows", got, "(expect [0,1,2,3])")
    if got != [0, 1, 2, 3]:
        fails.append(f"window=4 mask wrong: {got}")

    # 真实 config：层型 / 滑窗 / rope theta
    real = json.load(open(f"{MODEL_DIR}/config.json"))
    real_cfg = types.SimpleNamespace(**real)
    types_ = layer_types(real_cfg)
    n_full, n_slide = types_.count("full_attention"), types_.count("sliding_attention")
    print(f"real config: {len(types_)} layers = {n_full} full + {n_slide} sliding")
    if (len(types_), n_full, n_slide) != (88, 25, 63):
        fails.append(f"layer types {len(types_)}/{n_full}/{n_slide} != 88/25/63")
    exp_full = [0] + [1 + 4 * i for i in range(21)] + [85, 86, 87]
    got_full = [i for i, t in enumerate(types_) if t == "full_attention"]
    print("full layers:", got_full)
    if got_full != exp_full:
        fails.append(f"full layer idx {got_full} != {exp_full}")

    for idx, theta in ((0, 1e6), (32, 1e4), (85, 1e6)):
        torch.manual_seed(0)
        attn = IQuestQ1Attention(real_cfg, idx)
        if abs(attn.rope_theta - theta) > 1e-6:
            fails.append(f"layer {idx} rope_theta {attn.rope_theta} != {theta}")
        if (attn.sliding_window is None) != (idx in exp_full):
            fails.append(f"layer {idx} sliding_window {attn.sliding_window} 与层型不符")
    print("rope theta / sliding window per layer: checked 0, 32, 85")

    # 整层前向（真实 config 形状太大，用 fake config 跑一遍 residual 结构）
    cfg = fake_config(num_layers=2, window=2)
    torch.manual_seed(0)
    layer = IQuestQ1DecoderLayer(cfg, 1).eval()
    layer.mlp = torch.nn.Identity()  # 只看 attention 残差
    with torch.no_grad():
        out = layer(torch.arange(8), torch.randn(1, 8, cfg.hidden_size))
    assert torch.isfinite(out).all(), "decoder layer NaN/Inf"
    print("decoder layer forward: finite")

    if fails:
        print("\nFAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
