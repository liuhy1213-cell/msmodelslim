#!/usr/bin/env python3
"""生成 K 中心化用的 μ，写进成品包的 ``<pkg>/kv_center/``。

只标定 **full-attention 层**（25 个）：这些层的 K 存在明显的逐通道直流偏置，
会把 MX 的 32 通道共享尺度撑大。滑窗层的 K 分布不同，实测给它们做中心化反而
退化（GPQA 92.9% -> 87.5%），因此不生成、运行时也不读。

每个层写一份**不切分**的表：

    <pkg>/kv_center/{layer}.pt = tensor([mu, mean_abs_k])      # [2, num_kv_heads, head_dim]

运行时按实际的 TP 布局自己切（KV 头按 TP 连续分片），所以换 TP / 加 DP 都不
需要重新标定。运行时只读第 0 项，写 KV 前做 ``K - mu``，sink key 同步减 mu；
该变换对 softmax 是等价的（<q,k-mu> = <q,k> - <q,mu>），不改变模型语义。
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO = os.environ.get("MSMODELSLIM_REPO", "/mnt/sfs_turbo/work/liuhaiyang/modelslim_full")
MODEL = os.environ.get("SRC_MODEL", "/mnt/sfs_turbo/models/IQuest-Q1")
sys.path.insert(0, REPO)

from msmodelslim.core.const import DeviceType  # noqa: E402
import msmodelslim.model.iquest_q1.loader  # noqa: E402,F401
from msmodelslim.model.iquest_q1.model_adapter import IQuestQ1ModelAdapter  # noqa: E402
from msmodelslim.model.iquest_q1.iquest_q1_model import layer_types  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True, help="量化成品包目录，μ 写到 <pkg>/kv_center/")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--data", default=f"{REPO}/lab_calib/qwen3_cot_w4a4.json")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=2048)

    ap.add_argument("--device", default="npu", choices=["npu", "cpu"])
    ap.add_argument("--layers", default="full", choices=["full", "all"],
                    help="full=只算 25 个全注意力层（默认、生产口径）；all=88 层全算（实验用）")
    args = ap.parse_args()

    out_dir = Path(args.pkg) / "kv_center"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.set_grad_enabled(False)
    dev = torch.device("npu:0") if args.device == "npu" else torch.device("cpu")
    adapter = IQuestQ1ModelAdapter("IQuest-Q1", Path(args.model), True)
    cfg = adapter.config
    types = layer_types(cfg)
    if args.layers == "all":
        target = list(range(cfg.num_hidden_layers))
    else:
        target = [i for i, t in enumerate(types) if t == "full_attention"]

    tok = adapter._load_tokenizer(trust_remote_code=True)
    prompts = json.load(open(args.data))[: args.samples]
    ids = [tok(p, return_tensors="pt", truncation=True, max_length=args.seq_len)["input_ids"].to(dev)
           for p in prompts]
    model = adapter.init_model(DeviceType.NPU if args.device == "npu" else DeviceType.CPU)
    model.eval()
    embed = model.model.embed_tokens.to(dev)
    hs = [embed(i) for i in ids]
    pos = [torch.arange(i.shape[1], device=dev) for i in ids]

    nkv = cfg.num_key_value_heads
    print(f"目标层 {len(target)}，kv heads {nkv}，写不切分的整体表（运行时按 TP 自己切）")

    for idx in range(cfg.num_hidden_layers):
        layer = adapter.load_decoder_if_not_exist(model, idx).to(dev)
        if idx in target:
            attn = layer.self_attn
            acc = torch.zeros(attn.num_kv_heads, attn.head_dim, device=dev, dtype=torch.float64)
            acc_abs = torch.zeros_like(acc)
            ntok = 0
            for j in range(len(ids)):
                x = layer.attention_norm(hs[j])
                b, t, _ = x.shape
                k = attn.k_proj(x).view(b, t, attn.num_kv_heads, attn.head_dim)
                k = attn.k_norm(k)
                cos, sin = attn._rope_cos_sin(pos[j], k.dtype)
                k = attn._apply_rope(k, cos, sin)              # post-rope，与运行时一致
                acc += k.to(torch.float64).sum(dim=(0, 1))
                acc_abs += k.abs().to(torch.float64).sum(dim=(0, 1))
                ntok += b * t
            mu = (acc / ntok).float()
            mean_abs = (acc_abs / ntok).float()
            # 只写不切分的整体表：运行时按实际 TP 自己切。
            torch.save(torch.stack([mu, mean_abs]), out_dir / f"{idx}.pt")
            print(f"layer {idx:>2}: mu [{float(mu.min()):.3f}, {float(mu.max()):.3f}] "
                  f"|mu| 均值 {float(mu.abs().mean()):.4f}", flush=True)
        hs = [layer(pos[j], hs[j]) for j in range(len(ids))]
        model.model.layers.pop(-1)
        del layer
        if args.device == "npu":
            torch.npu.empty_cache()

    print(f"完成：{len(target)} 层 → {out_dir}")


if __name__ == "__main__":
    main()
