#!/usr/bin/env python3
"""生成 K 中心化用的 μ，写进成品包的 ``<pkg>/kv_center/``。

对每个 full-attention 层统计 post-rope K 在 (batch, token) 维上的逐通道均值，
形状 ``[num_kv_heads, head_dim]``；再按 TP rank 切片，每 rank 存

    <pkg>/kv_center/{layer}-{rank}.pt = tensor([mu_rank, mean_abs_k_rank])   # [2, 1, head_dim]

运行时（vllm-ascend 的 C8-MXFP attention）只读第 0 项，写 KV 前做 ``K - mu``，
并把 sink key 同步减 mu。该变换对 softmax 是等价的（<q,k-mu> = <q,k> - <q,mu>，
后者只与 query 有关），因此不改变模型语义，只让 MX 的 32 通道共享尺度不再被
通道均值偏移撑大。

默认给 88 层全算（--layers all）：滑窗层现在也走 C8-MXFP，同样需要 μ，
否则运行时 C8_KV_CENTER=1 会因为找不到 <layer>-<rank>.pt 直接报错。
--layers full 只保留 25 个全注意力层，供旧配置（滑窗层留 BF16 KV）使用。
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
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--device", default="npu", choices=["npu", "cpu"])
    ap.add_argument("--layers", default="all", choices=["full", "all"],
                    help="all=88 层全算（默认，滑窗层也走 C8）；full=只算 25 个全注意力层")
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
    if nkv % args.tp:
        raise SystemExit(f"kv heads {nkv} 不能被 tp {args.tp} 整除")
    per_rank = nkv // args.tp
    print(f"目标层 {len(target)}，kv heads {nkv}，tp {args.tp} → 每 rank {per_rank} 个 head")

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
            for r in range(args.tp):
                sl = slice(r * per_rank, (r + 1) * per_rank)
                torch.save(torch.stack([mu[sl], mean_abs[sl]]),
                           out_dir / f"{idx}-{r}.pt")
            print(f"layer {idx:>2}: mu [{float(mu.min()):.3f}, {float(mu.max()):.3f}] "
                  f"|mu| 均值 {float(mu.abs().mean()):.4f}", flush=True)
        hs = [layer(pos[j], hs[j]) for j in range(len(ids))]
        model.model.layers.pop(-1)
        del layer
        if args.device == "npu":
            torch.npu.empty_cache()

    print(f"完成：{len(target)} 层 × {args.tp} rank → {out_dir}")


if __name__ == "__main__":
    main()
