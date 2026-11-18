#!/usr/bin/env python3
"""把 msmodelslim 直接吐出来的 C8 包整理成发布形态（数值一律不改）。

做四件事，全部是格式层面：
  1. 25 个 ``fa_v.scale`` 从 [1024,1] 压成 [1024]，挪到单独的
     ``c8_kv_v_scales.safetensors``（和参考包 IQuest-Q1-W8A8C8-mxfp8 一致）；
  2. 从 index 里摘掉 25 个 ``fa_v.offset``（全零，运行时 suffix_map 直接映射成
     None，没有任何消费者）；tensor 字节留在原分片里，不再被引用；
  3. ``quant_model_description.json`` 补全局 ``kv_cache_type``，
     并删掉 ``fa_v.offset`` 条目；
  4. 补 ``chat_template.jinja`` 和 ``mtp/``（从 BF16 源模型硬链接），
     msmodelslim 只处理 88 层主干，这两样不在它的输出里。

权重分片一律硬链接，不复制。默认输出到 ``<pkg>_pkg``，``--in-place`` 原地改。
可重复执行。
"""
import argparse
import json
import os
import shutil
import struct
import sys

import torch
from safetensors.torch import save_file

KV_CACHE_TYPE = "K_DYNAMIC_V_STATIC_MXFP8_PER_CHANNEL"
SCALES_FILE = "c8_kv_v_scales.safetensors"
INDEX = "quant_model_weights.safetensors.index.json"
DESC = "quant_model_description.json"


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def read_tensor(pkg, shard, key, header, base):
    """按 header 里的偏移只读这一个 tensor 的字节，不加载整个分片。"""
    entry = header.get(key)
    if entry is None:
        raise KeyError(key)
    s, e = entry["data_offsets"]
    with open(os.path.join(pkg, shard), "rb") as f:
        f.seek(base + s)
        raw = f.read(e - s)
    return raw, entry


def link_or_copy(src, dst):
    if os.path.exists(dst):
        return "exists"
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-pkg", required=True)
    ap.add_argument("--out")
    ap.add_argument("--in-place", action="store_true")
    ap.add_argument("--mtp-src", default="/mnt/sfs_turbo/models/IQuest-Q1")
    ap.add_argument("--skip-mtp", action="store_true")
    args = ap.parse_args()

    src = os.path.abspath(args.src_pkg)
    dst = src if args.in_place else os.path.abspath(args.out or src.rstrip("/") + "_pkg")
    idx_path = os.path.join(src, INDEX)
    if not os.path.exists(idx_path):
        sys.exit(f"not a quant package: {idx_path} missing")

    weight_map = json.load(open(idx_path))["weight_map"]
    scale_keys = sorted(k for k in weight_map if k.endswith(".self_attn.fa_v.scale"))
    offset_keys = sorted(k for k in weight_map if k.endswith(".self_attn.fa_v.offset"))
    if not scale_keys:
        sys.exit("no fa_v.scale in index: is this really a C8 package?")
    print(f"fa_v.scale {len(scale_keys)} keys, fa_v.offset {len(offset_keys)} keys")

    # ---- 1. 收缩 + 外置 ----
    tensors, originals = {}, {}
    if dst != src:
        os.makedirs(dst, exist_ok=True)
        for shard in sorted({v for v in weight_map.values()}):
            act = link_or_copy(os.path.join(src, shard), os.path.join(dst, shard))
            if act == "copy":
                print(f"  WARN copied (not linked): {shard}")
    for key in scale_keys:
        shard = weight_map[key]
        header, base = read_header(os.path.join(src, shard))
        raw, entry = read_tensor(src, shard, key, header, base)
        if entry["dtype"] != "U8":
            sys.exit(f"{key}: dtype {entry['dtype']} != U8")
        t = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        if t.numel() != 1024:
            sys.exit(f"{key}: {t.numel()} channels != 1024")
        originals[key] = t.clone()
        tensors[key] = t.contiguous()          # [1024]
    save_file(tensors, os.path.join(dst, SCALES_FILE))

    # ---- 2. index ----
    for key in scale_keys:
        weight_map[key] = SCALES_FILE
    for key in offset_keys:
        weight_map.pop(key, None)
    json.dump({"metadata": json.load(open(idx_path)).get("metadata", {}),
               "weight_map": weight_map},
              open(os.path.join(dst, INDEX), "w"), indent=2)

    # ---- 3. description ----
    desc = json.load(open(os.path.join(src, DESC)))
    desc["kv_cache_type"] = KV_CACHE_TYPE
    dropped_desc = 0
    for key in offset_keys:
        if desc.pop(key, None) is not None:
            dropped_desc += 1
    for key in scale_keys:
        if key not in desc:
            sys.exit(f"{key} missing from description")
    json.dump(desc, open(os.path.join(dst, DESC), "w"), indent=1)

    # ---- 4. 主干之外的两样 ----
    for name in ("config.json", "configuration.json", "generation_config.json",
                 "chat_template.jinja", "special_tokens_map.json",
                 "tokenizer.json", "tokenizer_config.json"):
        s = os.path.join(args.mtp_src, name)
        if os.path.exists(s):
            link_or_copy(s, os.path.join(dst, name))
    if not args.skip_mtp:
        src_mtp = os.path.join(args.mtp_src, "mtp")
        if os.path.isdir(src_mtp):
            os.makedirs(os.path.join(dst, "mtp"), exist_ok=True)
            for name in sorted(os.listdir(src_mtp)):
                link_or_copy(os.path.join(src_mtp, name), os.path.join(dst, "mtp", name))

    # ---- 校验 ----
    from safetensors import safe_open
    with safe_open(os.path.join(dst, SCALES_FILE), framework="pt") as f:
        got = list(f.keys())
        assert sorted(got) == scale_keys, (len(got), len(scale_keys))
        for key in scale_keys[:3] + scale_keys[-2:]:
            t = f.get_tensor(key)
            assert t.dtype == torch.uint8 and tuple(t.shape) == (1024,), (key, t.shape)
            assert torch.equal(t, originals[key]), f"{key}: bytes changed"
    wm2 = json.load(open(os.path.join(dst, INDEX)))["weight_map"]
    assert not [k for k in wm2 if k.endswith(".fa_v.offset")], "offset still indexed"
    assert all(wm2[k] == SCALES_FILE for k in scale_keys)
    d2 = json.load(open(os.path.join(dst, DESC)))
    assert d2.get("kv_cache_type") == KV_CACHE_TYPE
    assert not [k for k in d2 if k.endswith(".fa_v.offset")]

    print(f"scales -> {SCALES_FILE} ({len(tensors)} x U8[1024])")
    print(f"dropped {len(offset_keys)} fa_v.offset from index, {dropped_desc} from description")
    print(f"kv_cache_type = {KV_CACHE_TYPE}")
    print("mtp/:", sorted(os.listdir(os.path.join(dst, "mtp"))) if os.path.isdir(os.path.join(dst, "mtp")) else "skipped")
    print(f"OK {dst}")


if __name__ == "__main__":
    main()
