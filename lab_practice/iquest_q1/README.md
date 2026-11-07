# IQuest-Q1 W8A8C8 (MXFP8) 量化

IQuest-Q1（88 层 MoE，256 专家，25 个 full-attention 层 + 63 个 sliding-window 层）
的 W8A8C8 量化适配：`modelslim_v1` 原生 PTQ 流水线
（`linear_quant` + `fa3_quant` + `ascendv1_saver`），与其他 MoE 模型（GLM-5.3、Qwen3.5-MoE 等）
同形的用法。

## 快速开始

```bash
msmodelslim quant \
  --model_path /path/to/IQuest-Q1 \
  --save_path  /path/to/output \
  --device npu \
  --device_id 0 1 2 3 4 5 6 7 \
  --model_type IQuest-Q1 \
  --quant_type w8a8c8 \
  --trust_remote_code true
```

- 8×NPU 分布式标定，88 层约 48 分钟（Ascend950DT，约 18 s/层）
- 标定集：`lab_calib/mix_calib.jsonl`（48 条，GSM8K 风格）。只有 `fa3_quant` 的
  fa_v 静态 per-channel 量化真正依赖标定数据；权重 per_block minmax 与 fa_q/fa_k
  per_block 均为动态量化，数据无关。

## 配方要点（`iquest_q1_w8a8c8_mxfp8.yaml`）

| 项 | 值 | 说明 |
|---|---|---|
| 权重/激活 | mxfp8 / per_block / minmax | attention + MoE 逐专家 + layer0 稠密 MLP；`*sink*` 排除 |
| KV C8 | `fa3_quant` details 分支 | fa_q / fa_k = per_block **动态**；fa_v = per_channel **静态**（E8M0 scale） |
| KV 层集合 | 仅 25 个 full-attention 层 | SWA 层保持 bf16 KV，见下文「SWA 层」 |
| 标定集 | `mix_calib.jsonl` | 相对名由 `FileDatasetLoader` 解析到 `lab_calib/` |
| save | `ascendv1_saver`, `part_file_size: 4` | 该字段按 **GB** 解释，4 → 每 4 GB 一个分片 |

## 适配器结构（`msmodelslim/model/iquest_q1/`）

| 文件 | 职责 |
|---|---|
| `configuration_iquest_q1.py` | `IQuestQ1Config`，注册进 transformers `AutoConfig`（`loader.py` import 时注册） |
| `loader.py` | `IQuestQ1AdapterLoader` 入口，`config.ini` 的 `[ModelAdapterEntryPoints]` 指向此处 |
| `model_adapter.py` | 标定数据加载（兼容 `text` / `prompt` / `inputs_pretokenized` 键名）、逐层加载与送 quant processor |
| `iquest_q1_model.py` | 标定用前向图：ForCausalLM 形态（`model.*` + `lm_head`）、稠密层 / MoE 层实现 |

## 产出与 vllm-ascend 对接的注意事项

saver 直接产出的是**逐专家**权重（`mlp.experts.fc.<eid>.{gate,up}_proj.weight` /
`mlp.experts.proj.<eid>.down_proj.weight`）与 `fa3_quant` 原始元数据。给
vllm-ascend 部署前需要后处理（参考已验证流程）：

1. **专家重排**：逐专家权重合成 vLLM 原生的 `routed_experts.w13/w2` 布局并分片。
2. **剔除 `self_attn.fa_v.offset`**：本配方下它是恒零张量，vllm-ascend 的 mxfp8
   路径将其映射为 `None` 丢弃；留在 checkpoint 里反而会让严格校验工具报多余参数。
3. **`fa_v.scale` 形状**：`[num_kv_heads*head_dim, 1]` 需 squeeze 成 `[1024]`（uint8 E8M0）。
4. **元数据归一**：saver 顶层写的是 `fa_quant_type: FAKQuant`；vllm-ascend 识别的
   KV 配方键是 `kv_cache_type: K_DYNAMIC_V_STATIC_MXFP8_PER_CHANNEL`，每层
   `self_attn.quant_type: QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL`，`fa_v.scale`
   条目改标 `FLOAT`。

### SWA 层为什么不进 fa3_quant include

vllm-ascend 侧 `_is_full_attention_layer` 只给 full-attention 层挂 C8 method，
SWA 层没有 `v_cache_scale` 参数；若 checkpoint 带上 SWA 层的 `fa_v.scale`，
加载时 KeyError / 多余参数报错。因此只对 25 个 full-attention 层
（0, 1, 5, 9, …, 85, 86, 87）做 KV C8，SWA 层 KV 保持 bf16。

## 实跑记录（2026-11-07，8×Ascend950DT）

- 量化 `SUCCESS rc=0`，83 分片 / 311 GiB / 135011 tensor
- 与独立路线（`modelslim_convert` + 自研 KV-C8 标定打包）产出的已验证 checkpoint
  对比：tensor 键集合一致、`quant_model_description.json` 键/值一致
  （仅 embed_tokens / lm_head / model.norm 为 F32 而非 BF16，符合
  `enable_lm_head_fp32: true`）
- TP8 起服加载 38.94 GB/rank，vLLM 正确识别 MXFP8 KV recipe，正常生成
