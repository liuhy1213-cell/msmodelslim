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
| 标定集 | `qwen3_cot_w4a4.json` | 42 条、1243~2661 token、合计 91k；相对名由 `FileDatasetLoader` 解析到 `lab_calib/`。旧的 `mix_calib.jsonl` 只有 48 条 / 中位 334 token，偏短 |
| save | `ascendv1_saver`, `part_file_size: 4` | 该字段按 **GB** 解释，4 → 每 4 GB 一个分片 |

## 适配器结构（`msmodelslim/model/iquest_q1/`）

| 文件 | 职责 |
|---|---|
| `configuration_iquest_q1.py` | `IQuestQ1Config`，注册进 transformers `AutoConfig`（`loader.py` import 时注册） |
| `loader.py` | `IQuestQ1AdapterLoader` 入口，`config.ini` 的 `[ModelAdapterEntryPoints]` 指向此处 |
| `model_adapter.py` | 标定数据加载（兼容 `text` / `prompt` / `inputs_pretokenized` 键名）、逐层加载与送 quant processor；结构照 deepseek_v4 适配器：`init_model` 只建 1 层壳，`generate_decoder_layer` 按需建层，`generate_model_forward` 用 layer0 pre-hook 截住 `(positions, hidden_states)` 后逐层 yield |
| `iquest_q1_model.py` | 标定前向图，与上游 `vllm_iquest_q1/model.py` 逐行对齐：causal + 逐层 sliding_window（含当前 token）+ sink 列、sandwich norm（layer0 额外 `ffn_out_norm`）、per-layer rope theta |

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

---

# 精度调优配方变体（2026-11-08）

按《量化精度调优指南》步骤 3「量化算法选择」准备的 W4A4C8 变体。
基线是 `iquest_q1_w4a4c8_mxfp4.yaml`（attention W8A8 + 专家 W4A4，全 minmax）。

| 变体 | attention W8A8 权重 | 专家 W4A4 权重 | GSM8K-200 实测（max_tokens=3072） |
|---|---|---|---|
| 基线 `iquest_q1_w4a4c8_mxfp4.yaml` | minmax | minmax | 194/200（97.0%，截断 7） |
| `iquest_q1_w4a4c8_mxfp4_tune_b1.yaml` | `mse_round` | `ceil_x`（关 search） | **197/200（98.5%，截断 2）= BF16 持平 ★ 采纳** |
| `iquest_q1_w4a4c8_mxfp4_tune_b2b.yaml` | minmax | `ceil_x`（关 search） | 194/200（97.0%，截断 3） |
| `iquest_q1_w4a4c8_mxfp4_tune_b3.yaml` | `mse_round` | minmax | 193/200（96.5%，截断 6） |
| BF16 上界 | — | — | 197/200（98.5%，截断 4） |

**单独换 method 都不动总分，组合（B-1）才拉平 BF16**——两个权重舍入改进的收益
在联合分布上才显影。B-1 vs BF16 真数值回归 0 题（变错 2 题全是截断）。
产物：`/mnt/sfs_turbo/models/IQuest-Q1-W4A4C8-tuneB1`。

## 为什么关掉 ceil_x 的 enable_search

官方参考配方（`gemma4_moe_mxfp8_mxfp4.yaml`）用 `enable_search: true`，
但 `ceil_x_search_best`（`core/quantizer/impl/ceil_x.py:107`）是对**整个权重张量**
做 25 个候选值的完整 quantize + dequantize + MSE 比较。IQuest 的专家权重是
**3D 融合张量** `[256, 2I, H]` → 计算量放大 256 倍，实测 30 分钟卡死在 layer 0。

改用固定 `ceil_x_value: 7.25`（`enable_search: false`）即可，
**在 IQuest 上不要开 search**。

## mse_round 暂不可用（上游 bug）

`mse_round` 在 IQuest 上直接崩：

```
RuntimeError: The size of tensor a (1024) must match the size of tensor b (96)
  at non-singleton dimension 1
  File ".../core/quantizer/impl/mse_round.py", line 94, in forward
  File ".../ir/api/impl/mx_quantization.py", line 112, in mxfp_per_block_dequantize
```

根因（不是配方写错）：`MXWeightPerBlockMseRound.init_weight` 把量化后的 storage
**还原成原形状**，但 `w_q_param.ext['scale']` 仍是**块形状**，`forward()` 用这两个
形状不匹配的量做反量化 → 广播失败。
对照 `MXWeightPerBlockMinmax`：它保留两份 storage（块形状用于 forward 反量化、
原形状用于 deploy），因此没有问题。

修复方案见工作区 `quant/kv_c8/patch_mse_round_blockshape.md`（待应用/上报上游）。
修好后 B-1/B-3 才可用。

## MX 格式下真正可选的量化方法（查 `core/quantizer/impl/` 注册表）

| dtype / scope | 可用 method |
|---|---|
| `mxfp8_per_block_sym` | `minmax`, `mse_round`（后者当前不可用） |
| `mxfp4_per_block_sym` | `minmax`, `ceil_x`, `fouroversix` |

INT8 常见的 `ssz` / `gptq` / `autoround` 在 MX 格式下**没有实现**。
激活侧同理：MX 的 act 是 data-free 运行时量化，`method` 只作用于权重。

---

# 适配器 v2（2026-11-16）

## 改了什么

初版适配器的标定前向是**手写的 `embed -> layers -> norm`**，两个硬伤：

1. **没有 causal / sliding_window 掩码**——整条序列双向注意力，滑窗层更是全可见；
2. **层类型按 `last_layers_types` 从尾部反向索引**，与上游
   `get_layer_sliding_window_size` 的 first→block→last 公式不一致。

v2 直接复用模型自己的 forward：`init_model` 只建 1 层壳，layer0 上挂 pre-hook 截住
`(positions, hidden_states)`，然后逐层 yield 给 quant processor。逐层语义逐行照抄上游：
88 层 = 25 full（0, 1, 5, 9, …, 81, 85, 86, 87）+ 63 sliding，25 个 full 层进 `fa3_quant`。

自检（不需要权重）：

```bash
IQUEST_MODEL_DIR=/path/to/IQuest-Q1 python script/quantize/iquest_q1_selftest.py
```

按扰动传播范围验证掩码（window=2 只有 0/1 行受影响，full 层全部行受影响，
T=1 不出 NaN），并核对真实 config 的 25/63 层型与逐层 rope theta。

## npu 与 cpu 两条量化路径不等价

`--device npu` 比 `cpu` 快约 6 倍（约 14 s/层 vs 80 s/层），但两版产物**不是同一个包**：

| 项 | 结果 |
|---|---|
| 权重 key 集合 | 一致 |
| `weight_scale`（per_block minmax，data-free） | **逐字节相同**（0/1179648 不同） |
| `fa_v.scale`（E8M0 per-channel） | 1024 通道里 6 个差 1 档 |
| fp8 权重数值 | gate_proj 有 665283/37748736（1.76%）元素不同 |

差异全部落在最后那次 fp→fp8 cast 上：**npu 是向零截断，cpu 是 round-to-nearest**
（全部差异元素满足 `|npu| <= |cpu|`、符号相同、比值 ∈ [0.5, 0.9375]），
所以 npu 包每个元素系统性偏小最多一个量化步。scale 一致说明 minmax 那步没问题。

要复现上面这张表：

```bash
# 两条路径各跑 2 层冒烟（IQUEST_MAX_LAYERS=2），再逐 tensor 比
python script/quantize/compare_checkpoints.py out_smoke out_smoke_npu
```

## 一条命令产出成品

```bash
MODEL_PATH=/path/to/IQuest-Q1 OUT=/path/to/out DEVICE=npu \
  ./script/quantize/iquest_q1_c8_oneshot.sh
```

量化完自动跑 `script/quantize/postfix_c8.py`：`fa_v.scale` 压成 `[1024]` 并外置到
`c8_kv_v_scales.safetensors`、从 index/description 摘掉 `fa_v.offset`、补全局
`kv_cache_type`、从源模型硬链接 `chat_template.jinja` 与 `mtp/`。`PKGCLEAN=0` 可跳过。
