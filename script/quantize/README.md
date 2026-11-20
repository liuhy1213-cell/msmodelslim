# IQuest-Q1 量化操作指导

从 BF16 原始权重出发，用 msmodelslim 在 CPU 或 NPU 上产出一份可直接被 vLLM-ascend
部署的 MXFP8 量化权重。**一条命令，一个成品。**

- 原始权重：`/mnt/sfs_turbo/models/IQuest-Q1`（bf16，88 层 MoE）
- 适配器仓库：`/mnt/sfs_turbo/work/liuhaiyang/modelslim_full`（本目录上一级的
  `msmodelslim_push/` 是同一份代码，可直接推到 fork）
- **必须在容器里跑**：`msmodelslim` 的 venv 依赖容器镜像里的 python，宿主机直接执行会
  `required file not found`。进容器后所有 `script/quantize/*` 都能直接用：

  ```bash
  ssh root@10.222.44.142 'docker exec -it iq-own bash'
  cd /mnt/sfs_turbo/work/liuhaiyang/modelslim_full
  bash script/quantize/quantize.sh c8
  ```

## 1. 两种配方

| 配方 | 配置文件 | 内容 | 耗时 |
|---|---|---|---|
| **v1** | `iquest_q1_w8a8_linear_mxfp8.yaml` | 线性 W8A8-MXFP8（attention + 逐专家 MoE + layer0 稠密 MLP），KV 保持 BF16 | ~38 min（全 data-free，CPU） |
| **C8** | `iquest_q1_w8a8c8_mxfp8.yaml` | v1 + KV 全 88 层走 C8，其中 25 个全注意力层额外带 V 的静态 per-channel E8M0 尺度 | ~56 min（需要校准前向，CPU） |

产物约 310 G。

### 1.1 C8 的层口径（容易搞混的地方）

| 项 | 覆盖范围 |
|---|---|
| KV 量化 dtype（部署时 `--kv-cache-dtype mxfp8`） | **全部 88 层**，滑窗层同样走 C8 |
| `fa_v.scale`（V 的静态 per-channel 尺度） | **只有 25 个全注意力层**（层号 0,1,5,9,…,85,86,87） |
| 63 个滑窗层 | 不写 `fa_v.scale`，运行时用中性尺度（E8M0 = 127，等价于 1.0）；实测与逐层标定差 2~5e-6 |
| K 中心化 μ（`kv_center/`） | **只有 25 个全注意力层** |

滑窗层**不要**加中心化：实测给 63 个滑窗层也做中心化会让 GPQA 从 92.9% 掉到 87.5%，
缺 ANSWER 比例从 0.51% 涨到 4.42%。

## 2. 环境准备（一次性）

```bash
REPO=/mnt/sfs_turbo/work/liuhaiyang/modelslim_full
cd $REPO
bash install.sh                 # 建 venv 并装依赖（已装过可跳过）
ls $REPO/../venv/bin/msmodelslim    # 确认可执行文件在
```

确认跑的是修补过的适配器（不是上游原版）：

```bash
grep -q "Top-k dispatch" $REPO/msmodelslim/model/iquest_q1/iquest_q1_model.py && echo OK
grep -q "Calibration shell must be bf16" $REPO/msmodelslim/model/iquest_q1/model_adapter.py && echo OK
```

## 3. 操作步骤

### 3.1 先冒烟（强烈建议，1 分钟）

```bash
bash script/quantize/smoke.sh c8
```

只跑前 2 层，验证适配器/配方/环境没坏。看到 `EXIT=0` 再往下走。

### 3.2 一条命令出成品

```bash
bash script/quantize/quantize.sh c8 [输出目录]     # 默认 <REPO>/out_full_c8
```

脚本做三步（`v1` 只做第 1 步）：

1. **量化**：`msmodelslim quant`，配 C8 配方；
2. **包整理**（`postfix_c8.py`）：25 个 `fa_v.scale` 压平外置到
   `c8_kv_v_scales.safetensors`、从 index 摘掉全零 `fa_v.offset`、补全局
   `kv_cache_type`、补 `chat_template.jinja` 与 `mtp/`；
3. **K 中心化**（`kv_center_calib.py`）：用 16 条标定样本统计 25 个全注意力层的
   K 逐通道均值，写 `<pkg>/kv_center/{layer}.pt`。

可调环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `DEVICE` | `cpu` | `npu` 快约 3~5 倍（npu ~14 s/层，cpu ~80 s/层）；两者 weight_scale 逐字节相同，fp8 数值差 ~1 LSB |
| `SKIP_POSTFIX` | 0 | 1=跳过包整理（调试） |
| `SKIP_KCENTER` | 0 | 1=跳过 K 中心化（调试） |
| `KCENTER_SAMPLES` | 16 | 统计 μ 的标定样本数 |
| `KCENTER_LAYERS` | `full` | `all`=88 层全算（**实验用，不要用于生产**） |

成功标志：最后一行 `完成 <size> <OUT>`，且 `grep -c EXIT=0` 无失败步。

### 3.3 自检产物

```bash
python3 script/quantize/verify_quant.py out_full_c8 --mode c8
```

检查：shard 数量、index 引用完整、`routed_experts.*` 融合布局、C8 的 `fa_v.scale`
是否正好 25 层、`kv_center/` 是否正好 25 份。全过会打印 `[PASS]`。

### 3.4 起服务验证

```bash
MODEL=/path/to/out_full_c8 bash /mnt/sfs_turbo/work/liuhaiyang/PD/serve_kc25.sh
curl -s http://127.0.0.1:8008/v1/models
```

C8 部署只需要 `--kv-cache-dtype mxfp8`：**不要**再带
`--kv-cache-dtype-skip-layers sliding_window`（滑窗层现在也走 C8），也不设置任何
`C8_*` 环境变量。

## 4. 产物结构

```
out_full_c8/
  config.json                          # 原始 config
  quant_model_description.json         # 量化描述（逐 key 的 quant type + 全局 kv_cache_type）
  quant_model_weights.safetensors.index.json
  quant_model_weights-000NN-of-00088.safetensors
  c8_kv_v_scales.safetensors           # 25 个全注意力层的 fa_v.scale，压平为 [1024]
  kv_center/
    <layer>.pt                         # [2, num_kv_heads, head_dim] = [mu, mean_abs_k]，未切分
  mtp/                                 # EAGLE draft 权重（从 BF16 源模型硬链接）
  chat_template.jinja
  tokenizer.json / tokenizer_config.json / generation_config.json ...
```

`quant_model_description.json` 里的关键键：

| key | 值 | 说明 |
|---|---|---|
| `model.layers.N.self_attn.quant_type` | `QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL` | 25 个全注意力层各一个 |
| `kv_cache_type`（全局） | `K_DYNAMIC_V_STATIC_MXFP8_PER_CHANNEL` | 包整理时补上 |
| `fa_quant_type` | `FAKQuant` | 与 MXFP8 KV 并存时 loader 自动 stand down |

`kv_center/<layer>.pt` 存的是**整张表**（不按 TP rank 切分）：运行时按实际 TP 布局切片，
换 TP 或加 DP 都不需要重新标定。缺任何一层的文件，运行时会直接报错（不静默降级）。

## 5. K 中心化是什么、为什么只做 25 层

K 的逐通道存在明显直流偏置，会把 MX 的 32 通道共享尺度撑大；把偏置减掉再量化能显著
降低误差。该变换对 softmax 是等价的（`<q,k-μ> = <q,k> - <q,μ>`，把 `-<q,μ>` 并进
sink logits），所以不改变模型语义。运行时写 KV 前做 `K - μ`，sink key 同步减 μ。

滑窗层的 K 分布不同，实测做中心化反而退化（见 1.1），所以只做 25 个全注意力层。

## 6. 四个坑（改适配器时踩过的，回归时先看这里）

1. **shell 必须是 bf16**
   `load_state_dict` 用 `copy_`，不转 dtype。`init_model` 若在默认 fp32 下构造 shell，
   参数会停在 fp32，第一个校准前向就把 fp32 激活喂进 bf16 量化线性 →
   `expected m1 and m2 to have the same dtype, but got float != BFloat16`。
   代码里已用 `default_dtype(torch.bfloat16)` + 断言锁死。
   **v1 全 data-free、不跑前向，所以这个 bug 只有 C8 会暴露。**

2. **MoE router 两边都要 fp32**
   `F.linear(x.to(torch.float32), router.weight.to(torch.float32))`，只转一边会 dtype 报错。

3. **融合专家张量不能丢**
   `IQuestQ1CalibMoE._apply` 只在移到 `meta` 时丢弃 `_gate3/_up3/_down3`。
   `LoadProcessor(device=cpu)` 的 `.to(cpu)` 也会触发 `_apply`，丢了就报
   `MoE fused weights are not available on the compute device`。

4. **MoE 校准前向必须按 top-k 派发**
   按 256 专家稠密算是 **78 分钟/层**，87 层要跑几天；改成按 top-k 分组后
   **15 秒/层**，数值逐位一致（未选中专家权重为 0）。
   改完用 `tests/test_moe_topk_equiv.py` 验等价性。

## 7. 改了哪些文件（推 fork 用）

`msmodelslim_push/iquest_q1.patch` 是一次性补丁，包含全部改动：

```
msmodelslim/model/iquest_q1/iquest_q1_model.py              | 50 ++++++++-------
msmodelslim/model/iquest_q1/model_adapter.py                | 57 +++++++++++++--
msmodelslim/model/iquest_q1/expert_fusion.py                | 新增（routed_experts 融合）
msmodelslim/core/quant_service/modelslim_v1/save/ascendv1.py            | 21 +++-
msmodelslim/core/quant_service/modelslim_v1/save/interface.py           | 36 +++++-
msmodelslim/core/quant_service/modelslim_v1/save/ascendv1_distributed.py|  3 +
lab_practice/iquest_q1/iquest_q1_w8a8_linear_mxfp8.yaml     | 新增（v1 配方）
lab_practice/iquest_q1/iquest_q1_w8a8c8_mxfp8.yaml          | 新增（C8 配方）
```

在 fork 仓库里：

```bash
git apply /path/to/iquest_q1.patch      # 或直接把 msmodelslim_push/ 下的文件按路径覆盖
git add -A && git commit
```

## 8. 常见失败对照

| 报错 | 原因 | 处理 |
|---|---|---|
| `expected m1 and m2 to have the same dtype` | 坑 1 / 坑 2 | 确认适配器是修补版；检查 shell dtype 断言 |
| `MoE fused weights are not available` | 坑 3 | 确认 `_apply` 只在 meta 时丢弃 |
| 单层校准 > 10 分钟 | 坑 4（稠密专家） | 确认走 top-k 派发分支 |
| `Checkpoint key not found` | 权重目录不是 IQuest-Q1 原始 BF16 | 用 `/mnt/sfs_turbo/models/IQuest-Q1` |
| 起服务 `KeyError: v_cache_scale` | 给滑窗层也写了 `fa_v.scale` | 配方 include 列表只能是 25 个全注意力层 |
| 起服务报 `kv_center ... is missing` | 成品包里没有 `kv_center/`（跳过或漏跑第 3 步） | 重跑 `kv_center_calib.py`，禁止静默降级 |
| 加载后直接回退 BF16 / 精度崩 | `--quantization ascend` 没加，或没带 `--kv-cache-dtype mxfp8` | 见 3.4 |
