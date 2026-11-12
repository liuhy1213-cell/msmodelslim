# IQuest-Q1 量化操作指导

从 BF16 原始权重出发，用 msmodelslim 在 CPU 上产出一份可直接被 vLLM-ascend 部署的
MXFP8 量化权重。**一条命令，一个成品。**

- 原始权重：`/mnt/sfs_turbo/models/IQuest-Q1`（bf16，88 层 MoE）
- 适配器仓库：`/mnt/sfs_turbo/work/liuhaiyang/modelslim_full`（本目录上一级的
  `msmodelslim_push/` 是同一份代码，可直接推到 fork）
- 量化设备：CPU（不需要 NPU 卡），一台 192 核节点足够
- **必须在容器里跑**：`msmodelslim` 的 venv 依赖容器镜像里的 python，宿主机直接执行会
  `required file not found`。进容器后所有 `script/quantize/*` 都能直接用：

  ```bash
  ssh root@10.222.44.142 'docker exec -it iq-own bash'
  cd /mnt/sfs_turbo/work/liuhaiyang && bash script/quantize/quantize.sh c8
  ```

## 1. 两种配方

| 配方 | 配置文件 | 内容 | 产物 | 耗时 |
|---|---|---|---|---|
| **v1** | `iquest_q1_w8a8_linear_mxfp8.yaml` | 线性 W8A8-MXFP8（attention + 逐专家 MoE + layer0 稠密 MLP），KV 保持 BF16 | ~310G | ~38 min（全 data-free） |
| **C8** | `iquest_q1_w8a8c8_mxfp8.yaml` | v1 + 25 个全注意力层的 KV C8（Q/K 动态 per-block，V 静态 per-channel E8M0） | ~310G | ~56 min（需要校准前向） |

C8 只处理 25 个 full-attention 层（层号 0,1,5,9,…,85,86,87），另外 63 个滑窗层
**不能**带 `fa_v.scale`——vLLM-ascend 不会给 SWA 层挂 C8 method，多出来的参数会加载报错。

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

### 3.2 正式量化

```bash
bash script/quantize/quantize.sh c8     # 或 v1
```

等价的底层命令（脚本就是包了一层）：

```bash
cd /mnt/sfs_turbo/work/liuhaiyang/modelslim_full
PYTHONPATH=$PWD OMP_NUM_THREADS=64 MKL_NUM_THREADS=64 \
  /mnt/sfs_turbo/work/liuhaiyang/venv/bin/msmodelslim quant \
  --model_path /mnt/sfs_turbo/models/IQuest-Q1 \
  --save_path $PWD/out_full_c8 \
  --device cpu \
  --config_path $PWD/lab_practice/iquest_q1/iquest_q1_w8a8c8_mxfp8.yaml \
  --model_type IQuest-Q1 --trust_remote_code true
```

成功标志：最后一行 `===========SUCCESS===========`，日志 `EXIT=0`。

### 3.3 自检产物

```bash
python3 script/quantize/verify_quant.py out_full_c8 --mode c8
```
检查：shard 数量、index 引用完整、`routed_experts.*` 融合布局、C8 的 `fa_v.scale`
是否正好 25 层。全过会打印 `[PASS]`。

### 3.4 起服务验证（可选但推荐）

```bash
bash /mnt/sfs_turbo/work/liuhaiyang/PD/serve_c8.sh     # 端口 8008
# 或 v1: PD/serve_single.sh（端口 8007）
curl -s http://127.0.0.1:8008/v1/models
```
C8 必须带 `--kv-cache-dtype mxfp8 --kv-cache-dtype-skip-layers sliding_window`。

## 4. 产物结构

```
out_full_c8/
  config.json                          # 原始 config
  quant_model_description.json         # 量化描述（逐 key 的 quant type）
  quant_model_weights.safetensors.index.json
  quant_model_weights-000NN-of-00088.safetensors
  tokenizer.json / tokenizer_config.json / generation_config.json ...
```

`quant_model_description.json` 里 C8 的关键键：

| key | 值 | 说明 |
|---|---|---|
| `model.layers.N.self_attn.quant_type` | `QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL` | 25 个全注意力层各一个 |
| `model.layers.N.self_attn.fa_v.scale` | `FAQuant` | uint8 E8M0，形状 `[8*128]` |
| `model.layers.N.self_attn.fa_v.offset` | `FAQuant` | 加载时被 loader 丢弃（无 consumer） |
| `fa_quant_type` | `FAKQuant` | 与 MXFP8 KV 并存时 loader 会自动 stand down |

官方参考权重用的是「模型级 `kv_cache_type` + 无 offset」的老格式；新版 loader 两种都吃，
逐层 `quant_type` 是首选格式，不要为了对齐参考产物去改。

## 5. 四个坑（改适配器时踩过的，回归时先看这里）

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
   改完用 `tests/test_moe_topk_equiv.py` 验等价性（原始实现 vs top-k 派发，逐位比较）。

## 6. 改了哪些文件（推 fork 用）

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

`msmodelslim_push/` 已经按仓库相对路径排好了，也可以整目录 rsync 覆盖。

## 7. 常见失败对照

| 报错 | 原因 | 处理 |
|---|---|---|
| `expected m1 and m2 to have the same dtype` | 坑 1 / 坑 2 | 确认适配器是修补版；检查 shell dtype 断言 |
| `MoE fused weights are not available` | 坑 3 | 确认 `_apply` 只在 meta 时丢弃 |
| 单层校准 > 10 分钟 | 坑 4（稠密专家） | 确认走 top-k 派发分支 |
| `Checkpoint key not found` | 权重目录不是 IQuest-Q1 原始 BF16 | 用 `/mnt/sfs_turbo/models/IQuest-Q1` |
| C8 起服务 `KeyError: v_cache_scale` | 给滑窗层也写了 `fa_v.scale` | 配方 include 列表只能是 25 个全注意力层 |
| 加载后直接回退 BF16 / 精度崩 | `--quantization ascend` 没加，或没带 `--kv-cache-dtype mxfp8` | 见 3.4 |
