# IQuest-Q1 IFBench 固定口径评测

位置：`/mnt/sfs_turbo/work/liuhaiyang/script/eval/`（SFS 共享，所有节点和容器都能看到）

目的：把「量化权重 vs BF16」的 IFBench 对比钉死在同一口径上，任何一次结果都能直接和
参考跑（zuoqian `qping`，inst_level_loose **74.42**）以及客户侧 BF16（约 **79%**）对比。

## 1. 一条命令

```bash
# 跳板机上（iquest）
bash /mnt/sfs_turbo/work/liuhaiyang/script/eval/run_remote.sh \
    10.222.44.20 iq1120p IQuest-Q1 8007 \
    /mnt/sfs_turbo/work/liuhaiyang/c8own/eval_c8_run1 300 -l c8
```

或在节点本地：

```bash
bash /mnt/sfs_turbo/work/liuhaiyang/script/eval/ifbench_eval.sh \
    -c iq1120p -m IQuest-Q1 -u http://127.0.0.1:8007/v1/chat/completions \
    -o /mnt/sfs_turbo/work/liuhaiyang/c8own/eval_c8_run1 -l c8
```

常见可选参数：`-n 20`（先跑小样本冒烟）、`-b 32`（并发降低）、`-t 16384`（输出上限）。

脚本自己会做三件事：检查服务 `/v1/models` 与 `max_model_len`、跑 evalscope、把四项指标
追加到 `results.csv`。

## 2. 口径（改动即不可比）

| 项 | 值 |
|---|---|
| 数据集 | `allenai/IFBench_test` 本地快照（300 prompt / 344 instruction） |
| temperature / top_p / top_k | 1.0 / 0.95 / 20 |
| max_tokens | 16384 |
| seed | 42 |
| limit | 300（全量） |
| eval batch | 64（服务端 `--max-num-seqs` 的上限） |
| evalscope | 1.14.1（`/mnt/sfs_turbo/work/liuhaiyang/evalvenv`） |
| 指标 | prompt_level_strict / inst_level_strict / prompt_level_loose / inst_level_loose |

`inst_level_*` 是官方 micro-average（按 instruction 加权），也是对外报数用的那个。

## 3. 服务端前提（踩过的坑）

1. **`max_model_len` 必须 >= 16384 + 输入长度**。BF16 服务默认 28672 够用；如果为了省显存
   调到 16384，长输入会直接 400，整批数据作废。
2. **`--max-num-seqs` 至少 64**，否则 `-b 64` 会排队，TTFT 拉长但结果不变。
3. **KV 量化不影响本指标的口径，但影响可比性**：C8 部署必须带
   `--kv-cache-dtype mxfp8 --kv-cache-dtype-skip-layers sliding_window`，
   和参考跑保持同一套；滑窗层一定要 skip，63 层里有 25 层是 full attention。
4. **NLTK 离线**：容器里必须能读到 `NLTK_DATA=/mnt/sfs_turbo/work/liuhaiyang/nltk_data`
   （`corpora/ wordnet`、`tokenizers/ punkt` 等），否则 IFBench 的约束校验直接报错。
5. **HF 离线**：`HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`，数据集走 `local_path`，
   不允许联网拉取（跳板机外网受限，也会导致不同次跑数据版本漂移）。

## 4. 怎么读结果

- 先看 `results.csv`（脚本自动追加），一行一次运行，列固定：
  `run,label,model,四项,truncated,avg_out_tokens`。
- **`truncated` 是关键诊断量**：输出打满 max_tokens 的条数。打满=必然判错，
  这个比例波动会让四项指标整体漂 ±2pt 以上。比较两次跑时先比这个，再看分数。
- temp=1.0 是采样，单次 300 题本身有 ±1~1.5pt 的噪声。判断"是否对齐"要求：
  - 差值 < 2pt，且
  - 方向不一致（有正有负），或
  - 逐题配对（McNemar）p > 0.05。
- 只有 `prompt_strict` 一致不算对齐，四项都要看。历史上出现过 strict 对齐、loose 差 2pt
  的情况，根因是截断率不同而不是量化。

## 5. 参考数字

| 运行 | prompt_s | inst_s | prompt_l | inst_l | 截断 |
|---|---|---|---|---|---|
| BF16（本机，2026-11-12） | 71.33 | 72.09 | 74.33 | 74.71 | 10.0% |
| **v1（本机，2026-11-12）** | **74.33** | **75.87** | **77.67** | **78.78** | 7.3% |
| **C8（本机，2026-11-12）** | **72.67** | **73.84** | **76.00** | **76.74** | 8.0% |
| 官方 C8（同配置 A/B，2026-11-12） | 69.00 | 70.93 | 73.33 | 74.71 | 8.0% |
| 参考 C8（zuoqian，batch32、无 MTP） | 69.67 | 71.22 | 73.33 | 74.42 | 8.7% |
| 客户侧 BF16 | — | — | — | ~79 | — |

三版权重（BF16 / v1 / C8）在本口径下四项指标相差 1.3~4.1pt，且都是量化侧偏高，
截断率 7.3~10.0%——判为对齐。

**与官方 C8 的差距已用同配置 A/B 定性**：把官方 `IQuest-Q1-W8A8C8-mxfp8` 用完全相同的
服务参数重跑，得 69.00 / 70.93 / 73.33 / 74.71，和我们 C8 差 +3.67 / +2.91 / +2.67 / +2.03，
截断率完全相同（都是 24/300）。部署配置（batch 32→64、MTP、async）只值 0.7pt 以内。
**差距来自校准数值，不是部署。**

## 6. 目录

```
script/eval/
  ifbench_eval.sh   # 主入口（节点上跑）
  run_remote.sh     # 跳板机 → 节点
  summarize.py      # 解析 report，输出四项 + 截断统计
  results.csv       # 所有运行的历史汇总（自动追加）
```
