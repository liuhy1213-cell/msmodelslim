# script/ —— IQuest-Q1 量化与评测操作集

两条主线：**产出一版权重** 和 **证明这版权重和 BF16 对齐**。

```
script/
  quantize/      # BF16 -> MXFP8 量化：冒烟、一条命令出成品、产物自检
  eval/          # IFBench 固定口径评测：跳板机一条命令投到任意节点
msmodelslim_push/  # 可直接推到 fork 的适配器改动（patch + 按路径排好的文件）
```

## 最短路径

```bash
# 1) 量化（CPU，约 56 分钟；一条命令 = 量化 + C8 包整理 + K 中心化）
bash script/quantize/smoke.sh c8         # 1 分钟冒烟，先确认环境没坏
bash script/quantize/quantize.sh c8      # 产出 out_full_c8
python3 script/quantize/verify_quant.py out_full_c8 --mode c8

# 2) 起服务（见 quantize/README 3.4），然后评测
bash script/eval/run_remote.sh 10.222.44.20 iq1120p IQuest-Q1 8008 \
     /mnt/sfs_turbo/work/liuhaiyang/c8own/eval_c8 300 -l c8
```

## 当前生产口径（2026-11-20）

| 项 | 口径 |
|---|---|
| 权重 | 88 层全 C8（KV 量化覆盖全部层） |
| V 尺度 | 只有 25 个全注意力层带 `fa_v.scale`；63 个滑窗层用中性 E8M0=127 |
| K 中心化 | 只有 25 个全注意力层，`kv_center/{layer}.pt` 存整表（不按 TP 切片） |
| 部署 | `--kv-cache-dtype mxfp8`，**不带** `--kv-cache-dtype-skip-layers` |
| 环境变量 | 无（不设任何 `C8_*`），不带 p_scale |
| 成品包 | `/mnt/sfs_turbo/models/IQuest-Q1-W8A8C8-mxfp8-allc8-kc25` |

改动细节与踩坑见 `quantize/README.md`。

## 历史对齐结论（v1，2026-11-12，同一口径 temp1.0 / max_tokens 16384 / batch64）

| 运行 | prompt_s | inst_s | prompt_l | inst_l | 截断 |
|---|---|---|---|---|---|
| BF16（客户基线参考） | 71.33 | 72.09 | 74.33 | 74.71 | 10.0% |
| **v1（我们量化，KV 保持 BF16）** | **74.33** | **75.87** | **77.67** | **78.78** | 7.3% |
| 官方 C8 参考跑（zuoqian，batch32） | 69.67 | 71.22 | 73.33 | 74.42 | 8.7% |

v1 与 BF16 四项同级（差值 2~4pt，方向一致偏高），且明显高于官方 C8 那次参考跑——
说明差距主要来自部署配置与单次采样噪声，不是量化本身。

这张表是 **v1（KV 不量化）** 的历史数据，用作适配器正确性的基线；C8 生产口径的
对齐数据按数据集分别记录在评测结果目录，判读方法见 `eval/README.md` 第 4 节。
