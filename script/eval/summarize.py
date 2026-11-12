#!/usr/bin/env python3
"""从 evalscope 的 work_dir 里提取 IFBench 四项指标，输出表格 + 单行 CSV。

用法:
    python3 summarize.py <work_dir|run_dir> [--csv-out FILE]

work_dir 可以指向 run 的父目录（自动取最新子目录），也可以直接指向
`<work_dir>/<timestamp>` 这一层。
"""
import argparse
import glob
import json
import os
import sys

METRICS = ["prompt_level_strict", "inst_level_strict", "prompt_level_loose", "inst_level_loose"]


def find_run_dir(path: str) -> str:
    if glob.glob(os.path.join(path, "reports", "*", "*.json")):
        return path
    subs = [d for d in glob.glob(os.path.join(path, "*")) if os.path.isdir(d)]
    subs = [d for d in subs if glob.glob(os.path.join(d, "reports", "*", "*.json"))]
    if not subs:
        sys.exit(f"找不到 evalscope run 目录: {path}")
    return max(subs, key=os.path.getmtime)


def load_report(run_dir: str) -> dict:
    files = glob.glob(os.path.join(run_dir, "reports", "*", "*.json"))
    if not files:
        sys.exit(f"没有 report json: {run_dir}/reports/*/*.json")
    data = json.load(open(files[0]))
    return data


def truncation_stats(run_dir: str, max_tokens: int):
    """返回 (总数, 输出打满 max_tokens 的条数, 平均输出 token)。"""
    files = glob.glob(os.path.join(run_dir, "predictions", "*", "*.jsonl"))
    if not files:
        return None
    total = 0
    burned = 0
    out_tokens = 0
    for f in files:
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            usage = (rec.get("model_output") or {}).get("usage") or {}
            n = usage.get("output_tokens") or 0
            total += 1
            out_tokens += n
            if max_tokens and n >= max_tokens:
                burned += 1
    if not total:
        return None
    return total, burned, out_tokens / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work_dir")
    ap.add_argument("--csv-out", default=None)
    ap.add_argument("--max-tokens", type=int, default=16384,
                    help="用于统计截断，须与本次评测的 generation max_tokens 一致")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    run_dir = find_run_dir(args.work_dir)
    report = load_report(run_dir)
    scores = {}
    nums = {}
    for m in report.get("metrics", []):
        name = (m.get("identity") or {}).get("name")
        if name in METRICS:
            scores[name] = m.get("score")
            nums[name] = m.get("num")

    model = report.get("model_name", "?")
    ts = os.path.basename(run_dir.rstrip("/"))

    print(f"model      : {model}")
    print(f"run_dir    : {run_dir}")
    print(f"label      : {args.label or '-'}")
    print()
    print(f"{'metric':<22}{'num':>6}{'score':>9}")
    print("-" * 37)
    for name in METRICS:
        s = scores.get(name)
        n = nums.get(name, "")
        print(f"{name:<22}{n:>6}{(s * 100 if s is not None else float('nan')):>8.2f}%" if s is not None
              else f"{name:<22}{n:>6}{'N/A':>9}")

    stats = truncation_stats(run_dir, args.max_tokens)
    if stats:
        total, burned, avg = stats
        print()
        print(f"截断(>= {args.max_tokens} tok): {burned}/{total} = {burned / total * 100:.1f}%"
              f"   平均输出 {avg:.0f} tok")

    row = [ts, args.label, model] + [
        f"{scores[n] * 100:.2f}" if scores.get(n) is not None else "" for n in METRICS
    ]
    if stats:
        row += [str(stats[1]), f"{stats[2]:.0f}"]
    line = ",".join(row)
    if args.csv_out:
        new_file = not os.path.exists(args.csv_out)
        with open(args.csv_out, "a") as fh:
            if new_file:
                fh.write("run,label,model," + ",".join(METRICS) + ",truncated,avg_out_tokens\n")
            fh.write(line + "\n")
    print()
    print("CSV: " + line)


if __name__ == "__main__":
    main()
