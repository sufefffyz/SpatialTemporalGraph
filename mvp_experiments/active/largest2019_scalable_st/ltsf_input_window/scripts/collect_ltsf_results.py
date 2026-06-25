#!/usr/bin/env python3
"""Collect LargeST-LTSF smoke/pilot metrics from BasicTS and BiST outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


BASICTS_NAME_RE = re.compile(r"(?P<dataset>SD|GBA|GLA)(?:_\d+)?_L(?P<input>\d+)_H(?P<horizon>\d+)")
BIST_NAME_RE = re.compile(r"BiST_(?P<dataset>SD|GBA|GLA)_L(?P<input>\d+)_H(?P<horizon>\d+)")


def max_memory_mib(mem_path: Path) -> str:
    if not mem_path.exists():
        return ""
    values = []
    for line in mem_path.read_text(encoding="utf-8", errors="ignore").splitlines()[1:]:
        parts = line.split(",")
        if len(parts) >= 2 and parts[1].strip().isdigit():
            values.append(int(parts[1].strip()))
    return str(max(values)) if values else ""


def wall_seconds(log_path: Path) -> str:
    if not log_path.exists():
        return ""
    match = re.search(r"WALL_SECONDS=(\d+)", log_path.read_text(encoding="utf-8", errors="ignore"))
    return match.group(1) if match else ""


def flatten_basic_metrics(metrics: dict) -> tuple[str, str, str]:
    overall = metrics.get("overall", metrics)
    return (
        str(overall.get("MAE", "")),
        str(overall.get("RMSE", "")),
        str(overall.get("MAPE", "")),
    )


def collect_basicts(exp_dir: Path, phase: str) -> list[dict]:
    rows = []
    checkpoint_root = Path("BasicTS") / "checkpoints" / "LargeSTLTSF"
    metrics_files = sorted(checkpoint_root.glob(f"*/*_{phase}*/**/test_metrics.json"))
    for metrics_path in metrics_files:
        try:
            rel_parts = metrics_path.relative_to(checkpoint_root).parts
        except ValueError:
            rel_parts = metrics_path.parts
        if len(rel_parts) < 3:
            continue
        model = rel_parts[0]
        exp_name = rel_parts[1]
        match = BASICTS_NAME_RE.search(exp_name)
        if not match:
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        mae, rmse, mape = flatten_basic_metrics(metrics)
        dataset = match.group("dataset")
        input_len = match.group("input")
        horizon = match.group("horizon")
        stem = f"{model}_{dataset}_L{input_len}_H{horizon}_ltsf_{phase}"
        log_path = exp_dir / "outputs" / "logs" / "basicts" / phase / f"{stem}.log"
        mem_path = exp_dir / "outputs" / "gpu_memory" / "basicts" / phase / f"{stem}.csv"
        rows.append(
            {
                "framework": "BasicTS",
                "model": model,
                "dataset": dataset,
                "input_len": input_len,
                "horizon": horizon,
                "phase": phase,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "wall_seconds": wall_seconds(log_path),
                "peak_memory_mib": max_memory_mib(mem_path),
                "artifact": str(metrics_path),
            }
        )
    return rows


def parse_bist_log(log_path: Path) -> tuple[str, str, str]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    candidates = [
        r"Average Test.*?MAE[:\s]+([0-9.]+).*?RMSE[:\s]+([0-9.]+).*?MAPE[:\s]+([0-9.]+)",
        r"Average Test.*?([0-9.]+)\s*/\s*([0-9.]+)\s*/\s*([0-9.]+)",
        r"Test.*?MAE[:\s]+([0-9.]+).*?RMSE[:\s]+([0-9.]+).*?MAPE[:\s]+([0-9.]+)",
    ]
    for pattern in candidates:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if matches:
            return tuple(matches[-1])
    return "", "", ""


def collect_bist(exp_dir: Path, phase: str) -> list[dict]:
    rows = []
    log_dir = exp_dir / "outputs" / "logs" / "bist" / phase
    for log_path in sorted(log_dir.glob("BiST_*_L*_H*_*.log")):
        match = BIST_NAME_RE.search(log_path.name)
        if not match:
            continue
        mae, rmse, mape = parse_bist_log(log_path)
        if not (mae and rmse and mape):
            continue
        mem_path = exp_dir / "outputs" / "gpu_memory" / "bist" / phase / f"{log_path.stem}.csv"
        rows.append(
            {
                "framework": "LargeST-BiST",
                "model": "BiST",
                "dataset": match.group("dataset"),
                "input_len": match.group("input"),
                "horizon": match.group("horizon"),
                "phase": phase,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "wall_seconds": wall_seconds(log_path),
                "peak_memory_mib": max_memory_mib(mem_path),
                "artifact": str(log_path),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="pilot")
    parser.add_argument("--exp-dir", default="mvp_experiments/active/largest2019_scalable_st/ltsf_input_window")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    rows = collect_basicts(exp_dir, args.phase) + collect_bist(exp_dir, args.phase)
    out_path = Path(args.out) if args.out else exp_dir / "outputs" / "summaries" / args.phase / "summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "framework",
        "model",
        "dataset",
        "input_len",
        "horizon",
        "phase",
        "mae",
        "rmse",
        "mape",
        "wall_seconds",
        "peak_memory_mib",
        "artifact",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path} with {len(rows)} rows")


if __name__ == "__main__":
    main()
