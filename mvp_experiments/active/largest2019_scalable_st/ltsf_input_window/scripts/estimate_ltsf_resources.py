#!/usr/bin/env python3
"""Estimate rough LargeST-LTSF memory and complexity for input-window probes."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


DATASETS = {
    "SD": {"nodes": 716, "features": 3, "tpd": 96},
    "GBA": {"nodes": 2352, "features": 3, "tpd": 96},
    "GLA": {"nodes": 3834, "features": 3, "tpd": 96},
    "CA": {"nodes": 8600, "features": 3, "tpd": 96},
}


@dataclass(frozen=True)
class ModelSpec:
    batch_size: int
    formula: str
    scalable_full: bool
    note: str


MODELS = {
    "STID": ModelSpec(
        64,
        "O(B*N*(L*C*d + R*d^2 + H*d))",
        True,
        "Per-node temporal compression plus node/time embeddings; no pairwise spatial interaction.",
    ),
    "DLinear": ModelSpec(
        64,
        "O(B*N*L*H)",
        True,
        "Target-only linear trend/seasonal projection; cheap but low-capacity sanity baseline.",
    ),
    "CycleNet": ModelSpec(
        64,
        "O(B*N*(L*d + H*d))",
        True,
        "Daily cycle template plus shared per-node MLP; tests periodic compression.",
    ),
    "TimeMixer": ModelSpec(
        16,
        "O(B*N*d*sum_i L/2^i + B*N*d*H)",
        True,
        "Multiscale temporal mixing with channel independence; more costly than DLinear/CycleNet.",
    ),
    "PatchTST": ModelSpec(
        16,
        "O(B*N*R*(P^2*d + P*d*d_ff)), P≈L/stride",
        False,
        "Patch attention is per node but attention scores grow with patch count; SD/probe first.",
    ),
    "iTransformer": ModelSpec(
        4,
        "O(B*N^2*d + B*N*d*H)",
        False,
        "Variable-token attention is quadratic in nodes; only useful as small-scale/OOM evidence.",
    ),
    "BiST": ModelSpec(
        64,
        "O(B*N*(L*d + H*d) + B*N^2*d*rp)",
        True,
        "Base temporal MLP plus residual propagation; adaptive dense kernel can dominate at large N.",
    ),
}


def mib(num_bytes: float) -> float:
    return num_bytes / 1024 / 1024


def estimate_row(dataset: str, model: str, input_len: int, horizon: int) -> dict:
    ds = DATASETS[dataset]
    spec = MODELS[model]
    batch = spec.batch_size
    nodes = ds["nodes"]
    features = ds["features"]
    raw_batch_mib = mib(4 * batch * nodes * (input_len * features + horizon))

    hidden_dim = 128
    if model == "STID":
        activation_mib = mib(4 * batch * nodes * (input_len * features + hidden_dim + horizon))
        risk = "low" if dataset in {"SD", "GBA"} else "medium"
    elif model == "DLinear":
        activation_mib = raw_batch_mib
        risk = "low"
    elif model == "CycleNet":
        activation_mib = mib(4 * batch * nodes * (input_len + horizon + 512))
        risk = "low" if dataset != "CA" else "medium"
    elif model == "TimeMixer":
        activation_mib = mib(4 * batch * nodes * (input_len * 16 + horizon))
        risk = "medium" if dataset in {"SD", "GBA"} else "high"
    elif model == "PatchTST":
        patch_len, stride, layers, heads, d_model = 32, 16, 3, 16, 128
        patch_num = max(1, int((input_len - patch_len) / stride + 1) + 1)
        attention_mib = mib(4 * batch * nodes * layers * heads * patch_num * patch_num)
        hidden_mib = mib(4 * batch * nodes * patch_num * d_model)
        activation_mib = raw_batch_mib + attention_mib + hidden_mib
        risk = "medium" if dataset == "SD" else "high"
    elif model == "iTransformer":
        activation_mib = raw_batch_mib + mib(4 * batch * nodes * nodes * 8)
        risk = "high"
    elif model == "BiST":
        dense_kernel_mib = mib(4 * nodes * nodes)
        prop_workspace_mib = mib(4 * batch * nodes * hidden_dim)
        activation_mib = raw_batch_mib + dense_kernel_mib + prop_workspace_mib
        risk = "medium" if dataset == "SD" else "high"
    else:
        activation_mib = raw_batch_mib
        risk = "unknown"

    return {
        "dataset": dataset,
        "model": model,
        "input_len": input_len,
        "horizon": horizon,
        "batch_size": batch,
        "nodes": nodes,
        "raw_batch_mib": f"{raw_batch_mib:.1f}",
        "rough_activation_mib": f"{activation_mib:.1f}",
        "risk": risk,
        "full_candidate": str(spec.scalable_full).lower(),
        "complexity": spec.formula,
        "note": spec.note,
    }


def write_markdown(rows: list[dict], path: Path) -> None:
    header = [
        "dataset",
        "model",
        "input_len",
        "horizon",
        "batch_size",
        "raw_batch_mib",
        "rough_activation_mib",
        "risk",
        "full_candidate",
    ]
    lines = [
        "# LargeST-LTSF Scalability Table",
        "",
        "Memory numbers are first-order estimates for A6000 48GB triage, not profiler truth.",
        "",
        "|" + "|".join(header) + "|",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for row in rows:
        lines.append("|" + "|".join(str(row[k]) for k in header) + "|")
    lines.extend(
        [
            "",
            "## Complexity Notes",
            "",
        ]
    )
    for model, spec in MODELS.items():
        lines.append(f"- `{model}`: `{spec.formula}`. {spec.note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_csv_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="SD,GBA,GLA")
    parser.add_argument("--models", default="STID,DLinear,CycleNet,TimeMixer,PatchTST,BiST")
    parser.add_argument("--input-lengths", default="96,192,336,672")
    parser.add_argument("--horizon", type=int, default=672)
    parser.add_argument(
        "--out-dir",
        default="mvp_experiments/active/largest2019_scalable_st/ltsf_input_window/outputs/resource_estimates",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in parse_csv_list(args.datasets):
        if dataset not in DATASETS:
            raise KeyError(f"Unknown dataset: {dataset}")
        for model in parse_csv_list(args.models):
            if model not in MODELS:
                raise KeyError(f"Unknown model: {model}")
            for input_len in [int(x) for x in parse_csv_list(args.input_lengths)]:
                rows.append(estimate_row(dataset, model, input_len, args.horizon))

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = out_dir / "scalability_table.md"
    write_markdown(rows, md_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
