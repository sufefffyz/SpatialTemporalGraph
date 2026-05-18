#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


METRICS = [
    ("avg_low_MAE", "Low MAE", "lower"),
    ("avg_high_MAE", "High MAE", "lower"),
    ("avg_high_over_full_MAE", "High/Full MAE", "lower"),
    ("avg_high_W1", "High W1", "lower"),
    ("abs_avg_high_temporal_lag1_corr", "Abs High Lag1", "lower"),
    ("avg_high_edge_corr", "High Edge Corr", "lower"),
    ("avg_high_dirichlet", "High Dirichlet", "lower"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize baseline ranking changes between decomposition methods.")
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-method", default="moving_average")
    parser.add_argument("--comparison-method", default="fft_lowpass")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fp:
        return [dict(row) for row in csv.DictReader(fp)]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(row: dict, key: str) -> float:
    if key == "abs_avg_high_temporal_lag1_corr":
        return abs(to_float(row, "avg_high_temporal_lag1_corr"))
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def fmt(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.4f}"


def rank_values(values: dict[str, float], direction: str) -> dict[str, float]:
    finite = [(system, value) for system, value in values.items() if math.isfinite(value)]
    reverse = direction == "higher"
    finite.sort(key=lambda item: item[1], reverse=reverse)
    ranks: dict[str, float] = {}
    idx = 0
    while idx < len(finite):
        j = idx + 1
        while j < len(finite) and finite[j][1] == finite[idx][1]:
            j += 1
        avg_rank = (idx + 1 + j) / 2.0
        for k in range(idx, j):
            ranks[finite[k][0]] = avg_rank
        idx = j
    for system in values:
        ranks.setdefault(system, float("nan"))
    return ranks


def kendall_tau(rank_a: dict[str, float], rank_b: dict[str, float]) -> float:
    systems = [system for system in rank_a if math.isfinite(rank_a[system]) and math.isfinite(rank_b.get(system, float("nan")))]
    concordant = 0
    discordant = 0
    for i, left in enumerate(systems):
        for right in systems[i + 1 :]:
            da = rank_a[left] - rank_a[right]
            db = rank_b[left] - rank_b[right]
            if da == 0 or db == 0:
                continue
            if da * db > 0:
                concordant += 1
            else:
                discordant += 1
    denom = concordant + discordant
    return (concordant - discordant) / denom if denom else float("nan")


def order_string(ranks: dict[str, float]) -> str:
    ordered = sorted(ranks.items(), key=lambda item: (item[1], item[0]))
    return " > ".join(f"{int(rank)}:{system}" if rank.is_integer() else f"{rank:.1f}:{system}" for system, rank in ordered)


def main() -> int:
    args = parse_args()
    rows = read_csv(args.summary_csv.expanduser().resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_method = {(row["decomposition_method"], row["system"]): row for row in rows}
    systems = sorted({row["system"] for row in rows})
    ranking_rows: list[dict] = []
    metric_rows: list[dict] = []

    for key, label, direction in METRICS:
        baseline_values = {
            system: to_float(by_method.get((args.baseline_method, system), {}), key) for system in systems
        }
        comparison_values = {
            system: to_float(by_method.get((args.comparison_method, system), {}), key) for system in systems
        }
        baseline_ranks = rank_values(baseline_values, direction)
        comparison_ranks = rank_values(comparison_values, direction)
        tau = kendall_tau(baseline_ranks, comparison_ranks)
        changed = 0
        largest_shift = 0.0
        for system in systems:
            rank_delta = comparison_ranks[system] - baseline_ranks[system]
            changed += int(rank_delta != 0)
            largest_shift = max(largest_shift, abs(rank_delta))
            ranking_rows.append(
                {
                    "metric": key,
                    "metric_label": label,
                    "system": system,
                    "baseline_method": args.baseline_method,
                    "baseline_value": baseline_values[system],
                    "baseline_rank": baseline_ranks[system],
                    "comparison_method": args.comparison_method,
                    "comparison_value": comparison_values[system],
                    "comparison_rank": comparison_ranks[system],
                    "rank_delta": rank_delta,
                }
            )
        metric_rows.append(
            {
                "metric": key,
                "metric_label": label,
                "baseline_method": args.baseline_method,
                "baseline_order": order_string(baseline_ranks),
                "comparison_method": args.comparison_method,
                "comparison_order": order_string(comparison_ranks),
                "kendall_tau": tau,
                "changed_systems": changed,
                "largest_abs_rank_shift": largest_shift,
            }
        )

    write_csv(args.output_dir / "decomposition_rank_changes.csv", ranking_rows)
    write_csv(args.output_dir / "decomposition_rank_orders.csv", metric_rows)

    lines = [
        "# Decomposition Ranking Change Summary",
        "",
        f"- Baseline method: `{args.baseline_method}`",
        f"- Comparison method: `{args.comparison_method}`",
        f"- Source: `{args.summary_csv}`",
        "",
        "Lower rank is better for every metric in this report. `Abs High Lag1` uses absolute lag-1 residual correlation.",
        "",
        "| Metric | Kendall tau | Changed Models | Largest Shift | Moving-Average Order | FFT Low-Pass Order |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in metric_rows:
        lines.append(
            "| {metric} | {tau} | {changed} | {shift} | {base} | {comp} |".format(
                metric=row["metric_label"],
                tau=fmt(float(row["kendall_tau"])),
                changed=row["changed_systems"],
                shift=fmt(float(row["largest_abs_rank_shift"])),
                base=row["baseline_order"],
                comp=row["comparison_order"],
            )
        )
    lines.append("")
    (args.output_dir / "decomposition_ranking_report.md").write_text("\n".join(lines), encoding="utf-8")
    print({"output_dir": str(args.output_dir), "metrics": len(METRICS), "systems": len(systems)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
