#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COMPONENTS = ["full", "low", "high"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot and summarize decoupled ST diagnostic CSV outputs.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--plot-format", choices=["png", "pdf", "both"], default="png")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as fp:
        return [dict(row) for row in csv.DictReader(fp)]


def to_float(row: dict, key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def to_int(row: dict, key: str) -> int:
    value = row.get(key, "")
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


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


def fmt(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{value:.4f}"


def ordered_systems(component_rows: list[dict]) -> list[str]:
    full_rows = [row for row in component_rows if row.get("component") == "full"]
    values = {}
    for row in full_rows:
        system = row["system"]
        values.setdefault(system, []).append(to_float(row, "MAE"))
    return [system for system, _ in sorted(values.items(), key=lambda item: float(np.nanmean(item[1])))]


def horizons(rows: list[dict]) -> list[int]:
    return sorted({to_int(row, "horizon") for row in rows if to_int(row, "horizon") > 0})


def component_lookup(rows: list[dict], metric: str) -> dict[tuple[str, int, str], float]:
    return {(row["system"], to_int(row, "horizon"), row["component"]): to_float(row, metric) for row in rows}


def peak_lookup(rows: list[dict], metric: str) -> dict[tuple[str, int, str], float]:
    return {(row["system"], to_int(row, "horizon"), row["window_type"]): to_float(row, metric) for row in rows}


def residual_lookup(rows: list[dict], metric: str) -> dict[tuple[str, int, str], float]:
    return {(row["system"], to_int(row, "horizon"), row["component"]): to_float(row, metric) for row in rows}


def average(values: list[float]) -> float:
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    return float(np.mean(arr)) if arr.size else float("nan")


def build_summary_rows(component_rows: list[dict], peak_rows: list[dict], residual_rows: list[dict], spatial_rows: list[dict]) -> list[dict]:
    systems = ordered_systems(component_rows)
    hs = horizons(component_rows)
    comp_mae = component_lookup(component_rows, "MAE")
    comp_w1 = component_lookup(component_rows, "W1")
    peak_mae = peak_lookup(peak_rows, "MAE")
    peak_w1 = peak_lookup(peak_rows, "W1")
    resid_lag = residual_lookup(residual_rows, "temporal_lag1_corr")
    resid_under = residual_lookup(residual_rows, "under_prediction_rate")
    spatial_corr = residual_lookup(spatial_rows, "edge_residual_corr")
    spatial_dir = residual_lookup(spatial_rows, "residual_dirichlet")

    rows = []
    for system in systems:
        full_mae = average([comp_mae.get((system, h, "full"), float("nan")) for h in hs])
        low_mae = average([comp_mae.get((system, h, "low"), float("nan")) for h in hs])
        high_mae = average([comp_mae.get((system, h, "high"), float("nan")) for h in hs])
        normal_mae = average([peak_mae.get((system, h, "normal"), float("nan")) for h in hs])
        peak_key = [key for key in {row["window_type"] for row in peak_rows} if key.startswith("peak_q")]
        peak_name = peak_key[0] if peak_key else "peak_q0.90"
        peak_mae_avg = average([peak_mae.get((system, h, peak_name), float("nan")) for h in hs])
        rows.append(
            {
                "system": system,
                "avg_full_MAE": full_mae,
                "avg_low_MAE": low_mae,
                "avg_high_MAE": high_mae,
                "avg_high_over_full_MAE": high_mae / full_mae if full_mae and np.isfinite(full_mae) else float("nan"),
                "avg_full_W1": average([comp_w1.get((system, h, "full"), float("nan")) for h in hs]),
                "avg_low_W1": average([comp_w1.get((system, h, "low"), float("nan")) for h in hs]),
                "avg_high_W1": average([comp_w1.get((system, h, "high"), float("nan")) for h in hs]),
                "avg_normal_MAE": normal_mae,
                "avg_peak_MAE": peak_mae_avg,
                "avg_peak_over_normal_MAE": peak_mae_avg / normal_mae if normal_mae and np.isfinite(normal_mae) else float("nan"),
                "avg_peak_W1": average([peak_w1.get((system, h, peak_name), float("nan")) for h in hs]),
                "avg_high_temporal_lag1_corr": average([resid_lag.get((system, h, "high"), float("nan")) for h in hs]),
                "avg_high_under_prediction_rate": average([resid_under.get((system, h, "high"), float("nan")) for h in hs]),
                "avg_high_edge_corr": average([spatial_corr.get((system, h, "high"), float("nan")) for h in hs]),
                "avg_high_dirichlet": average([spatial_dir.get((system, h, "high"), float("nan")) for h in hs]),
            }
        )
    return rows


def save_fig(fig: plt.Figure, output_dir: Path, stem: str, plot_format: str) -> None:
    formats = ["png", "pdf"] if plot_format == "both" else [plot_format]
    for suffix in formats:
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight", pad_inches=0.04)


def matrix_for(rows: list[dict], metric: str, systems: list[str], hs: list[int], component: str) -> np.ndarray:
    lookup = component_lookup(rows, metric)
    return np.asarray([[lookup.get((system, h, component), float("nan")) for h in hs] for system in systems], dtype=float)


def plot_component_heatmaps(rows: list[dict], systems: list[str], hs: list[int], output_dir: Path, plot_format: str) -> None:
    for metric in ["MAE", "W1"]:
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2), sharex=True, sharey=True)
        for ax, component in zip(axes, COMPONENTS):
            matrix = matrix_for(rows, metric, systems, hs, component)
            im = ax.imshow(matrix, aspect="auto", cmap="viridis")
            ax.set_title(f"{component} {metric}")
            ax.set_xticks(np.arange(len(hs)), [f"H{h}" for h in hs], rotation=45)
            ax.set_yticks(np.arange(len(systems)), systems)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        fig.tight_layout()
        save_fig(fig, output_dir, f"component_{metric.lower()}_heatmaps", plot_format)
        plt.close(fig)


def plot_component_ratio(rows: list[dict], systems: list[str], hs: list[int], output_dir: Path, plot_format: str) -> None:
    lookup = component_lookup(rows, "MAE")
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    colors = plt.cm.tab10.colors
    for idx, system in enumerate(systems):
        values = []
        for h in hs:
            high = lookup.get((system, h, "high"), float("nan"))
            full = lookup.get((system, h, "full"), float("nan"))
            values.append(high / full if full and np.isfinite(full) else float("nan"))
        ax.plot(hs, values, marker="o", linewidth=1.8, markersize=3.5, label=system, color=colors[idx % len(colors)])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("High-component MAE / full MAE")
    ax.set_xticks(hs)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=7, frameon=False)
    fig.tight_layout()
    save_fig(fig, output_dir, "high_over_full_mae_by_horizon", plot_format)
    plt.close(fig)


def plot_peak(rows: list[dict], systems: list[str], hs: list[int], output_dir: Path, plot_format: str) -> None:
    lookup_mae = peak_lookup(rows, "MAE")
    lookup_w1 = peak_lookup(rows, "W1")
    peak_names = [key for key in {row["window_type"] for row in rows} if key.startswith("peak_q")]
    peak_name = peak_names[0] if peak_names else "peak_q0.90"
    colors = plt.cm.tab10.colors
    for metric_name, lookup, ylabel, stem in [
        ("ratio", lookup_mae, "Peak MAE / normal MAE", "peak_over_normal_mae_by_horizon"),
        ("w1", lookup_w1, "Peak W1", "peak_w1_by_horizon"),
    ]:
        fig, ax = plt.subplots(figsize=(8.2, 4.2))
        for idx, system in enumerate(systems):
            values = []
            for h in hs:
                peak = lookup.get((system, h, peak_name), float("nan"))
                if metric_name == "ratio":
                    normal = lookup.get((system, h, "normal"), float("nan"))
                    values.append(peak / normal if normal and np.isfinite(normal) else float("nan"))
                else:
                    values.append(peak)
            ax.plot(hs, values, marker="o", linewidth=1.8, markersize=3.5, label=system, color=colors[idx % len(colors)])
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel(ylabel)
        ax.set_xticks(hs)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(ncol=2, fontsize=7, frameon=False)
        fig.tight_layout()
        save_fig(fig, output_dir, stem, plot_format)
        plt.close(fig)


def plot_residual_heatmaps(residual_rows: list[dict], spatial_rows: list[dict], systems: list[str], hs: list[int], output_dir: Path, plot_format: str) -> None:
    panels = [
        ("temporal_lag1_corr", residual_rows, "High residual lag1 corr"),
        ("under_prediction_rate", residual_rows, "High under-prediction rate"),
        ("edge_residual_corr", spatial_rows, "High edge residual corr"),
        ("residual_dirichlet", spatial_rows, "High residual Dirichlet"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.2), sharex=True, sharey=True)
    for ax, (metric, rows, title) in zip(axes.ravel(), panels):
        lookup = residual_lookup(rows, metric)
        matrix = np.asarray([[lookup.get((system, h, "high"), float("nan")) for h in hs] for system in systems], dtype=float)
        im = ax.imshow(matrix, aspect="auto", cmap="magma")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(hs)), [f"H{h}" for h in hs], rotation=45)
        ax.set_yticks(np.arange(len(systems)), systems)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    save_fig(fig, output_dir, "high_residual_structure_heatmaps", plot_format)
    plt.close(fig)


def plot_summary_bars(summary_rows: list[dict], output_dir: Path, plot_format: str) -> None:
    metrics = [
        ("avg_high_MAE", "High MAE"),
        ("avg_peak_MAE", "Peak MAE"),
        ("avg_full_W1", "Full W1"),
        ("avg_high_temporal_lag1_corr", "High residual lag1 corr"),
    ]
    systems = [row["system"] for row in summary_rows]
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.8), sharey=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(systems)))
    for ax, (key, title) in zip(axes, metrics):
        values = [to_float(row, key) if isinstance(row[key], str) else row[key] for row in summary_rows]
        ax.barh(systems, values, color=colors)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(True, axis="x", alpha=0.25)
        if ax is not axes[0]:
            ax.tick_params(axis="y", labelleft=False)
    fig.tight_layout()
    save_fig(fig, output_dir, "decoupled_summary_bars", plot_format)
    plt.close(fig)


def markdown_summary(path: Path, summary_rows: list[dict], report: dict) -> None:
    lines = [
        "# Decoupled SD Baseline Diagnostic Report",
        "",
        f"- Input directory: `{report['input_dir']}`",
        f"- Systems: {report['num_systems']}",
        f"- Horizons: {', '.join(f'H{h}' for h in report['horizons'])}",
        "",
        "## Average Diagnostics Across Horizons",
        "",
        "| System | Full MAE | Low MAE | High MAE | High/Full | Full W1 | High W1 | Peak MAE | Peak/Normal | Peak W1 | High Lag1 | High Edge Corr | High Dirichlet |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {system} | {full} | {low} | {high} | {ratio} | {full_w1} | {high_w1} | {peak} | {peak_ratio} | {peak_w1} | {lag} | {edge} | {dirichlet} |".format(
                system=row["system"],
                full=fmt(row["avg_full_MAE"]),
                low=fmt(row["avg_low_MAE"]),
                high=fmt(row["avg_high_MAE"]),
                ratio=fmt(row["avg_high_over_full_MAE"]),
                full_w1=fmt(row["avg_full_W1"]),
                high_w1=fmt(row["avg_high_W1"]),
                peak=fmt(row["avg_peak_MAE"]),
                peak_ratio=fmt(row["avg_peak_over_normal_MAE"]),
                peak_w1=fmt(row["avg_peak_W1"]),
                lag=fmt(row["avg_high_temporal_lag1_corr"]),
                edge=fmt(row["avg_high_edge_corr"]),
                dirichlet=fmt(row["avg_high_dirichlet"]),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    component_rows = read_csv(input_dir / "component_metrics.csv")
    peak_rows = read_csv(input_dir / "peak_window_metrics.csv")
    residual_rows = read_csv(input_dir / "residual_structure_metrics.csv")
    spatial_rows = read_csv(input_dir / "spatial_residual_metrics.csv")
    if not component_rows:
        raise FileNotFoundError(f"Missing component metrics under {input_dir}")

    systems = ordered_systems(component_rows)
    hs = horizons(component_rows)
    summary_rows = build_summary_rows(component_rows, peak_rows, residual_rows, spatial_rows)
    write_csv(input_dir / "decoupled_average_summary.csv", summary_rows)
    report = {"input_dir": str(input_dir), "num_systems": len(systems), "horizons": hs}
    (input_dir / "decoupled_plot_summary.json").write_text(json.dumps(report, indent=2, default=json_default), encoding="utf-8")
    markdown_summary(input_dir / "decoupled_diagnostic_report.md", summary_rows, report)

    plot_component_heatmaps(component_rows, systems, hs, input_dir, args.plot_format)
    plot_component_ratio(component_rows, systems, hs, input_dir, args.plot_format)
    plot_peak(peak_rows, systems, hs, input_dir, args.plot_format)
    if residual_rows and spatial_rows:
        plot_residual_heatmaps(residual_rows, spatial_rows, systems, hs, input_dir, args.plot_format)
    plot_summary_bars(summary_rows, input_dir, args.plot_format)
    print(json.dumps({"input_dir": str(input_dir), "num_systems": len(systems)}, indent=2))


if __name__ == "__main__":
    main()
