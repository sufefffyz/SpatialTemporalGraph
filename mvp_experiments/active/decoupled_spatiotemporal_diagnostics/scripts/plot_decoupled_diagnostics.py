#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    plt = None
    HAS_MATPLOTLIB = False


COMPONENTS = ["full", "low", "high"]
DEFAULT_METHOD = "moving_average"


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


def method_of(row: dict) -> str:
    return row.get("decomposition_method") or row.get("method") or DEFAULT_METHOD


def ordered_methods(rows: list[dict]) -> list[str]:
    methods: list[str] = []
    for row in rows:
        method = method_of(row)
        if method not in methods:
            methods.append(method)
    return methods or [DEFAULT_METHOD]


def safe_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


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


def ordered_systems(component_rows: list[dict], standard_rows: list[dict] | None = None) -> list[str]:
    if standard_rows:
        values = {}
        for row in standard_rows:
            system = row["system"]
            values.setdefault(system, []).append(to_float(row, "MAE"))
        return [system for system, _ in sorted(values.items(), key=lambda item: float(np.nanmean(item[1])))]
    full_rows = [row for row in component_rows if row.get("component") == "full"]
    values = {}
    for row in full_rows:
        system = row["system"]
        values.setdefault(system, []).append(to_float(row, "MAE"))
    return [system for system, _ in sorted(values.items(), key=lambda item: float(np.nanmean(item[1])))]


def horizons(rows: list[dict]) -> list[int]:
    return sorted({to_int(row, "horizon") for row in rows if to_int(row, "horizon") > 0})


def component_lookup(rows: list[dict], metric: str) -> dict[tuple[str, str, int, str], float]:
    return {
        (method_of(row), row["system"], to_int(row, "horizon"), row["component"]): to_float(row, metric)
        for row in rows
    }


def peak_lookup(rows: list[dict], metric: str) -> dict[tuple[str, str, int, str], float]:
    return {
        (method_of(row), row["system"], to_int(row, "horizon"), row["window_type"]): to_float(row, metric)
        for row in rows
    }


def residual_lookup(rows: list[dict], metric: str) -> dict[tuple[str, str, int, str], float]:
    return {
        (method_of(row), row["system"], to_int(row, "horizon"), row["component"]): to_float(row, metric)
        for row in rows
    }


def standard_lookup(rows: list[dict], metric: str) -> dict[tuple[str, int], float]:
    return {(row["system"], to_int(row, "horizon")): to_float(row, metric) for row in rows}


def average(values: list[float]) -> float:
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    return float(np.mean(arr)) if arr.size else float("nan")


def build_standard_summary_rows(standard_rows: list[dict], rank_summary_rows: list[dict]) -> list[dict]:
    if not standard_rows:
        return []
    systems = ordered_systems([], standard_rows)
    hs = horizons(standard_rows)
    rank_lookup = {row["system"]: row for row in rank_summary_rows}
    metric_names = [
        "MAE",
        "RMSE",
        "WAPE",
        "W1",
        "MASE",
        "RMSSE",
        "worst_pct_MAE",
        "normal_MAE",
        "peak_MAE",
        "peak_over_normal_MAE",
        "peak_precision",
        "peak_recall",
        "peak_F1",
    ]
    lookups = {metric: standard_lookup(standard_rows, metric) for metric in metric_names}
    rows = []
    for system in systems:
        row = {"system": system}
        for metric in metric_names:
            row[f"avg_{metric}"] = average([lookups[metric].get((system, h), float("nan")) for h in hs])
        rank_row = rank_lookup.get(system, {})
        row["avg_standard_rank"] = to_float(rank_row, "avg_standard_rank")
        row["rank_count"] = to_int(rank_row, "rank_count")
        rows.append(row)
    return rows


def build_summary_rows(component_rows: list[dict], residual_rows: list[dict]) -> list[dict]:
    systems = ordered_systems(component_rows)
    hs = horizons(component_rows)
    method_names = ordered_methods(component_rows)
    comp_mae = component_lookup(component_rows, "MAE")
    comp_w1 = component_lookup(component_rows, "W1")
    resid_lag = residual_lookup(residual_rows, "temporal_lag1_corr")
    resid_under = residual_lookup(residual_rows, "under_prediction_rate")

    rows = []
    for method in method_names:
        for system in systems:
            low_mae = average([comp_mae.get((method, system, h, "low"), float("nan")) for h in hs])
            high_mae = average([comp_mae.get((method, system, h, "high"), float("nan")) for h in hs])
            rows.append(
                {
                    "decomposition_method": method,
                    "system": system,
                    "avg_low_MAE": low_mae,
                    "avg_high_MAE": high_mae,
                    "avg_high_over_low_MAE": high_mae / low_mae if low_mae and np.isfinite(low_mae) else float("nan"),
                    "avg_low_W1": average([comp_w1.get((method, system, h, "low"), float("nan")) for h in hs]),
                    "avg_high_W1": average([comp_w1.get((method, system, h, "high"), float("nan")) for h in hs]),
                    "avg_high_temporal_lag1_corr": average([resid_lag.get((method, system, h, "high"), float("nan")) for h in hs]),
                    "avg_high_under_prediction_rate": average([resid_under.get((method, system, h, "high"), float("nan")) for h in hs]),
                }
            )
    return rows


def save_fig(fig: plt.Figure, output_dir: Path, stem: str, plot_format: str) -> None:
    formats = ["png", "pdf"] if plot_format == "both" else [plot_format]
    for suffix in formats:
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight", pad_inches=0.04)


def matrix_for(rows: list[dict], metric: str, method: str, systems: list[str], hs: list[int], component: str) -> np.ndarray:
    lookup = component_lookup(rows, metric)
    return np.asarray([[lookup.get((method, system, h, component), float("nan")) for h in hs] for system in systems], dtype=float)


def plot_component_heatmaps(rows: list[dict], methods: list[str], systems: list[str], hs: list[int], output_dir: Path, plot_format: str) -> None:
    for method in methods:
        suffix = "" if len(methods) == 1 else f"_{safe_stem(method)}"
        for metric in ["MAE", "W1"]:
            fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2), sharex=True, sharey=True)
            for ax, component in zip(axes, COMPONENTS):
                matrix = matrix_for(rows, metric, method, systems, hs, component)
                im = ax.imshow(matrix, aspect="auto", cmap="viridis")
                ax.set_title(f"{method}: {component} {metric}")
                ax.set_xticks(np.arange(len(hs)), [f"H{h}" for h in hs], rotation=45)
                ax.set_yticks(np.arange(len(systems)), systems)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            fig.tight_layout()
            save_fig(fig, output_dir, f"component_{metric.lower()}_heatmaps{suffix}", plot_format)
            plt.close(fig)


def plot_component_ratio(rows: list[dict], methods: list[str], systems: list[str], hs: list[int], output_dir: Path, plot_format: str) -> None:
    lookup = component_lookup(rows, "MAE")
    colors = plt.cm.tab10.colors
    for method in methods:
        fig, ax = plt.subplots(figsize=(8.2, 4.2))
        for idx, system in enumerate(systems):
            values = []
            for h in hs:
                high = lookup.get((method, system, h, "high"), float("nan"))
                low = lookup.get((method, system, h, "low"), float("nan"))
                values.append(high / low if low and np.isfinite(low) else float("nan"))
            ax.plot(hs, values, marker="o", linewidth=1.8, markersize=3.5, label=system, color=colors[idx % len(colors)])
        ax.set_title(method)
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel("High-component MAE / low-component MAE")
        ax.set_xticks(hs)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(ncol=2, fontsize=7, frameon=False)
        fig.tight_layout()
        suffix = "" if len(methods) == 1 else f"_{safe_stem(method)}"
        save_fig(fig, output_dir, f"high_over_low_mae_by_horizon{suffix}", plot_format)
        plt.close(fig)


def plot_peak(rows: list[dict], systems: list[str], hs: list[int], output_dir: Path, plot_format: str) -> None:
    lookup_mae = {(row["system"], to_int(row, "horizon"), row["window_type"]): to_float(row, "MAE") for row in rows}
    lookup_w1 = {(row["system"], to_int(row, "horizon"), row["window_type"]): to_float(row, "W1") for row in rows}
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
        ax.set_title("Standard peak windows")
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel(ylabel)
        ax.set_xticks(hs)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(ncol=2, fontsize=7, frameon=False)
        fig.tight_layout()
        save_fig(fig, output_dir, stem, plot_format)
        plt.close(fig)


def plot_residual_heatmaps(
    residual_rows: list[dict],
    spatial_rows: list[dict],
    methods: list[str],
    systems: list[str],
    hs: list[int],
    output_dir: Path,
    plot_format: str,
) -> None:
    panels = [
        ("temporal_lag1_corr", residual_rows, "High residual lag1 corr"),
        ("under_prediction_rate", residual_rows, "High under-prediction rate"),
        ("edge_residual_corr", spatial_rows, "High edge residual corr"),
        ("residual_dirichlet", spatial_rows, "High residual Dirichlet"),
    ]
    for method in methods:
        fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.2), sharex=True, sharey=True)
        for ax, (metric, rows, title) in zip(axes.ravel(), panels):
            lookup = residual_lookup(rows, metric)
            matrix = np.asarray(
                [[lookup.get((method, system, h, "high"), float("nan")) for h in hs] for system in systems],
                dtype=float,
            )
            im = ax.imshow(matrix, aspect="auto", cmap="magma")
            ax.set_title(f"{method}: {title}")
            ax.set_xticks(np.arange(len(hs)), [f"H{h}" for h in hs], rotation=45)
            ax.set_yticks(np.arange(len(systems)), systems)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        fig.tight_layout()
        suffix = "" if len(methods) == 1 else f"_{safe_stem(method)}"
        save_fig(fig, output_dir, f"high_residual_structure_heatmaps{suffix}", plot_format)
        plt.close(fig)


def plot_summary_bars(summary_rows: list[dict], output_dir: Path, plot_format: str) -> None:
    metrics = [
        ("avg_low_MAE", "Low MAE"),
        ("avg_high_MAE", "High MAE"),
        ("avg_low_W1", "Low W1"),
        ("avg_high_W1", "High W1"),
    ]
    labels = [f"{row.get('decomposition_method', DEFAULT_METHOD)}:{row['system']}" for row in summary_rows]
    fig_height = max(3.8, 0.22 * len(labels) + 1.2)
    fig, axes = plt.subplots(1, len(metrics), figsize=(12.0, fig_height), sharey=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    for ax, (key, title) in zip(axes, metrics):
        values = [to_float(row, key) if isinstance(row[key], str) else row[key] for row in summary_rows]
        ax.barh(labels, values, color=colors)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(True, axis="x", alpha=0.25)
        if ax is not axes[0]:
            ax.tick_params(axis="y", labelleft=False)
    fig.tight_layout()
    save_fig(fig, output_dir, "decoupled_summary_bars", plot_format)
    plt.close(fig)


def plot_standard_bars(summary_rows: list[dict], output_dir: Path, plot_format: str) -> None:
    if not summary_rows:
        return
    metrics = [
        ("avg_MAE", "MAE"),
        ("avg_MASE", "MASE"),
        ("avg_worst_pct_MAE", "Worst-window MAE"),
        ("avg_peak_F1", "Peak F1"),
        ("avg_standard_rank", "Avg rank"),
    ]
    labels = [row["system"] for row in summary_rows]
    fig_height = max(3.8, 0.24 * len(labels) + 1.2)
    fig, axes = plt.subplots(1, len(metrics), figsize=(14.0, fig_height), sharey=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    for ax, (key, title) in zip(axes, metrics):
        values = [float(row.get(key, float("nan"))) for row in summary_rows]
        ax.barh(labels, values, color=colors)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(True, axis="x", alpha=0.25)
        if ax is not axes[0]:
            ax.tick_params(axis="y", labelleft=False)
    fig.tight_layout()
    save_fig(fig, output_dir, "standard_performance_summary_bars", plot_format)
    plt.close(fig)


def build_method_comparison_rows(summary_rows: list[dict]) -> list[dict]:
    methods = ordered_methods(summary_rows)
    if len(methods) < 2:
        return []
    baseline_method = methods[0]
    lookup = {(row["decomposition_method"], row["system"]): row for row in summary_rows}
    systems = sorted({row["system"] for row in summary_rows})
    metric_keys = [
        "avg_low_MAE",
        "avg_high_MAE",
        "avg_low_W1",
        "avg_high_W1",
        "avg_high_temporal_lag1_corr",
    ]
    rows: list[dict] = []
    for method in methods[1:]:
        for system in systems:
            base = lookup.get((baseline_method, system))
            comp = lookup.get((method, system))
            if base is None or comp is None:
                continue
            row: dict[str, str | float] = {
                "baseline_method": baseline_method,
                "comparison_method": method,
                "system": system,
            }
            for key in metric_keys:
                base_value = float(base.get(key, float("nan")))
                comp_value = float(comp.get(key, float("nan")))
                delta = comp_value - base_value if np.isfinite(base_value) and np.isfinite(comp_value) else float("nan")
                row[f"delta_{key}"] = delta
                row[f"pct_delta_{key}"] = (delta / base_value) if base_value and np.isfinite(delta) else float("nan")
            rows.append(row)
    return rows


def plot_method_comparison(summary_rows: list[dict], output_dir: Path, plot_format: str) -> None:
    methods = ordered_methods(summary_rows)
    if len(methods) < 2:
        return
    systems = sorted({row["system"] for row in summary_rows})
    lookup = {(row["decomposition_method"], row["system"]): row for row in summary_rows}
    panels = [
        ("avg_low_MAE", "Average low MAE"),
        ("avg_high_MAE", "Average high MAE"),
        ("avg_low_W1", "Average low W1"),
        ("avg_high_W1", "Average high W1"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(13.0, max(3.8, 0.22 * len(systems) + 1.3)), sharey=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (key, title) in zip(axes, panels):
        matrix = np.asarray(
            [[float(lookup.get((method, system), {}).get(key, float("nan"))) for method in methods] for system in systems],
            dtype=float,
        )
        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(methods)), methods, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(systems)), systems)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    save_fig(fig, output_dir, "decomposition_method_metric_heatmaps", plot_format)
    plt.close(fig)


def markdown_summary(path: Path, standard_summary_rows: list[dict], summary_rows: list[dict], report: dict) -> None:
    lines = [
        "# Decoupled SD Baseline Diagnostic Report",
        "",
        f"- Input directory: `{report['input_dir']}`",
        f"- Systems: {report['num_systems']}",
        f"- Decomposition methods: {', '.join(report['decomposition_methods'])}",
        f"- Horizons: {', '.join(f'H{h}' for h in report['horizons'])}",
        "",
        "## Standard Performance Metrics",
        "",
        "These metrics do not depend on a low/high-frequency decomposition.",
        "",
        "| System | MAE | RMSE | WAPE | MASE | RMSSE | Worst MAE | Peak MAE | Peak F1 | Full W1 | Avg Rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in standard_summary_rows:
        lines.append(
            "| {system} | {mae} | {rmse} | {wape} | {mase} | {rmsse} | {worst} | {peak} | {f1} | {w1} | {rank} |".format(
                system=row["system"],
                mae=fmt(row["avg_MAE"]),
                rmse=fmt(row["avg_RMSE"]),
                wape=fmt(row["avg_WAPE"]),
                mase=fmt(row["avg_MASE"]),
                rmsse=fmt(row["avg_RMSSE"]),
                worst=fmt(row["avg_worst_pct_MAE"]),
                peak=fmt(row["avg_peak_MAE"]),
                f1=fmt(row["avg_peak_F1"]),
                w1=fmt(row["avg_W1"]),
                rank=fmt(row["avg_standard_rank"]),
            )
        )

    lines.extend(
        [
            "",
            "## Decomposition-Dependent Low/High Metrics",
            "",
            "These metrics should be interpreted within each decomposition method.",
            "",
            "| Method | System | Low MAE | High MAE | High/Low | Low W1 | High W1 | High Lag1 | High Under Rate |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| {method} | {system} | {low} | {high} | {ratio} | {low_w1} | {high_w1} | {lag} | {under} |".format(
                method=row.get("decomposition_method", DEFAULT_METHOD),
                system=row["system"],
                low=fmt(row["avg_low_MAE"]),
                high=fmt(row["avg_high_MAE"]),
                ratio=fmt(row["avg_high_over_low_MAE"]),
                low_w1=fmt(row["avg_low_W1"]),
                high_w1=fmt(row["avg_high_W1"]),
                lag=fmt(row["avg_high_temporal_lag1_corr"]),
                under=fmt(row["avg_high_under_prediction_rate"]),
            )
        )
    method_comparison_rows = build_method_comparison_rows(summary_rows)
    if method_comparison_rows:
        lines.extend(
            [
                "",
                "## Method Comparison",
                "",
                "Positive deltas mean the comparison decomposition is larger than the baseline decomposition.",
                "",
                "| Baseline | Comparison | System | Delta Low MAE | Delta High MAE | Delta Low W1 | Delta High W1 | Delta High Lag1 |",
                "|---|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in method_comparison_rows:
            lines.append(
                "| {base} | {comp} | {system} | {low_mae} | {high_mae} | {low_w1} | {high_w1} | {lag} |".format(
                    base=row["baseline_method"],
                    comp=row["comparison_method"],
                    system=row["system"],
                    low_mae=fmt(float(row["delta_avg_low_MAE"])),
                    high_mae=fmt(float(row["delta_avg_high_MAE"])),
                    low_w1=fmt(float(row["delta_avg_low_W1"])),
                    high_w1=fmt(float(row["delta_avg_high_W1"])),
                    lag=fmt(float(row["delta_avg_high_temporal_lag1_corr"])),
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
    standard_rows = read_csv(input_dir / "standard_performance_metrics.csv")
    rank_summary_rows = read_csv(input_dir / "standard_rank_summary.csv")
    component_rows = read_csv(input_dir / "component_metrics.csv")
    peak_rows = read_csv(input_dir / "peak_window_metrics.csv")
    residual_rows = read_csv(input_dir / "residual_structure_metrics.csv")
    spatial_rows = read_csv(input_dir / "spatial_residual_metrics.csv")
    if not component_rows:
        raise FileNotFoundError(f"Missing component metrics under {input_dir}")

    systems = ordered_systems(component_rows, standard_rows)
    method_names = ordered_methods(component_rows)
    hs = horizons(component_rows)
    standard_summary_rows = build_standard_summary_rows(standard_rows, rank_summary_rows)
    summary_rows = build_summary_rows(component_rows, residual_rows)
    write_csv(input_dir / "standard_average_summary.csv", standard_summary_rows)
    write_csv(input_dir / "decomposition_average_summary.csv", summary_rows)
    write_csv(input_dir / "decoupled_average_summary.csv", summary_rows)
    method_comparison_rows = build_method_comparison_rows(summary_rows)
    write_csv(input_dir / "decoupled_method_comparison.csv", method_comparison_rows)
    report = {
        "input_dir": str(input_dir),
        "num_systems": len(systems),
        "decomposition_methods": method_names,
        "horizons": hs,
    }
    (input_dir / "decoupled_plot_summary.json").write_text(json.dumps(report, indent=2, default=json_default), encoding="utf-8")
    markdown_summary(input_dir / "decoupled_diagnostic_report.md", standard_summary_rows, summary_rows, report)

    if not HAS_MATPLOTLIB:
        print(
            json.dumps(
                {
                    "input_dir": str(input_dir),
                    "num_systems": len(systems),
                    "decomposition_methods": method_names,
                    "plots": "skipped_missing_matplotlib",
                },
                indent=2,
            )
        )
        return

    plot_component_heatmaps(component_rows, method_names, systems, hs, input_dir, args.plot_format)
    plot_component_ratio(component_rows, method_names, systems, hs, input_dir, args.plot_format)
    plot_peak(peak_rows, systems, hs, input_dir, args.plot_format)
    if residual_rows and spatial_rows:
        plot_residual_heatmaps(residual_rows, spatial_rows, method_names, systems, hs, input_dir, args.plot_format)
    plot_method_comparison(summary_rows, input_dir, args.plot_format)
    plot_summary_bars(summary_rows, input_dir, args.plot_format)
    plot_standard_bars(standard_summary_rows, input_dir, args.plot_format)
    print(json.dumps({"input_dir": str(input_dir), "num_systems": len(systems), "decomposition_methods": method_names}, indent=2))


if __name__ == "__main__":
    main()
