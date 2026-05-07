#!/usr/bin/env python3
"""Plot multi-method LargeST delay-audit outputs.

The audit CSVs can be large, so this script streams edge rows and aggregates
histograms before plotting. It is intended to run after
``run_largest_5min_delay_multimethod_audit.py`` without recomputing delays.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Any


DATASET_ORDER = ["SD", "GLA", "GBA"]
METHOD_ORDER = ["mcc_5min_resid", "stdde_spline_fft_mcc", "lift_fft_abs"]
METHOD_LABELS = {
    "mcc_5min_resid": "5-min MCC",
    "stdde_spline_fft_mcc": "STDDE-style spline MCC",
    "lift_fft_abs": "LIFT-style FFT abs",
}
METHOD_COLORS = {
    "mcc_5min_resid": "#4E79A7",
    "stdde_spline_fft_mcc": "#F28E2B",
    "lift_fft_abs": "#59A14F",
}
FILTERED_LAG_SENTINEL = -5
FILTERED_LAG_LABEL = "Filtered"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate figures from strict multi-method LargeST delay-audit CSVs."
    )
    parser.add_argument(
        "--audit-dir",
        default="delay_selective/outputs/largest_5min_delay_multimethod_audit",
        help="Directory containing all_summary.csv and per-dataset edge_delay_scores.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        default="delay_selective/figures/largest_5min_delay_multimethod_audit",
    )
    parser.add_argument("--formats", default="pdf,png", help="Comma-separated output formats.")
    parser.add_argument("--count-y-scale", default="log", choices=["log", "linear"])
    parser.add_argument("--max-lag-minutes", type=int, default=60)
    parser.add_argument(
        "--distance-bins-km",
        default="0,1,2,5,10,20,50,100,inf",
        help="Comma-separated edge distance bins for distance-conditioned delay plots.",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        default=["month", "week_daily"],
        choices=["month", "week_daily", "train_prefix"],
    )
    return parser.parse_args()


def parse_bins(spec: str) -> list[float]:
    values = []
    for raw in spec.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        values.append(float("inf") if item in {"inf", "infinity"} else float(item))
    if len(values) < 2:
        raise ValueError("Need at least two distance-bin boundaries.")
    return values


def distance_bin_label(distance_km: float | None, bins: list[float]) -> str | None:
    if distance_km is None:
        return None
    for lo, hi in zip(bins[:-1], bins[1:]):
        if math.isinf(hi):
            if distance_km >= lo:
                return f"[{lo:g}, inf)"
        elif distance_km >= lo and distance_km < hi:
            return f"[{lo:g}, {hi:g})"
    return None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def safe_int(value: Any) -> int | None:
    f = safe_float(value)
    if f is None:
        return None
    return int(round(f))


def setup_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
        }
    )
    return plt


def save_figure(fig: Any, output_dir: Path, stem: str, formats: list[str]) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path)
        made.append(str(path))
    return made


def ordered_unique(values: set[str], preferred: list[str]) -> list[str]:
    ordered = [v for v in preferred if v in values]
    ordered.extend(sorted(values - set(ordered)))
    return ordered


def aggregate_edge_histograms(
    audit_dir: Path,
    windows: set[str],
    max_lag_minutes: int,
    distance_bins_km: list[float],
) -> tuple[
    dict[tuple[str, str, str, str, str, int], int],
    dict[tuple[str, str, str, str, str, str, int], int],
    dict[tuple[str, str, str, str], int],
]:
    """Return histogram counts and total edge counts.

    Histogram key:
    (dataset, window, window_label, method, lag_field, lag_minutes)
    """

    hist: Counter[tuple[str, str, str, str, str, int]] = Counter()
    distance_hist: Counter[tuple[str, str, str, str, str, str, int]] = Counter()
    totals: Counter[tuple[str, str, str, str]] = Counter()

    for edge_path in sorted(audit_dir.glob("*/edge_delay_scores.csv")):
        with edge_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                window = row.get("window", "")
                if window not in windows:
                    continue
                dataset = row.get("dataset", edge_path.parent.name)
                window_label = row.get("window_label", "")
                method = row.get("method", "")
                dist_label = distance_bin_label(safe_float(row.get("edge_distance_km")), distance_bins_km)
                total_key = (dataset, window, window_label, method)
                totals[total_key] += 1
                for field in ["best_lag_minutes", "effective_lag_minutes"]:
                    lag = safe_int(row.get(field))
                    if lag is None and field == "effective_lag_minutes":
                        lag = FILTERED_LAG_SENTINEL
                    if lag is None:
                        continue
                    if lag != FILTERED_LAG_SENTINEL and (lag < 0 or lag > max_lag_minutes):
                        continue
                    hist[(dataset, window, window_label, method, field, lag)] += 1
                    if dist_label is not None:
                        distance_hist[(dataset, window, window_label, method, field, dist_label, lag)] += 1
    return dict(hist), dict(distance_hist), dict(totals)


def plot_month_delay_histograms(
    plt: Any,
    hist: dict[tuple[str, str, str, str, str, int], int],
    output_dir: Path,
    formats: list[str],
    count_y_scale: str,
    lag_field: str,
) -> list[str]:
    rows = [k for k in hist if k[1] == "month" and k[4] == lag_field]
    datasets = ordered_unique({k[0] for k in rows}, DATASET_ORDER)
    methods = ordered_unique({k[3] for k in rows}, METHOD_ORDER)
    if not datasets or not methods:
        return []

    fig, axes = plt.subplots(
        len(datasets),
        len(methods),
        figsize=(3.4 * len(methods), 2.45 * len(datasets)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for r, dataset in enumerate(datasets):
        for c, method in enumerate(methods):
            ax = axes[r][c]
            pairs = [
                (lag, count)
                for (ds, window, _label, mt, field, lag), count in hist.items()
                if ds == dataset and window == "month" and mt == method and field == lag_field
            ]
            pairs.sort()
            if pairs:
                lags = [p[0] for p in pairs]
                counts = [p[1] for p in pairs]
                step = min([b - a for a, b in zip(lags[:-1], lags[1:]) if b > a] or [5])
                ax.bar(
                    lags,
                    counts,
                    width=max(0.8, step * 0.85),
                    color=METHOD_COLORS.get(method, "#4E79A7"),
                    edgecolor="white",
                    linewidth=0.4,
                )
            if count_y_scale == "log":
                ax.set_yscale("log")
                ax.set_ylim(bottom=0.8)
            if lag_field == "effective_lag_minutes":
                ax.set_xlim(FILTERED_LAG_SENTINEL - 3, 61)
                ax.set_xticks([FILTERED_LAG_SENTINEL, 0, 15, 30, 45, 60])
                ax.set_xticklabels([FILTERED_LAG_LABEL, "0", "15", "30", "45", "60"])
            else:
                ax.set_xlim(-1, 61)
            if r == 0:
                ax.set_title(METHOD_LABELS.get(method, method))
            if c == 0:
                ax.set_ylabel(f"{dataset}\n# edges")
            if r == len(datasets) - 1:
                ax.set_xlabel("Delay lag (minutes)")
    field_label = "effective" if lag_field == "effective_lag_minutes" else "raw_best"
    fig.tight_layout()
    return save_figure(
        fig,
        output_dir,
        f"month_{field_label}_delay_hist_{count_y_scale}_count",
        formats,
    )


def plot_month_ratio_bars(
    plt: Any,
    summary_rows: list[dict[str, str]],
    output_dir: Path,
    formats: list[str],
) -> list[str]:
    rows = [r for r in summary_rows if r.get("window") == "month"]
    datasets = ordered_unique({r.get("dataset", "") for r in rows}, DATASET_ORDER)
    methods = ordered_unique({r.get("method", "") for r in rows}, METHOD_ORDER)
    if not datasets or not methods:
        return []

    metrics = [
        ("corr_filtered_ratio", "corr < 0.8"),
        ("high_conf_nonzero_ratio", "accepted nonzero"),
        ("low_improvement_nonzero_ratio", "nonzero, weak gain"),
        ("effective_zero_ratio", "effective zero"),
    ]
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(3.5 * len(datasets), 3.0),
        squeeze=False,
        sharey=True,
    )
    x = list(range(len(methods)))
    width = 0.19
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for ax, dataset in zip(axes[0], datasets):
        dataset_rows = {
            r.get("method", ""): r
            for r in rows
            if r.get("dataset") == dataset
        }
        for (metric, label), offset in zip(metrics, offsets):
            values = [
                safe_float(dataset_rows.get(method, {}).get(metric)) or 0.0
                for method in methods
            ]
            ax.bar(
                [i + offset for i in x],
                values,
                width=width,
                label=label,
                edgecolor="white",
                linewidth=0.5,
            )
        ax.set_title(dataset)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=25, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Edge ratio")
    axes[0][-1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    return save_figure(fig, output_dir, "month_delay_acceptance_ratios", formats)


def plot_week_daily_high_conf(
    plt: Any,
    summary_rows: list[dict[str, str]],
    output_dir: Path,
    formats: list[str],
) -> list[str]:
    rows = [r for r in summary_rows if r.get("window") == "week_daily"]
    datasets = ordered_unique({r.get("dataset", "") for r in rows}, DATASET_ORDER)
    methods = ordered_unique({r.get("method", "") for r in rows}, METHOD_ORDER)
    if not datasets or not methods:
        return []

    fig, axes = plt.subplots(
        len(datasets),
        1,
        figsize=(6.2, 2.2 * len(datasets)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for ax, dataset in zip(axes[:, 0], datasets):
        for method in methods:
            method_rows = [
                r for r in rows
                if r.get("dataset") == dataset and r.get("method") == method
            ]
            method_rows.sort(key=lambda r: safe_int(r.get("day_in_window")) or 0)
            days = [safe_int(r.get("day_in_window")) for r in method_rows]
            values = [safe_float(r.get("high_conf_nonzero_ratio")) for r in method_rows]
            days_clean = [d for d, v in zip(days, values) if d is not None and v is not None]
            values_clean = [v for d, v in zip(days, values) if d is not None and v is not None]
            if days_clean:
                ax.plot(
                    days_clean,
                    values_clean,
                    marker="o",
                    linewidth=1.5,
                    markersize=4,
                    label=METHOD_LABELS.get(method, method),
                    color=METHOD_COLORS.get(method),
                )
        ax.set_title(dataset)
        ax.set_ylabel("Accepted\nnonzero ratio")
        ax.set_ylim(0, 1)
    axes[-1][0].set_xlabel("Day in first week")
    axes[0][0].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    return save_figure(fig, output_dir, "week_daily_high_conf_nonzero_ratio", formats)


def plot_distance_bin_heatmap(
    plt: Any,
    distance_rows: list[dict[str, str]],
    output_dir: Path,
    formats: list[str],
) -> list[str]:
    rows = [
        r for r in distance_rows
        if r.get("window") == "month" and r.get("method") == "mcc_5min_resid"
    ]
    datasets = ordered_unique({r.get("dataset", "") for r in rows}, DATASET_ORDER)
    bins = []
    for r in rows:
        label = r.get("distance_bin_km", "")
        if label and label not in bins:
            bins.append(label)
    if not datasets or not bins:
        return []

    import numpy as np

    values = np.full((len(datasets), len(bins)), np.nan, dtype=float)
    for r in rows:
        dataset = r.get("dataset", "")
        label = r.get("distance_bin_km", "")
        if dataset in datasets and label in bins:
            v = safe_float(r.get("high_conf_nonzero_ratio"))
            if v is not None:
                values[datasets.index(dataset), bins.index(label)] = v

    fig, ax = plt.subplots(figsize=(0.78 * len(bins) + 1.8, 2.8))
    im = ax.imshow(values, aspect="auto", cmap="viridis", vmin=0, vmax=max(0.01, float(np.nanmax(values))))
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(bins, rotation=35, ha="right")
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets)
    ax.set_xlabel("Edge distance bin (km)")
    ax.set_ylabel("Dataset")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Accepted nonzero ratio")
    fig.tight_layout()
    return save_figure(fig, output_dir, "month_distance_bin_high_conf_heatmap_mcc", formats)


def plot_month_delay_by_distance_heatmaps(
    plt: Any,
    distance_hist: dict[tuple[str, str, str, str, str, str, int], int],
    output_dir: Path,
    formats: list[str],
    lag_field: str,
    distance_bins_km: list[float],
) -> list[str]:
    rows = [k for k in distance_hist if k[1] == "month" and k[4] == lag_field]
    datasets = ordered_unique({k[0] for k in rows}, DATASET_ORDER)
    methods = ordered_unique({k[3] for k in rows}, METHOD_ORDER)
    if not datasets or not methods:
        return []

    import numpy as np
    from matplotlib.colors import LogNorm

    distance_labels = []
    for lo, hi in zip(distance_bins_km[:-1], distance_bins_km[1:]):
        if math.isinf(hi):
            distance_labels.append(f"[{lo:g}, inf)")
        else:
            distance_labels.append(f"[{lo:g}, {hi:g})")
    distance_labels = [
        label for label in distance_labels
        if any(k[5] == label for k in rows)
    ]
    lag_values = sorted({k[6] for k in rows})
    if not distance_labels or not lag_values:
        return []

    matrices: dict[tuple[str, str], np.ndarray] = {}
    max_count = 1
    for dataset in datasets:
        for method in methods:
            mat = np.zeros((len(distance_labels), len(lag_values)), dtype=float)
            for (ds, window, _label, mt, field, dist_label, lag), count in distance_hist.items():
                if ds != dataset or window != "month" or mt != method or field != lag_field:
                    continue
                if dist_label not in distance_labels or lag not in lag_values:
                    continue
                mat[distance_labels.index(dist_label), lag_values.index(lag)] += count
            max_count = max(max_count, int(np.nanmax(mat)) if mat.size else 1)
            matrices[(dataset, method)] = mat

    fig, axes = plt.subplots(
        len(datasets),
        len(methods),
        figsize=(3.4 * len(methods), 2.65 * len(datasets)),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    last_im = None
    norm = LogNorm(vmin=1, vmax=max_count)
    for r, dataset in enumerate(datasets):
        for c, method in enumerate(methods):
            ax = axes[r][c]
            mat = matrices[(dataset, method)]
            masked = np.ma.masked_where(mat <= 0, mat)
            last_im = ax.imshow(
                masked,
                aspect="auto",
                origin="lower",
                cmap="magma",
                norm=norm,
                extent=[
                    min(lag_values) - 2.5,
                    max(lag_values) + 2.5,
                    -0.5,
                    len(distance_labels) - 0.5,
                ],
            )
            if r == 0:
                ax.set_title(METHOD_LABELS.get(method, method))
            if c == 0:
                ax.set_ylabel(f"{dataset}\nDistance bin (km)")
                ax.set_yticks(range(len(distance_labels)))
                ax.set_yticklabels(distance_labels)
            if r == len(datasets) - 1:
                ax.set_xlabel("Delay lag (minutes)")
            ax.set_xlim(min(lag_values) - 2.5, max(lag_values) + 2.5)
            if lag_field == "effective_lag_minutes" and FILTERED_LAG_SENTINEL in lag_values:
                ticks = [FILTERED_LAG_SENTINEL, 0, 15, 30, 45, 60]
                ax.set_xticks(ticks)
                ax.set_xticklabels([FILTERED_LAG_LABEL, "0", "15", "30", "45", "60"])
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.015)
        cbar.set_label("Edge count")
    field_label = "effective" if lag_field == "effective_lag_minutes" else "raw_best"
    return save_figure(
        fig,
        output_dir,
        f"month_{field_label}_delay_by_distance_bin_log_count",
        formats,
    )


def main() -> None:
    args = parse_args()
    audit_dir = Path(args.audit_dir)
    output_dir = Path(args.output_dir)
    formats = [item.strip() for item in args.formats.split(",") if item.strip()]
    distance_bins_km = parse_bins(args.distance_bins_km)
    plt = setup_matplotlib()

    summary_rows = read_csv_rows(audit_dir / "all_summary.csv")
    distance_rows = read_csv_rows(audit_dir / "all_distance_bin_summary.csv")
    hist, distance_hist, totals = aggregate_edge_histograms(
        audit_dir=audit_dir,
        windows=set(args.windows),
        max_lag_minutes=args.max_lag_minutes,
        distance_bins_km=distance_bins_km,
    )

    made: list[str] = []
    made.extend(
        plot_month_delay_histograms(
            plt,
            hist=hist,
            output_dir=output_dir,
            formats=formats,
            count_y_scale=args.count_y_scale,
            lag_field="best_lag_minutes",
        )
    )
    made.extend(
        plot_month_delay_histograms(
            plt,
            hist=hist,
            output_dir=output_dir,
            formats=formats,
            count_y_scale=args.count_y_scale,
            lag_field="effective_lag_minutes",
        )
    )
    made.extend(plot_month_ratio_bars(plt, summary_rows, output_dir, formats))
    made.extend(plot_week_daily_high_conf(plt, summary_rows, output_dir, formats))
    made.extend(plot_distance_bin_heatmap(plt, distance_rows, output_dir, formats))
    made.extend(
        plot_month_delay_by_distance_heatmaps(
            plt,
            distance_hist=distance_hist,
            output_dir=output_dir,
            formats=formats,
            lag_field="best_lag_minutes",
            distance_bins_km=distance_bins_km,
        )
    )
    made.extend(
        plot_month_delay_by_distance_heatmaps(
            plt,
            distance_hist=distance_hist,
            output_dir=output_dir,
            formats=formats,
            lag_field="effective_lag_minutes",
            distance_bins_km=distance_bins_km,
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "plot_manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["figure_path"])
        writer.writeheader()
        for path in made:
            writer.writerow({"figure_path": path})
    with (output_dir / "histogram_totals.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "window", "window_label", "method", "num_edge_rows"],
        )
        writer.writeheader()
        for (dataset, window, window_label, method), count in sorted(totals.items()):
            writer.writerow(
                {
                    "dataset": dataset,
                    "window": window,
                    "window_label": window_label,
                    "method": method,
                    "num_edge_rows": count,
                }
            )
    print("Generated figures:")
    for path in made:
        print(path)


if __name__ == "__main__":
    main()
