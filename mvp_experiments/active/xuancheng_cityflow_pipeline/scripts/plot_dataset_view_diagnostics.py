#!/usr/bin/env python3
"""Plot lane-level and road-level Xuancheng dataset diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FEATURES = [
    "entered_veh",
    "exited_veh",
    "mean_active_veh",
    "mean_speed_kmh",
    "density_veh_per_km",
]

COLORS = {
    "entered_veh": "#0072B2",
    "exited_veh": "#D55E00",
    "mean_speed_kmh": "#009E73",
    "mean_active_veh": "#CC79A7",
    "density_veh_per_km": "#E69F00",
    "in_degree": "#0072B2",
    "out_degree": "#D55E00",
    "total_degree": "#000000",
}


@dataclass
class DegreeStats:
    ids: list[str]
    in_degree: np.ndarray
    out_degree: np.ndarray

    @property
    def total_degree(self) -> np.ndarray:
        return self.in_degree + self.out_degree


@dataclass
class ViewStats:
    granularity: str
    unit_ids: list[str]
    feature_names: list[str]
    file_count: int
    dates: list[str]
    shape_per_day: list[int]
    total_cells_per_feature: int
    zero_counts: dict[str, int]
    per_unit_zero_rate: dict[str, np.ndarray]
    samples: dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate lane-level and road-level degree/flow/speed/zero-rate diagnostics."
    )
    parser.add_argument("--lane-dir", required=True, help="Directory with lane-level daily NPZ files.")
    parser.add_argument("--road-dir", required=True, help="Directory with road-level daily NPZ files.")
    parser.add_argument("--output-dir", required=True, help="Directory to write figures and summaries.")
    parser.add_argument("--roadnet", default=None, help="CityFlow roadnet JSON. Defaults to NPZ source_config.")
    parser.add_argument("--lane-pattern", default="xuancheng_*_lane_agg_60s.npz")
    parser.add_argument("--road-pattern", default="xuancheng_*_road_agg_from_lane_60s.npz")
    parser.add_argument("--max-samples-per-feature", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def configure_style(dpi: int) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "text.usetex": False,
            "mathtext.fontset": "stix",
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path)
        print(f"[figure] {path}")
    plt.close(fig)


def scalar_to_str(value: np.ndarray) -> str:
    if value.shape == ():
        return str(value.tolist())
    raise ValueError(f"expected scalar array, got shape {value.shape}")


def infer_date(path: Path, archive: np.lib.npyio.NpzFile) -> str:
    if "date" in archive:
        return scalar_to_str(archive["date"])
    match = re.search(r"xuancheng_(\d{4}-\d{2}-\d{2})_", path.name)
    if match:
        return match.group(1)
    raise ValueError(f"cannot infer date from {path}")


def resolve_cityflow_path(config_path: Path, config: dict[str, Any], field: str) -> Path:
    raw = Path(str(config[field]))
    if raw.is_absolute():
        return raw
    candidates = []
    if config.get("dir"):
        candidates.append(Path(str(config["dir"])) / raw)
    candidates.append(config_path.parent / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def infer_roadnet_from_npz(npz_path: Path) -> Path:
    with np.load(npz_path, allow_pickle=False) as archive:
        if "source_config" not in archive:
            raise ValueError(f"{npz_path} has no source_config; pass --roadnet explicitly")
        config_path = Path(scalar_to_str(archive["source_config"]))
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return resolve_cityflow_path(config_path, config, "roadnetFile")


def load_roadnet_degree(roadnet_path: Path, lane_ids: list[str], road_ids: list[str]) -> tuple[DegreeStats, DegreeStats]:
    with roadnet_path.open("r", encoding="utf-8") as f:
        roadnet = json.load(f)

    road_set = set(road_ids)
    lane_set = set(lane_ids)
    road_edges: set[tuple[str, str]] = set()
    lane_edges: set[tuple[str, str]] = set()

    for intersection in roadnet.get("intersections", []):
        for road_link in intersection.get("roadLinks", []):
            start_road = str(road_link.get("startRoad", ""))
            end_road = str(road_link.get("endRoad", ""))
            if start_road in road_set and end_road in road_set:
                road_edges.add((start_road, end_road))
            for lane_link in road_link.get("laneLinks", []):
                start_lane = f"{start_road}_{lane_link.get('startLaneIndex')}"
                end_lane = f"{end_road}_{lane_link.get('endLaneIndex')}"
                if start_lane in lane_set and end_lane in lane_set:
                    lane_edges.add((start_lane, end_lane))

    def degree_from_edges(ids: list[str], edges: set[tuple[str, str]]) -> DegreeStats:
        index = {unit_id: idx for idx, unit_id in enumerate(ids)}
        in_degree = np.zeros(len(ids), dtype=np.int32)
        out_degree = np.zeros(len(ids), dtype=np.int32)
        for source, target in edges:
            if source in index:
                out_degree[index[source]] += 1
            if target in index:
                in_degree[index[target]] += 1
        return DegreeStats(ids=ids, in_degree=in_degree, out_degree=out_degree)

    return degree_from_edges(lane_ids, lane_edges), degree_from_edges(road_ids, road_edges)


def sample_values(values: np.ndarray, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    values = values[np.isfinite(values)]
    if values.size <= max_samples:
        return values.astype(np.float32, copy=False)
    sample_idx = rng.choice(values.size, size=max_samples, replace=False)
    return values[sample_idx].astype(np.float32, copy=False)


def load_view_stats(
    granularity: str,
    data_dir: Path,
    pattern: str,
    id_key: str,
    max_samples_per_feature: int,
    rng: np.random.Generator,
) -> ViewStats:
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise ValueError(f"no files match {data_dir / pattern}")

    per_file_sample = max(1, math.ceil(max_samples_per_feature / len(files)))
    feature_names: list[str] | None = None
    unit_ids: list[str] | None = None
    dates: list[str] = []
    shape_per_day: list[int] | None = None
    zero_counts: dict[str, int] | None = None
    total_cells_per_feature = 0
    per_unit_zero_counts: dict[str, np.ndarray] = {}
    sample_chunks: dict[str, list[np.ndarray]] = {name: [] for name in FEATURES}

    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            data = archive["data"]
            current_features = [str(value) for value in archive["feature_names"].tolist()]
            current_ids = [str(value) for value in archive[id_key].tolist()]
            date = infer_date(path, archive)
            dates.append(date)

            if feature_names is None:
                feature_names = current_features
                unit_ids = current_ids
                shape_per_day = list(data.shape)
                zero_counts = {name: 0 for name in feature_names}
                for name in ("entered_veh", "exited_veh", "mean_speed_kmh"):
                    per_unit_zero_counts[name] = np.zeros(len(unit_ids), dtype=np.int64)
            else:
                if current_features != feature_names:
                    raise ValueError(f"{path} has different feature_names")
                if current_ids != unit_ids:
                    raise ValueError(f"{path} has different {id_key} order")
                if list(data.shape) != shape_per_day:
                    raise ValueError(f"{path} shape {data.shape} differs from {shape_per_day}")

            assert feature_names is not None
            assert zero_counts is not None
            total_cells_per_feature += data.shape[0] * data.shape[1]
            feature_index = {name: idx for idx, name in enumerate(feature_names)}
            for name in feature_names:
                values = data[:, :, feature_index[name]]
                zero_counts[name] += int(np.count_nonzero(values == 0))

            for name in per_unit_zero_counts:
                values = data[:, :, feature_index[name]]
                per_unit_zero_counts[name] += np.count_nonzero(values == 0, axis=0)

            for name in ("entered_veh", "exited_veh"):
                values = data[:, :, feature_index[name]].ravel()
                sample_chunks[name].append(sample_values(values[values > 0], per_file_sample, rng))
            speed_values = data[:, :, feature_index["mean_speed_kmh"]].ravel()
            sample_chunks["mean_speed_kmh"].append(
                sample_values(speed_values[speed_values > 0], per_file_sample, rng)
            )

    assert feature_names is not None
    assert unit_ids is not None
    assert shape_per_day is not None
    assert zero_counts is not None

    total_time = len(files) * shape_per_day[0]
    per_unit_zero_rate = {
        name: counts.astype(np.float64) / float(total_time)
        for name, counts in per_unit_zero_counts.items()
    }
    samples = {
        name: np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
        for name, chunks in sample_chunks.items()
    }
    return ViewStats(
        granularity=granularity,
        unit_ids=unit_ids,
        feature_names=feature_names,
        file_count=len(files),
        dates=dates,
        shape_per_day=shape_per_day,
        total_cells_per_feature=total_cells_per_feature,
        zero_counts=zero_counts,
        per_unit_zero_rate=per_unit_zero_rate,
        samples=samples,
    )


def plot_degree(stats: DegreeStats, granularity: str, output_dir: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(5.1, 3.0))
    series = [
        ("In-degree", stats.in_degree, COLORS["in_degree"]),
        ("Out-degree", stats.out_degree, COLORS["out_degree"]),
        ("Total degree", stats.total_degree, COLORS["total_degree"]),
    ]
    max_degree = max(int(values.max()) for _, values, _ in series)
    x = np.arange(max_degree + 1)
    for label, values, color in series:
        counts = np.bincount(values, minlength=max_degree + 1)
        ax.plot(x, counts, marker="o", linewidth=1.4, markersize=3.0, label=label, color=color)
    ax.set_xlabel("Graph degree")
    ax.set_ylabel(f"Number of {granularity}s")
    ax.set_yscale("log")
    ax.set_xlim(left=-0.3, right=max_degree + 0.3)
    ax.legend(frameon=False)
    save_figure(fig, output_dir, f"{granularity}_degree_distribution")


def percentile_bins(values: np.ndarray, bins: int, lower: float = 0.5, upper: float = 99.5) -> np.ndarray:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.linspace(0.0, 1.0, bins + 1)
    lo, hi = np.percentile(values, [lower, upper])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(float(lo), float(hi), bins + 1)


def plot_flow_speed(stats: ViewStats, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    entered = stats.samples["entered_veh"]
    exited = stats.samples["exited_veh"]
    combined_flow = np.concatenate([arr for arr in (entered, exited) if arr.size > 0])
    flow_bins = percentile_bins(combined_flow, bins=40, lower=0.0, upper=99.5)
    axes[0].hist(
        entered,
        bins=flow_bins,
        histtype="step",
        linewidth=1.6,
        color=COLORS["entered_veh"],
        label="Entered",
    )
    axes[0].hist(
        exited,
        bins=flow_bins,
        histtype="step",
        linewidth=1.6,
        color=COLORS["exited_veh"],
        label="Exited",
    )
    axes[0].set_xlabel("Non-zero flow (vehicles/min)")
    axes[0].set_ylabel("Sample frequency")
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False)

    speed = stats.samples["mean_speed_kmh"]
    speed_bins = percentile_bins(speed, bins=50, lower=0.5, upper=99.5)
    axes[1].hist(
        speed,
        bins=speed_bins,
        color=COLORS["mean_speed_kmh"],
        alpha=0.85,
        linewidth=0.0,
    )
    axes[1].set_xlabel("Non-zero mean speed (km/h)")
    axes[1].set_ylabel("Sample frequency")
    axes[1].set_yscale("log")

    fig.tight_layout(w_pad=2.2)
    save_figure(fig, output_dir, f"{stats.granularity}_flow_speed_distribution")


def plot_zero_rate(stats: ViewStats, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    labels = stats.feature_names
    rates = [stats.zero_counts[name] / stats.total_cells_per_feature for name in labels]
    x = np.arange(len(labels))
    bar_colors = [COLORS.get(name, "#999999") for name in labels]
    axes[0].bar(x, rates, color=bar_colors, width=0.72)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        ["entered", "exited", "active", "speed", "density"],
        rotation=30,
        ha="right",
    )
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Global zero rate")

    bins = np.linspace(0.0, 1.0, 41)
    for name, label, color in [
        ("entered_veh", "Entered flow", COLORS["entered_veh"]),
        ("mean_speed_kmh", "Speed", COLORS["mean_speed_kmh"]),
    ]:
        axes[1].hist(
            stats.per_unit_zero_rate[name],
            bins=bins,
            histtype="step",
            linewidth=1.6,
            label=label,
            color=color,
        )
    axes[1].set_xlabel("Per-unit zero rate")
    axes[1].set_ylabel(f"Number of {stats.granularity}s")
    axes[1].legend(frameon=False)

    fig.tight_layout(w_pad=2.0)
    save_figure(fig, output_dir, f"{stats.granularity}_zero_rate")


def summarize_degree(stats: DegreeStats) -> dict[str, float]:
    total = stats.total_degree.astype(np.float64)
    return {
        "degree_mean": float(total.mean()),
        "degree_median": float(np.median(total)),
        "degree_p90": float(np.percentile(total, 90)),
        "degree_max": float(total.max()),
        "isolated_rate": float(np.mean(total == 0)),
    }


def summarize_samples(values: np.ndarray, prefix: str) -> dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}_sample_count": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_p99": 0.0,
        }
    return {
        f"{prefix}_sample_count": float(values.size),
        f"{prefix}_p50": float(np.percentile(values, 50)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
        f"{prefix}_p99": float(np.percentile(values, 99)),
    }


def write_summary(output_dir: Path, view_stats: list[ViewStats], degree_stats: dict[str, DegreeStats]) -> None:
    rows: list[dict[str, Any]] = []
    json_payload: dict[str, Any] = {}
    for stats in view_stats:
        degree = degree_stats[stats.granularity]
        row: dict[str, Any] = {
            "granularity": stats.granularity,
            "num_units": len(stats.unit_ids),
            "num_days": stats.file_count,
            "date_start": min(stats.dates),
            "date_end": max(stats.dates),
            "shape_per_day": "x".join(str(value) for value in stats.shape_per_day),
        }
        row.update(summarize_degree(degree))
        for name in stats.feature_names:
            row[f"zero_rate_{name}"] = stats.zero_counts[name] / stats.total_cells_per_feature
        row.update(summarize_samples(stats.samples["entered_veh"], "entered_nonzero"))
        row.update(summarize_samples(stats.samples["mean_speed_kmh"], "speed_nonzero"))
        rows.append(row)
        json_payload[stats.granularity] = row

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "dataset_view_diagnostics_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_dir / "dataset_view_diagnostics_summary.json"
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[summary] {csv_path}")
    print(f"[summary] {json_path}")


def main() -> int:
    args = parse_args()
    configure_style(args.dpi)
    rng = np.random.default_rng(args.seed)

    lane_dir = Path(args.lane_dir).expanduser().resolve()
    road_dir = Path(args.road_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    lane_stats = load_view_stats(
        "lane",
        lane_dir,
        args.lane_pattern,
        "lane_ids",
        args.max_samples_per_feature,
        rng,
    )
    road_stats = load_view_stats(
        "road",
        road_dir,
        args.road_pattern,
        "road_ids",
        args.max_samples_per_feature,
        rng,
    )

    roadnet_path = Path(args.roadnet).expanduser().resolve() if args.roadnet else infer_roadnet_from_npz(
        sorted(lane_dir.glob(args.lane_pattern))[0]
    )
    print(f"[info] roadnet={roadnet_path}")
    lane_degree, road_degree = load_roadnet_degree(roadnet_path, lane_stats.unit_ids, road_stats.unit_ids)

    for stats, degree in [(lane_stats, lane_degree), (road_stats, road_degree)]:
        plot_degree(degree, stats.granularity, output_dir)
        plot_flow_speed(stats, output_dir)
        plot_zero_rate(stats, output_dir)

    write_summary(output_dir, [lane_stats, road_stats], {"lane": lane_degree, "road": road_degree})
    return 0


if __name__ == "__main__":
    sys.exit(main())
