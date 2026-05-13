#!/usr/bin/env python3
"""Plot city-traffic-M/L target and degree distributions from profile outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate publication-style figures from "
            "profile_city_traffic_datasets.py outputs."
        )
    )
    parser.add_argument("--profile-dir", default=str(EXPERIMENT_DIR / "outputs" / "dataset_profile"))
    parser.add_argument("--output-dir", default=str(EXPERIMENT_DIR / "figures" / "dataset_profile"))
    parser.add_argument("--formats", default="pdf,png", help="Comma-separated output formats.")
    parser.add_argument(
        "--include-subgraphs",
        action="store_true",
        help="Allow category subgraphs when canonical full city-M/L files are absent.",
    )
    parser.add_argument(
        "--target-y-scale",
        default="log",
        choices=["linear", "log"],
        help="Y scale for speed/volume histograms.",
    )
    parser.add_argument(
        "--degree-y-scale",
        default="log",
        choices=["linear", "log"],
        help="Y scale for degree histograms.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


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


def city_from_name(name: str, summary: dict[str, Any]) -> str | None:
    text = f"{name} {summary.get('path', '')}".lower()
    paper_key = (
        summary.get("paper_context", {})
        .get("paper_dataset_key")
    )
    if paper_key == "city_traffic_m" or "city_traffic_m" in text or "city-traffic-m" in text:
        return "M"
    if paper_key == "city_traffic_l" or "city_traffic_l" in text or "city-traffic-l" in text:
        return "L"
    return None


def target_kind_from_name(name: str, summary: dict[str, Any]) -> str | None:
    text = f"{name} {summary.get('path', '')}".lower()
    if "speed" in text:
        return "speed"
    if "volume" in text:
        return "volume"
    return None


def is_subgraph(summary: dict[str, Any]) -> bool:
    return bool(
        summary.get("paper_context", {})
        .get("is_likely_category_subgraph", False)
    )


def dataset_priority(name: str, summary: dict[str, Any]) -> tuple[int, int, str]:
    """Lower priority tuple is better."""
    full_count_penalty = 0 if not is_subgraph(summary) else 10
    name_penalty = 0 if "full" in name.lower() else 1
    return (full_count_penalty, name_penalty, name)


def discover_profiles(profile_dir: Path, include_subgraphs: bool) -> dict[tuple[str, str], dict[str, Any]]:
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for summary_path in sorted(profile_dir.glob("*/summary.json")):
        summary = read_json(summary_path)
        name = str(summary.get("dataset", summary_path.parent.name))
        if is_subgraph(summary) and not include_subgraphs:
            continue
        city = city_from_name(name, summary)
        kind = target_kind_from_name(name, summary)
        if city is None or kind is None:
            continue
        current = selected.get((city, kind))
        candidate = {"name": name, "dir": summary_path.parent, "summary": summary}
        if current is None or dataset_priority(name, summary) < dataset_priority(
            current["name"], current["summary"]
        ):
            selected[(city, kind)] = candidate
    return selected


def load_histogram(profile: dict[str, Any]) -> tuple[list[float], list[float], list[float]]:
    rows = read_csv_rows(Path(profile["dir"]) / "target_histogram.csv")
    left = [float(r["bin_left"]) for r in rows]
    right = [float(r["bin_right"]) for r in rows]
    count = [float(r["count"]) for r in rows]
    return left, right, count


def load_degree_rows(profile: dict[str, Any]) -> dict[str, tuple[list[int], list[float]]]:
    rows = read_csv_rows(Path(profile["dir"]) / "degree_distribution.csv")
    out: dict[str, list[tuple[int, int]]] = {"in": [], "out": [], "total": []}
    for r in rows:
        degree_type = r["degree_type"]
        if degree_type not in out:
            continue
        out[degree_type].append((int(r["degree"]), int(r["num_nodes"])))
    packed: dict[str, tuple[list[int], list[float]]] = {}
    for degree_type, pairs in out.items():
        pairs = sorted(pairs)
        packed[degree_type] = ([p[0] for p in pairs], [float(p[1]) for p in pairs])
    return packed


def setup_matplotlib() -> Any:
    try:
        import matplotlib
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it in the server environment "
            "or run with a Python environment that already has matplotlib."
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "font.size": 9,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
        }
    )
    return plt


def save_figure(fig: Any, output_dir: Path, stem: str, formats: list[str]) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        fmt = fmt.strip().lstrip(".")
        if not fmt:
            continue
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path)
        paths.append(str(path))
    return paths


def format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.1f}%"


def plot_target_distributions(
    profiles: dict[tuple[str, str], dict[str, Any]],
    output_dir: Path,
    formats: list[str],
    y_scale: str,
) -> list[str]:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8))
    colors = {"speed": "#4E79A7", "volume": "#F28E2B"}
    panel_specs = [
        ("M", "speed"),
        ("M", "volume"),
        ("L", "speed"),
        ("L", "volume"),
    ]

    for ax, (city, kind) in zip(axes.reshape(-1), panel_specs):
        profile = profiles.get((city, kind))
        if profile is None:
            ax.text(0.5, 0.5, "missing profile", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"City-{city} {kind}")
            ax.set_axis_off()
            continue

        left, right, count = load_histogram(profile)
        total = float(sum(count))
        width = [r - l for l, r in zip(left, right)]
        probability = [c / total for c in count] if total > 0 else count
        ax.bar(
            left,
            probability,
            width=width,
            align="edge",
            color=colors[kind],
            edgecolor="white",
            linewidth=0.25,
            alpha=0.88,
        )
        ax.set_title(f"City-{city} {kind.capitalize()}")
        ax.set_xlabel("Speed" if kind == "speed" else "Volume")
        ax.set_ylabel("Probability mass")
        if y_scale == "log":
            positive = [p for p in probability if p > 0]
            if positive:
                ax.set_yscale("log")
                ax.set_ylim(bottom=max(min(positive) * 0.7, 1e-8))

        dist = profile["summary"].get("target_distribution_sample", {})
        p50 = safe_float(dist.get("p50"))
        p95 = safe_float(dist.get("p95"))
        nan_ratio = safe_float(dist.get("nan_ratio"))
        zero_ratio = safe_float(dist.get("zero_ratio"))
        sample_values = dist.get("sample_values")
        stats = [
            f"p50={p50:.2f}" if p50 is not None else "p50=n/a",
            f"p95={p95:.2f}" if p95 is not None else "p95=n/a",
            f"missing={format_percent(nan_ratio)}",
            f"zero={format_percent(zero_ratio)}",
        ]
        if sample_values:
            stats.append(f"n={int(sample_values):,}")
        ax.text(
            0.98,
            0.96,
            "\n".join(stats),
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.74, "pad": 2},
        )

    fig.suptitle("Target distributions in city-traffic-M/L", y=1.02, fontsize=11)
    fig.tight_layout()
    paths = save_figure(fig, output_dir, "city_traffic_target_distributions", formats)
    plt.close(fig)
    return paths


def select_graph_profiles(
    profiles: dict[tuple[str, str], dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for city in ["M", "L"]:
        selected[city] = profiles.get((city, "speed")) or profiles.get((city, "volume"))
    return {k: v for k, v in selected.items() if v is not None}


def plot_degree_distributions(
    profiles: dict[tuple[str, str], dict[str, Any]],
    output_dir: Path,
    formats: list[str],
    y_scale: str,
) -> list[str]:
    plt = setup_matplotlib()
    graph_profiles = select_graph_profiles(profiles)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=False)
    colors = {"in": "#4E79A7", "out": "#F28E2B", "total": "#59A14F"}
    labels = {"in": "In-degree", "out": "Out-degree", "total": "Total degree"}

    for ax, city in zip(axes, ["M", "L"]):
        profile = graph_profiles.get(city)
        if profile is None:
            ax.text(0.5, 0.5, "missing profile", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"City-{city}")
            ax.set_axis_off()
            continue

        degree_rows = load_degree_rows(profile)
        for degree_type in ["in", "out", "total"]:
            x, y = degree_rows[degree_type]
            ax.plot(
                x,
                y,
                marker="o",
                markersize=2.6,
                linewidth=1.1,
                color=colors[degree_type],
                label=labels[degree_type],
            )
        if y_scale == "log":
            ax.set_yscale("log")
        ax.set_xlabel("Degree")
        ax.set_ylabel("Number of nodes")
        ax.set_title(f"City-{city} graph degree distribution")
        degree_summary = profile["summary"].get("degree", {})
        num_nodes = degree_summary.get("num_nodes")
        num_edges = degree_summary.get("num_edges")
        isolated = degree_summary.get("num_isolated_nodes")
        annotation = []
        if num_nodes is not None:
            annotation.append(f"|V|={int(num_nodes):,}")
        if num_edges is not None:
            annotation.append(f"|E|={int(num_edges):,}")
        if isolated is not None:
            annotation.append(f"isolated={int(isolated):,}")
        ax.text(
            0.98,
            0.96,
            "\n".join(annotation),
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.74, "pad": 2},
        )
    handles, labels_list = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_list, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    paths = save_figure(fig, output_dir, "city_traffic_degree_distributions", formats)
    plt.close(fig)
    return paths


def write_summary_tables(
    profiles: dict[tuple[str, str], dict[str, Any]],
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "city_traffic_target_distribution_summary.csv"
    degree_path = output_dir / "city_traffic_degree_summary.csv"

    target_fields = [
        "city",
        "target",
        "dataset",
        "sample_values",
        "finite_count",
        "nan_ratio",
        "zero_ratio",
        "mean",
        "std",
        "p50",
        "p95",
        "p99",
        "min",
        "max",
    ]
    with target_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=target_fields)
        writer.writeheader()
        for city, kind in [("M", "speed"), ("M", "volume"), ("L", "speed"), ("L", "volume")]:
            profile = profiles.get((city, kind))
            if profile is None:
                continue
            dist = profile["summary"].get("target_distribution_sample", {})
            writer.writerow(
                {
                    "city": city,
                    "target": kind,
                    "dataset": profile["name"],
                    **{field: dist.get(field) for field in target_fields if field not in {"city", "target", "dataset"}},
                }
            )

    degree_fields = [
        "city",
        "dataset",
        "num_nodes",
        "num_edges",
        "num_isolated_nodes",
        "in_degree_p50",
        "in_degree_p95",
        "out_degree_p50",
        "out_degree_p95",
        "total_degree_p50",
        "total_degree_p95",
        "total_degree_max",
    ]
    with degree_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=degree_fields)
        writer.writeheader()
        for city, profile in select_graph_profiles(profiles).items():
            degree = profile["summary"].get("degree", {})
            writer.writerow(
                {
                    "city": city,
                    "dataset": profile["name"],
                    "num_nodes": degree.get("num_nodes"),
                    "num_edges": degree.get("num_edges"),
                    "num_isolated_nodes": degree.get("num_isolated_nodes"),
                    "in_degree_p50": degree.get("in_degree", {}).get("p50"),
                    "in_degree_p95": degree.get("in_degree", {}).get("p95"),
                    "out_degree_p50": degree.get("out_degree", {}).get("p50"),
                    "out_degree_p95": degree.get("out_degree", {}).get("p95"),
                    "total_degree_p50": degree.get("total_degree", {}).get("p50"),
                    "total_degree_p95": degree.get("total_degree", {}).get("p95"),
                    "total_degree_max": degree.get("total_degree", {}).get("max"),
                }
            )
    return [str(target_path), str(degree_path)]


def main() -> None:
    args = parse_args()
    profile_dir = Path(args.profile_dir)
    output_dir = Path(args.output_dir)
    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]

    profiles = discover_profiles(profile_dir, include_subgraphs=args.include_subgraphs)
    required = [("M", "speed"), ("M", "volume"), ("L", "speed"), ("L", "volume")]
    missing = [f"city-{city} {kind}" for city, kind in required if (city, kind) not in profiles]
    if missing:
        raise SystemExit(
            "Missing canonical profile outputs for: "
            + ", ".join(missing)
            + f". Expected files under {profile_dir}. "
            + "Run run_city_traffic_profile.sh first, or pass --include-subgraphs."
        )

    made = []
    made.extend(plot_target_distributions(profiles, output_dir, formats, args.target_y_scale))
    made.extend(plot_degree_distributions(profiles, output_dir, formats, args.degree_y_scale))
    made.extend(write_summary_tables(profiles, output_dir))

    print("Generated city-traffic profile figures/tables:")
    for path in made:
        print(f"  {path}")


if __name__ == "__main__":
    main()
