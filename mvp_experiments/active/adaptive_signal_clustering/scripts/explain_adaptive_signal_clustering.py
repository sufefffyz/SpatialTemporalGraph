#!/usr/bin/env python3
"""Post-hoc explanation figures for daily signal clustering outputs.

The clustering runner writes daily profiles and aligned labels. This script
only reads those artifacts and adds explanation-oriented diagnostics:

- global center IQR bands,
- selected-day center IQR bands,
- day-to-day transition matrices,
- uncertain / low-confidence timelines,
- cluster fraction timelines,
- dominant-pattern maps when coordinates are available.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[3]
    default_run = (
        repo_root
        / "mvp_experiments"
        / "active"
        / "adaptive_signal_clustering"
        / "outputs"
        / "traffic_daily_zshape_fixed_merge_30d_20260604"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=default_run)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--output-subdir", default="explanations")
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--selected-days", nargs="*", type=int, default=None, help="1-indexed days. Defaults to first/middle/last.")
    parser.add_argument("--max-map-days", type=int, default=8)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def discover_datasets(run_dir: Path, requested: list[str] | None) -> list[Path]:
    if requested:
        return [run_dir / name for name in requested]
    return sorted(p for p in run_dir.iterdir() if p.is_dir() and (p / "daily_profiles_time_zshape.npy").exists())


def discover_methods(dataset_dir: Path, requested: list[str] | None) -> list[Path]:
    if requested:
        return [dataset_dir / name for name in requested]
    return sorted(p for p in dataset_dir.iterdir() if p.is_dir() and (p / "labels_global.npy").exists())


def cluster_names(k: int, include_uncertain: bool) -> list[str]:
    names = [f"C{i}" for i in range(k)]
    if include_uncertain:
        names.append("U")
    return names


def label_to_index(label: int, k: int, include_uncertain: bool) -> int | None:
    if label >= 0:
        return int(label)
    if include_uncertain:
        return k
    return None


def selected_days(num_days: int, requested: Iterable[int] | None) -> list[int]:
    if requested:
        days = []
        for day in requested:
            idx = int(day) - 1
            if 0 <= idx < num_days:
                days.append(idx)
        return sorted(set(days))
    candidates = [0, num_days // 2, num_days - 1]
    return sorted(set(idx for idx in candidates if 0 <= idx < num_days))


def safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    den_b = np.broadcast_to(den, num.shape)
    out = np.zeros(num.shape, dtype=np.float64)
    np.divide(num, den_b, out=out, where=den_b > 0)
    return out


def compute_transition(labels: np.ndarray, include_uncertain: bool = True) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    k = int(labels[labels >= 0].max()) + 1 if np.any(labels >= 0) else 0
    size = k + int(include_uncertain)
    counts = np.zeros((size, size), dtype=np.int64)
    rows = []
    for day in range(labels.shape[0] - 1):
        daily = np.zeros((size, size), dtype=np.int64)
        for src, dst in zip(labels[day], labels[day + 1]):
            si = label_to_index(int(src), k, include_uncertain)
            di = label_to_index(int(dst), k, include_uncertain)
            if si is None or di is None:
                continue
            daily[si, di] += 1
        counts += daily
        total = max(1, int(daily.sum()))
        stay = int(np.trace(daily[:k, :k])) if k else 0
        from_uncertain = int(daily[k, :].sum()) if include_uncertain and size else 0
        to_uncertain = int(daily[:, k].sum()) if include_uncertain and size else 0
        row_sum = daily.sum(axis=1)
        prob = safe_divide(daily, row_sum[:, None])
        entropy = 0.0
        active_rows = 0
        for r in range(size):
            if row_sum[r] > 0:
                vals = prob[r][prob[r] > 0]
                entropy += float(-(vals * np.log(vals + 1e-12)).sum())
                active_rows += 1
        rows.append(
            {
                "day_from": day + 1,
                "day_to": day + 2,
                "stay_fraction": stay / total,
                "from_uncertain_fraction": from_uncertain / total,
                "to_uncertain_fraction": to_uncertain / total,
                "mean_row_entropy": entropy / max(1, active_rows),
            }
        )
    probs = safe_divide(counts, counts.sum(axis=1, keepdims=True))
    return counts, probs, pd.DataFrame(rows)


def plot_transition_matrix(probs: np.ndarray, names: list[str], title: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(5.2, 0.52 * len(names)), max(4.4, 0.46 * len(names))))
    im = ax.imshow(probs, cmap="mako" if "mako" in plt.colormaps() else "viridis", vmin=0.0, vmax=max(0.2, float(np.nanmax(probs))))
    ax.set_xticks(np.arange(len(names)), labels=names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(names)), labels=names)
    ax.set_xlabel("next day global pattern")
    ax.set_ylabel("current day global pattern")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="row-normalized probability")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_transition_timeline(rows: pd.DataFrame, dataset: str, method: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    ax.plot(rows["day_to"], rows["stay_fraction"], marker="o", markersize=3, linewidth=1.4, label="stay")
    ax.plot(rows["day_to"], rows["mean_row_entropy"], marker="s", markersize=3, linewidth=1.4, label="row entropy")
    ax.set_xlabel("Day")
    ax.set_ylabel("fraction / entropy")
    ax.set_title(f"{dataset} {method}: day-to-day transition stability")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def profile_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "q25": np.quantile(values, 0.25, axis=0),
        "median": np.quantile(values, 0.50, axis=0),
        "q75": np.quantile(values, 0.75, axis=0),
        "mean": values.mean(axis=0),
    }


def plot_global_center_iqr(profiles: np.ndarray, labels: np.ndarray, dataset: str, method: str, output: Path) -> dict[str, int]:
    k = int(labels[labels >= 0].max()) + 1 if np.any(labels >= 0) else 0
    cols = min(4, max(1, k))
    rows = int(math.ceil(max(1, k) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 2.9 * rows), squeeze=False, sharex=True, sharey=True)
    x = np.arange(profiles.shape[2])
    cmap = plt.get_cmap("tab20", max(20, k))
    counts = {}
    flat_profiles = profiles.reshape(-1, profiles.shape[2])
    flat_labels = labels.reshape(-1)
    for idx, ax in enumerate(axes.ravel()):
        if idx >= k:
            ax.axis("off")
            continue
        vals = flat_profiles[flat_labels == idx]
        counts[f"C{idx}"] = int(len(vals))
        if len(vals) == 0:
            ax.axis("off")
            continue
        stats = profile_stats(vals)
        ax.fill_between(x, stats["q25"], stats["q75"], color=cmap(idx), alpha=0.22, linewidth=0)
        ax.plot(x, stats["median"], color=cmap(idx), linewidth=2.0, label="median")
        ax.plot(x, stats["mean"], color="black", linewidth=1.0, alpha=0.55, label="mean")
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.28)
        ax.set_title(f"C{idx} n={len(vals)}")
        ax.grid(True, alpha=0.18)
    for ax in axes[-1, :]:
        ax.set_xlabel("15-min step in day")
    for ax in axes[:, 0]:
        ax.set_ylabel("z-shape")
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_, loc="upper right", frameon=False)
    fig.suptitle(f"{dataset} {method}: global pattern IQR bands", fontsize=13)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return counts


def plot_day_center_iqr(profiles: np.ndarray, labels: np.ndarray, day: int, dataset: str, method: str, output: Path) -> None:
    k = int(labels[labels >= 0].max()) + 1 if np.any(labels >= 0) else 0
    cols = min(4, max(1, k))
    rows = int(math.ceil(max(1, k) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 2.9 * rows), squeeze=False, sharex=True, sharey=True)
    x = np.arange(profiles.shape[2])
    cmap = plt.get_cmap("tab20", max(20, k))
    for idx, ax in enumerate(axes.ravel()):
        if idx >= k:
            ax.axis("off")
            continue
        vals = profiles[day, labels[day] == idx]
        if len(vals) == 0:
            ax.axis("off")
            continue
        stats = profile_stats(vals)
        ax.fill_between(x, stats["q25"], stats["q75"], color=cmap(idx), alpha=0.22, linewidth=0)
        ax.plot(x, stats["median"], color=cmap(idx), linewidth=2.0)
        ax.plot(x, stats["mean"], color="black", linewidth=1.0, alpha=0.55)
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.28)
        ax.set_title(f"C{idx} n={len(vals)}")
        ax.grid(True, alpha=0.18)
    for ax in axes[-1, :]:
        ax.set_xlabel("15-min step in day")
    for ax in axes[:, 0]:
        ax.set_ylabel("z-shape")
    fig.suptitle(f"{dataset} {method}: day {day + 1} pattern IQR bands", fontsize=13)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_uncertain_timeline(labels: np.ndarray, confidence: np.ndarray | None, threshold: float | None, dataset: str, method: str, output: Path) -> pd.DataFrame:
    rows = []
    for day in range(labels.shape[0]):
        row = {
            "day": day + 1,
            "assigned_uncertain_ratio": float(np.mean(labels[day] < 0)),
        }
        if confidence is not None:
            row["mean_confidence"] = float(np.mean(confidence[day]))
            row["bottom20_confidence"] = float(np.quantile(confidence[day], 0.2))
            if threshold is not None:
                row["low_confidence_ratio"] = float(np.mean(confidence[day] < threshold))
        rows.append(row)
    df = pd.DataFrame(rows)
    fig, ax1 = plt.subplots(figsize=(8.2, 3.4))
    ax1.plot(df["day"], df["assigned_uncertain_ratio"], marker="o", markersize=3, linewidth=1.4, label="assigned uncertain")
    if "low_confidence_ratio" in df:
        ax1.plot(df["day"], df["low_confidence_ratio"], marker="s", markersize=3, linewidth=1.4, label=f"confidence < {threshold:g}")
    ax1.set_xlabel("Day")
    ax1.set_ylabel("ratio")
    ax1.set_ylim(bottom=0.0)
    ax1.grid(True, alpha=0.25)
    if "mean_confidence" in df:
        ax2 = ax1.twinx()
        ax2.plot(df["day"], df["mean_confidence"], color="#4d4d4d", linewidth=1.2, alpha=0.75, label="mean confidence")
        ax2.set_ylabel("confidence")
        lines, labels_ = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels_ + labels2, frameon=False, loc="upper right")
    else:
        ax1.legend(frameon=False, loc="upper right")
    ax1.set_title(f"{dataset} {method}: uncertain and confidence timeline")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return df


def plot_cluster_fraction_timeline(labels: np.ndarray, dataset: str, method: str, output: Path) -> pd.DataFrame:
    k = int(labels[labels >= 0].max()) + 1 if np.any(labels >= 0) else 0
    rows = []
    fractions = []
    for day in range(labels.shape[0]):
        vals = []
        for cluster in range(k):
            frac = float(np.mean(labels[day] == cluster))
            vals.append(frac)
            rows.append({"day": day + 1, "cluster": cluster, "fraction": frac})
        unc = float(np.mean(labels[day] < 0))
        vals.append(unc)
        rows.append({"day": day + 1, "cluster": -1, "fraction": unc})
        fractions.append(vals)
    arr = np.asarray(fractions).T
    names = cluster_names(k, include_uncertain=True)
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    x = np.arange(1, labels.shape[0] + 1)
    cmap = plt.get_cmap("tab20", max(20, len(names)))
    ax.stackplot(x, arr, labels=names, colors=[cmap(i) for i in range(len(names))], alpha=0.88)
    ax.set_xlim(1, labels.shape[0])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Day")
    ax.set_ylabel("node fraction")
    ax.set_title(f"{dataset} {method}: global pattern composition over days")
    ax.legend(ncol=min(5, len(names)), fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def dominant_pattern(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dominant = np.full(labels.shape[1], -1, dtype=int)
    purity = np.zeros(labels.shape[1], dtype=np.float64)
    for node in range(labels.shape[1]):
        vals = [int(v) for v in labels[:, node] if int(v) >= 0]
        if not vals:
            continue
        label, count = Counter(vals).most_common(1)[0]
        dominant[node] = label
        purity[node] = count / labels.shape[0]
    return dominant, purity


def plot_dominant_map(meta: pd.DataFrame, dominant: np.ndarray, purity: np.ndarray, dataset: str, method: str, output: Path) -> None:
    k = int(dominant[dominant >= 0].max()) + 1 if np.any(dominant >= 0) else 0
    cmap = plt.get_cmap("tab20", max(20, k))
    lon = meta["_lon"].to_numpy()
    lat = meta["_lat"].to_numpy()
    size = np.clip(10.0 + 35.0 * purity, 8.0, 46.0)
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    uncertain = dominant < 0
    if np.any(~uncertain):
        sc = ax.scatter(
            lon[~uncertain],
            lat[~uncertain],
            c=dominant[~uncertain],
            s=size[~uncertain],
            cmap=cmap,
            vmin=-0.5,
            vmax=max(0.5, k - 0.5),
            alpha=0.88,
            linewidths=0,
        )
        cbar = fig.colorbar(sc, ax=ax, ticks=np.arange(k))
        cbar.set_label("dominant global pattern")
    if np.any(uncertain):
        ax.scatter(lon[uncertain], lat[uncertain], c="#bdbdbd", s=12, alpha=0.35, linewidths=0)
    ax.set_xlabel("Lon")
    ax.set_ylabel("Lat")
    ax.set_title(f"{dataset} {method}: dominant 30-day pattern map\nmarker size = dominant-day ratio")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.18, linewidth=0.5)
    fig.savefig(output, dpi=230, bbox_inches="tight")
    plt.close(fig)


def plot_category_composition(meta: pd.DataFrame, labels: np.ndarray, output_dir: Path, dataset: str, method: str) -> list[str]:
    candidates = []
    for col in meta.columns:
        low = str(col).lower()
        if low.startswith("_"):
            continue
        if any(key in low for key in ["freeway", "direction", "type", "road", "route", "district"]):
            nunique = meta[col].nunique(dropna=True)
            if 1 < nunique <= 30:
                candidates.append(col)
    if not candidates:
        return []
    dominant, _ = dominant_pattern(labels)
    k = int(dominant[dominant >= 0].max()) + 1 if np.any(dominant >= 0) else 0
    made = []
    for col in candidates[:4]:
        df = pd.DataFrame({"cluster": dominant, "category": meta[col].astype(str)})
        df = df[df["cluster"] >= 0]
        if df.empty:
            continue
        table = pd.crosstab(df["cluster"], df["category"], normalize="index")
        csv_path = output_dir / f"category_composition_{col}.csv"
        table.to_csv(csv_path)
        fig, ax = plt.subplots(figsize=(max(6.2, 0.42 * table.shape[1]), max(3.0, 0.36 * max(1, k))))
        im = ax.imshow(table.to_numpy(), aspect="auto", cmap="viridis", vmin=0)
        ax.set_xticks(np.arange(table.shape[1]), labels=table.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(table.shape[0]), labels=[f"C{i}" for i in table.index])
        ax.set_xlabel(col)
        ax.set_ylabel("dominant global pattern")
        ax.set_title(f"{dataset} {method}: pattern composition by {col}")
        fig.colorbar(im, ax=ax, label="row fraction")
        png_path = output_dir / f"category_composition_{col}.png"
        fig.savefig(png_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        made.append(str(png_path.name))
    return made


def load_confidence_threshold(run_dir: Path, override: float | None) -> float | None:
    if override is not None:
        return override
    args_path = run_dir / "args.json"
    if args_path.exists():
        args = read_json(args_path)
        val = args.get("confidence_threshold")
        if val is not None:
            return float(val)
    return None


def explain_method(dataset_dir: Path, method_dir: Path, args: argparse.Namespace, confidence_threshold: float | None) -> dict:
    dataset = dataset_dir.name
    method = method_dir.name
    profiles = np.load(dataset_dir / "daily_profiles_time_zshape.npy")
    labels = np.load(method_dir / "labels_global.npy")
    confidence_path = method_dir / "confidence.npy"
    confidence = np.load(confidence_path) if confidence_path.exists() else None
    if profiles.shape[:2] != labels.shape:
        raise ValueError(f"{dataset}/{method}: profile shape {profiles.shape} incompatible with labels {labels.shape}")

    output_dir = method_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    k = int(labels[labels >= 0].max()) + 1 if np.any(labels >= 0) else 0
    counts, probs, transition_rows = compute_transition(labels, include_uncertain=True)
    names = cluster_names(k, include_uncertain=True)
    pd.DataFrame(counts, index=names, columns=names).to_csv(output_dir / "transition_counts_avg.csv")
    pd.DataFrame(probs, index=names, columns=names).to_csv(output_dir / "transition_probs_avg.csv")
    transition_rows.to_csv(output_dir / "transition_timeline.csv", index=False)
    plot_transition_matrix(probs, names, f"{dataset} {method}: average day-to-day transitions", output_dir / "transition_matrix_avg.png")
    plot_transition_timeline(transition_rows, dataset, method, output_dir / "transition_timeline.png")

    center_counts = plot_global_center_iqr(profiles, labels, dataset, method, output_dir / "center_profiles_global_iqr.png")
    for day in selected_days(labels.shape[0], args.selected_days):
        plot_day_center_iqr(profiles, labels, day, dataset, method, output_dir / f"center_profiles_day_{day + 1:02d}_iqr.png")

    uncertain_rows = plot_uncertain_timeline(
        labels=labels,
        confidence=confidence,
        threshold=confidence_threshold,
        dataset=dataset,
        method=method,
        output=output_dir / "uncertain_ratio_timeline.png",
    )
    uncertain_rows.to_csv(output_dir / "uncertain_ratio_timeline.csv", index=False)
    fraction_rows = plot_cluster_fraction_timeline(labels, dataset, method, output_dir / "cluster_fraction_timeline.png")
    fraction_rows.to_csv(output_dir / "cluster_fraction_timeline.csv", index=False)

    dominant, purity = dominant_pattern(labels)
    dominant_df = pd.DataFrame({"node_index": np.arange(labels.shape[1]), "dominant_pattern": dominant, "dominant_day_ratio": purity})
    dominant_df.to_csv(output_dir / "node_dominant_pattern.csv", index=False)
    category_figures: list[str] = []
    meta_path = dataset_dir / "resolved_meta.csv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        if {"_lon", "_lat"}.issubset(meta.columns) and len(meta) == labels.shape[1]:
            plot_dominant_map(meta, dominant, purity, dataset, method, output_dir / "dominant_pattern_map.png")
        category_figures = plot_category_composition(meta, labels, output_dir, dataset, method)

    summary = {
        "dataset": dataset,
        "method": method,
        "num_days": int(labels.shape[0]),
        "num_nodes": int(labels.shape[1]),
        "num_global_patterns": int(k),
        "mean_stay_fraction": float(transition_rows["stay_fraction"].mean()) if len(transition_rows) else float("nan"),
        "mean_transition_entropy": float(transition_rows["mean_row_entropy"].mean()) if len(transition_rows) else float("nan"),
        "mean_assigned_uncertain_ratio": float(uncertain_rows["assigned_uncertain_ratio"].mean()),
        "mean_low_confidence_ratio": float(uncertain_rows["low_confidence_ratio"].mean()) if "low_confidence_ratio" in uncertain_rows else None,
        "mean_dominant_day_ratio": float(np.mean(purity)),
        "center_counts": center_counts,
        "category_figures": category_figures,
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "explanation_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    if not args.run_dir.exists():
        raise FileNotFoundError(args.run_dir)
    confidence_threshold = load_confidence_threshold(args.run_dir, args.confidence_threshold)
    summaries = []
    for dataset_dir in discover_datasets(args.run_dir, args.datasets):
        for method_dir in discover_methods(dataset_dir, args.methods):
            print(f"[explain] {dataset_dir.name}/{method_dir.name}", flush=True)
            summaries.append(explain_method(dataset_dir, method_dir, args, confidence_threshold))
    if summaries:
        df = pd.DataFrame(summaries)
        df.to_csv(args.run_dir / f"explanation_summary_{args.output_subdir}.csv", index=False)
        print(df[["dataset", "method", "num_global_patterns", "mean_stay_fraction", "mean_dominant_day_ratio"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
