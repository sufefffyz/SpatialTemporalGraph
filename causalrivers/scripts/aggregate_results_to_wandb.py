#!/usr/bin/env python3
"""Aggregate CausalRivers benchmark outputs into flat tables and optionally upload to Weights & Biases."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency in local env
    yaml = None


PRIMARY_METRICS = ("AUROC", "Max F1", "Max Acc", "Individual AUROC")
RESULT_FILENAMES = ("scoring.csv", "runtime.csv", "config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan causalrivers/results, flatten scoring/runtime outputs into tables, "
            "write analysis summaries, and optionally upload them to wandb."
        )
    )
    repo_root = Path(__file__).resolve().parents[1]
    default_results_root = repo_root / "results"
    default_output_root = repo_root / "outputs" / "aggregated_results" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    parser.add_argument("--results-root", type=Path, default=default_results_root, help="Root directory containing benchmark result folders.")
    parser.add_argument("--output-dir", type=Path, default=default_output_root, help="Directory to store aggregated CSV/JSON/Markdown outputs.")
    parser.add_argument("--wandb-project", default="causalrivers-results", help="wandb project name.")
    parser.add_argument("--wandb-entity", default=None, help="wandb entity/team. Omit to use your default account.")
    parser.add_argument("--wandb-run-name", default=None, help="Explicit wandb run name. Defaults to the output directory name.")
    parser.add_argument("--wandb-tags", nargs="*", default=["causalrivers", "result-aggregation"], help="wandb tags to attach to the run.")
    parser.add_argument("--no-wandb", action="store_true", help="Skip wandb upload and only write local files.")
    parser.add_argument(
        "--primary-metrics",
        nargs="*",
        default=list(PRIMARY_METRICS),
        help="Metrics to highlight in the markdown analysis.",
    )
    return parser.parse_args()


def normalize_resolution(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "false":
        return None
    return text.lower().replace("hours", "h").replace("hour", "h")


def parse_scalar(raw: str) -> Any:
    text = raw.strip()
    if text == "":
        return None
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if "." not in text and "e" not in lowered:
            return int(text)
        return float(text)
    except ValueError:
        return text


def load_config_subset(config_path: Path) -> dict[str, Any]:
    if yaml is not None:
        with config_path.open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        data_preprocess = cfg.get("data_preprocess") or {}
        return {
            "label_path": cfg.get("label_path"),
            "data_path": cfg.get("data_path"),
            "save_path": cfg.get("save_path"),
            "load_mode": cfg.get("load_mode"),
            "dt_preprocess": cfg.get("dt_preprocess"),
            "resolution": data_preprocess.get("resolution"),
            "normalize": data_preprocess.get("normalize"),
            "interpolate": data_preprocess.get("interpolate"),
            "subset_month": data_preprocess.get("subset_month"),
            "subset_year": data_preprocess.get("subset_year"),
            "subsample": data_preprocess.get("subsample"),
            "remove_trailing_nans_early": data_preprocess.get("remove_trailing_nans_early"),
            "fill_remaining_nans_with_zero": data_preprocess.get("fill_remaining_nans_with_zero"),
        }

    # Fallback parser for environments without PyYAML.
    top_level_keys = {"label_path", "data_path", "save_path", "load_mode", "dt_preprocess"}
    dp_keys = {
        "resolution",
        "normalize",
        "interpolate",
        "subset_month",
        "subset_year",
        "subsample",
        "remove_trailing_nans_early",
        "fill_remaining_nans_with_zero",
    }
    out: dict[str, Any] = {}
    section: str | None = None
    with config_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                continue
            indent = len(raw_line) - len(raw_line.lstrip(" "))
            stripped = raw_line.strip()
            if indent == 0:
                section = stripped[:-1] if stripped.endswith(":") else None
                if ":" in stripped and not stripped.endswith(":"):
                    key, value = stripped.split(":", 1)
                    if key in top_level_keys:
                        out[key] = parse_scalar(value)
            elif section == "data_preprocess" and ":" in stripped:
                key, value = stripped.split(":", 1)
                if key in dp_keys:
                    out[key] = parse_scalar(value)
    return out


def parse_label_group(label_group: str) -> tuple[str | None, int | None, str | None]:
    match = re.match(r"^(?P<strategy>.+)_(?P<n_vars>\d+)(?:_(?P<label_tag>.+))?$", label_group)
    if not match:
        return None, None, None
    strategy = match.group("strategy")
    n_vars = int(match.group("n_vars"))
    label_tag = match.group("label_tag")
    return strategy, n_vars, label_tag


def parse_dataset_resolution_hint(dataset_name: str | None) -> str | None:
    if not dataset_name:
        return None
    match = re.search(r"_(\d+(?:min|h))$", dataset_name.lower())
    if match:
        return match.group(1)
    return None


def dataset_family(dataset_name: str | None) -> str:
    if not dataset_name:
        return "unknown"
    lowered = dataset_name.lower()
    if lowered.startswith("traffic"):
        return "traffic"
    if lowered.startswith("rivers"):
        return "rivers"
    return "other"


def extract_label_metadata(label_path: str | None) -> dict[str, Any]:
    meta = {
        "label_dataset_name": None,
        "label_group": None,
        "strategy": None,
        "n_vars": None,
        "label_tag": None,
        "label_file": None,
        "label_resolution_hint": None,
    }
    if not label_path:
        return meta
    path = Path(label_path)
    meta["label_file"] = path.stem
    parent = path.parent
    meta["label_group"] = parent.name if parent.name else None
    dataset_dir = parent.parent if parent.parent != parent else None
    meta["label_dataset_name"] = dataset_dir.name if dataset_dir is not None else None
    meta["label_resolution_hint"] = parse_dataset_resolution_hint(meta["label_dataset_name"])
    if meta["label_group"]:
        strategy, n_vars, label_tag = parse_label_group(meta["label_group"])
        meta["strategy"] = strategy
        meta["n_vars"] = n_vars
        meta["label_tag"] = label_tag
    return meta


DEDUP_ID_FIELDS = (
    "data_dataset_name",
    "label_dataset_name",
    "dataset_family",
    "label_group",
    "strategy",
    "n_vars",
    "label_tag",
    "config_resolution",
    "method",
)


def build_compact_metric_rows(metric_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in metric_summary:
        rows.append(
            {
                "data_dataset_name": row.get("data_dataset_name"),
                "strategy": row.get("strategy"),
                "n_vars": row.get("n_vars"),
                "method": row.get("method"),
                "metric": row.get("metric"),
                "value": row.get("mean"),
            }
        )
    rows.sort(
        key=lambda row: (
            row.get("data_dataset_name") or "",
            row.get("strategy") or "",
            row.get("n_vars") or -1,
            row.get("method") or "",
            row.get("metric") or "",
        )
    )
    return rows


def _metric_signature(rows: list[dict[str, Any]]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted((str(row.get("metric")), row.get("value")) for row in rows))


def deduplicate_metric_rows(metric_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    run_method_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        run_method_groups[(str(row.get("run_dir")), str(row.get("method")))].append(row)

    signature_groups: dict[tuple[Any, ...], list[tuple[tuple[str, str], list[dict[str, Any]]]]] = defaultdict(list)
    for run_method_key, rows in run_method_groups.items():
        exemplar = rows[0]
        group_key = tuple(exemplar.get(field) for field in DEDUP_ID_FIELDS) + (_metric_signature(rows),)
        signature_groups[group_key].append((run_method_key, rows))

    dedup_rows: list[dict[str, Any]] = []
    run_method_meta: dict[tuple[str, str], dict[str, Any]] = {}
    for group_index, grouped_runs in enumerate(signature_groups.values(), start=1):
        dedup_group_id = f"dedup_{group_index:06d}"
        merged_run_dirs = sorted({run_dir for (run_dir, _method), _rows in grouped_runs})
        merged_run_dirs_str = ";".join(merged_run_dirs)
        duplicate_run_count = len(grouped_runs)
        representative_rows = grouped_runs[0][1]

        for row in representative_rows:
            new_row = dict(row)
            new_row["duplicate_run_count"] = duplicate_run_count
            new_row["merged_run_dirs"] = merged_run_dirs_str
            new_row["dedup_group_id"] = dedup_group_id
            dedup_rows.append(new_row)

        for run_method_key, _rows in grouped_runs:
            run_method_meta[run_method_key] = {
                "dedup_group_id": dedup_group_id,
                "duplicate_run_count": duplicate_run_count,
                "merged_run_dirs": merged_run_dirs_str,
            }

    dedup_rows.sort(
        key=lambda row: (
            row.get("data_dataset_name") or "",
            row.get("label_group") or "",
            row.get("method") or "",
            row.get("metric") or "",
        )
    )
    return dedup_rows, run_method_meta


def deduplicate_runtime_rows(
    runtime_rows: list[dict[str, Any]],
    run_method_meta: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    passthrough_rows: list[dict[str, Any]] = []

    for row in runtime_rows:
        method = str(row.get("method"))
        if method == "overall":
            passthrough_rows.append(dict(row))
            continue
        meta = run_method_meta.get((str(row.get("run_dir")), method))
        if meta is None:
            passthrough_rows.append(dict(row))
            continue
        grouped[meta["dedup_group_id"]].append(row)

    dedup_rows: list[dict[str, Any]] = []
    for dedup_group_id, rows in grouped.items():
        exemplar = dict(rows[0])
        runtime_values = [row.get("runtime_seconds") for row in rows if row.get("runtime_seconds") is not None]
        exemplar["dedup_group_id"] = dedup_group_id
        exemplar["duplicate_run_count"] = len(rows)
        exemplar["merged_run_dirs"] = ";".join(sorted({str(row.get("run_dir")) for row in rows}))
        exemplar["runtime_seconds"] = mean(runtime_values) if runtime_values else None
        exemplar["runtime_str"] = None
        dedup_rows.append(exemplar)

    passthrough_rows.sort(key=lambda row: (row.get("data_dataset_name") or "", row.get("label_group") or "", row.get("method") or ""))
    dedup_rows.sort(key=lambda row: (row.get("data_dataset_name") or "", row.get("label_group") or "", row.get("method") or ""))
    return dedup_rows + passthrough_rows


def parse_runtime_seconds(raw: str | None) -> float | None:
    if not raw:
        return None
    text = raw.strip()
    day_match = re.match(r"^(?P<days>\d+)\s+days?\s+(?P<rest>\d+:\d+:\d+(?:\.\d+)?)$", text)
    day_offset = 0
    if day_match:
        day_offset = int(day_match.group("days")) * 86400
        text = day_match.group("rest")
    try:
        hours, minutes, seconds = text.split(":")
        return day_offset + int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except ValueError:
        return None


def maybe_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "":
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def discover_run_dirs(results_root: Path) -> list[Path]:
    run_dirs: list[Path] = []
    if not results_root.exists():
        return run_dirs
    for scoring_path in results_root.rglob("scoring.csv"):
        run_dir = scoring_path.parent
        if all((run_dir / filename).exists() for filename in RESULT_FILENAMES):
            run_dirs.append(run_dir)
    return sorted(set(run_dirs))


def load_runtime_map(runtime_path: Path, methods: list[str]) -> dict[str, tuple[str | None, float | None]]:
    runtime_map: dict[str, tuple[str | None, float | None]] = {}
    with runtime_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        return runtime_map

    if "name" in (reader.fieldnames or []):
        for row in rows:
            method_name = row.get("name")
            runtime_str = row.get("runtime")
            runtime_map[method_name] = (runtime_str, parse_runtime_seconds(runtime_str))
        return runtime_map

    # Single-run files are written via pandas.DataFrame(...).to_csv() and look like:
    # ,runtime
    # 0,0 days 00:00:00.285902
    if len(methods) == 1:
        runtime_str = rows[0].get("runtime")
        runtime_map[methods[0]] = (runtime_str, parse_runtime_seconds(runtime_str))
    return runtime_map


def aggregate_results(results_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    metric_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for run_dir in discover_run_dirs(results_root):
        config_path = run_dir / "config.yaml"
        scoring_path = run_dir / "scoring.csv"
        runtime_path = run_dir / "runtime.csv"

        cfg = load_config_subset(config_path)
        label_meta = extract_label_metadata(cfg.get("label_path"))
        data_dataset_name = Path(cfg["data_path"]).name if cfg.get("data_path") else None
        data_resolution_hint = parse_dataset_resolution_hint(data_dataset_name)
        config_resolution = normalize_resolution(cfg.get("resolution"))
        label_dataset_name = label_meta["label_dataset_name"]
        label_resolution_hint = label_meta["label_resolution_hint"]

        base_meta = {
            "run_dir": str(run_dir),
            "run_group": run_dir.parent.name,
            "run_timestamp": run_dir.name,
            "results_kind": "multi" if run_dir.parent.name.startswith("multi_") else "single",
            "data_dataset_name": data_dataset_name,
            "label_dataset_name": label_dataset_name,
            "dataset_family": dataset_family(data_dataset_name or label_dataset_name),
            "label_group": label_meta["label_group"],
            "strategy": label_meta["strategy"],
            "n_vars": label_meta["n_vars"],
            "label_tag": label_meta["label_tag"],
            "label_file": label_meta["label_file"],
            "data_path": cfg.get("data_path"),
            "label_path": cfg.get("label_path"),
            "save_path": cfg.get("save_path"),
            "load_mode": cfg.get("load_mode"),
            "dt_preprocess": cfg.get("dt_preprocess"),
            "config_resolution": config_resolution,
            "data_resolution_hint": data_resolution_hint,
            "label_resolution_hint": label_resolution_hint,
            "resolution_hint_mismatch": bool(config_resolution and data_resolution_hint and config_resolution != data_resolution_hint),
            "dataset_name_mismatch": bool(label_dataset_name and data_dataset_name and label_dataset_name != data_dataset_name),
            "normalize": cfg.get("normalize"),
            "interpolate": cfg.get("interpolate"),
            "subset_month": cfg.get("subset_month"),
            "subset_year": cfg.get("subset_year"),
            "subsample": cfg.get("subsample"),
            "remove_trailing_nans_early": cfg.get("remove_trailing_nans_early"),
            "fill_remaining_nans_with_zero": cfg.get("fill_remaining_nans_with_zero"),
        }

        if base_meta["dataset_name_mismatch"]:
            warnings.append(
                f"Dataset name mismatch in {run_dir}: label dataset={label_dataset_name}, data dataset={data_dataset_name}"
            )
        if base_meta["resolution_hint_mismatch"]:
            warnings.append(
                f"Resolution mismatch in {run_dir}: config={config_resolution}, data hint={data_resolution_hint}"
            )

        with scoring_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            rows = list(reader)
        if not rows:
            warnings.append(f"Empty scoring.csv in {run_dir}")
            continue
        methods = rows[0][1:]
        runtime_map = load_runtime_map(runtime_path, methods)
        for method_name, (runtime_str, runtime_seconds) in runtime_map.items():
            runtime_rows.append(
                {
                    **base_meta,
                    "method": method_name,
                    "runtime_str": runtime_str,
                    "runtime_seconds": runtime_seconds,
                }
            )
        for row in rows[1:]:
            if not row:
                continue
            metric_name = row[0]
            if metric_name.startswith("Null "):
                null_metric_name = metric_name.replace("Null ", "", 1)
                null_values = []
                for raw_value in row[1:]:
                    value = maybe_float(raw_value)
                    if value is not None:
                        null_values.append(value)
                unique_null_values = sorted(set(null_values))
                if len(unique_null_values) > 1:
                    warnings.append(
                        f"Null metric values differ across methods in {run_dir} for {metric_name}: {unique_null_values}"
                    )
                metric_rows.append(
                    {
                        **base_meta,
                        "method": "NULL",
                        "metric": null_metric_name,
                        "value": unique_null_values[0] if unique_null_values else None,
                        "runtime_str": None,
                        "runtime_seconds": None,
                    }
                )
                continue
            for method_name, raw_value in zip(methods, row[1:]):
                value = maybe_float(raw_value)
                runtime_str, runtime_seconds = runtime_map.get(method_name, (None, None))
                metric_rows.append(
                    {
                        **base_meta,
                        "method": method_name,
                        "metric": metric_name,
                        "value": value,
                        "runtime_str": runtime_str,
                        "runtime_seconds": runtime_seconds,
                    }
                )
    return metric_rows, runtime_rows, warnings


def summarize_rows(rows: list[dict[str, Any]], group_fields: list[str], value_field: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    exemplars: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        value = row.get(value_field)
        if value is None:
            continue
        key = tuple(row.get(field) for field in group_fields)
        grouped[key].append(value)
        exemplars.setdefault(key, {field: row.get(field) for field in group_fields})

    summary_rows: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        exemplar = exemplars[key]
        summary_rows.append(
            {
                **exemplar,
                "count": len(values),
                "mean": mean(values),
                "std": pstdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }
        )
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_analysis(metric_summary: list[dict[str, Any]], runtime_summary: list[dict[str, Any]], warnings: list[str], primary_metrics: list[str]) -> str:
    lines: list[str] = ["# Aggregated CausalRivers Results", ""]
    run_keys = {
        (
            row.get("data_dataset_name"),
            row.get("label_group"),
            row.get("config_resolution"),
            row.get("method"),
        )
        for row in metric_summary
    }
    lines.append(f"- Metric summary rows: {len(metric_summary)}")
    lines.append(f"- Runtime summary rows: {len(runtime_summary)}")
    lines.append(f"- Unique dataset/label_group/resolution/method combinations: {len(run_keys)}")
    lines.append("")

    if warnings:
        lines.append("## Warnings")
        for warning in sorted(set(warnings)):
            lines.append(f"- {warning}")
        lines.append("")

    lines.append("## Best Methods By Primary Metric")
    for metric in primary_metrics:
        rows = [row for row in metric_summary if row.get("metric") == metric]
        if not rows:
            continue
        lines.append(f"### {metric}")
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            key = (row.get("data_dataset_name"), row.get("label_group"))
            grouped[key].append(row)
        for (dataset_name, label_group), candidates in sorted(grouped.items()):
            best = max(candidates, key=lambda item: item["mean"])
            baseline = next((item for item in candidates if item.get("method") == "var"), None)
            delta_text = ""
            if baseline is not None and baseline.get("method") != best.get("method"):
                delta = best["mean"] - baseline["mean"]
                delta_text = f", delta vs var={delta:+.4f}"
            lines.append(
                f"- dataset={dataset_name}, label_group={label_group}: best={best['method']} "
                f"(mean={best['mean']:.4f}, std={best['std']:.4f}, n={best['count']}{delta_text})"
            )
        lines.append("")

    lines.append("## Fastest Methods")
    runtime_grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in runtime_summary:
        if row.get("method") == "overall":
            continue
        key = (row.get("data_dataset_name"), row.get("label_group"))
        runtime_grouped[key].append(row)
    for (dataset_name, label_group), rows in sorted(runtime_grouped.items()):
        fastest = min(rows, key=lambda item: item["mean"])
        lines.append(
            f"- dataset={dataset_name}, label_group={label_group}: fastest={fastest['method']} "
            f"(mean_runtime={fastest['mean']:.3f}s, n={fastest['count']})"
        )
    lines.append("")

    return "\n".join(lines)


def build_wandb_table(rows: list[dict[str, Any]]):
    import wandb

    if not rows:
        return wandb.Table(columns=[])
    columns = list(rows[0].keys())
    data = [[row.get(column) for column in columns] for row in rows]
    return wandb.Table(columns=columns, data=data)


def upload_to_wandb(
    args: argparse.Namespace,
    output_dir: Path,
    metric_rows_raw: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    metric_summary: list[dict[str, Any]],
    compact_metric_rows: list[dict[str, Any]],
    runtime_rows_raw: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
    runtime_summary: list[dict[str, Any]],
    analysis_path: Path,
    manifest: dict[str, Any],
) -> str:
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError("wandb is not installed. Install it or pass --no-wandb.") from exc

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or output_dir.name,
        job_type="result_aggregation",
        tags=args.wandb_tags,
        config={
            "results_root": str(args.results_root),
            "output_dir": str(output_dir),
            "primary_metrics": args.primary_metrics,
            "num_metric_rows": len(metric_rows),
            "num_metric_rows_raw": len(metric_rows_raw),
            "num_runtime_rows": len(runtime_rows),
            "num_runtime_rows_raw": len(runtime_rows_raw),
        },
    )

    run.summary["num_metric_rows"] = len(metric_rows)
    run.summary["num_metric_rows_raw"] = len(metric_rows_raw)
    run.summary["num_runtime_rows"] = len(runtime_rows)
    run.summary["num_runtime_rows_raw"] = len(runtime_rows_raw)
    run.summary["num_runs_scanned"] = manifest["num_run_dirs"]

    wandb.log(
        {
            "tables/metrics_long": build_wandb_table(metric_rows),
            "tables/metrics_long_raw": build_wandb_table(metric_rows_raw),
            "tables/metrics_compact": build_wandb_table(compact_metric_rows),
            "tables/metrics_summary": build_wandb_table(metric_summary),
            "tables/runtime_long": build_wandb_table(runtime_rows),
            "tables/runtime_long_raw": build_wandb_table(runtime_rows_raw),
            "tables/runtime_summary": build_wandb_table(runtime_summary),
        }
    )

    artifact = wandb.Artifact(f"{run.name}-aggregated-results", type="aggregated-results")
    for output_file in sorted(output_dir.iterdir()):
        artifact.add_file(str(output_file), name=output_file.name)
    run.log_artifact(artifact)

    with analysis_path.open("r", encoding="utf-8") as handle:
        run.summary["analysis_preview"] = handle.read()

    run.finish()
    return getattr(run, "url", "") or ""


def main() -> int:
    args = parse_args()
    results_root = args.results_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows_raw, runtime_rows_raw, warnings = aggregate_results(results_root)
    metric_rows, run_method_meta = deduplicate_metric_rows(metric_rows_raw)
    runtime_rows = deduplicate_runtime_rows(runtime_rows_raw, run_method_meta)

    metric_summary = summarize_rows(
        metric_rows,
        [
            "data_dataset_name",
            "label_dataset_name",
            "dataset_family",
            "label_group",
            "strategy",
            "n_vars",
            "label_tag",
            "config_resolution",
            "method",
            "metric",
        ],
        "value",
    )
    runtime_summary = summarize_rows(
        runtime_rows,
        [
            "data_dataset_name",
            "label_dataset_name",
            "dataset_family",
            "label_group",
            "strategy",
            "n_vars",
            "label_tag",
            "config_resolution",
            "method",
        ],
        "runtime_seconds",
    )
    compact_metric_rows = build_compact_metric_rows(metric_summary)

    metrics_long_raw_path = output_dir / "metrics_long_raw.csv"
    metrics_long_path = output_dir / "metrics_long.csv"
    metrics_compact_path = output_dir / "metrics_compact.csv"
    metrics_summary_path = output_dir / "metrics_summary.csv"
    runtime_long_raw_path = output_dir / "runtime_long_raw.csv"
    runtime_long_path = output_dir / "runtime_long.csv"
    runtime_summary_path = output_dir / "runtime_summary.csv"
    analysis_path = output_dir / "analysis.md"
    manifest_path = output_dir / "manifest.json"

    write_csv(metrics_long_raw_path, metric_rows_raw)
    write_csv(metrics_long_path, metric_rows)
    write_csv(metrics_compact_path, compact_metric_rows)
    write_csv(metrics_summary_path, metric_summary)
    write_csv(runtime_long_raw_path, runtime_rows_raw)
    write_csv(runtime_long_path, runtime_rows)
    write_csv(runtime_summary_path, runtime_summary)

    analysis = build_analysis(metric_summary, runtime_summary, warnings, args.primary_metrics)
    analysis_path.write_text(analysis, encoding="utf-8")

    manifest = {
        "created_at": datetime.now().isoformat(),
        "results_root": str(results_root),
        "output_dir": str(output_dir),
        "num_run_dirs": len(discover_run_dirs(results_root)),
        "num_metric_rows_raw": len(metric_rows_raw),
        "num_metric_rows": len(metric_rows),
        "num_compact_metric_rows": len(compact_metric_rows),
        "num_runtime_rows_raw": len(runtime_rows_raw),
        "num_runtime_rows": len(runtime_rows),
        "num_metric_summary_rows": len(metric_summary),
        "num_runtime_summary_rows": len(runtime_summary),
        "warnings": sorted(set(warnings)),
        "files": {
            "metrics_long_raw": str(metrics_long_raw_path),
            "metrics_long": str(metrics_long_path),
            "metrics_compact": str(metrics_compact_path),
            "metrics_summary": str(metrics_summary_path),
            "runtime_long_raw": str(runtime_long_raw_path),
            "runtime_long": str(runtime_long_path),
            "runtime_summary": str(runtime_summary_path),
            "analysis": str(analysis_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    wandb_url = ""
    if not args.no_wandb:
        wandb_url = upload_to_wandb(
            args,
            output_dir,
            metric_rows_raw,
            metric_rows,
            metric_summary,
            compact_metric_rows,
            runtime_rows_raw,
            runtime_rows,
            runtime_summary,
            analysis_path,
            manifest,
        )

    print(f"Aggregated outputs written to: {output_dir}")
    print(f"Metric rows: {len(metric_rows)} (raw={len(metric_rows_raw)})")
    print(f"Compact metric rows: {len(compact_metric_rows)}")
    print(f"Runtime rows: {len(runtime_rows)} (raw={len(runtime_rows_raw)})")
    if warnings:
        print(f"Warnings: {len(set(warnings))}")
    if wandb_url:
        print(f"wandb URL: {wandb_url}")
    print()
    print(analysis)
    return 0


if __name__ == "__main__":
    sys.exit(main())
