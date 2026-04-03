#!/usr/bin/env python3
"""Build project-level advisor tables from both CausalRivers and BasicTS results."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required to run build_advisor_tables.py") from exc

from aggregate_results_to_wandb import (
    aggregate_results,
    deduplicate_metric_rows,
    normalize_resolution,
    parse_dataset_resolution_hint,
)


CAUSAL_PRIMARY_METRICS = ["AUROC", "Max F1", "Max Acc", "Individual AUROC"]
FORECAST_PRIMARY_METRICS = ["MAE", "RMSE", "MAPE", "WAPE", "SMAPE"]
FORECAST_SCOPES = ("overall", "horizon_3", "horizon_6", "horizon_12")
TRACK_MODEL_ORDER = {
    "A-causal": ["VAR", "CC", "LagCC", "RP", "Combo", "PCMCI", "VARLiNGAM", "NULL"],
    "B-causal": ["AGCRN", "D2STGNN", "GWNET", "MTGNN", "GTS", "NULL"],
}
CONFIG_DIR_PATTERN = re.compile(
    r"^(?P<dataset>.+)_(?P<epochs>\d+)_(?P<input_len>\d+)_(?P<output_len>\d+)$"
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Build advisor-facing project summary tables from causalrivers and BasicTS results."
    )
    parser.add_argument(
        "--causal-results-root",
        type=Path,
        action="append",
        help="Root directory containing causalrivers benchmark results. Can be passed multiple times.",
    )
    parser.add_argument(
        "--forecast-checkpoints-root",
        type=Path,
        action="append",
        help="Root directory containing BasicTS checkpoint folders with test_metrics.json files. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "outputs" / "advisor_tables",
        help="Project-level directory for advisor-facing outputs.",
    )
    parser.add_argument(
        "--include-null",
        action="store_true",
        default=True,
        help="Include NULL baseline rows in causal tables. Enabled by default.",
    )
    parser.add_argument(
        "--exclude-null",
        action="store_true",
        help="Exclude NULL baseline rows from causal tables.",
    )
    return parser.parse_args()


def _find_named_ancestor(path: Path, name: str) -> Path | None:
    current = path if path.is_dir() else path.parent
    for candidate in (current, *current.parents):
        if candidate.name == name:
            return candidate
    return None


def _unique_existing_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    existing: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if not resolved.exists() or resolved in seen:
            continue
        seen.add(resolved)
        existing.append(resolved)
    return existing


def _auto_discover_causal_roots(repo_root: Path) -> list[Path]:
    roots: list[Path] = []
    for config_path in repo_root.rglob("config.yaml"):
        ancestor = _find_named_ancestor(config_path, "results")
        if ancestor is not None:
            roots.append(ancestor)
    return _unique_existing_paths(roots)


def _auto_discover_forecast_roots(repo_root: Path) -> list[Path]:
    roots: list[Path] = []
    for metrics_path in repo_root.rglob("test_metrics.json"):
        ancestor = _find_named_ancestor(metrics_path, "checkpoints")
        if ancestor is not None:
            roots.append(ancestor)
    return _unique_existing_paths(roots)


def _resolve_causal_roots(args: argparse.Namespace) -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    if args.causal_results_root:
        return _unique_existing_paths(list(args.causal_results_root))
    default_root = repo_root / "causalrivers" / "results"
    if default_root.exists():
        return [default_root.resolve()]
    return _auto_discover_causal_roots(repo_root)


def _resolve_forecast_roots(args: argparse.Namespace) -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    if args.forecast_checkpoints_root:
        return _unique_existing_paths(list(args.forecast_checkpoints_root))
    default_root = repo_root / "BasicTS" / "checkpoints"
    if default_root.exists():
        return [default_root.resolve()]
    return _auto_discover_forecast_roots(repo_root)


def _normalize_model(name: str | None) -> str | None:
    if not name:
        return None
    mapping = {
        "GraphWaveNet": "GWNET",
        "graphwavenet": "GWNET",
        "GWNET": "GWNET",
        "AGCRN": "AGCRN",
        "D2STGNN": "D2STGNN",
        "MTGNN": "MTGNN",
        "GTS": "GTS",
    }
    return mapping.get(name, name)


def _infer_stgnn_model_from_path(learned_graph_path: str | None) -> str | None:
    if not learned_graph_path:
        return None
    parts = Path(learned_graph_path).parts
    if "checkpoints" in parts:
        idx = parts.index("checkpoints")
        if idx + 1 < len(parts):
            return _normalize_model(parts[idx + 1])
    return None


def _load_full_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _build_run_metadata(results_root: Path) -> dict[str, dict[str, Any]]:
    run_meta: dict[str, dict[str, Any]] = {}
    for config_path in results_root.rglob("config.yaml"):
        run_dir = str(config_path.parent.resolve())
        cfg = _load_full_config(config_path)
        method_cfg = cfg.get("method") or {}
        methods_cfg = cfg.get("methods") or []

        source_method = method_cfg.get("name")
        if source_method is None and methods_cfg:
            source_method = "multi"

        learned_graph_path = method_cfg.get("learned_graph_path")
        stgnn_model = _infer_stgnn_model_from_path(learned_graph_path)
        run_track = "B-causal" if source_method == "stgnn_precomputed" else "A-causal"

        run_meta[run_dir] = {
            "source_method": source_method,
            "learned_graph_path": learned_graph_path,
            "stgnn_model": stgnn_model,
            "track": run_track,
        }
    return run_meta


def _display_method(row: dict[str, Any], run_meta: dict[str, dict[str, Any]]) -> str:
    run_dir = str(Path(str(row["run_dir"])).resolve())
    meta = run_meta.get(run_dir, {})
    method = str(row.get("method"))
    if method == "stgnn_precomputed":
        return meta.get("stgnn_model") or "STGNN"
    if method == "NULL":
        return "NULL"

    mapping = {
        "var": "VAR",
        "corr": "CC",
        "lagcorr": "LagCC",
        "cc": "CC",
        "rp": "RP",
        "combo": "Combo",
        "pcmci": "PCMCI",
        "varlingam": "VARLiNGAM",
    }
    return mapping.get(method, method)


def _row_track(row: dict[str, Any], run_meta: dict[str, dict[str, Any]]) -> str:
    run_dir = str(Path(str(row["run_dir"])).resolve())
    meta = run_meta.get(run_dir, {})
    return meta.get("track", "A-causal")


def _meaningful_dataset_name(data_name: Any, label_name: Any) -> str | None:
    data_text = str(data_name).strip() if data_name is not None else ""
    label_text = str(label_name).strip() if label_name is not None else ""
    if data_text and data_text.lower() not in {"datasets", "none"}:
        return data_text
    if label_text and label_text.lower() not in {"datasets", "none"}:
        return label_text
    return None


def _describe_dataset(raw_name: str | None, resolution_hint: Any = None) -> dict[str, str | None]:
    name = (raw_name or "").strip()
    lowered = name.lower().removesuffix(".csv")
    resolution = parse_dataset_resolution_hint(name) or normalize_resolution(resolution_hint)

    alias = name if name else None
    signal = None
    family = "other"

    if lowered.startswith("traffic_volume_") or lowered.startswith("traffic_city_traffic_m_volume__category__1_0"):
        alias = "UTB-m"
        signal = "volume"
        family = "traffic"
        resolution = resolution or "5min"
    elif lowered.startswith("traffic_speed_") or lowered.startswith("traffic_city_traffic_m_speed__category__1_0"):
        alias = "UTB-m"
        signal = "speed"
        family = "traffic"
        resolution = resolution or "5min"
    elif lowered.startswith("rivers_east_germany_") or lowered.startswith("rivers_ts_east_germany"):
        alias = "Rivers-East"
        signal = "discharge"
        family = "rivers"
    elif name:
        alias = name.replace(".csv", "")

    return {
        "dataset": alias,
        "resolution": resolution,
        "signal": signal,
        "family": family,
    }


def _enrich_causal_rows(rows: list[dict[str, Any]], run_meta: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        dataset_name = _meaningful_dataset_name(row.get("data_dataset_name"), row.get("label_dataset_name"))
        dataset_meta = _describe_dataset(dataset_name, row.get("config_resolution"))
        metric_names = [row.get("metric")]
        model_name = _display_method(row, run_meta)

        if model_name == "NULL":
            metric_name = str(row.get("metric"))
            if metric_name == "Acc":
                metric_names = ["Max Acc"]
            elif metric_name == "F1":
                metric_names = ["Max F1"]
            elif metric_name == "AUROC":
                metric_names = ["AUROC", "Individual AUROC"]

        for metric_name in metric_names:
            new_row = dict(row)
            new_row["metric"] = metric_name
            new_row["track"] = _row_track(row, run_meta)
            new_row["model"] = model_name
            new_row["dataset"] = dataset_meta["dataset"]
            new_row["resolution"] = dataset_meta["resolution"]
            new_row["signal"] = dataset_meta["signal"]
            new_row["family"] = dataset_meta["family"]
            enriched.append(new_row)
    return enriched


def _filter_causal_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("strategy") or "") == "debug_set":
            continue
        filtered.append(row)
    return filtered


def _build_causal_core(metric_rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not metric_rows:
        return pd.DataFrame(
            columns=[
                "track",
                "dataset",
                "resolution",
                "signal",
                "strategy",
                "n_vars",
                "model",
                *CAUSAL_PRIMARY_METRICS,
            ]
        )

    metric_df = pd.DataFrame(metric_rows)
    group_cols = ["track", "dataset", "resolution", "signal", "strategy", "n_vars", "model"]
    metric_summary = (
        metric_df.groupby(group_cols + ["metric"], dropna=False)["value"]
        .mean()
        .reset_index()
    )
    wide = (
        metric_summary.pivot_table(
            index=group_cols,
            columns="metric",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns.name = None
    preferred_cols = group_cols + CAUSAL_PRIMARY_METRICS
    ordered_cols = [col for col in preferred_cols if col in wide.columns] + [
        col for col in wide.columns if col not in preferred_cols
    ]
    wide = wide[ordered_cols]
    sort_cols = ["dataset", "resolution", "strategy", "n_vars", "track", "AUROC", "Max F1"]
    existing_sort_cols = [col for col in sort_cols if col in wide.columns]
    ascending = [True, True, True, True, True, False, False][: len(existing_sort_cols)]
    return wide.sort_values(existing_sort_cols, ascending=ascending).reset_index(drop=True)


def _find_config_dir_name(path: Path) -> str | None:
    for part in path.parts:
        if CONFIG_DIR_PATTERN.match(part):
            return part
    return None


def _parse_forecast_run(path: Path) -> dict[str, Any] | None:
    parts = path.parts
    if "checkpoints" not in parts:
        return None
    idx = parts.index("checkpoints")
    if idx + 1 >= len(parts):
        return None

    model_dir = parts[idx + 1]
    model = _normalize_model(model_dir)
    config_dir_name = _find_config_dir_name(path)
    if config_dir_name is None:
        return None

    match = CONFIG_DIR_PATTERN.match(config_dir_name)
    if match is None:
        return None

    dataset_name = match.group("dataset")
    metrics_raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metrics_raw, dict):
        return None

    dataset_meta = _describe_dataset(dataset_name)
    return {
        "track": "B-forecast",
        "model": model,
        "dataset": dataset_meta["dataset"],
        "resolution": dataset_meta["resolution"],
        "signal": dataset_meta["signal"],
        "family": dataset_meta["family"],
        "dataset_name": dataset_name,
        "epochs": int(match.group("epochs")),
        "input_len": int(match.group("input_len")),
        "output_len": int(match.group("output_len")),
        "metrics": metrics_raw,
        "run_dir": str(path.parent.resolve()),
        "metrics_path": str(path.resolve()),
    }


def _discover_forecast_runs(checkpoints_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if not checkpoints_root.exists():
        return runs
    for metrics_path in checkpoints_root.rglob("test_metrics.json"):
        parsed = _parse_forecast_run(metrics_path)
        if parsed is not None:
            runs.append(parsed)
    return sorted(runs, key=lambda row: (row["dataset_name"], row["model"], row["run_dir"]))


def _build_forecast_long_rows(forecast_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in forecast_runs:
        for scope in FORECAST_SCOPES:
            if scope == "overall":
                scope_metrics = run["metrics"].get("overall", run["metrics"])
            else:
                scope_metrics = run["metrics"].get(scope)
            if not isinstance(scope_metrics, dict):
                continue
            for metric_name, value in scope_metrics.items():
                if not isinstance(value, (int, float)):
                    continue
                rows.append(
                    {
                        "track": run["track"],
                        "dataset": run["dataset"],
                        "resolution": run["resolution"],
                        "signal": run["signal"],
                        "model": run["model"],
                        "scope": scope,
                        "metric": metric_name,
                        "value": float(value),
                    }
                )
    return rows


def _build_forecast_core(forecast_long_rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not forecast_long_rows:
        return pd.DataFrame(columns=["track", "dataset", "resolution", "signal", "scope", "model"])

    forecast_df = pd.DataFrame(forecast_long_rows)
    group_cols = ["track", "dataset", "resolution", "signal", "scope", "model"]
    summary = (
        forecast_df.groupby(group_cols + ["metric"], dropna=False)["value"]
        .mean()
        .reset_index()
    )
    wide = (
        summary.pivot_table(
            index=group_cols,
            columns="metric",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns.name = None
    metric_order = FORECAST_PRIMARY_METRICS + [
        col for col in wide.columns if col not in group_cols and col not in FORECAST_PRIMARY_METRICS
    ]
    ordered_cols = group_cols + [col for col in metric_order if col in wide.columns]
    wide = wide[ordered_cols]
    scope_order = {scope: idx for idx, scope in enumerate(FORECAST_SCOPES)}
    wide["scope_order"] = wide["scope"].map(scope_order).fillna(999)
    sort_cols = ["dataset", "resolution", "scope_order"]
    if "MAE" in wide.columns:
        sort_cols.append("MAE")
    sort_cols.append("model")
    ascending = [True, True, True, True, True][: len(sort_cols)]
    return wide.sort_values(sort_cols, ascending=ascending).drop(columns=["scope_order"]).reset_index(drop=True)


def _round_numeric_df(df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    rounded = df.copy()
    numeric_cols = rounded.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        rounded[numeric_cols] = rounded[numeric_cols].round(digits)
    return rounded


def _format_plain(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def _ordered_model_columns(track: str, present_models: list[str]) -> list[str]:
    preferred = TRACK_MODEL_ORDER.get(track, [])
    ordered = [model for model in preferred if model in present_models]
    extras = sorted(set(present_models) - set(preferred))
    return ordered + extras


def _build_track_matrix(core_df: pd.DataFrame, track: str) -> pd.DataFrame:
    subset = core_df[core_df["track"] == track].copy()
    if subset.empty:
        return pd.DataFrame(columns=["dataset", "resolution", "signal", "strategy", "n_vars", "metric"])

    model_cols = _ordered_model_columns(track, subset["model"].dropna().unique().tolist())
    metric_cols = [col for col in CAUSAL_PRIMARY_METRICS if col in subset.columns]
    dataset_frames: list[pd.DataFrame] = []
    for dataset_key, dataset_df in subset.groupby(["dataset", "resolution", "signal"], dropna=False):
        long_df = dataset_df.melt(
            id_vars=["strategy", "n_vars", "model"],
            value_vars=metric_cols,
            var_name="metric",
            value_name="value",
        )
        strategy_pairs = (
            long_df[["strategy", "n_vars"]]
            .drop_duplicates()
            .sort_values(["strategy", "n_vars"], na_position="last")
        )
        index_tuples = [
            (row.strategy, row.n_vars, metric)
            for row in strategy_pairs.itertuples(index=False)
            for metric in metric_cols
        ]
        full_index = pd.MultiIndex.from_tuples(index_tuples, names=["strategy", "n_vars", "metric"])
        pivot = long_df.pivot_table(
            index=["strategy", "n_vars", "metric"],
            columns="model",
            values="value",
            aggfunc="first",
        )
        pivot = pivot.reindex(full_index)
        pivot = pivot.reindex(columns=model_cols)
        pivot = pivot.reset_index()
        pivot.insert(0, "signal", dataset_key[2])
        pivot.insert(0, "resolution", dataset_key[1])
        pivot.insert(0, "dataset", dataset_key[0])
        dataset_frames.append(pivot)

    matrix = pd.concat(dataset_frames, ignore_index=True)
    numeric_cols = [col for col in model_cols if col in matrix.columns]
    if numeric_cols:
        matrix[numeric_cols] = matrix[numeric_cols].round(4)
    return matrix


def _format_matrix_df(df: pd.DataFrame, model_cols: list[str], lower_is_better: bool = False) -> pd.DataFrame:
    formatted = df.copy()
    for row_idx, row in formatted.iterrows():
        present_values = []
        for col in model_cols:
            value = row.get(col)
            if pd.notna(value):
                present_values.append(float(value))
        ordered_values = sorted(set(present_values), reverse=not lower_is_better)
        best = ordered_values[0] if ordered_values else None
        second = ordered_values[1] if len(ordered_values) > 1 else None

        for col in model_cols:
            value = row.get(col)
            if pd.isna(value):
                formatted.at[row_idx, col] = "/"
                continue
            text = f"{float(value):.4f}"
            if best is not None and float(value) == best:
                text = f"**{text}**"
            elif second is not None and float(value) == second:
                text = f"<u>{text}</u>"
            formatted.at[row_idx, col] = text

    for col in formatted.columns:
        if col not in model_cols:
            formatted[col] = formatted[col].map(_format_plain)
    return formatted


def _blank_repeated_labels(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    compact = df.copy()
    if compact.empty:
        return compact
    for col in columns:
        previous = object()
        for idx, value in compact[col].items():
            if value == previous:
                compact.at[idx, col] = ""
            else:
                previous = value
    return compact


def _styled_metric_table(
    df: pd.DataFrame,
    metric_cols: list[str],
    lower_is_better: bool = False,
) -> pd.DataFrame:
    styled = df.copy()
    rankable = [
        col
        for col in metric_cols
        if col in styled.columns and pd.api.types.is_numeric_dtype(styled[col])
    ]

    for col in rankable:
        values = styled[col].dropna().unique().tolist()
        values = sorted(values, reverse=not lower_is_better)
        best = values[0] if values else None
        second = values[1] if len(values) > 1 else None
        column_values: list[str] = []
        for value in styled[col]:
            if pd.isna(value):
                column_values.append("")
                continue
            text = f"{float(value):.4f}"
            if best is not None and value == best:
                text = f"**{text}**"
            elif second is not None and value == second:
                text = f"<u>{text}</u>"
            column_values.append(text)
        styled[col] = column_values

    for col in styled.columns:
        if col not in rankable:
            styled[col] = styled[col].map(_format_plain)
    return styled


def _markdown_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df.empty:
        return "_No rows found._"
    shown = df.head(max_rows).copy()
    shown = shown.fillna("")
    headers = list(shown.columns)
    rows = [[str(value) for value in row] for row in shown.to_numpy()]
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def fmt_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    lines = [
        fmt_row(headers),
        "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |",
    ]
    lines.extend(fmt_row(row) for row in rows)
    if len(df) > max_rows:
        lines.append("")
        lines.append(f"_Showing first {max_rows} of {len(df)} rows._")
    return "\n".join(lines)


def _section_title(dataset: str | None, resolution: str | None, signal: str | None) -> str:
    pieces = [piece for piece in [dataset, resolution, signal] if piece]
    return " | ".join(pieces) if pieces else "Unknown Dataset"


def _scope_label(scope: str | None) -> str:
    mapping = {
        "overall": "Overall",
        "horizon_3": "H3",
        "horizon_6": "H6",
        "horizon_12": "H12",
    }
    return mapping.get(scope or "", scope or "Unknown")


def _ordered_forecast_models(present_models: list[str]) -> list[str]:
    preferred = [model for model in TRACK_MODEL_ORDER["B-causal"] if model != "NULL"]
    ordered = [model for model in preferred if model in present_models]
    extras = sorted(set(present_models) - set(preferred))
    return ordered + extras


def _styled_forecast_dataset_table(dataset_df: pd.DataFrame) -> pd.DataFrame:
    if dataset_df.empty:
        return pd.DataFrame(columns=["scope", "model", *FORECAST_PRIMARY_METRICS])

    model_cols = [
        col
        for col in ["model", "MAE", "RMSE", "MAPE", "WAPE", "SMAPE"]
        if col in dataset_df.columns
    ]
    working = dataset_df.copy()
    scope_order = {scope: idx for idx, scope in enumerate(FORECAST_SCOPES)}
    model_order = {
        model: idx for idx, model in enumerate(_ordered_forecast_models(working["model"].dropna().unique().tolist()))
    }
    working["scope_order"] = working["scope"].map(scope_order).fillna(999)
    working["model_order"] = working["model"].map(model_order).fillna(999)
    working = working.sort_values(["scope_order", "model_order", "model"]).reset_index(drop=True)

    styled_parts: list[pd.DataFrame] = []
    for scope, scope_df in working.groupby("scope", dropna=False, sort=False):
        scope_table = _styled_metric_table(
            scope_df[model_cols].reset_index(drop=True),
            metric_cols=FORECAST_PRIMARY_METRICS,
            lower_is_better=True,
        )
        scope_table.insert(0, "scope", _scope_label(scope))
        styled_parts.append(scope_table)

    return pd.concat(styled_parts, ignore_index=True)


def _dataset_matrix_sections(matrix_df: pd.DataFrame, track: str) -> list[str]:
    lines: list[str] = []
    if matrix_df.empty:
        lines.append("_No rows found._")
        lines.append("")
        return lines

    model_cols = _ordered_model_columns(track, [col for col in matrix_df.columns if col not in {"dataset", "resolution", "signal", "strategy", "n_vars", "metric"}])
    model_cols = [col for col in model_cols if col in matrix_df.columns]
    for (dataset, resolution, signal), dataset_df in matrix_df.groupby(["dataset", "resolution", "signal"], dropna=False):
        lines.append(f"### {_section_title(dataset, resolution, signal)}")
        lines.append(
            _markdown_table(
                _blank_repeated_labels(
                    _format_matrix_df(
                        dataset_df[["strategy", "n_vars", "metric", *model_cols]].copy(),
                        model_cols=model_cols,
                        lower_is_better=False,
                    ),
                    ["strategy", "n_vars"],
                ),
                max_rows=120,
            )
        )
        lines.append("")
    return lines


def _html_escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _html_table_with_rowspan(df: pd.DataFrame, merge_cols: list[str], model_cols: list[str]) -> str:
    if df.empty:
        return "<p><em>No rows found.</em></p>"

    headers = list(df.columns)
    rows = df.reset_index(drop=True)

    rowspan_maps: dict[str, dict[int, int]] = {col: {} for col in merge_cols}
    for col in merge_cols:
        start = 0
        while start < len(rows):
            value = rows.at[start, col]
            end = start + 1
            while end < len(rows) and rows.at[end, col] == value:
                end += 1
            rowspan_maps[col][start] = end - start
            start = end

    html_lines = [
        '<table class="advisor-table">',
        "  <thead>",
        "    <tr>",
    ]
    for header in headers:
        html_lines.append(f"      <th>{_html_escape(header)}</th>")
    html_lines.extend(["    </tr>", "  </thead>", "  <tbody>"])

    skip_cells: dict[str, set[int]] = {col: set() for col in merge_cols}
    for row_idx in range(len(rows)):
        html_lines.append("    <tr>")
        for col in headers:
            value = rows.at[row_idx, col]
            if col in merge_cols:
                if row_idx in skip_cells[col]:
                    continue
                rowspan = rowspan_maps[col].get(row_idx, 1)
                for skipped in range(row_idx + 1, row_idx + rowspan):
                    skip_cells[col].add(skipped)
                html_lines.append(
                    f'      <td rowspan="{rowspan}">{_html_escape(value)}</td>'
                )
            else:
                cell = _html_escape(value)
                if col in model_cols and cell == "/":
                    html_lines.append('      <td class="missing">/</td>')
                else:
                    html_lines.append(f"      <td>{cell}</td>")
        html_lines.append("    </tr>")

    html_lines.extend(["  </tbody>", "</table>"])
    return "\n".join(html_lines)


def _forecast_html_sections(forecast_core: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    if forecast_core.empty:
        lines.append("<p><em>No forecasting rows found.</em></p>")
        return lines

    for (dataset, resolution, signal), dataset_df in forecast_core.groupby(
        ["dataset", "resolution", "signal"], dropna=False
    ):
        lines.append(f"<h3>{_html_escape(_section_title(dataset, resolution, signal))}</h3>")
        working = _styled_forecast_dataset_table(dataset_df)
        lines.append(
            _html_table_with_rowspan(
                working,
                merge_cols=["scope"],
                model_cols=[col for col in FORECAST_PRIMARY_METRICS if col in working.columns],
            )
        )
    return lines


def _matrix_html_sections(matrix_df: pd.DataFrame, track: str) -> list[str]:
    lines: list[str] = []
    if matrix_df.empty:
        lines.append("<p><em>No rows found.</em></p>")
        return lines

    model_cols = _ordered_model_columns(
        track,
        [col for col in matrix_df.columns if col not in {"dataset", "resolution", "signal", "strategy", "n_vars", "metric"}],
    )
    model_cols = [col for col in model_cols if col in matrix_df.columns]
    for (dataset, resolution, signal), dataset_df in matrix_df.groupby(["dataset", "resolution", "signal"], dropna=False):
        lines.append(f"<h3>{_html_escape(_section_title(dataset, resolution, signal))}</h3>")
        working = _format_matrix_df(
            dataset_df[["strategy", "n_vars", "metric", *model_cols]].copy(),
            model_cols=model_cols,
            lower_is_better=False,
        )
        lines.append(_html_table_with_rowspan(working, merge_cols=["strategy", "n_vars"], model_cols=model_cols))
    return lines


def _build_html_report(causal_core: pd.DataFrame, forecast_core: pd.DataFrame) -> str:
    a_matrix = _build_track_matrix(causal_core, "A-causal")
    b_matrix = _build_track_matrix(causal_core, "B-causal")

    sections: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="zh-CN">',
        "<head>",
        '  <meta charset="utf-8">',
        "  <title>Advisor Summary</title>",
        "  <style>",
        "body { font-family: Arial, sans-serif; margin: 24px; line-height: 1.4; }",
        "h1, h2, h3 { margin-top: 24px; }",
        ".advisor-table { border-collapse: collapse; width: 100%; margin: 12px 0 24px; }",
        ".advisor-table th, .advisor-table td { border: 1px solid #999; padding: 6px 8px; text-align: center; vertical-align: middle; }",
        ".advisor-table th { background: #f3f4f6; }",
        ".missing { color: #666; }",
        "ul { margin-top: 0; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <h1>Advisor Summary</h1>",
        "  <h2>1. A-line Causal Summary</h2>",
        "  <ul><li>每个数据集单独汇总成矩阵表。</li><li>行：strategy / n_vars / metric。</li><li>列：模型。</li><li>debug_set 已排除，缺失项显示为 /。</li></ul>",
        *_matrix_html_sections(a_matrix, "A-causal"),
        "  <h2>2. B-line Causal Summary</h2>",
        "  <ul><li>先训练 STGNN learned graph，再用 causal benchmark 回评。</li><li>NULL 作为共同下界基线保留。</li><li>缺失项显示为 /。</li></ul>",
        *_matrix_html_sections(b_matrix, "B-causal"),
        "  <h2>3. Forecasting Summary</h2>",
        "  <ul><li>汇总 BasicTS checkpoints 的 Overall + H3 + H6 + H12。</li></ul>",
        *_forecast_html_sections(forecast_core),
        "</body>",
        "</html>",
    ]
    return "\n".join(sections)


def _build_report(causal_core: pd.DataFrame, forecast_core: pd.DataFrame) -> str:
    a_matrix = _build_track_matrix(causal_core, "A-causal")
    b_matrix = _build_track_matrix(causal_core, "B-causal")

    lines: list[str] = [
        "# Advisor Summary",
        "",
        "## 1. A-line Causal Summary",
        "",
        "- 每个数据集单独汇总成矩阵表。",
        "- 行：`strategy / n_vars / metric`。",
        "- 列：模型。",
        "- `debug_set` 已排除。",
        "- 没测到的组合显示为 `/`。",
        "",
    ]
    lines.extend(_dataset_matrix_sections(a_matrix, "A-causal"))

    lines.extend(
        [
            "## 2. B-line Causal Summary",
            "",
            "- 先训练 STGNN learned graph，再用 causal benchmark 回评。",
            "- `NULL` 作为共同下界基线保留。",
            "- `debug_set` 已排除，缺失项显示为 `/`。",
            "",
        ]
    )
    lines.extend(_dataset_matrix_sections(b_matrix, "B-causal"))

    lines.extend(
        [
            "## 3. Forecasting Summary",
            "",
            "- 这里汇总的是 `BasicTS/checkpoints/**/test_metrics.json` 的 `Overall + H3 + H6 + H12` 预测指标。",
            "",
        ]
    )

    if forecast_core.empty:
        lines.append("_No forecasting rows found._")
        lines.append("")
    else:
        for (dataset, resolution, signal), dataset_df in forecast_core.groupby(
            ["dataset", "resolution", "signal"], dropna=False
        ):
            lines.append(f"### {_section_title(dataset, resolution, signal)}")
            lines.append("")
            lines.append(
                _markdown_table(
                    _blank_repeated_labels(
                        _styled_forecast_dataset_table(dataset_df),
                        ["scope"],
                    ),
                    max_rows=80,
                )
            )
            lines.append("")

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    causal_results_roots = _resolve_causal_roots(args)
    forecast_checkpoints_roots = _resolve_forecast_roots(args)

    metric_rows_raw: list[dict[str, Any]] = []
    runtime_rows_raw: list[dict[str, Any]] = []
    warnings: list[str] = []
    for causal_root in causal_results_roots:
        root_metric_rows, root_runtime_rows, root_warnings = aggregate_results(causal_root)
        metric_rows_raw.extend(root_metric_rows)
        runtime_rows_raw.extend(root_runtime_rows)
        warnings.extend(root_warnings)

    metric_rows, _run_method_meta = deduplicate_metric_rows(metric_rows_raw)
    run_meta: dict[str, dict[str, Any]] = {}
    for causal_root in causal_results_roots:
        run_meta.update(_build_run_metadata(causal_root))
    metric_rows = _enrich_causal_rows(metric_rows, run_meta)
    metric_rows = _filter_causal_rows(metric_rows)

    include_null = args.include_null and not args.exclude_null
    if not include_null:
        metric_rows = [row for row in metric_rows if row.get("model") != "NULL"]

    causal_core = _round_numeric_df(_build_causal_core(metric_rows))
    a_matrix = _build_track_matrix(causal_core, "A-causal")
    b_matrix = _build_track_matrix(causal_core, "B-causal")

    forecast_runs: list[dict[str, Any]] = []
    for forecast_root in forecast_checkpoints_roots:
        forecast_runs.extend(_discover_forecast_runs(forecast_root))
    forecast_long_rows = _build_forecast_long_rows(forecast_runs)
    forecast_core = _round_numeric_df(_build_forecast_core(forecast_long_rows))

    causal_path = output_dir / "advisor_causal_core.csv"
    a_matrix_path = output_dir / "advisor_a_line_matrix.csv"
    b_matrix_path = output_dir / "advisor_b_line_matrix.csv"
    forecast_path = output_dir / "advisor_forecasting_core.csv"
    report_path = output_dir / "advisor_report.md"
    html_report_path = output_dir / "advisor_report.html"
    warnings_path = output_dir / "advisor_warnings.txt"

    causal_core.to_csv(causal_path, index=False)
    a_matrix.to_csv(a_matrix_path, index=False, na_rep="/")
    b_matrix.to_csv(b_matrix_path, index=False, na_rep="/")
    forecast_core.to_csv(forecast_path, index=False)
    report_path.write_text(_build_report(causal_core, forecast_core), encoding="utf-8")
    html_report_path.write_text(_build_html_report(causal_core, forecast_core), encoding="utf-8")

    cleaned_warnings = sorted(
        {
            warning
            for warning in warnings
            if "label dataset=datasets" not in warning
        }
    )
    if cleaned_warnings:
        warnings_path.write_text("\n".join(cleaned_warnings), encoding="utf-8")

    if causal_results_roots:
        print("Causal roots:")
        for root in causal_results_roots:
            print(f"  - {root}")
    else:
        print("Causal roots: none found")

    if forecast_checkpoints_roots:
        print("Forecast roots:")
        for root in forecast_checkpoints_roots:
            print(f"  - {root}")
    else:
        print("Forecast roots: none found")

    print(f"Wrote causal table    : {causal_path}")
    print(f"Wrote A-line matrix   : {a_matrix_path}")
    print(f"Wrote B-line matrix   : {b_matrix_path}")
    print(f"Wrote forecast table  : {forecast_path}")
    print(f"Wrote report          : {report_path}")
    print(f"Wrote HTML report     : {html_report_path}")
    if cleaned_warnings:
        print(f"Wrote warnings        : {warnings_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
