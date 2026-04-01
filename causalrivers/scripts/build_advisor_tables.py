#!/usr/bin/env python3
"""Build advisor-facing summary tables from causalrivers result directories.

This script is designed for two result families:
1. A-line: causal discovery from time series (var / cc / pcmci / varlingam / ...)
2. B-line: STGNN learned-graph evaluation routed through benchmark.py via
   `method=stgnn_precomputed`.

It scans `results/`, deduplicates identical runs, restores the concrete STGNN
model name from `method.learned_graph_path`, and writes advisor-friendly CSV and
Markdown tables.
"""

from __future__ import annotations

import argparse
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
    deduplicate_runtime_rows,
)


PRIMARY_METRICS = ["AUROC", "Max F1", "Max Acc", "Individual AUROC"]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build advisor-facing result tables from causalrivers benchmark outputs."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=repo_root / "results",
        help="Root directory containing result folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "outputs" / "advisor_tables",
        help="Directory for CSV and Markdown outputs.",
    )
    parser.add_argument(
        "--include-null",
        action="store_true",
        help="Include NULL baseline rows in summary tables.",
    )
    return parser.parse_args()


def _normalize_stgnn_model(name: str | None) -> str | None:
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
            return _normalize_stgnn_model(parts[idx + 1])
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
        run_line = "B-line" if source_method == "stgnn_precomputed" else "A-line"

        run_meta[run_dir] = {
            "source_method": source_method,
            "learned_graph_path": learned_graph_path,
            "stgnn_model": stgnn_model,
            "line": run_line,
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
        "stgnn_precomputed": meta.get("stgnn_model") or "STGNN",
    }
    return mapping.get(method, method)


def _row_line(row: dict[str, Any], run_meta: dict[str, dict[str, Any]]) -> str:
    run_dir = str(Path(str(row["run_dir"])).resolve())
    meta = run_meta.get(run_dir, {})
    return meta.get("line", "A-line")


def _enrich_rows(rows: list[dict[str, Any]], run_meta: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        new_row["line"] = _row_line(row, run_meta)
        new_row["method_display"] = _display_method(row, run_meta)
        enriched.append(new_row)
    return enriched


def _build_summary_table(metric_rows: list[dict[str, Any]], runtime_rows: list[dict[str, Any]]) -> pd.DataFrame:
    metric_df = pd.DataFrame(metric_rows)
    runtime_df = pd.DataFrame(runtime_rows)

    group_cols = [
        "line",
        "data_dataset_name",
        "label_group",
        "strategy",
        "n_vars",
        "label_tag",
        "config_resolution",
        "method_display",
    ]

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

    if not runtime_df.empty:
        runtime_filtered = runtime_df[runtime_df["method"] != "overall"].copy()
        runtime_summary = (
            runtime_filtered.groupby(group_cols, dropna=False)["runtime_seconds"]
            .mean()
            .reset_index()
            .rename(columns={"runtime_seconds": "runtime_seconds_mean"})
        )
        wide = wide.merge(runtime_summary, on=group_cols, how="left")

    preferred_cols = group_cols + PRIMARY_METRICS + ["runtime_seconds_mean"]
    ordered_cols = [col for col in preferred_cols if col in wide.columns] + [
        col for col in wide.columns if col not in preferred_cols
    ]
    wide = wide[ordered_cols]
    return wide.sort_values(group_cols).reset_index(drop=True)


def _compact_summary_view(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    preferred = [
        "data_dataset_name",
        "label_group",
        "strategy",
        "n_vars",
        "method_display",
        "AUROC",
        "Max F1",
        "Max Acc",
        "runtime_seconds_mean",
    ]
    columns = [column for column in preferred if column in df.columns]
    compact = df[columns].copy()
    sort_cols = [column for column in ["data_dataset_name", "label_group", "AUROC", "Max F1"] if column in compact.columns]
    ascending = [True, True, False, False][: len(sort_cols)]
    return compact.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def _best_rows_by_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if df.empty or metric not in df.columns:
        return df.iloc[0:0].copy()
    group_cols = [column for column in ["line", "data_dataset_name", "label_group", "strategy", "n_vars"] if column in df.columns]
    idx = (
        df.groupby(group_cols, dropna=False)[metric]
        .idxmax()
        .dropna()
        .astype(int)
    )
    best = df.loc[idx].copy()
    sort_cols = [column for column in ["line", "data_dataset_name", "label_group", metric] if column in best.columns]
    ascending = [True, True, True, False][: len(sort_cols)]
    return best.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def _a_b_comparison(df: pd.DataFrame, metric: str = "AUROC") -> pd.DataFrame:
    if df.empty or metric not in df.columns:
        return df.iloc[0:0].copy()
    group_cols = ["data_dataset_name", "label_group", "strategy", "n_vars"]
    subset_cols = group_cols + ["line", "method_display", metric]
    available_cols = [column for column in subset_cols if column in df.columns]
    working = df[available_cols].copy()
    best = _best_rows_by_metric(working, metric)
    if best.empty or "line" not in best.columns:
        return best
    a_best = best[best["line"] == "A-line"].copy()
    b_best = best[best["line"] == "B-line"].copy()
    if a_best.empty or b_best.empty:
        return best.iloc[0:0].copy()
    merged = a_best.merge(
        b_best,
        on=group_cols,
        suffixes=("_a", "_b"),
        how="inner",
    )
    if merged.empty:
        return merged
    merged["delta_b_minus_a"] = merged[f"{metric}_b"] - merged[f"{metric}_a"]
    order_cols = group_cols + [
        "method_display_a",
        f"{metric}_a",
        "method_display_b",
        f"{metric}_b",
        "delta_b_minus_a",
    ]
    existing = [column for column in order_cols if column in merged.columns]
    return merged[existing].sort_values(group_cols).reset_index(drop=True)


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


def _write_report(
    output_path: Path,
    long_df: pd.DataFrame,
    a_df: pd.DataFrame,
    b_df: pd.DataFrame,
    combined_df: pd.DataFrame,
) -> None:
    a_best = _best_rows_by_metric(a_df, "AUROC")
    b_best = _best_rows_by_metric(b_df, "AUROC")
    ab_compare = _a_b_comparison(combined_df, metric="AUROC")
    a_compact = _compact_summary_view(a_df)
    b_compact = _compact_summary_view(b_df)
    combined_compact = _compact_summary_view(combined_df)

    lines: list[str] = [
        "# Advisor Report Tables",
        "",
        f"- Total metric rows: {len(long_df)}",
        f"- A-line summary rows: {len(a_df)}",
        f"- B-line summary rows: {len(b_df)}",
        "",
        "## Topline",
        "- A-line: time-series causal discovery baselines.",
        "- B-line: STGNN learned-graph results evaluated as causal graphs.",
        "",
        "## A-line Best By AUROC",
        _markdown_table(a_best, max_rows=30),
        "",
        "## B-line Best By AUROC",
        _markdown_table(b_best, max_rows=30),
        "",
        "## A-vs-B Best AUROC Comparison",
        _markdown_table(ab_compare, max_rows=30),
        "",
        "## A-line Compact Table",
        _markdown_table(a_compact, max_rows=40),
        "",
        "## B-line Compact Table",
        _markdown_table(b_compact, max_rows=40),
        "",
        "## Combined Compact Table",
        _markdown_table(combined_compact, max_rows=50),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    results_root = args.results_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows_raw, runtime_rows_raw, warnings = aggregate_results(results_root)
    metric_rows, run_method_meta = deduplicate_metric_rows(metric_rows_raw)
    runtime_rows = deduplicate_runtime_rows(runtime_rows_raw, run_method_meta)

    run_meta = _build_run_metadata(results_root)
    metric_rows = _enrich_rows(metric_rows, run_meta)
    runtime_rows = _enrich_rows(runtime_rows, run_meta)

    if not args.include_null:
        metric_rows = [row for row in metric_rows if row.get("method_display") != "NULL"]
        runtime_rows = [row for row in runtime_rows if row.get("method_display") != "NULL"]

    long_df = pd.DataFrame(metric_rows).sort_values(
        ["line", "data_dataset_name", "label_group", "method_display", "metric"]
    )
    combined_df = _build_summary_table(metric_rows, runtime_rows)
    a_df = combined_df[combined_df["line"] == "A-line"].reset_index(drop=True)
    b_df = combined_df[combined_df["line"] == "B-line"].reset_index(drop=True)

    long_path = output_dir / "advisor_results_long.csv"
    combined_path = output_dir / "advisor_combined_summary.csv"
    a_path = output_dir / "advisor_a_line_summary.csv"
    b_path = output_dir / "advisor_b_line_summary.csv"
    a_best_path = output_dir / "advisor_a_line_best_auroc.csv"
    b_best_path = output_dir / "advisor_b_line_best_auroc.csv"
    compare_path = output_dir / "advisor_a_vs_b_best_auroc.csv"
    report_path = output_dir / "advisor_report.md"
    warnings_path = output_dir / "advisor_warnings.txt"

    a_best = _best_rows_by_metric(a_df, "AUROC")
    b_best = _best_rows_by_metric(b_df, "AUROC")
    compare_df = _a_b_comparison(combined_df, metric="AUROC")

    long_df.to_csv(long_path, index=False)
    combined_df.to_csv(combined_path, index=False)
    a_df.to_csv(a_path, index=False)
    b_df.to_csv(b_path, index=False)
    a_best.to_csv(a_best_path, index=False)
    b_best.to_csv(b_best_path, index=False)
    compare_df.to_csv(compare_path, index=False)
    _write_report(report_path, long_df, a_df, b_df, combined_df)

    if warnings:
        warnings_path.write_text("\n".join(sorted(set(warnings))), encoding="utf-8")

    print(f"Wrote long table    : {long_path}")
    print(f"Wrote combined table: {combined_path}")
    print(f"Wrote A-line table  : {a_path}")
    print(f"Wrote B-line table  : {b_path}")
    print(f"Wrote A best table  : {a_best_path}")
    print(f"Wrote B best table  : {b_best_path}")
    print(f"Wrote A/B compare   : {compare_path}")
    print(f"Wrote report        : {report_path}")
    if warnings:
        print(f"Wrote warnings      : {warnings_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
