#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINTS_ROOT = REPO_ROOT / "BasicTS" / "checkpoints"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmark" / "eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize completed old-SD experiment results into report-friendly tables."
    )
    parser.add_argument(
        "--checkpoints-root",
        type=Path,
        default=DEFAULT_CHECKPOINTS_ROOT,
        help="BasicTS checkpoints root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store the summary CSV/Markdown files.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["GWNet", "DCRNN"],
        help="Model directories to scan under checkpoints root.",
    )
    parser.add_argument(
        "--result-path",
        action="append",
        default=[],
        help="Explicit result path(s): either a test_metrics.json file or a run directory containing it. Can be repeated.",
    )
    parser.add_argument(
        "--result",
        action="append",
        default=[],
        help="Labeled result in the form experiment=PATH or model:experiment=PATH. Can be repeated.",
    )
    return parser.parse_args()


def classify_experiment(model_name: str, path_text: str) -> str | None:
    lower = path_text.lower()
    if model_name.lower() == "gwnet":
        if "phys_adaptive_forward" in lower or "forward_adaptive" in lower:
            return "phys_forward+adaptive"
        if "phys_adaptive" in lower:
            return "phys+adaptive"
        if "adaptive" in lower and "phys" not in lower:
            return "adaptive"
        if "original_fixed" in lower:
            return "distthre"
        if "original" in lower:
            return "distthre+adaptive"
        if "directed" in lower:
            return "phys_dir"
        if "bidir" in lower:
            return "phys_bidir"
    if model_name.lower() == "dcrnn":
        if "original" in lower:
            return "distthre"
        if "directed" in lower:
            return "phys_dir"
        if "bidir" in lower:
            return "phys_bidir"
    return None


def load_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def get_metric(payload: dict, section: str, key: str):
    return payload.get(section, {}).get(key)


def flatten_metrics(payload: dict) -> dict:
    flattened = {}
    for section, metrics in payload.items():
        if not isinstance(metrics, dict):
            continue
        prefix = "overall" if section == "overall" else section
        for metric_name, value in metrics.items():
            flattened[f"{prefix}_{metric_name}"] = value
    return flattened


def infer_model_from_path(path: Path) -> str | None:
    parts = [part.lower() for part in path.parts]
    if "gwnet" in parts:
        return "GWNet"
    if "dcrnn" in parts:
        return "DCRNN"
    return None


def build_row(model_name: str, metrics_path: Path, path_text: str) -> dict | None:
    experiment = classify_experiment(model_name, path_text)
    if experiment is None:
        return None

    payload = load_metrics(metrics_path)
    row = {
        "model": model_name,
        "experiment": experiment,
        "run_dir": str(metrics_path.parent),
        "metrics_path": str(metrics_path),
    }
    row.update(flatten_metrics(payload))
    return row


def collect_rows(checkpoints_root: Path, models: list[str]) -> list[dict]:
    rows: list[dict] = []
    for model_name in models:
        model_root = checkpoints_root / model_name
        if not model_root.exists():
            continue
        for metrics_path in sorted(model_root.rglob("test_metrics.json")):
            relative_text = str(metrics_path.relative_to(model_root))
            row = build_row(model_name, metrics_path, relative_text)
            if row is not None:
                rows.append(row)
    return rows


def resolve_metrics_path(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "test_metrics.json"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"No test_metrics.json under directory: {path}")
    if path.is_file():
        if path.name != "test_metrics.json":
            raise ValueError(f"Expected test_metrics.json or a run directory, got: {path}")
        return path
    raise FileNotFoundError(f"Result path not found: {path}")


def collect_rows_from_explicit_paths(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for raw_path in paths:
        resolved_input = Path(raw_path).expanduser().resolve()
        metrics_path = resolve_metrics_path(resolved_input)
        model_name = infer_model_from_path(metrics_path)
        if model_name is None:
            raise ValueError(f"Unable to infer model name from path: {metrics_path}")
        row = build_row(model_name, metrics_path, str(metrics_path))
        if row is not None:
            rows.append(row)
    return rows


def parse_labeled_result(raw_value: str) -> tuple[str | None, str, Path]:
    if "=" not in raw_value:
        raise ValueError(
            f"Expected labeled result in the form experiment=PATH or model:experiment=PATH, got: {raw_value}"
        )
    label, raw_path = raw_value.split("=", 1)
    label = label.strip()
    raw_path = raw_path.strip()
    if not label or not raw_path:
        raise ValueError(f"Invalid labeled result: {raw_value}")

    if ":" in label:
        model_name, experiment = label.split(":", 1)
        model_name = model_name.strip()
        experiment = experiment.strip()
        if not model_name or not experiment:
            raise ValueError(f"Invalid labeled result: {raw_value}")
        return model_name, experiment, Path(raw_path).expanduser().resolve()

    return None, label, Path(raw_path).expanduser().resolve()


def collect_rows_from_labeled_results(items: list[str]) -> list[dict]:
    rows: list[dict] = []
    for raw_item in items:
        model_name_hint, experiment, resolved_input = parse_labeled_result(raw_item)
        metrics_path = resolve_metrics_path(resolved_input)
        model_name = model_name_hint or infer_model_from_path(metrics_path)
        if model_name is None:
            raise ValueError(f"Unable to infer model name from path: {metrics_path}")

        payload = load_metrics(metrics_path)
        row = {
            "model": model_name,
            "experiment": experiment,
            "run_dir": str(metrics_path.parent),
            "metrics_path": str(metrics_path),
        }
        row.update(flatten_metrics(payload))
        rows.append(row)
    return rows


def make_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No matching old-SD results found.\n"

    display = df.copy()
    numeric_cols = [col for col in display.columns if col not in {"model", "experiment"}]
    for col in numeric_cols:
        display[col] = display[col].map(lambda value: f"{value:.4f}" if pd.notna(value) else "/")
    return display.to_markdown(index=False) + "\n"


def metric_sort_key(column_name: str) -> tuple[int, int, int, str]:
    if column_name.startswith("overall_"):
        metric_name = column_name[len("overall_") :]
        metric_order = {"MAE": 0, "RMSE": 1, "MAPE": 2}.get(metric_name, 99)
        return (0, 0, metric_order, metric_name)
    if column_name.startswith("horizon_"):
        parts = column_name.split("_")
        try:
            horizon = int(parts[1])
        except ValueError:
            horizon = 999
        metric_name = "_".join(parts[2:])
        metric_order = {"MAE": 0, "RMSE": 1, "MAPE": 2}.get(metric_name, 99)
        return (1, horizon, metric_order, metric_name)
    return (2, 999, 999, column_name)


def build_summary_columns(df: pd.DataFrame) -> list[str]:
    metric_columns = [col for col in df.columns if col not in {"model", "experiment", "run_dir", "metrics_path"}]
    return ["model", "experiment", *sorted(metric_columns, key=metric_sort_key)]


def make_grouped_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No matching old-SD results found.\n"

    graph_rows = ["distthre", "phys_dir", "phys_bidir"]
    adaptive_rows = ["adaptive", "distthre+adaptive", "phys+adaptive", "phys_forward+adaptive"]
    summary_columns = build_summary_columns(df)
    sections = []

    for model_name in sorted(df["model"].unique().tolist()):
        model_df = df[df["model"] == model_name].copy()
        model_df = model_df[summary_columns]
        sections.append(f"## {model_name}\n")

        graph_df = model_df[model_df["experiment"].isin(graph_rows)].copy()
        if not graph_df.empty:
            graph_df["experiment"] = pd.Categorical(graph_df["experiment"], categories=graph_rows, ordered=True)
            graph_df = graph_df.sort_values("experiment")
            sections.append("### Graph Structure Comparison\n")
            sections.append(make_markdown_table(graph_df))

        adaptive_df = model_df[model_df["experiment"].isin(adaptive_rows)].copy()
        if not adaptive_df.empty:
            adaptive_df["experiment"] = pd.Categorical(adaptive_df["experiment"], categories=adaptive_rows, ordered=True)
            adaptive_df = adaptive_df.sort_values("experiment")
            sections.append("### Adaptive Comparison\n")
            sections.append(make_markdown_table(adaptive_df))

    return "\n".join(sections)


def iter_metric_horizon_rows(df: pd.DataFrame):
    metric_names = ["MAE", "RMSE", "MAPE"]
    horizon_pairs = [("overall", "overall"), ("horizon_3", "h3"), ("horizon_6", "h6"), ("horizon_12", "h12")]
    for metric_name in metric_names:
        first_row = True
        for section_name, horizon_label in horizon_pairs:
            column_name = f"{section_name}_{metric_name}"
            if column_name in df.columns:
                yield {
                    "metric": metric_name if first_row else "",
                    "horizon": horizon_label,
                    "column_name": column_name,
                }
                first_row = False


def build_pivot_table(model_df: pd.DataFrame) -> pd.DataFrame:
    experiments = model_df["experiment"].tolist()
    rows = []
    for row_def in iter_metric_horizon_rows(model_df):
        row = {
            "metric": row_def["metric"],
            "horizon": row_def["horizon"],
        }
        for _, experiment_row in model_df.iterrows():
            row[experiment_row["experiment"]] = experiment_row.get(row_def["column_name"])
        rows.append(row)
    return pd.DataFrame(rows, columns=["metric", "horizon", *experiments])


def format_pivot_for_markdown(pivot_df: pd.DataFrame) -> pd.DataFrame:
    display = pivot_df.copy()
    for column in display.columns:
        if column in {"metric", "horizon"}:
            continue
        display[column] = display[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "/")
    return display


def main() -> None:
    args = parse_args()
    checkpoints_root = args.checkpoints_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.result:
        rows = collect_rows_from_labeled_results(args.result)
    elif args.result_path:
        rows = collect_rows_from_explicit_paths(args.result_path)
    else:
        rows = collect_rows(checkpoints_root, args.models)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["model", "experiment"]).reset_index(drop=True)

    full_csv = output_dir / "old_sd_results_full.csv"
    summary_csv = output_dir / "old_sd_results_summary.csv"
    summary_md = output_dir / "old_sd_results_summary.md"
    grouped_md = output_dir / "old_sd_results_grouped.md"
    pivot_paths = []

    if df.empty:
        pd.DataFrame().to_csv(full_csv, index=False)
        pd.DataFrame().to_csv(summary_csv, index=False)
        summary_md.write_text("No matching old-SD results found.\n", encoding="utf-8")
        grouped_md.write_text("No matching old-SD results found.\n", encoding="utf-8")
        print(json.dumps({"rows": 0, "output_dir": str(output_dir)}, indent=2))
        return

    df.to_csv(full_csv, index=False)

    summary_df = df[build_summary_columns(df)].copy()
    summary_df.to_csv(summary_csv, index=False)
    summary_md.write_text(make_markdown_table(summary_df), encoding="utf-8")
    grouped_md.write_text(make_grouped_markdown(df), encoding="utf-8")

    for model_name in sorted(df["model"].unique().tolist()):
        model_df = df[df["model"] == model_name].copy().sort_values("experiment").reset_index(drop=True)
        pivot_df = build_pivot_table(model_df)
        pivot_csv = output_dir / f"old_sd_results_{model_name.lower()}_pivot.csv"
        pivot_md = output_dir / f"old_sd_results_{model_name.lower()}_pivot.md"
        pivot_df.to_csv(pivot_csv, index=False)
        pivot_md.write_text(format_pivot_for_markdown(pivot_df).to_markdown(index=False) + "\n", encoding="utf-8")
        pivot_paths.append({"model": model_name, "csv": str(pivot_csv), "md": str(pivot_md)})

    print(
        json.dumps(
            {
                "rows": int(len(df)),
                "full_csv": str(full_csv),
                "summary_csv": str(summary_csv),
                "summary_md": str(summary_md),
                "grouped_md": str(grouped_md),
                "pivot_tables": pivot_paths,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
