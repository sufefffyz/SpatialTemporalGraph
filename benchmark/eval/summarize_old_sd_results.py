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
    return {
        "model": model_name,
        "experiment": experiment,
        "run_dir": str(metrics_path.parent),
        "metrics_path": str(metrics_path),
        "overall_MAE": get_metric(payload, "overall", "MAE"),
        "overall_RMSE": get_metric(payload, "overall", "RMSE"),
        "overall_MAPE": get_metric(payload, "overall", "MAPE"),
        "h3_MAE": get_metric(payload, "horizon_3", "MAE"),
        "h3_RMSE": get_metric(payload, "horizon_3", "RMSE"),
        "h3_MAPE": get_metric(payload, "horizon_3", "MAPE"),
        "h6_MAE": get_metric(payload, "horizon_6", "MAE"),
        "h6_RMSE": get_metric(payload, "horizon_6", "RMSE"),
        "h6_MAPE": get_metric(payload, "horizon_6", "MAPE"),
        "h12_MAE": get_metric(payload, "horizon_12", "MAE"),
        "h12_RMSE": get_metric(payload, "horizon_12", "RMSE"),
        "h12_MAPE": get_metric(payload, "horizon_12", "MAPE"),
    }


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
        rows.append(
            {
                "model": model_name,
                "experiment": experiment,
                "run_dir": str(metrics_path.parent),
                "metrics_path": str(metrics_path),
                "overall_MAE": get_metric(payload, "overall", "MAE"),
                "overall_RMSE": get_metric(payload, "overall", "RMSE"),
                "overall_MAPE": get_metric(payload, "overall", "MAPE"),
                "h3_MAE": get_metric(payload, "horizon_3", "MAE"),
                "h3_RMSE": get_metric(payload, "horizon_3", "RMSE"),
                "h3_MAPE": get_metric(payload, "horizon_3", "MAPE"),
                "h6_MAE": get_metric(payload, "horizon_6", "MAE"),
                "h6_RMSE": get_metric(payload, "horizon_6", "RMSE"),
                "h6_MAPE": get_metric(payload, "horizon_6", "MAPE"),
                "h12_MAE": get_metric(payload, "horizon_12", "MAE"),
                "h12_RMSE": get_metric(payload, "horizon_12", "RMSE"),
                "h12_MAPE": get_metric(payload, "horizon_12", "MAPE"),
            }
        )
    return rows


def make_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No matching old-SD results found.\n"

    display = df.copy()
    numeric_cols = [col for col in display.columns if col not in {"model", "experiment"}]
    for col in numeric_cols:
        display[col] = display[col].map(lambda value: f"{value:.4f}" if pd.notna(value) else "/")
    return display.to_markdown(index=False) + "\n"


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

    if df.empty:
        pd.DataFrame().to_csv(full_csv, index=False)
        pd.DataFrame().to_csv(summary_csv, index=False)
        summary_md.write_text("No matching old-SD results found.\n", encoding="utf-8")
        print(json.dumps({"rows": 0, "output_dir": str(output_dir)}, indent=2))
        return

    df.to_csv(full_csv, index=False)

    summary_df = df[
        [
            "model",
            "experiment",
            "overall_MAE",
            "overall_RMSE",
            "overall_MAPE",
            "h3_MAE",
            "h6_MAE",
            "h12_MAE",
        ]
    ].copy()
    summary_df.to_csv(summary_csv, index=False)
    summary_md.write_text(make_markdown_table(summary_df), encoding="utf-8")

    print(
        json.dumps(
            {
                "rows": int(len(df)),
                "full_csv": str(full_csv),
                "summary_csv": str(summary_csv),
                "summary_md": str(summary_md),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
