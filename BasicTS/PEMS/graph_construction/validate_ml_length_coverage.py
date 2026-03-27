"""Validate whether PeMS ML Length matches midpoint-derived coverage intervals.

The script treats each ML sensor as covering a mainline interval in postmile
space. For an interior ML sensor, the interval starts at the midpoint between
the current sensor and the previous ML sensor, and ends at the midpoint between
the current sensor and the next ML sensor.

It compares that midpoint-derived interval length against the PeMS Length field
and writes both per-sensor details and per-corridor summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ABS_TOL = 0.05
DEFAULT_REL_TOL = 0.10


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    for col in df.columns:
        cl = col.strip().lower()
        if cl == "id":
            col_map[col] = "ID"
        elif cl == "fwy":
            col_map[col] = "Fwy"
        elif cl in ("dir", "direction"):
            col_map[col] = "Dir"
        elif cl == "type":
            col_map[col] = "Type"
        elif cl in ("latitude", "lat"):
            col_map[col] = "Latitude"
        elif cl in ("longitude", "lng", "lon"):
            col_map[col] = "Longitude"
        elif cl == "name":
            col_map[col] = "Name"
        elif cl == "abs_pm":
            col_map[col] = "Abs_PM"
        elif cl == "length":
            col_map[col] = "Length"
    return df.rename(columns=col_map)


def load_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", encoding="latin-1")
    df = normalize_columns(df)

    required = {"ID", "Fwy", "Dir", "Type", "Abs_PM", "Length"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce")
    df["Fwy"] = pd.to_numeric(df["Fwy"], errors="coerce")
    df["Abs_PM"] = pd.to_numeric(df["Abs_PM"], errors="coerce")
    df["Length"] = pd.to_numeric(df["Length"], errors="coerce")
    return df


def classify_validation_case(has_prev: bool, has_next: bool) -> str:
    if has_prev and has_next:
        return "interior"
    if has_next:
        return "leading_boundary"
    if has_prev:
        return "trailing_boundary"
    return "singleton"


def evaluate_group(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values(["Abs_PM", "ID"], kind="mergesort").reset_index(drop=True)

    prev_pm = group["Abs_PM"].shift(1)
    next_pm = group["Abs_PM"].shift(-1)

    has_prev = prev_pm.notna()
    has_next = next_pm.notna()

    midpoint_start = (prev_pm + group["Abs_PM"]) / 2.0
    midpoint_end = (group["Abs_PM"] + next_pm) / 2.0

    result = group.copy()
    result["prev_ml_id"] = group["ID"].shift(1)
    result["next_ml_id"] = group["ID"].shift(-1)
    result["prev_abs_pm"] = prev_pm
    result["next_abs_pm"] = next_pm
    result["has_prev_ml"] = has_prev
    result["has_next_ml"] = has_next
    result["validation_case"] = [
        classify_validation_case(bool(p), bool(n)) for p, n in zip(has_prev, has_next)
    ]

    result["midpoint_start_pm"] = midpoint_start
    result["midpoint_end_pm"] = midpoint_end
    result["midpoint_interval_length"] = midpoint_end - midpoint_start

    result["length_center_start_pm"] = result["Abs_PM"] - result["Length"] / 2.0
    result["length_center_end_pm"] = result["Abs_PM"] + result["Length"] / 2.0

    result["start_delta_pm"] = result["midpoint_start_pm"] - result["length_center_start_pm"]
    result["end_delta_pm"] = result["midpoint_end_pm"] - result["length_center_end_pm"]
    result["length_residual_pm"] = result["midpoint_interval_length"] - result["Length"]
    result["abs_length_residual_pm"] = result["length_residual_pm"].abs()
    result["rel_length_residual"] = result["abs_length_residual_pm"] / result["Length"].replace(0, np.nan)

    result["one_sided_symmetric_length"] = np.nan
    lead_mask = (~has_prev) & has_next
    trail_mask = has_prev & (~has_next)
    result.loc[lead_mask, "one_sided_symmetric_length"] = (
        2.0 * (result.loc[lead_mask, "midpoint_end_pm"] - result.loc[lead_mask, "Abs_PM"])
    )
    result.loc[trail_mask, "one_sided_symmetric_length"] = (
        2.0 * (result.loc[trail_mask, "Abs_PM"] - result.loc[trail_mask, "midpoint_start_pm"])
    )
    result["one_sided_length_residual_pm"] = result["one_sided_symmetric_length"] - result["Length"]

    duplicated_pm = result["Abs_PM"].duplicated(keep=False)
    result["same_postmile_tie"] = duplicated_pm
    return result


def build_summary(details: pd.DataFrame, abs_tol: float, rel_tol: float) -> pd.DataFrame:
    summaries = []

    for (fwy, direction), group in details.groupby(["Fwy", "Dir"], dropna=False):
        interior = group[group["validation_case"] == "interior"].copy()
        boundary = group[group["validation_case"].isin(["leading_boundary", "trailing_boundary"])]

        within_tol = interior[
            (interior["abs_length_residual_pm"] <= abs_tol)
            | (interior["rel_length_residual"] <= rel_tol)
        ]

        summaries.append(
            {
                "Fwy": fwy,
                "Dir": direction,
                "ml_count": int(len(group)),
                "interior_count": int(len(interior)),
                "boundary_count": int(len(boundary)),
                "tie_count": int(group["same_postmile_tie"].sum()),
                "length_missing_count": int(group["Length"].isna().sum()),
                "interior_mean_abs_residual_pm": interior["abs_length_residual_pm"].mean(),
                "interior_median_abs_residual_pm": interior["abs_length_residual_pm"].median(),
                "interior_p90_abs_residual_pm": interior["abs_length_residual_pm"].quantile(0.9) if len(interior) else np.nan,
                "interior_mean_rel_residual": interior["rel_length_residual"].mean(),
                "interior_within_tolerance_rate": len(within_tol) / len(interior) if len(interior) else np.nan,
                "boundary_mean_one_sided_residual_pm": boundary["one_sided_length_residual_pm"].abs().mean(),
            }
        )

    summary_df = pd.DataFrame(summaries)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["interior_mean_abs_residual_pm", "ml_count"], ascending=[False, False])
    return summary_df


def build_global_report(details: pd.DataFrame, summary: pd.DataFrame, abs_tol: float, rel_tol: float) -> dict:
    interior = details[details["validation_case"] == "interior"].copy()
    boundary = details[details["validation_case"].isin(["leading_boundary", "trailing_boundary"])]
    within_tol = interior[
        (interior["abs_length_residual_pm"] <= abs_tol)
        | (interior["rel_length_residual"] <= rel_tol)
    ]

    return {
        "ml_count": int(len(details)),
        "corridor_count": int(len(summary)),
        "interior_count": int(len(interior)),
        "boundary_count": int(len(boundary)),
        "singleton_count": int((details["validation_case"] == "singleton").sum()),
        "tie_count": int(details["same_postmile_tie"].sum()),
        "length_missing_count": int(details["Length"].isna().sum()),
        "interior_mean_abs_residual_pm": float(interior["abs_length_residual_pm"].mean()) if len(interior) else None,
        "interior_median_abs_residual_pm": float(interior["abs_length_residual_pm"].median()) if len(interior) else None,
        "interior_mean_rel_residual": float(interior["rel_length_residual"].mean()) if len(interior) else None,
        "interior_within_tolerance_rate": float(len(within_tol) / len(interior)) if len(interior) else None,
        "boundary_mean_one_sided_residual_pm": float(boundary["one_sided_length_residual_pm"].abs().mean()) if len(boundary) else None,
        "abs_tolerance_pm": abs_tol,
        "relative_tolerance": rel_tol,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ML Length against midpoint-derived coverage intervals")
    parser.add_argument("metadata_file", help="PeMS metadata file in tab-separated format")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <metadata_dir>/length_validation_<metadata_stem>",
    )
    parser.add_argument("--abs-tol", type=float, default=DEFAULT_ABS_TOL, help="Absolute tolerance in postmile units")
    parser.add_argument("--rel-tol", type=float, default=DEFAULT_REL_TOL, help="Relative tolerance against Length")
    parser.add_argument("--top-k", type=int, default=50, help="Number of worst residual rows to export separately")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata_file).expanduser().resolve()
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = metadata_path.parent / f"length_validation_{metadata_path.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_metadata(metadata_path)
    ml = df[df["Type"] == "ML"].copy()
    ml = ml[ml["Abs_PM"].notna()].copy()

    details = []
    for _, group in ml.groupby(["Fwy", "Dir"], dropna=False):
        details.append(evaluate_group(group))

    if details:
        details_df = pd.concat(details, ignore_index=True)
    else:
        details_df = pd.DataFrame()

    summary_df = build_summary(details_df, args.abs_tol, args.rel_tol) if not details_df.empty else pd.DataFrame()
    report = build_global_report(details_df, summary_df, args.abs_tol, args.rel_tol) if not details_df.empty else {}

    details_path = output_dir / "ml_length_coverage_details.csv"
    summary_path = output_dir / "ml_length_coverage_summary.csv"
    report_path = output_dir / "ml_length_coverage_report.json"
    worst_path = output_dir / "ml_length_coverage_worst_cases.csv"

    details_df.to_csv(details_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if not details_df.empty:
        worst_cases = details_df.sort_values(
            ["abs_length_residual_pm", "rel_length_residual"],
            ascending=[False, False],
            na_position="last",
        ).head(args.top_k)
        worst_cases.to_csv(worst_path, index=False)

    print(f"metadata_file={metadata_path}")
    print(f"output_dir={output_dir}")
    if report:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("No ML rows with valid Abs_PM were found.")


if __name__ == "__main__":
    main()