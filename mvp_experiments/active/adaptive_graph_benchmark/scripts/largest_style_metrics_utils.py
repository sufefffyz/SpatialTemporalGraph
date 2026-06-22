"""Helpers for LargeST-style horizon-averaged metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re


DEFAULT_METRIC_NAMES = ("MAE", "RMSE", "MAPE")


def infer_horizons(metrics: Mapping[str, Mapping[str, float]]) -> list[int]:
    """Infer sorted horizon ids from BasicTS metric keys."""
    horizons: list[int] = []
    for key in metrics:
        match = re.fullmatch(r"horizon_(\d+)", key)
        if match:
            horizons.append(int(match.group(1)))
    if not horizons:
        raise ValueError("No horizon_<n> metrics found.")
    return sorted(horizons)


def compute_largest_average(
    metrics: Mapping[str, Mapping[str, float]],
    horizons: Sequence[int] | None = None,
    metric_names: Sequence[str] = DEFAULT_METRIC_NAMES,
) -> dict[str, float]:
    """Average per-horizon metrics with the LargeST convention.

    LargeST reports each forecast horizon separately and then computes the
    arithmetic mean across horizons. This intentionally ignores BasicTS
    ``overall`` metrics, whose RMSE/MAPE aggregation semantics differ.
    """
    selected_horizons = list(horizons) if horizons is not None else infer_horizons(metrics)
    if not selected_horizons:
        raise ValueError("At least one horizon is required.")

    averages: dict[str, float] = {}
    for metric_name in metric_names:
        values: list[float] = []
        for horizon in selected_horizons:
            horizon_key = f"horizon_{horizon}"
            if horizon_key not in metrics:
                raise KeyError(f"Missing {horizon_key} in metric file.")
            if metric_name not in metrics[horizon_key]:
                raise KeyError(f"Missing {metric_name} in {horizon_key}.")
            values.append(float(metrics[horizon_key][metric_name]))
        averages[metric_name] = sum(values) / len(values)
    return averages


def format_summary_cell(metrics: Mapping[str, float]) -> str:
    """Format one table cell as MAE / RMSE / MAPE."""
    return "{:.4f} / {:.4f} / {:.4f}".format(
        float(metrics["MAE"]),
        float(metrics["RMSE"]),
        float(metrics["MAPE"]),
    )
