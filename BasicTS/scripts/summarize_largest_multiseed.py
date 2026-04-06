import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev


def _load_json(path: Path):
    with path.open("r") as fp:
        return json.load(fp)


def _round4(value):
    if value is None:
        return None
    return round(float(value), 4)


def _collect_metric(run_summary, metric_path: Path | None, section: str, name: str):
    if metric_path is None or not metric_path.exists():
        return None
    payload = _load_json(metric_path)
    return payload.get(section, {}).get(name)


def _aggregate(values):
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": _round4(mean(valid)),
        "std": _round4(pstdev(valid)) if len(valid) > 1 else 0.0,
        "n": len(valid),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="append", required=True, help="Path to profile_summary.json")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    metric_keys = [
        ("overall", "MAE"),
        ("overall", "RMSE"),
        ("overall", "MAPE"),
        ("horizon_3", "MAE"),
        ("horizon_3", "RMSE"),
        ("horizon_3", "MAPE"),
        ("horizon_6", "MAE"),
        ("horizon_6", "RMSE"),
        ("horizon_6", "MAPE"),
        ("horizon_12", "MAE"),
        ("horizon_12", "RMSE"),
        ("horizon_12", "MAPE"),
    ]

    run_entries = []
    duration_values = []
    memory_values = []
    metric_buckets = {f"{section}.{name}": [] for section, name in metric_keys}

    for raw_path in args.summary:
        summary_path = Path(raw_path).resolve()
        summary = _load_json(summary_path)
        metrics_path_raw = summary.get("test_metrics_path")
        metrics_path = Path(metrics_path_raw).resolve() if metrics_path_raw else None

        entry = {
            "summary_path": str(summary_path),
            "seed": summary.get("seed"),
            "duration_seconds": summary.get("duration_seconds"),
            "peak_gpu_memory_mb": summary.get("peak_gpu_memory_mb"),
            "test_metrics_path": str(metrics_path) if metrics_path else None,
        }

        duration_values.append(summary.get("duration_seconds"))
        memory_values.append(summary.get("peak_gpu_memory_mb"))

        for section, name in metric_keys:
            value = _collect_metric(summary, metrics_path, section, name)
            entry[f"{section}.{name}"] = _round4(value) if value is not None else None
            metric_buckets[f"{section}.{name}"].append(value)

        run_entries.append(entry)

    output = {
        "runs": run_entries,
        "aggregates": {
            "duration_seconds": _aggregate(duration_values),
            "peak_gpu_memory_mb": _aggregate(memory_values),
        },
    }

    for key, values in metric_buckets.items():
        output["aggregates"][key] = _aggregate(values)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fp:
        json.dump(output, fp, indent=2)

    print(f"Saved multi-seed summary to {output_path}")
    print(json.dumps(output["aggregates"], indent=2))


if __name__ == "__main__":
    main()
