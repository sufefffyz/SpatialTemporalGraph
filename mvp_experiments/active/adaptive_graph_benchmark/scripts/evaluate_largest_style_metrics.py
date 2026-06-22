#!/usr/bin/env python3
"""Re-evaluate adaptive graph checkpoints with LargeST-style metrics.

The script does not retrain models. It reloads full-training checkpoints, asks
BasicTS to evaluate every forecast horizon, and then averages horizon metrics
with the LargeST convention.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

from largest_style_metrics_utils import compute_largest_average, format_summary_cell


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
BASICTS_ROOT = REPO_ROOT / "BasicTS"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "mvp_experiments"
    / "active"
    / "adaptive_graph_benchmark"
    / "outputs"
    / "largest_style_metrics"
)


MODEL_SPECS = {
    "GWNet": {
        "config": "baselines/GWNet/AdaptiveImportance.py",
        "variant_env": "BASICTS_GWNET_VARIANT",
        "variants": {
            "original": "GraphWaveNetAdaptiveImportance",
            "no_adaptive_from_scratch": "GraphWaveNetNoAdaptiveImportance",
            "frozen_random_adaptive_from_scratch": "GraphWaveNetFrozenRandomAdaptiveImportance",
            "learned_no_relu": "GraphWaveNetLearnedNoReluAdaptiveImportance",
            "signal_mlp_relu": "GraphWaveNetSignalMLPReluAdaptiveImportance",
            "signal_mlp_no_relu": "GraphWaveNetSignalMLPNoReluAdaptiveImportance",
        },
    },
    "AGCRN": {
        "config": "baselines/AGCRN/AdaptiveImportance.py",
        "variant_env": "BASICTS_AGCRN_VARIANT",
        "variants": {
            "original": "AGCRNAdaptiveImportanceOriginal",
            "identity_support": "AGCRNAdaptiveImportanceIdentitySupport",
            "frozen_random_embedding": "AGCRNAdaptiveImportanceFrozenRandomEmbedding",
            "graph_no_relu": "AGCRNAdaptiveImportanceGraphNoRelu",
        },
    },
    "MTGNN": {
        "config": "baselines/MTGNN/AdaptiveImportance.py",
        "variant_env": "BASICTS_MTGNN_VARIANT",
        "variants": {
            "original": "MTGNNAdaptiveImportanceOriginal",
            "fixed_physical": "MTGNNAdaptiveImportanceFixedPhysical",
            "frozen_random_graph": "MTGNNAdaptiveImportanceFrozenRandomGraph",
            "no_relu_score": "MTGNNAdaptiveImportanceNoReluScore",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["gwnet", "agcrn-mtgnn", "all"], default="all")
    parser.add_argument("--datasets", nargs="+", default=["METR-LA", "PEMS04", "PEMS07", "SD"])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=BASICTS_ROOT / "checkpoints")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--device-type", default="gpu")
    parser.add_argument("--batch-size", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Only discover checkpoints.")

    parser.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--variant", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dataset", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--job-output-dir", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def selected_models(scope: str) -> list[str]:
    if scope == "gwnet":
        return ["GWNet"]
    if scope == "agcrn-mtgnn":
        return ["AGCRN", "MTGNN"]
    return ["GWNet", "AGCRN", "MTGNN"]


def checkpoint_patterns(
    checkpoint_root: Path,
    dataset: str,
    variant: str,
    model_name: str,
) -> list[str]:
    # Prefer the exact full-run, 100-epoch naming used by the adaptive-importance scripts.
    exact = (
        checkpoint_root
        / model_name
        / f"{dataset}_full_{variant}*det0_cudnndet0_100_*"
        / "*"
        / f"{model_name}_best_val_MAE.pt"
    )
    # Fallback keeps the full-stage constraint but accepts small naming drift.
    fallback = (
        checkpoint_root
        / model_name
        / f"{dataset}_full_{variant}*det0_cudnndet0*"
        / "*"
        / f"{model_name}_best_val_MAE.pt"
    )
    return [str(exact), str(fallback)]


def find_checkpoint(
    checkpoint_root: Path,
    dataset: str,
    variant: str,
    model_name: str,
) -> tuple[Path | None, str]:
    seen: set[Path] = set()
    candidates: list[Path] = []
    patterns = checkpoint_patterns(checkpoint_root, dataset, variant, model_name)
    for pattern in patterns:
        for path in checkpoint_root.parent.glob(os.path.relpath(pattern, checkpoint_root.parent)):
            if path not in seen:
                candidates.append(path)
                seen.add(path)
        if candidates:
            break
    if not candidates:
        return None, patterns[0]
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1], patterns[0]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_job_env(model: str, dataset: str, variant: str) -> dict[str, str]:
    spec = MODEL_SPECS[model]
    env = os.environ.copy()
    env.update(
        {
            "BASICTS_DATA_NAME": dataset,
            "BASICTS_RUN_STAGE": "full",
            "BASICTS_NUM_EPOCHS": "100",
            "BASICTS_ENV_TAG": "det0_cudnndet0",
            "BASICTS_DETERMINISTIC": "0",
            "BASICTS_CUDNN_DETERMINISTIC": "0",
            "BASICTS_CUDNN_BENCHMARK": "1",
            "BASICTS_EARLY_STOPPING_PATIENCE": "30",
            "WANDB_MODE": os.environ.get("WANDB_MODE", "offline"),
            "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "adaptive_graph_importance_ablation"),
            spec["variant_env"]: variant,
        }
    )
    if model == "GWNet":
        env.update(
            {
                "BASICTS_ADAPTIVE_EVAL_MODE": "learned",
                "BASICTS_ADAPTIVE_KEEP_RATIO": "1.0",
                "BASICTS_ADAPTIVE_KEEP_TAG": "1p000",
            }
        )
    return env


def output_len_from_cfg(cfg: Any) -> int:
    dataset_param = cfg.get("DATASET", {}).get("PARAM", {})
    if "output_len" in dataset_param:
        return int(dataset_param["output_len"])
    model_param = cfg.get("MODEL", {}).get("PARAM", {})
    for key in ("out_dim", "horizon"):
        if key in model_param:
            return int(model_param[key])
    raise KeyError("Cannot infer OUTPUT_LEN from config.")


def run_single_eval(args: argparse.Namespace) -> int:
    if args.model is None or args.variant is None or args.dataset is None:
        raise ValueError("--single requires --model, --variant, and --dataset.")
    if args.config is None or args.checkpoint is None or args.job_output_dir is None:
        raise ValueError("--single requires --config, --checkpoint, and --job-output-dir.")

    sys.path.insert(0, str(BASICTS_ROOT))
    os.chdir(BASICTS_ROOT)

    from easytorch.config import init_cfg
    from easytorch.device import set_device_type
    from easytorch.utils import get_logger, set_visible_devices

    cfg = init_cfg(args.config, save=True)
    output_len = output_len_from_cfg(cfg)
    horizons = list(range(1, output_len + 1))
    cfg.EVAL.HORIZONS = horizons
    cfg.EVAL.SAVE_RESULTS = False

    set_device_type(args.device_type)
    if args.device_type != "cpu":
        set_visible_devices(args.gpu)

    logger = get_logger("largest-style-eval")
    logger.info("Initializing runner '%s'", cfg["RUNNER"])
    runner = cfg["RUNNER"](cfg)
    runner.init_logger(logger_name="largest-style-evaluation", log_file_name="evaluation_log")

    if args.batch_size is not None:
        cfg.TEST.DATA.BATCH_SIZE = int(args.batch_size)

    if runner.need_setup_graph:
        runner.setup_graph(cfg=cfg, train=False)
        runner.need_setup_graph = False

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")

    job_output_dir = args.job_output_dir.resolve()
    job_output_dir.mkdir(parents=True, exist_ok=True)
    runner.ckpt_save_dir = str(job_output_dir)

    logger.info("Loading model checkpoint from %s", checkpoint)
    runner.load_model(ckpt_path=str(checkpoint), strict=True)
    metrics = runner.test_pipeline(cfg=cfg, save_metrics=True, save_results=False)
    metrics_path = job_output_dir / "test_metrics.json"
    if metrics is None:
        with metrics_path.open() as f:
            metrics = json.load(f)

    largest_avg = compute_largest_average(metrics, horizons=horizons)
    record = {
        "dataset": args.dataset,
        "model": args.model,
        "variant": args.variant,
        "config": args.config,
        "checkpoint": str(checkpoint),
        "test_metrics_path": str(metrics_path),
        "horizons": horizons,
        "largest_avg": largest_avg,
        "basic_overall": metrics.get("overall", {}),
        "horizon_metrics": {f"horizon_{h}": metrics[f"horizon_{h}"] for h in horizons},
    }
    with (job_output_dir / "record.json").open("w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return 0


def collect_jobs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jobs: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for model in selected_models(args.scope):
        spec = MODEL_SPECS[model]
        for dataset in args.datasets:
            for variant, model_name in spec["variants"].items():
                checkpoint, pattern = find_checkpoint(
                    args.checkpoint_root.resolve(),
                    dataset,
                    variant,
                    model_name,
                )
                row = {
                    "dataset": dataset,
                    "model": model,
                    "variant": variant,
                    "model_name": model_name,
                    "config": spec["config"],
                    "pattern": pattern,
                }
                if checkpoint is None:
                    missing.append({**row, "reason": "checkpoint_not_found"})
                else:
                    jobs.append({**row, "checkpoint": str(checkpoint.resolve())})
    return jobs, missing


def job_output_dir(run_dir: Path, job: dict[str, Any]) -> Path:
    safe_variant = job["variant"].replace("/", "_")
    return run_dir / "eval_runs" / job["model"] / job["dataset"] / safe_variant


def run_batch(args: argparse.Namespace) -> int:
    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root.resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    jobs, missing = collect_jobs(args)
    missing_fieldnames = ["dataset", "model", "variant", "model_name", "config", "pattern", "reason"]
    write_csv(run_dir / "missing.csv", missing, missing_fieldnames)

    manifest_fieldnames = [
        "dataset",
        "model",
        "variant",
        "model_name",
        "config",
        "checkpoint",
        "pattern",
    ]
    write_csv(run_dir / "manifest.csv", jobs, manifest_fieldnames)

    if args.dry_run:
        print(f"Run dir: {run_dir}")
        print(f"Discovered jobs: {len(jobs)}")
        print(f"Missing jobs: {len(missing)}")
        for job in jobs[:20]:
            print(f"FOUND {job['dataset']} {job['model']} {job['variant']} -> {job['checkpoint']}")
        if len(jobs) > 20:
            print(f"... {len(jobs) - 20} more")
        return 0

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    metrics_jsonl = run_dir / "metrics.jsonl"
    if metrics_jsonl.exists():
        metrics_jsonl.unlink()

    for index, job in enumerate(jobs, start=1):
        print(
            f"[{index}/{len(jobs)}] Evaluating "
            f"{job['dataset']} {job['model']} {job['variant']}",
            flush=True,
        )
        output_dir = job_output_dir(run_dir, job)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--single",
            "--model",
            job["model"],
            "--dataset",
            job["dataset"],
            "--variant",
            job["variant"],
            "--config",
            job["config"],
            "--checkpoint",
            job["checkpoint"],
            "--job-output-dir",
            str(output_dir),
            "--gpu",
            args.gpu,
            "--device-type",
            args.device_type,
        ]
        if args.batch_size is not None:
            cmd.extend(["--batch-size", str(args.batch_size)])
        env = build_job_env(job["model"], job["dataset"], job["variant"])
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False)
        if proc.returncode != 0:
            failures.append({**job, "reason": f"returncode_{proc.returncode}"})
            continue

        record_path = output_dir / "record.json"
        with record_path.open() as f:
            record = json.load(f)
        records.append(record)
        append_jsonl(metrics_jsonl, record)

    if failures:
        failure_fields = ["dataset", "model", "variant", "model_name", "config", "checkpoint", "reason"]
        write_csv(run_dir / "failures.csv", failures, failure_fields)

    write_summaries(run_dir, records)
    print(f"Run dir: {run_dir}")
    print(f"Evaluated records: {len(records)}")
    print(f"Missing jobs: {len(missing)}")
    print(f"Failed jobs: {len(failures)}")
    return 1 if failures else 0


def write_summaries(run_dir: Path, records: list[dict[str, Any]]) -> None:
    long_rows: list[dict[str, Any]] = []
    pivot: dict[str, dict[str, str]] = {}
    columns: list[str] = []

    for record in sorted(records, key=lambda r: (r["dataset"], r["model"], r["variant"])):
        avg = record["largest_avg"]
        cell = format_summary_cell(avg)
        column = f"{record['model']}:{record['variant']}"
        if column not in columns:
            columns.append(column)
        pivot.setdefault(record["dataset"], {})[column] = cell
        long_rows.append(
            {
                "dataset": record["dataset"],
                "model": record["model"],
                "variant": record["variant"],
                "MAE": f"{avg['MAE']:.6f}",
                "RMSE": f"{avg['RMSE']:.6f}",
                "MAPE": f"{avg['MAPE']:.6f}",
                "cell": cell,
                "checkpoint": record["checkpoint"],
                "test_metrics_path": record["test_metrics_path"],
            }
        )

    write_csv(
        run_dir / "summary_long.csv",
        long_rows,
        ["dataset", "model", "variant", "MAE", "RMSE", "MAPE", "cell", "checkpoint", "test_metrics_path"],
    )

    dataset_order = sorted(pivot)
    pivot_rows = []
    for dataset in dataset_order:
        row = {"dataset": dataset}
        for column in columns:
            row[column] = pivot[dataset].get(column, "/")
        pivot_rows.append(row)
    write_csv(run_dir / "summary.csv", pivot_rows, ["dataset", *columns])


def main() -> int:
    args = parse_args()
    try:
        if args.single:
            return run_single_eval(args)
        return run_batch(args)
    except BaseException:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
