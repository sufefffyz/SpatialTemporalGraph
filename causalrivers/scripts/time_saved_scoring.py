#!/usr/bin/env python3
"""Time pure causalrivers scoring on saved benchmark outputs.

This script isolates the scoring stage from the rest of `benchmark.py`.
It loads:
  - `config.yaml`
  - `preds.p`
from one or more saved run directories, rebuilds the label tensors, and
measures wall-clock time spent inside `tools.scoring_tools.score(...)`.

Example:

    python scripts/time_saved_scoring.py \
      --run-dir results/stgnn_precomputed_random_3/2026-04-01_12:34:56.123456 \
      --run-dir results/stgnn_precomputed_random_3/2026-04-01_12:40:00.654321
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from tools.scoring_tools import score
from tools.tools import graph_to_label_tensor, load_label_graphs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure pure scoring time from saved causalrivers benchmark outputs."
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        dest="run_dirs",
        help="Saved benchmark run directory containing config.yaml and preds.p. Can be repeated.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Override scorer parallelism. Defaults to 1.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Override scorer chunk size. Defaults to 512.",
    )
    return parser.parse_args()


def _resolve_repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _load_preds(path: Path) -> list[np.ndarray]:
    with path.open("rb") as handle:
        preds = pickle.load(handle)
    return [np.asarray(pred) for pred in preds]


def _prepare_labels(cfg: Any) -> list[np.ndarray]:
    cfg.label_path = str(_resolve_repo_path(cfg.label_path))
    label_graphs = load_label_graphs(cfg)
    return [graph_to_label_tensor(graph, human_readable=False) for graph in label_graphs]


def _first_pred_stats(preds: list[np.ndarray]) -> dict[str, Any]:
    first = np.asarray(preds[0])
    return {
        "num_preds": len(preds),
        "first_shape": tuple(first.shape),
        "nan_count": int(np.isnan(first).sum()),
        "inf_count": int(np.isinf(first).sum()),
        "min": float(np.nanmin(first)),
        "max": float(np.nanmax(first)),
    }


def main() -> int:
    args = parse_args()
    summary_rows: list[dict[str, Any]] = []

    for raw_run_dir in args.run_dirs:
        run_dir = Path(raw_run_dir).expanduser().resolve()
        config_path = run_dir / "config.yaml"
        preds_path = run_dir / "preds.p"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing config.yaml under {run_dir}")
        if not preds_path.is_file():
            raise FileNotFoundError(f"Missing preds.p under {run_dir}")

        cfg = OmegaConf.load(config_path)
        preds = _load_preds(preds_path)
        labels = _prepare_labels(cfg)

        if len(preds) != len(labels):
            raise ValueError(
                f"Prediction count mismatch for {run_dir}: {len(preds)} preds vs {len(labels)} labels."
            )

        stats = _first_pred_stats(preds)
        print(f"\n=== {run_dir.name} ===")
        print(f"run_dir: {run_dir}")
        print(f"label_path: {cfg.label_path}")
        print(f"method: {cfg.method.name}")
        for key, value in stats.items():
            print(f"{key}: {value}")

        t0 = time.perf_counter()
        out = score(
            preds,
            labels,
            remove_autoregressive=bool(cfg.remove_diagonal),
            name=str(cfg.method.name),
            n_jobs=args.n_jobs,
            chunk_size=args.chunk_size,
        )
        elapsed = time.perf_counter() - t0

        print(out)
        print(f"score_only_seconds: {elapsed:.6f}")

        summary_rows.append(
            {
                "run_dir": str(run_dir),
                "method": str(cfg.method.name),
                "label_path": str(cfg.label_path),
                "num_preds": len(preds),
                "first_shape": str(stats["first_shape"]),
                "pred_min": stats["min"],
                "pred_max": stats["max"],
                "score_only_seconds": elapsed,
            }
        )

    if len(summary_rows) > 1:
        print("\n=== Summary ===")
        print(pd.DataFrame(summary_rows).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
