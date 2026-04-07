#!/usr/bin/env python3
"""
Run-side utilities for the PeMS MVP pilot inside SpatialTemporalGraph.

Subcommands:
- prepare-configs: generate fixed-graph GWNet configs for already-prepared datasets
- summarize: summarize checkpoint results into CSV tables
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_ROOT = Path(__file__).resolve().parent
PEMS_ROOT = SCRIPT_ROOT.parent
BASICTS_ROOT = PEMS_ROOT.parent
DEFAULT_RESULTS_ROOT = PEMS_ROOT / "mvp_results"


def log(message: str) -> None:
    print(message, flush=True)


def resolve_existing_path(path_str: str, desc: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{desc} 不存在: {path}")
    return path


def dataset_names_for_args(districts: list[int], year: int, last_days: int | None, full: bool) -> list[str]:
    names = []
    for district in districts:
        base = f"PEMSD{district}_{year}_full" if full else f"PEMSD{district}_{year}_last{last_days}"
        names.extend([f"{base}_phys", f"{base}_knn"])
    return names


def build_run_specs(train_window_days: list[int] | None, include_full_train: bool) -> list[dict[str, object]]:
    specs = []
    if include_full_train or not train_window_days:
        specs.append({"run_tag": "fulltrain", "train_window_days": None})
    if train_window_days:
        for days in sorted(set(train_window_days), reverse=True):
            if days <= 0:
                raise ValueError("--train-window-days 必须为正整数")
            specs.append({"run_tag": f"train{days}d", "train_window_days": int(days)})
    return specs


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_gwnet_config(
    basicts_root: Path,
    dataset_name: str,
    run_tag: str,
    epochs: int,
    batch_size: int,
    train_window_days: int | None,
) -> Path:
    desc = load_json(basicts_root / "datasets" / dataset_name / "desc.json")
    config_name = f"{dataset_name}_{run_tag}"
    config_path = basicts_root / "baselines" / "GWNet" / f"{config_name}.py"

    config_text = textwrap.dedent(
        f"""\
        import os
        import sys
        import torch
        from easydict import EasyDict
        sys.path.append(os.path.abspath(__file__ + '/../../..'))

        from basicts.metrics import masked_mae, masked_mape, masked_rmse, masked_wape
        from basicts.data import RecentWindowTimeSeriesForecastingDataset
        from basicts.runners import SimpleTimeSeriesForecastingRunner
        from basicts.scaler import ZScoreScaler
        from basicts.utils import get_regular_settings, load_adj, load_dataset_desc

        from .arch import GraphWaveNet

        DATA_NAME = '{dataset_name}'
        regular_settings = get_regular_settings(DATA_NAME)
        INPUT_LEN = regular_settings['INPUT_LEN']
        OUTPUT_LEN = regular_settings['OUTPUT_LEN']
        TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
        NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
        RESCALE = regular_settings['RESCALE']
        NULL_VAL = regular_settings['NULL_VAL']

        MODEL_ARCH = GraphWaveNet
        adj_mx, _ = load_adj('datasets/' + DATA_NAME + '/adj_mx.pkl', 'doubletransition')
        desc = load_dataset_desc(DATA_NAME)
        NUM_NODES = desc['num_nodes']

        MODEL_PARAM = {{
            'num_nodes': NUM_NODES,
            'supports': [torch.tensor(i, dtype=torch.float32) for i in adj_mx],
            'dropout': 0.3,
            'gcn_bool': True,
            'addaptadj': False,
            'aptinit': None,
            'in_dim': 3,
            'out_dim': OUTPUT_LEN,
            'residual_channels': 32,
            'dilation_channels': 32,
            'skip_channels': 128,
            'end_channels': 256,
            'kernel_size': 2,
            'blocks': 4,
            'layers': 2,
        }}

        CFG = EasyDict()
        CFG.DESCRIPTION = 'PeMS MVP pilot on ' + DATA_NAME + ' ({run_tag})'
        CFG.GPU_NUM = 1
        CFG.RUNNER = SimpleTimeSeriesForecastingRunner

        CFG.ENV = EasyDict()
        CFG.ENV.SEED = 42
        CFG.ENV.DETERMINISTIC = True
        CFG.ENV.CUDNN = EasyDict()
        CFG.ENV.CUDNN.ENABLED = True
        CFG.ENV.CUDNN.BENCHMARK = True
        CFG.ENV.CUDNN.DETERMINISTIC = True

        CFG.DATASET = EasyDict()
        CFG.DATASET.NAME = DATA_NAME
        CFG.DATASET.TYPE = RecentWindowTimeSeriesForecastingDataset
        CFG.DATASET.PARAM = EasyDict({{
            'dataset_name': DATA_NAME,
            'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
            'input_len': INPUT_LEN,
            'output_len': OUTPUT_LEN,
            'train_recent_days': {repr(train_window_days)},
        }})

        CFG.SCALER = EasyDict()
        CFG.SCALER.TYPE = ZScoreScaler
        CFG.SCALER.PARAM = EasyDict({{
            'dataset_name': DATA_NAME,
            'train_ratio': TRAIN_VAL_TEST_RATIO[0],
            'norm_each_channel': NORM_EACH_CHANNEL,
            'rescale': RESCALE,
        }})

        CFG.MODEL = EasyDict()
        CFG.MODEL.NAME = MODEL_ARCH.__name__
        CFG.MODEL.ARCH = MODEL_ARCH
        CFG.MODEL.PARAM = MODEL_PARAM
        CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
        CFG.MODEL.TARGET_FEATURES = [0]

        CFG.METRICS = EasyDict()
        CFG.METRICS.FUNCS = EasyDict({{
            'MAE': masked_mae,
            'MAPE': masked_mape,
            'RMSE': masked_rmse,
            'WAPE': masked_wape,
        }})
        CFG.METRICS.TARGET = 'MAE'
        CFG.METRICS.NULL_VAL = NULL_VAL

        CFG.TRAIN = EasyDict()
        CFG.TRAIN.NUM_EPOCHS = {epochs}
        CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
            'checkpoints',
            MODEL_ARCH.__name__,
            '_'.join([DATA_NAME, '{run_tag}', str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
        )
        CFG.TRAIN.LOSS = masked_mae
        CFG.TRAIN.OPTIM = EasyDict()
        CFG.TRAIN.OPTIM.TYPE = 'Adam'
        CFG.TRAIN.OPTIM.PARAM = {{
            'lr': 0.002,
            'weight_decay': 0.0001,
        }}
        CFG.TRAIN.LR_SCHEDULER = EasyDict()
        CFG.TRAIN.LR_SCHEDULER.TYPE = 'MultiStepLR'
        CFG.TRAIN.LR_SCHEDULER.PARAM = {{
            'milestones': [10, 20],
            'gamma': 0.5,
        }}
        CFG.TRAIN.DATA = EasyDict()
        CFG.TRAIN.DATA.BATCH_SIZE = {batch_size}
        CFG.TRAIN.DATA.SHUFFLE = True
        CFG.TRAIN.CLIP_GRAD_PARAM = {{'max_norm': 5.0}}
        CFG.TRAIN.EARLY_STOPPING_PATIENCE = 10

        CFG.VAL = EasyDict()
        CFG.VAL.INTERVAL = 1
        CFG.VAL.DATA = EasyDict()
        CFG.VAL.DATA.BATCH_SIZE = {batch_size}

        CFG.TEST = EasyDict()
        CFG.TEST.INTERVAL = 1
        CFG.TEST.DATA = EasyDict()
        CFG.TEST.DATA.BATCH_SIZE = {batch_size}

        CFG.EVAL = EasyDict()
        CFG.EVAL.HORIZONS = [3, 6, 12]
        CFG.EVAL.USE_GPU = True
        """
    )
    config_path.write_text(config_text, encoding="utf-8")
    return config_path


def find_checkpoint_dir(basicts_root: Path, model_name: str, dataset_name: str, run_tag: str) -> Path:
    parent = basicts_root / "checkpoints" / model_name
    if not parent.exists():
        raise FileNotFoundError(f"checkpoint 根目录不存在: {parent}")
    matches = sorted(parent.glob(f"{dataset_name}_{run_tag}_*"))
    matches = [path for path in matches if path.is_dir()]
    if not matches:
        raise FileNotFoundError(f"未找到 {dataset_name} ({run_tag}) 的 checkpoint 目录")
    return max(matches, key=lambda path: path.stat().st_mtime)


def infer_num_test_samples(pred_path: Path, output_len: int, num_nodes: int) -> int:
    bytes_per_sample = np.dtype(np.float32).itemsize * output_len * num_nodes
    total_bytes = pred_path.stat().st_size
    if total_bytes % bytes_per_sample != 0:
        raise ValueError(f"预测结果文件大小与预期形状不一致: {pred_path}")
    return total_bytes // bytes_per_sample


def load_test_memmaps(pred_path: Path, tgt_path: Path, output_len: int, num_nodes: int) -> tuple[np.memmap, np.memmap]:
    num_samples = infer_num_test_samples(pred_path, output_len, num_nodes)
    shape = (num_samples, output_len, num_nodes, 1)
    pred = np.memmap(pred_path, dtype=np.float32, mode="r", shape=shape)
    tgt = np.memmap(tgt_path, dtype=np.float32, mode="r", shape=shape)
    return pred, tgt


def compute_group_mae(pred: np.ndarray, tgt: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return float("nan")
    err = np.abs(pred[:, :, mask, :] - tgt[:, :, mask, :])
    return float(err.mean())


def stratified_eval(dataset_dir: Path, ckpt_dir: Path) -> dict[str, float]:
    sensor_catalog_path = dataset_dir / "sensor_catalog.csv"
    pred_path = ckpt_dir / "test_results" / "predictions.npy"
    tgt_path = ckpt_dir / "test_results" / "targets.npy"
    if not (sensor_catalog_path.exists() and pred_path.exists() and tgt_path.exists()):
        return {}

    desc = load_json(dataset_dir / "desc.json")
    output_len = int(desc["regular_settings"]["OUTPUT_LEN"])
    num_nodes = int(desc["num_nodes"])
    pred, tgt = load_test_memmaps(pred_path, tgt_path, output_len, num_nodes)

    catalog = pd.read_csv(sensor_catalog_path)
    if "Type" not in catalog.columns:
        return {}

    sensor_type = catalog["Type"].astype(str).str.upper().fillna("")
    masks = {
        "mae_all": np.ones(num_nodes, dtype=bool),
        "mae_ml": (sensor_type == "ML").to_numpy(),
        "mae_or": (sensor_type == "OR").to_numpy(),
        "mae_fr": (sensor_type == "FR").to_numpy(),
        "mae_ramp": sensor_type.isin(["OR", "FR"]).to_numpy(),
    }
    return {name: compute_group_mae(pred, tgt, mask) for name, mask in masks.items()}


def parse_dataset_name(dataset_name: str) -> dict[str, str]:
    graph_type = "phys" if dataset_name.endswith("_phys") else "knn" if dataset_name.endswith("_knn") else "unknown"
    district = dataset_name.split("_")[0]
    return {
        "dataset": dataset_name,
        "district": district,
        "graph_type": graph_type,
    }


def summarize_dataset(
    basicts_root: Path,
    dataset_name: str,
    model_name: str,
    run_tag: str,
    train_window_days: int | None,
) -> dict[str, object]:
    dataset_dir = basicts_root / "datasets" / dataset_name
    ckpt_dir = find_checkpoint_dir(basicts_root, model_name, dataset_name, run_tag)
    metrics_path = ckpt_dir / "test_metrics.json"
    metrics = load_json(metrics_path)
    overall = metrics.get("overall", {})

    row = {
        **parse_dataset_name(dataset_name),
        "run_tag": run_tag,
        "train_window_days": train_window_days if train_window_days is not None else "full_train",
        "checkpoint_dir": str(ckpt_dir),
        "overall_mae": overall.get("MAE"),
        "overall_rmse": overall.get("RMSE"),
        "overall_mape": overall.get("MAPE"),
        "overall_wape": overall.get("WAPE"),
    }

    for horizon_key, prefix in (("horizon_3", "h3"), ("horizon_6", "h6"), ("horizon_12", "h12")):
        horizon_metrics = metrics.get(horizon_key, {})
        row[f"{prefix}_mae"] = horizon_metrics.get("MAE")
        row[f"{prefix}_rmse"] = horizon_metrics.get("RMSE")

    row.update(stratified_eval(dataset_dir, ckpt_dir))
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run-side utilities for the PeMS MVP pilot.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-configs", help="Generate fixed-graph GWNet configs and print train commands.")
    prepare.add_argument("--districts", type=int, nargs="+", required=True, help="District IDs, e.g. 3 4")
    prepare.add_argument("--year", type=int, default=2025, help="Target year, default 2025")
    prepare.add_argument("--full", action="store_true", help="Use dataset names in *_full_{phys,knn} form")
    prepare.add_argument("--last-days", type=int, default=60, help="Number of trailing days used in dataset names")
    prepare.add_argument("--train-window-days", type=int, nargs="*", default=None, help="Optional recent-train windows in days, e.g. 60 30 7")
    prepare.add_argument("--include-full-train", action="store_true", help="Also generate a config using the whole train split")
    prepare.add_argument("--basicts-root", type=Path, default=BASICTS_ROOT, help="BasicTS root")
    prepare.add_argument("--epochs", type=int, default=30, help="GWNet training epochs")
    prepare.add_argument("--batch-size", type=int, default=64, help="GWNet batch size")
    prepare.add_argument("--gpus", default="0", help="GPU string for printed commands")

    summarize = subparsers.add_parser("summarize", help="Summarize checkpoint results into CSV tables.")
    summarize.add_argument("--districts", type=int, nargs="+", required=True, help="District IDs, e.g. 3 4")
    summarize.add_argument("--year", type=int, default=2025, help="Target year, default 2025")
    summarize.add_argument("--full", action="store_true", help="Use dataset names in *_full_{phys,knn} form")
    summarize.add_argument("--last-days", type=int, default=60, help="Number of trailing days used in dataset names")
    summarize.add_argument("--train-window-days", type=int, nargs="*", default=None, help="Optional recent-train windows in days, e.g. 60 30 7")
    summarize.add_argument("--include-full-train", action="store_true", help="Also summarize the config using the whole train split")
    summarize.add_argument("--basicts-root", type=Path, default=BASICTS_ROOT, help="BasicTS root")
    summarize.add_argument("--model-name", default="GraphWaveNet", help="Checkpoint subdir name, default GraphWaveNet")
    summarize.add_argument("--output-root", type=Path, default=DEFAULT_RESULTS_ROOT, help="Directory for summary CSVs")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.basicts_root = resolve_existing_path(str(args.basicts_root), "BasicTS root")

    if not args.full and args.last_days <= 0:
        raise ValueError("--last-days 必须为正整数，或者改用 --full")

    dataset_names = dataset_names_for_args(args.districts, args.year, args.last_days, args.full)
    run_specs = build_run_specs(args.train_window_days, args.include_full_train)

    if args.command == "prepare-configs":
        generated = []
        for dataset_name in dataset_names:
            for spec in run_specs:
                config_path = write_gwnet_config(
                    basicts_root=args.basicts_root,
                    dataset_name=dataset_name,
                    run_tag=str(spec["run_tag"]),
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    train_window_days=spec["train_window_days"],
                )
                generated.append((dataset_name, spec["run_tag"], config_path))

        log("已生成 config:")
        for dataset_name, run_tag, config_path in generated:
            log(f"- {dataset_name} [{run_tag}]: {config_path}")

        log("")
        log(f"训练命令（在 {args.basicts_root} 下执行）:")
        log(f"cd {args.basicts_root}")
        for dataset_name, run_tag, _ in generated:
            log(f"python experiments/train.py -c baselines/GWNet/{dataset_name}_{run_tag}.py -g {args.gpus}")
        return 0

    args.output_root = args.output_root.expanduser().resolve()
    rows = []
    for dataset_name in dataset_names:
        for spec in run_specs:
            log(f"汇总 {dataset_name} [{spec['run_tag']}]")
            rows.append(
                summarize_dataset(
                    basicts_root=args.basicts_root,
                    dataset_name=dataset_name,
                    model_name=args.model_name,
                    run_tag=str(spec["run_tag"]),
                    train_window_days=spec["train_window_days"],
                )
            )

    summary_df = pd.DataFrame(rows).sort_values(["district", "graph_type", "run_tag"]).reset_index(drop=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "mvp_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    compare_cols = [
        col
        for col in (
            "district",
            "graph_type",
            "train_window_days",
            "overall_mae",
            "overall_rmse",
            "mae_ml",
            "mae_or",
            "mae_fr",
            "mae_ramp",
        )
        if col in summary_df.columns
    ]
    compare_df = summary_df[compare_cols].copy()
    compare_path = args.output_root / "mvp_compare_view.csv"
    compare_df.to_csv(compare_path, index=False)

    log("")
    log("MVP 结果摘要:")
    if not compare_df.empty:
        log(compare_df.round(4).to_string(index=False))
    log("")
    log(f"完整结果: {summary_path}")
    log(f"对比视图: {compare_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
