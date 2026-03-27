"""Grouped MAE/MSE/MAPE evaluator for Urban Traffic Benchmark.

Supports two prediction sources:
1) Naive methods (built-in):
   - prev-latest
   - prev-periodic
   - constant
   - per-node-constant
2) External baseline/model predictions (from any model):
   - provide val/test prediction files via --val-preds and --test-preds

Outputs overall metrics and per-road-type metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch

from dataset import Dataset
from utils import DummyHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grouped MAE/MSE/MAPE evaluator.")

    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset .npz")
    parser.add_argument("--prediction-horizon", type=int, default=12)
    parser.add_argument("--only-predict-at-end-of-horizon", action="store_true")
    parser.add_argument("--road-type-col", type=str, default="edge_type")
    parser.add_argument("--mape-eps", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument(
        "--pred-source",
        type=str,
        default="naive",
        choices=["naive", "external"],
        help="Use built-in naive predictors or externally generated baseline/model predictions.",
    )

    # Naive settings.
    parser.add_argument(
        "--naive-method",
        type=str,
        default="prev-latest",
        choices=["prev-latest", "prev-periodic", "constant", "per-node-constant"],
    )
    parser.add_argument("--period", type=int, default=12, help="Used for prev-periodic")
    parser.add_argument("--constant", type=str, default="mean", help="Used for constant")
    parser.add_argument(
        "--per-node-constant",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="Used for per-node-constant",
    )

    # External prediction settings.
    parser.add_argument("--val-preds", type=str, default=None, help="Path to val predictions (.npy/.pt/.npz)")
    parser.add_argument("--test-preds", type=str, default=None, help="Path to test predictions (.npy/.pt/.npz)")
    parser.add_argument("--val-preds-key", type=str, default="val_preds", help="npz key if val preds file is .npz")
    parser.add_argument("--test-preds-key", type=str, default="test_preds", help="npz key if test preds file is .npz")

    parser.add_argument("--output-csv", type=str, default=None)
    return parser.parse_args()


def load_road_types(dataset_npz_path: str, road_type_col: str) -> np.ndarray:
    with np.load(dataset_npz_path, allow_pickle=True) as data:
        names = [str(x) for x in data["spatial_node_feature_names"].tolist()]
        if road_type_col not in names:
            raise ValueError(
                f"road type column '{road_type_col}' is not found. Available: {names}"
            )

        idx = names.index(road_type_col)
        values = np.asarray(data["spatial_node_features"][0, :, idx])

    if np.issubdtype(values.dtype, np.floating):
        rounded = np.rint(values)
        if np.allclose(values, rounded, equal_nan=True):
            values = rounded.astype(np.int64)
    return values


def _load_tensor(path: str, npz_key: str) -> torch.Tensor:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prediction file not found: {p}")

    if p.suffix == ".pt":
        arr = torch.load(str(p), map_location="cpu", weights_only=False)
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu()
        raise ValueError(f".pt file must contain a Tensor: {p}")

    if p.suffix == ".npy":
        return torch.from_numpy(np.load(str(p)))

    if p.suffix == ".npz":
        with np.load(str(p), allow_pickle=True) as data:
            if npz_key not in data:
                raise ValueError(f"Key '{npz_key}' not found in {p}. Keys: {list(data.keys())}")
            return torch.from_numpy(data[npz_key])

    raise ValueError(f"Unsupported prediction file type: {p.suffix}")


def make_naive_preds(
    ds: Dataset,
    method: str,
    prediction_horizon: int,
    only_end: bool,
    period: int,
    constant: str,
    per_node_constant: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    val_targets, _ = ds.get_val_targets_for_metrics()
    test_targets, _ = ds.get_test_targets_for_metrics()

    if method == "prev-latest":
        val_preds = ds.targets[ds.val_timestamps]
        test_preds = ds.targets[ds.test_timestamps]
        if not only_end:
            val_preds = val_preds.unsqueeze(2).expand(-1, -1, prediction_horizon)
            test_preds = test_preds.unsqueeze(2).expand(-1, -1, prediction_horizon)

    elif method == "prev-periodic":
        if period < prediction_horizon:
            raise ValueError("period must be >= prediction_horizon to avoid leakage")

        if only_end:
            val_preds = ds.targets[ds.val_timestamps - period + prediction_horizon]
            test_preds = ds.targets[ds.test_timestamps - period + prediction_horizon]
        else:
            val_idx = ds.val_timestamps - period + 1
            val_idx = torch.cat(
                [val_idx, torch.arange(val_idx[-1] + 1, val_idx[-1] + prediction_horizon, dtype=torch.int64)], axis=0
            )
            val_preds = ds.targets[val_idx].unfold(dimension=0, size=prediction_horizon, step=1)

            test_idx = ds.test_timestamps - period + 1
            test_idx = torch.cat(
                [test_idx, torch.arange(test_idx[-1] + 1, test_idx[-1] + prediction_horizon, dtype=torch.int64)], axis=0
            )
            test_preds = ds.targets[test_idx].unfold(dimension=0, size=prediction_horizon, step=1)

    elif method == "constant":
        if constant in ("mean", "median"):
            train_targets = ds.targets[ds.all_train_timestamps]
            train_nan_mask = ds.targets_nan_mask[ds.all_train_timestamps]
            known = train_targets[~train_nan_mask]
            const = known.mean().item() if constant == "mean" else known.median().item()
        else:
            const = float(constant)

        val_preds = torch.tensor(const).expand_as(val_targets)
        test_preds = torch.tensor(const).expand_as(test_targets)

    elif method == "per-node-constant":
        train_targets = ds.targets[ds.all_train_timestamps].numpy().copy()
        train_nan_mask = ds.targets_nan_mask[ds.all_train_timestamps].numpy()
        train_targets[train_nan_mask] = np.nan

        if per_node_constant == "mean":
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Mean of empty slice")
                consts = np.nanmean(train_targets, axis=0)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="All-NaN slice encountered")
                consts = np.nanmedian(train_targets, axis=0)

        consts = torch.from_numpy(consts)[None, :, None]
        val_preds = consts.expand_as(val_targets)
        test_preds = consts.expand_as(test_targets)

    else:
        raise ValueError(f"Unknown naive method: {method}")

    return val_preds, test_preds


def masked_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    nan_mask: torch.Tensor,
    mape_eps: float,
    node_mask: torch.Tensor | None = None,
) -> tuple[float, float, float, int]:
    preds = preds.float()
    targets = targets.float()
    valid = ~nan_mask.bool()

    if node_mask is not None:
        if preds.ndim == 3:
            valid = valid & node_mask[None, :, None]
        else:
            valid = valid & node_mask[None, :]

    count = int(valid.sum().item())
    if count == 0:
        return float("nan"), float("nan"), float("nan"), 0

    err = preds - targets
    mae = float(torch.abs(err)[valid].mean().item())
    mse = float((err ** 2)[valid].mean().item())

    mape_valid = valid & (torch.abs(targets) > mape_eps)
    if int(mape_valid.sum().item()) == 0:
        mape = float("nan")
    else:
        mape = float((torch.abs(err[mape_valid] / targets[mape_valid]).mean().item()) * 100.0)

    return mae, mse, mape, count


def main() -> None:
    args = parse_args()

    ds = Dataset(
        name_or_path=args.dataset,
        state_handler=DummyHandler(Path("dummy_path"), Path("."), 1),
        prediction_horizon=args.prediction_horizon,
        only_predict_at_end_of_horizon=args.only_predict_at_end_of_horizon,
        drop_early_train_timestamps="none",
        targets_for_features_nan_imputation_strategy="prev",
        do_not_use_temporal_features=True,
        do_not_use_spatial_features=True,
        do_not_use_spatiotemporal_features=True,
        time_based_features_types=[],
        time_based_features_periods=[],
        device=args.device,
    )

    road_types = load_road_types(args.dataset, args.road_type_col)
    road_types_t = torch.from_numpy(road_types)

    val_targets, val_mask = ds.get_val_targets_for_metrics()
    test_targets, test_mask = ds.get_test_targets_for_metrics()

    if args.pred_source == "naive":
        val_preds, test_preds = make_naive_preds(
            ds=ds,
            method=args.naive_method,
            prediction_horizon=args.prediction_horizon,
            only_end=args.only_predict_at_end_of_horizon,
            period=args.period,
            constant=args.constant,
            per_node_constant=args.per_node_constant,
        )
        exp_name = f"naive:{args.naive_method}"
    else:
        if not args.val_preds or not args.test_preds:
            raise ValueError("--val-preds and --test-preds are required when --pred-source external")
        val_preds = _load_tensor(args.val_preds, args.val_preds_key)
        test_preds = _load_tensor(args.test_preds, args.test_preds_key)
        exp_name = "external"

    if tuple(val_preds.shape) != tuple(val_targets.shape):
        raise ValueError(f"val preds shape mismatch: preds={tuple(val_preds.shape)} targets={tuple(val_targets.shape)}")
    if tuple(test_preds.shape) != tuple(test_targets.shape):
        raise ValueError(
            f"test preds shape mismatch: preds={tuple(test_preds.shape)} targets={tuple(test_targets.shape)}"
        )

    rows: list[dict[str, object]] = []

    # Overall.
    v_mae, v_mse, v_mape, v_n = masked_metrics(val_preds, val_targets, val_mask, args.mape_eps)
    t_mae, t_mse, t_mape, t_n = masked_metrics(test_preds, test_targets, test_mask, args.mape_eps)
    rows.append(
        {
            "experiment": exp_name,
            "road_type": "ALL",
            "val_MAE": v_mae,
            "val_MSE": v_mse,
            "val_MAPE": v_mape,
            "val_count": v_n,
            "test_MAE": t_mae,
            "test_MSE": t_mse,
            "test_MAPE": t_mape,
            "test_count": t_n,
        }
    )

    # Per road type.
    for rt in np.unique(road_types):
        node_mask = road_types_t == rt
        v_mae, v_mse, v_mape, v_n = masked_metrics(val_preds, val_targets, val_mask, args.mape_eps, node_mask)
        t_mae, t_mse, t_mape, t_n = masked_metrics(test_preds, test_targets, test_mask, args.mape_eps, node_mask)

        rows.append(
            {
                "experiment": exp_name,
                "road_type": str(rt),
                "val_MAE": v_mae,
                "val_MSE": v_mse,
                "val_MAPE": v_mape,
                "val_count": v_n,
                "test_MAE": t_mae,
                "test_MSE": t_mse,
                "test_MAPE": t_mape,
                "test_count": t_n,
            }
        )

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\nSaved grouped metrics to: {out}")


if __name__ == "__main__":
    main()
