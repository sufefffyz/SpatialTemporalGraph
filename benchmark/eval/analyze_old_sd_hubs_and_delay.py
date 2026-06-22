#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SD_DIR = REPO_ROOT / "BasicTS" / "datasets" / "SD"
DEFAULT_SD_PHYS_DIR = REPO_ROOT / "BasicTS" / "datasets" / "SD_phys"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmark" / "eval" / "old_sd_hub_delay_analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze old-SD degree distribution, hub-node metrics, and edge delay distributions."
    )
    parser.add_argument("--sd-dir", type=Path, default=DEFAULT_SD_DIR, help="BasicTS old SD dataset directory.")
    parser.add_argument("--sd-phys-dir", type=Path, default=DEFAULT_SD_PHYS_DIR, help="BasicTS old SD_phys dataset directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for CSV/JSON/Markdown outputs.")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run spec: model:experiment=/abs/path/to/checkpoint_dir . Can be repeated.",
    )
    parser.add_argument(
        "--hub-ratio",
        type=float,
        default=0.15,
        help="Top fraction of nodes marked as hubs based on graph degree. Default: 0.15",
    )
    parser.add_argument(
        "--interp-minutes",
        type=int,
        default=5,
        help="Interpolation resolution in minutes for MCC delay computation. Default: 5",
    )
    parser.add_argument(
        "--max-lag-minutes",
        type=int,
        default=60,
        help="Maximum lag window in minutes searched by MCC. Default: 60",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="*",
        default=[3, 6, 12],
        help="Forecast horizons to report in addition to overall. Default: 3 6 12",
    )
    parser.add_argument(
        "--batch-chunk",
        type=int,
        default=256,
        help="Chunk size on sample dimension when computing group metrics. Default: 256",
    )
    parser.add_argument(
        "--default-graph-basis",
        default="distthre",
        choices=["distthre", "phys_dir", "phys_bidir", "identity"],
        help="Fallback graph basis for runs like adaptive that do not encode one explicitly. Default: distthre",
    )
    return parser.parse_args()


def resolve_existing_path(path: Path, desc: str) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{desc} not found: {path}")
    return path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_pkl(path: Path):
    with path.open("rb") as fp:
        return pickle.load(fp)


def unwrap_adj(payload) -> np.ndarray:
    if isinstance(payload, tuple) and len(payload) == 3:
        payload = payload[2]
    elif isinstance(payload, list) and len(payload) == 1:
        payload = payload[0]
    return np.asarray(payload, dtype=np.float32)


def load_dataset_flow(dataset_dir: Path) -> tuple[np.ndarray, dict]:
    desc = load_json(dataset_dir / "desc.json")
    shape = tuple(desc["shape"])
    data = np.memmap(dataset_dir / "data.dat", dtype="float32", mode="r", shape=shape)
    flow = np.asarray(data[..., 0]).copy()
    return flow, desc


def load_sensor_ids(dataset_dir: Path, num_nodes: int) -> np.ndarray:
    meta_path = dataset_dir / "meta.csv"
    if meta_path.exists():
        meta_df = pd.read_csv(meta_path)
        if "ID" in meta_df.columns and len(meta_df) == num_nodes:
            return pd.to_numeric(meta_df["ID"], errors="raise").astype(np.int64).to_numpy()
    return np.arange(num_nodes, dtype=np.int64)


def strip_self_loops(adj: np.ndarray) -> np.ndarray:
    adj = np.asarray(adj, dtype=np.float32).copy()
    np.fill_diagonal(adj, 0.0)
    return adj


def build_graph_matrices(sd_dir: Path, sd_phys_dir: Path) -> dict[str, np.ndarray]:
    distthre = strip_self_loops(unwrap_adj(load_pkl(sd_dir / "adj_mx.pkl")))
    phys_dir = strip_self_loops(unwrap_adj(load_pkl(sd_phys_dir / "adj_mx.pkl")))
    phys_bidir = np.maximum(phys_dir, phys_dir.T)
    identity = np.eye(distthre.shape[0], dtype=np.float32)
    np.fill_diagonal(identity, 0.0)
    return {
        "identity": identity,
        "distthre": distthre,
        "phys_dir": phys_dir,
        "phys_bidir": phys_bidir,
    }


def compute_degree_catalog(graph_name: str, adj: np.ndarray, sensor_ids: np.ndarray, hub_ratio: float) -> pd.DataFrame:
    adj = np.asarray(adj, dtype=np.float32)
    out_degree = (adj > 0).sum(axis=1).astype(int)
    in_degree = (adj > 0).sum(axis=0).astype(int)
    undirected_adj = np.maximum(adj, adj.T)
    undirected_degree = (undirected_adj > 0).sum(axis=1).astype(int)

    num_nodes = len(sensor_ids)
    hub_count = max(1, int(round(num_nodes * hub_ratio)))
    sort_order = np.argsort(-undirected_degree, kind="stable")
    is_hub = np.zeros(num_nodes, dtype=bool)
    if undirected_degree.max() > undirected_degree.min():
        is_hub[sort_order[:hub_count]] = True

    df = pd.DataFrame(
        {
            "graph": graph_name,
            "node_index": np.arange(num_nodes, dtype=int),
            "sensor_id": sensor_ids,
            "out_degree": out_degree,
            "in_degree": in_degree,
            "degree": undirected_degree,
            "group": np.where(is_hub, "hub", "normal"),
        }
    )
    return df


def infer_num_samples(npy_path: Path, output_len: int, num_nodes: int, channels: int = 1) -> int:
    bytes_per_sample = np.dtype(np.float32).itemsize * output_len * num_nodes * channels
    total_bytes = npy_path.stat().st_size
    if total_bytes % bytes_per_sample != 0:
        raise ValueError(f"File size does not match expected shape: {npy_path}")
    return total_bytes // bytes_per_sample


def load_memmap_array(path: Path, output_len: int, num_nodes: int) -> np.memmap:
    num_samples = infer_num_samples(path, output_len, num_nodes, channels=1)
    shape = (num_samples, output_len, num_nodes, 1)
    return np.memmap(path, dtype=np.float32, mode="r", shape=shape)


def parse_run_spec(spec: str) -> tuple[str, str, Path]:
    if "=" not in spec or ":" not in spec.split("=", 1)[0]:
        raise ValueError(f"--run format should be model:experiment=/abs/path/to/checkpoint_dir, got: {spec}")
    label, raw_path = spec.split("=", 1)
    model_name, experiment = label.split(":", 1)
    ckpt_dir = resolve_existing_path(Path(raw_path), f"checkpoint dir {label}")
    return model_name.strip(), experiment.strip(), ckpt_dir


def infer_graph_basis(experiment: str, default_basis: str) -> str:
    lower = experiment.lower()
    if "identity" in lower:
        return "identity"
    if "phys_bidir" in lower or "bidir" in lower:
        return "phys_bidir"
    if "phys" in lower or "directed" in lower:
        return "phys_dir"
    if "distthre" in lower or "original" in lower:
        return "distthre"
    return default_basis


def valid_mask(target: np.ndarray, null_val: float, for_mape: bool = False) -> np.ndarray:
    if math.isnan(null_val):
        mask = ~np.isnan(target)
    else:
        mask = ~np.isclose(target, null_val, atol=5e-5, rtol=0.0)
    if for_mape:
        mask &= ~np.isclose(target, 0.0, atol=5e-5, rtol=0.0)
    return mask


def compute_metrics_numpy(pred: np.ndarray, tgt: np.ndarray, null_val: float) -> dict[str, float]:
    mae_mask = valid_mask(tgt, null_val, for_mape=False)
    mape_mask = valid_mask(tgt, null_val, for_mape=True)

    mae = np.abs(pred[mae_mask] - tgt[mae_mask])
    rmse = (pred[mae_mask] - tgt[mae_mask]) ** 2
    mape = np.abs((pred[mape_mask] - tgt[mape_mask]) / tgt[mape_mask])

    return {
        "MAE": float(np.mean(mae)) if mae.size else float("nan"),
        "RMSE": float(np.sqrt(np.mean(rmse))) if rmse.size else float("nan"),
        "MAPE": float(np.mean(mape)) if mape.size else float("nan"),
    }


def evaluate_run_groups(
    model_name: str,
    experiment: str,
    ckpt_dir: Path,
    degree_catalogs: dict[str, pd.DataFrame],
    num_nodes: int,
    output_len: int,
    null_val: float,
    horizons: list[int],
) -> pd.DataFrame:
    pred_path = ckpt_dir / "test_results" / "predictions.npy"
    tgt_path = ckpt_dir / "test_results" / "targets.npy"
    if not pred_path.exists() or not tgt_path.exists():
        raise FileNotFoundError(
            f"{ckpt_dir} missing test_results/predictions.npy or targets.npy. "
            f"Please run BasicTS evaluate.py on the checkpoint first."
        )

    graph_basis = infer_graph_basis(experiment, default_basis="distthre")
    degree_df = degree_catalogs[graph_basis]

    pred = load_memmap_array(pred_path, output_len, num_nodes)
    tgt = load_memmap_array(tgt_path, output_len, num_nodes)

    rows = []
    for group_name in ["hub", "normal"]:
        mask = (degree_df["group"] == group_name).to_numpy()
        if mask.sum() == 0:
            continue

        group_pred = np.asarray(pred[:, :, mask, :])
        group_tgt = np.asarray(tgt[:, :, mask, :])
        overall = compute_metrics_numpy(group_pred, group_tgt, null_val)
        rows.append(
            {
                "model": model_name,
                "experiment": experiment,
                "graph_basis": graph_basis,
                "group": group_name,
                "num_nodes": int(mask.sum()),
                "horizon": "overall",
                **overall,
                "run_dir": str(ckpt_dir),
            }
        )

        for horizon in horizons:
            idx = horizon - 1
            if idx < 0 or idx >= output_len:
                continue
            pred_h = np.asarray(group_pred[:, idx : idx + 1, :, :])
            tgt_h = np.asarray(group_tgt[:, idx : idx + 1, :, :])
            metrics_h = compute_metrics_numpy(pred_h, tgt_h, null_val)
            rows.append(
                {
                    "model": model_name,
                    "experiment": experiment,
                    "graph_basis": graph_basis,
                    "group": group_name,
                    "num_nodes": int(mask.sum()),
                    "horizon": f"h{horizon}",
                    **metrics_h,
                    "run_dir": str(ckpt_dir),
                }
            )

    return pd.DataFrame(rows)


def interpolate_series_matrix(flow: np.ndarray, native_minutes: int, interp_minutes: int) -> tuple[np.ndarray, np.ndarray]:
    if interp_minutes == native_minutes:
        old_t = np.arange(flow.shape[0], dtype=np.float64) * native_minutes
        return flow.astype(np.float32), old_t

    if native_minutes % interp_minutes != 0 and interp_minutes % native_minutes != 0:
        raise ValueError(
            f"Interpolation resolution {interp_minutes} must align with native resolution {native_minutes}."
        )

    old_t = np.arange(flow.shape[0], dtype=np.float64) * native_minutes
    new_t = np.arange(0, old_t[-1] + interp_minutes, interp_minutes, dtype=np.float64)
    interp = np.empty((len(new_t), flow.shape[1]), dtype=np.float32)
    for node_idx in range(flow.shape[1]):
        interp[:, node_idx] = np.interp(new_t, old_t, flow[:, node_idx]).astype(np.float32)
    return interp, new_t


def max_cross_correlation_delay(src: np.ndarray, dst: np.ndarray, max_lag_steps: int) -> tuple[int, float]:
    best_lag = 0
    best_corr = -np.inf
    for lag in range(max_lag_steps + 1):
        if lag == 0:
            x = src
            y = dst
        else:
            x = src[:-lag]
            y = dst[lag:]
        if len(x) < 3:
            continue
        x_std = np.std(x)
        y_std = np.std(y)
        if x_std < 1e-8 or y_std < 1e-8:
            continue
        corr = float(np.corrcoef(x, y)[0, 1])
        if np.isnan(corr):
            continue
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    if not np.isfinite(best_corr):
        best_corr = float("nan")
    return best_lag, best_corr


def compute_delay_distribution(
    graph_name: str,
    adj: np.ndarray,
    flow: np.ndarray,
    sensor_ids: np.ndarray,
    native_minutes: int,
    interp_minutes: int,
    max_lag_minutes: int,
) -> pd.DataFrame:
    adj = strip_self_loops(adj)
    flow_interp, _ = interpolate_series_matrix(flow, native_minutes, interp_minutes)
    max_lag_steps = max(0, int(max_lag_minutes // interp_minutes))
    edges = np.argwhere(adj > 0)

    rows = []
    for edge_idx, (src_idx, dst_idx) in enumerate(edges, start=1):
        src = flow_interp[:, int(src_idx)]
        dst = flow_interp[:, int(dst_idx)]
        lag_steps, corr_peak = max_cross_correlation_delay(src, dst, max_lag_steps)
        rows.append(
            {
                "graph": graph_name,
                "source_index": int(src_idx),
                "target_index": int(dst_idx),
                "source_sensor_id": int(sensor_ids[int(src_idx)]),
                "target_sensor_id": int(sensor_ids[int(dst_idx)]),
                "delay_steps": int(lag_steps),
                "delay_minutes": int(lag_steps * interp_minutes),
                "corr_peak": corr_peak,
            }
        )
        if edge_idx % 1000 == 0:
            print(f"[{graph_name}] processed {edge_idx}/{len(edges)} edges", flush=True)

    return pd.DataFrame(rows)


def summarize_delay_distribution(delay_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for graph_name, graph_df in delay_df.groupby("graph"):
        rows.append(
            {
                "graph": graph_name,
                "num_directed_edges": int(len(graph_df)),
                "delay_mean_min": float(graph_df["delay_minutes"].mean()),
                "delay_std_min": float(graph_df["delay_minutes"].std()),
                "delay_p50_min": float(graph_df["delay_minutes"].quantile(0.5)),
                "delay_p90_min": float(graph_df["delay_minutes"].quantile(0.9)),
                "delay_max_min": float(graph_df["delay_minutes"].max()),
                "corr_peak_mean": float(graph_df["corr_peak"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("graph").reset_index(drop=True)


def make_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows.\n"
    display = df.copy()
    for column in display.columns:
        if pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "/")
    return display.to_markdown(index=False) + "\n"


def main() -> None:
    args = parse_args()
    sd_dir = resolve_existing_path(args.sd_dir, "old SD dataset dir")
    sd_phys_dir = resolve_existing_path(args.sd_phys_dir, "old SD_phys dataset dir")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    flow, desc = load_dataset_flow(sd_dir)
    num_nodes = int(desc["num_nodes"])
    output_len = int(desc["regular_settings"]["OUTPUT_LEN"])
    null_val = float(desc["regular_settings"].get("NULL_VAL", 0.0))
    native_minutes = int(desc["frequency (minutes)"])
    sensor_ids = load_sensor_ids(sd_dir, num_nodes)

    graph_matrices = build_graph_matrices(sd_dir, sd_phys_dir)

    degree_frames = []
    degree_catalogs = {}
    for graph_name, adj in graph_matrices.items():
        degree_df = compute_degree_catalog(graph_name, adj, sensor_ids, args.hub_ratio)
        degree_frames.append(degree_df)
        degree_catalogs[graph_name] = degree_df
    degree_df = pd.concat(degree_frames, ignore_index=True)

    delay_frames = []
    for graph_name in ["distthre", "phys_dir", "phys_bidir"]:
        delay_frames.append(
            compute_delay_distribution(
                graph_name=graph_name,
                adj=graph_matrices[graph_name],
                flow=flow,
                sensor_ids=sensor_ids,
                native_minutes=native_minutes,
                interp_minutes=args.interp_minutes,
                max_lag_minutes=args.max_lag_minutes,
            )
        )
    delay_df = pd.concat(delay_frames, ignore_index=True)
    delay_summary_df = summarize_delay_distribution(delay_df)

    group_metric_frames = []
    for spec in args.run:
        model_name, experiment, ckpt_dir = parse_run_spec(spec)
        group_metric_frames.append(
            evaluate_run_groups(
                model_name=model_name,
                experiment=experiment,
                ckpt_dir=ckpt_dir,
                degree_catalogs=degree_catalogs,
                num_nodes=num_nodes,
                output_len=output_len,
                null_val=null_val,
                horizons=args.horizons,
            )
        )
    group_metrics_df = (
        pd.concat(group_metric_frames, ignore_index=True)
        if group_metric_frames
        else pd.DataFrame(columns=["model", "experiment", "graph_basis", "group", "num_nodes", "horizon", "MAE", "RMSE", "MAPE", "run_dir"])
    )

    degree_csv = output_dir / "degree_catalog.csv"
    delay_csv = output_dir / "edge_delay_distribution.csv"
    delay_summary_csv = output_dir / "edge_delay_summary.csv"
    group_metrics_csv = output_dir / "hub_normal_metrics.csv"
    report_md = output_dir / "report.md"

    degree_df.to_csv(degree_csv, index=False)
    delay_df.to_csv(delay_csv, index=False)
    delay_summary_df.to_csv(delay_summary_csv, index=False)
    group_metrics_df.to_csv(group_metrics_csv, index=False)

    report_sections = [
        "# Old SD Hub / Delay Analysis\n",
        "## Delay Summary\n",
        make_markdown(delay_summary_df),
        "## Hub Counts by Graph\n",
        make_markdown(
            degree_df.groupby(["graph", "group"]).agg(num_nodes=("node_index", "count"), mean_degree=("degree", "mean")).reset_index()
        ),
    ]
    if not group_metrics_df.empty:
        report_sections.extend(
            [
                "## Hub vs Normal Metrics\n",
                make_markdown(group_metrics_df.sort_values(["model", "experiment", "horizon", "group"]).reset_index(drop=True)),
            ]
        )
    report_md.write_text("\n".join(report_sections), encoding="utf-8")

    print(
        json.dumps(
            {
                "degree_csv": str(degree_csv),
                "delay_csv": str(delay_csv),
                "delay_summary_csv": str(delay_summary_csv),
                "group_metrics_csv": str(group_metrics_csv),
                "report_md": str(report_md),
                "num_graphs": len(graph_matrices),
                "num_delay_rows": int(len(delay_df)),
                "num_group_metric_rows": int(len(group_metrics_df)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
