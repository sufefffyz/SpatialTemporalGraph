from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SD_DIR = REPO_ROOT / "BasicTS" / "datasets" / "SD"
DEFAULT_SD_PHYS_DIR = REPO_ROOT / "BasicTS" / "datasets" / "SD_phys"
DEFAULT_SD_5MIN_DIR = REPO_ROOT / "BasicTS" / "datasets" / "SD_5min_full"
DEFAULT_SD_5MIN_GRAPH_ROOT = REPO_ROOT / "graphs" / "SD"


def resolve_existing_path(path: str | Path, desc: str) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{desc} not found: {path}")
    return path


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_pkl(path: str | Path):
    with Path(path).open("rb") as fp:
        return pickle.load(fp)


def unwrap_adj(payload) -> np.ndarray:
    if isinstance(payload, tuple) and len(payload) == 3:
        payload = payload[2]
    elif isinstance(payload, list) and len(payload) == 1:
        payload = payload[0]
    return np.asarray(payload, dtype=np.float32)


def strip_self_loops(adj: np.ndarray) -> np.ndarray:
    adj = np.asarray(adj, dtype=np.float32).copy()
    np.fill_diagonal(adj, 0.0)
    return adj


def load_dataset_flow(dataset_dir: str | Path) -> tuple[np.ndarray, dict]:
    dataset_dir = resolve_existing_path(dataset_dir, "dataset dir")
    desc = load_json(dataset_dir / "desc.json")
    shape = tuple(desc["shape"])
    data = np.memmap(dataset_dir / "data.dat", dtype="float32", mode="r", shape=shape)
    flow = np.asarray(data[..., 0]).copy()
    return flow, desc


def load_sensor_ids(dataset_dir: str | Path, num_nodes: int) -> np.ndarray:
    dataset_dir = resolve_existing_path(dataset_dir, "dataset dir")
    meta_path = dataset_dir / "meta.csv"
    if meta_path.exists():
        meta_df = pd.read_csv(meta_path)
        if "ID" in meta_df.columns and len(meta_df) == num_nodes:
            return pd.to_numeric(meta_df["ID"], errors="raise").astype(np.int64).to_numpy()
    return np.arange(num_nodes, dtype=np.int64)


def build_graph_matrices(sd_dir: str | Path = DEFAULT_SD_DIR, sd_phys_dir: str | Path = DEFAULT_SD_PHYS_DIR) -> dict[str, np.ndarray]:
    sd_dir = resolve_existing_path(sd_dir, "SD dataset dir")
    sd_phys_dir = resolve_existing_path(sd_phys_dir, "SD_phys dataset dir")

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


def build_sd_5min_graph_matrices(
    dataset_dir: str | Path = DEFAULT_SD_5MIN_DIR,
    graph_root: str | Path = DEFAULT_SD_5MIN_GRAPH_ROOT,
) -> dict[str, np.ndarray]:
    dataset_dir = resolve_existing_path(dataset_dir, "SD_5min dataset dir")
    graph_root = Path(graph_root).expanduser().resolve()

    def _resolve_adj(*candidates: Path) -> Path:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "Unable to find adjacency file in any of: "
            + ", ".join(str(candidate) for candidate in candidates)
        )

    distthre_path = _resolve_adj(
        dataset_dir / "adj_mx_largeST_original.pkl",
        graph_root / "adj_mx_largeST_original.pkl",
    )
    phys_dir_path = _resolve_adj(
        dataset_dir / "adj_mx_physical_directed.pkl",
        graph_root / "adj_mx_physical_directed.pkl",
    )

    distthre = strip_self_loops(unwrap_adj(load_pkl(distthre_path)))
    phys_dir = strip_self_loops(unwrap_adj(load_pkl(phys_dir_path)))
    phys_bidir = np.maximum(phys_dir, phys_dir.T)
    identity = np.eye(distthre.shape[0], dtype=np.float32)
    np.fill_diagonal(identity, 0.0)
    return {
        "identity": identity,
        "distthre": distthre,
        "phys_dir": phys_dir,
        "phys_bidir": phys_bidir,
    }


def slice_flow_by_day_window(
    flow: np.ndarray,
    native_minutes: int,
    start_day: int,
    num_days: int,
) -> tuple[np.ndarray, dict[str, int]]:
    steps_per_day = (24 * 60) // native_minutes
    start_idx = int(start_day * steps_per_day)
    end_idx = int((start_day + num_days) * steps_per_day)
    if start_idx < 0 or start_idx >= flow.shape[0]:
        raise ValueError(f"start_day={start_day} is out of range for flow length {flow.shape[0]}.")
    end_idx = min(end_idx, flow.shape[0])
    if end_idx <= start_idx:
        raise ValueError(f"Invalid day window: start_day={start_day}, num_days={num_days}.")
    sliced = np.asarray(flow[start_idx:end_idx]).copy()
    meta = {
        "steps_per_day": int(steps_per_day),
        "start_day": int(start_day),
        "num_days_requested": int(num_days),
        "num_days_actual": int((end_idx - start_idx) // steps_per_day),
        "start_index": int(start_idx),
        "end_index": int(end_idx),
    }
    return sliced, meta


def compute_degree_catalog(graph_name: str, adj: np.ndarray, sensor_ids: np.ndarray, hub_ratio: float = 0.15) -> pd.DataFrame:
    adj = np.asarray(adj, dtype=np.float32)
    out_degree = (adj > 0).sum(axis=1).astype(int)
    in_degree = (adj > 0).sum(axis=0).astype(int)
    undirected_adj = np.maximum(adj, adj.T)
    degree = (undirected_adj > 0).sum(axis=1).astype(int)

    num_nodes = len(sensor_ids)
    hub_count = max(1, int(round(num_nodes * hub_ratio)))
    sort_order = np.argsort(-degree, kind="stable")
    is_hub = np.zeros(num_nodes, dtype=bool)
    if degree.max() > degree.min():
        is_hub[sort_order[:hub_count]] = True

    return pd.DataFrame(
        {
            "graph": graph_name,
            "node_index": np.arange(num_nodes, dtype=int),
            "sensor_id": sensor_ids,
            "out_degree": out_degree,
            "in_degree": in_degree,
            "degree": degree,
            "group": np.where(is_hub, "hub", "normal"),
        }
    )


def infer_num_samples(npy_path: Path, output_len: int, num_nodes: int, channels: int = 1) -> int:
    bytes_per_sample = np.dtype(np.float32).itemsize * output_len * num_nodes * channels
    total_bytes = npy_path.stat().st_size
    if total_bytes % bytes_per_sample != 0:
        raise ValueError(f"File size does not match expected shape: {npy_path}")
    return total_bytes // bytes_per_sample


def load_memmap_array(path: str | Path, output_len: int, num_nodes: int) -> np.memmap:
    path = Path(path)
    num_samples = infer_num_samples(path, output_len, num_nodes, channels=1)
    return np.memmap(path, dtype=np.float32, mode="r", shape=(num_samples, output_len, num_nodes, 1))


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


def infer_graph_basis(experiment: str, default_basis: str = "distthre") -> str:
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


def evaluate_run_groups(
    model_name: str,
    experiment: str,
    ckpt_dir: str | Path,
    degree_catalogs: dict[str, pd.DataFrame],
    num_nodes: int,
    output_len: int,
    null_val: float,
    horizons: list[int] | tuple[int, ...] = (3, 6, 12),
) -> pd.DataFrame:
    ckpt_dir = resolve_existing_path(ckpt_dir, f"checkpoint dir {model_name}:{experiment}")
    pred_path = ckpt_dir / "test_results" / "predictions.npy"
    tgt_path = ckpt_dir / "test_results" / "targets.npy"
    if not pred_path.exists() or not tgt_path.exists():
        raise FileNotFoundError(
            f"{ckpt_dir} missing test_results/predictions.npy or targets.npy. "
            f"Please run BasicTS evaluate.py on this checkpoint first."
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
            metrics_h = compute_metrics_numpy(
                np.asarray(group_pred[:, idx : idx + 1, :, :]),
                np.asarray(group_tgt[:, idx : idx + 1, :, :]),
                null_val,
            )
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


def interpolate_series_matrix(
    flow: np.ndarray,
    native_minutes: int,
    interp_minutes: int,
    method: str = "natural_cubic_spline",
) -> tuple[np.ndarray, np.ndarray]:
    if interp_minutes == native_minutes:
        old_t = np.arange(flow.shape[0], dtype=np.float64) * native_minutes
        return flow.astype(np.float32), old_t

    old_t = np.arange(flow.shape[0], dtype=np.float64) * native_minutes
    new_t = np.arange(0, old_t[-1] + interp_minutes, interp_minutes, dtype=np.float64)
    interp = np.empty((len(new_t), flow.shape[1]), dtype=np.float32)
    for node_idx in range(flow.shape[1]):
        if method == "linear":
            interp[:, node_idx] = np.interp(new_t, old_t, flow[:, node_idx]).astype(np.float32)
        elif method == "natural_cubic_spline":
            spline = CubicSpline(old_t, flow[:, node_idx], bc_type="natural")
            interp[:, node_idx] = spline(new_t).astype(np.float32)
        else:
            raise ValueError(f"Unsupported interpolation method: {method}")
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
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
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
    interp_minutes: int = 5,
    max_lag_minutes: int = 60,
    interpolation_method: str = "natural_cubic_spline",
    show_progress: bool = True,
) -> pd.DataFrame:
    adj = strip_self_loops(adj)
    flow_interp, _ = interpolate_series_matrix(
        flow,
        native_minutes,
        interp_minutes,
        method=interpolation_method,
    )
    max_lag_steps = max(0, int(max_lag_minutes // interp_minutes))
    edges = np.argwhere(adj > 0)

    rows = []
    iterator = edges
    if show_progress:
        iterator = tqdm(edges, total=len(edges), desc=f"{graph_name} edges", leave=False)

    for src_idx, dst_idx in iterator:
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
                "interpolation_method": interpolation_method,
            }
        )
    return pd.DataFrame(rows)


def compute_daily_delay_distributions(
    graph_name: str,
    adj: np.ndarray,
    flow: np.ndarray,
    sensor_ids: np.ndarray,
    native_minutes: int,
    start_day: int,
    num_days: int,
    interp_minutes: int = 5,
    max_lag_minutes: int = 60,
    interpolation_method: str = "natural_cubic_spline",
    show_progress: bool = True,
) -> pd.DataFrame:
    window_flow, meta = slice_flow_by_day_window(flow, native_minutes, start_day=start_day, num_days=num_days)
    steps_per_day = meta["steps_per_day"]
    frames = []
    iterator = range(meta["num_days_actual"])
    if show_progress:
        iterator = tqdm(iterator, total=meta["num_days_actual"], desc=f"{graph_name} daily windows", leave=False)

    for offset in iterator:
        day_start = offset * steps_per_day
        day_end = day_start + steps_per_day
        day_flow = np.asarray(window_flow[day_start:day_end]).copy()
        day_df = compute_delay_distribution(
            graph_name=graph_name,
            adj=adj,
            flow=day_flow,
            sensor_ids=sensor_ids,
            native_minutes=native_minutes,
            interp_minutes=interp_minutes,
            max_lag_minutes=max_lag_minutes,
            interpolation_method=interpolation_method,
            show_progress=show_progress,
        )
        day_df["day_offset"] = int(start_day + offset)
        day_df["day_in_window"] = int(offset)
        frames.append(day_df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


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
