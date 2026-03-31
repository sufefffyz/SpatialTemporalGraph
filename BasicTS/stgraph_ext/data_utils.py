import json
import os
import pickle
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def dataset_dir(dataset_name: str) -> str:
    return os.path.join("datasets", dataset_name)


def resolve_dataset_dir(dataset_name: str, output_root: str = "datasets") -> Path:
    output_root_path = Path(output_root)
    if not output_root_path.is_absolute():
        output_root_path = project_root() / output_root_path
    return output_root_path / dataset_name


def load_description(dataset_name: str) -> dict:
    with open(os.path.join(dataset_dir(dataset_name), "desc.json"), "r") as fp:
        return json.load(fp)


def load_data_memmap(dataset_name: str, mode: str = "r") -> np.memmap:
    description = load_description(dataset_name)
    data_path = os.path.join(dataset_dir(dataset_name), "data.dat")
    return np.memmap(data_path, dtype="float32", mode=mode, shape=tuple(description["shape"]))


def load_data_array(dataset_name: str) -> np.ndarray:
    return np.asarray(load_data_memmap(dataset_name)).copy()


def load_split_indices(dataset_name: str, split_filename: str = "split_indices.npz") -> dict[str, np.ndarray]:
    split_path = os.path.join(dataset_dir(dataset_name), split_filename)
    with np.load(split_path) as data:
        return {key: np.asarray(data[key], dtype=np.int64) for key in data.files}


def load_adjacency_matrix(dataset_name: str, filename: str = "adj_mx.pkl") -> np.ndarray:
    adj_path = os.path.join(dataset_dir(dataset_name), filename)
    with open(adj_path, "rb") as fp:
        obj = pickle.load(fp)
    if isinstance(obj, tuple) and len(obj) == 3:
        return np.asarray(obj[2], dtype=np.float32)
    return np.asarray(obj, dtype=np.float32)


def save_adjacency_matrix(adj: np.ndarray, path: str) -> None:
    with open(path, "wb") as fp:
        pickle.dump(np.asarray(adj, dtype=np.float32), fp)


def save_basicts_adjacency(adj: np.ndarray, path: str, node_ids: Iterable[Any] | None = None) -> None:
    adjacency = np.asarray(adj, dtype=np.float32)
    if node_ids is None:
        sensor_ids = [str(idx) for idx in range(adjacency.shape[0])]
    else:
        sensor_ids = [str(node_id) for node_id in node_ids]
    sensor_id_to_ind = {sensor_id: idx for idx, sensor_id in enumerate(sensor_ids)}
    with open(path, "wb") as fp:
        pickle.dump((sensor_ids, sensor_id_to_ind, adjacency), fp)


def build_temporal_feature_array(dt_index: pd.DatetimeIndex, num_nodes: int) -> tuple[np.ndarray, list[str]]:
    time_of_day = (
        (dt_index.hour.to_numpy(dtype=np.float32) * 60.0 + dt_index.minute.to_numpy(dtype=np.float32))
        / (24.0 * 60.0)
    )
    day_of_week = dt_index.dayofweek.to_numpy(dtype=np.float32) / 7.0
    time_of_day = np.tile(time_of_day[:, None, None], (1, num_nodes, 1))
    day_of_week = np.tile(day_of_week[:, None, None], (1, num_nodes, 1))
    return np.concatenate([time_of_day, day_of_week], axis=-1).astype(np.float32), ["time of day", "day of week"]


def build_multichannel_series(targets: np.ndarray, dt_index: pd.DatetimeIndex) -> tuple[np.ndarray, list[str]]:
    target_array = np.asarray(targets, dtype=np.float32)
    if target_array.ndim == 2:
        target_array = target_array[..., None]
    if target_array.ndim != 3 or target_array.shape[-1] != 1:
        raise ValueError(f"Expected targets with shape [L, N] or [L, N, 1], got {target_array.shape}.")
    temporal_features, feature_names = build_temporal_feature_array(dt_index, target_array.shape[1])
    data = np.concatenate([target_array, temporal_features], axis=-1).astype(np.float32)
    return data, ["target", *feature_names]


def save_memmap_array(path: str | Path, data: np.ndarray) -> None:
    array = np.asarray(data, dtype=np.float32)
    fp = np.memmap(path, dtype="float32", mode="w+", shape=array.shape)
    fp[:] = array[:]
    fp.flush()
    del fp


def save_description(path: str | Path, description: dict) -> None:
    with open(path, "w") as fp:
        json.dump(description, fp, indent=4)


def save_split_indices(path: str | Path, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> None:
    np.savez_compressed(
        path,
        train_idx=np.asarray(train_idx, dtype=np.int64),
        val_idx=np.asarray(val_idx, dtype=np.int64),
        test_idx=np.asarray(test_idx, dtype=np.int64),
    )


def infer_frequency_minutes(dt_index: pd.DatetimeIndex, fallback_minutes: int) -> int:
    if len(dt_index) < 2:
        return int(fallback_minutes)
    diffs = np.diff(dt_index.asi8)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return int(fallback_minutes)
    median_ns = int(np.median(diffs))
    return max(1, median_ns // (60 * 1_000_000_000))


def maybe_int(value: Any) -> Any:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return int(stripped)
        except ValueError:
            return stripped
    return value


def normalize_node_labels(values: Iterable[Any]) -> list[Any]:
    return [maybe_int(value) for value in values]


def extract_networkx_graph(obj: Any) -> nx.Graph:
    if isinstance(obj, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        return obj
    if isinstance(obj, dict):
        for key in ("graph", "G", "digraph"):
            if key in obj:
                return extract_networkx_graph(obj[key])
        for value in obj.values():
            if isinstance(value, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
                return value
    if isinstance(obj, (list, tuple)):
        for value in obj:
            if isinstance(value, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
                return value
    raise TypeError(f"Could not extract a networkx graph from object of type {type(obj)!r}.")


def build_directed_adjacency_from_graph(graph: nx.Graph, node_order: list[Any]) -> np.ndarray:
    adjacency = np.zeros((len(node_order), len(node_order)), dtype=np.float32)
    node_to_idx = {node: idx for idx, node in enumerate(node_order)}
    for source, target in graph.edges():
        if source in node_to_idx and target in node_to_idx:
            adjacency[node_to_idx[source], node_to_idx[target]] = 1.0
    return adjacency
