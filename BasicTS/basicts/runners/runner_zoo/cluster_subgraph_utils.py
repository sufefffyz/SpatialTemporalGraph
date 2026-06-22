import csv
import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import torch


@dataclass
class ClusterAssignment:
    clusters: List[List[int]]
    labels: List[int]
    cluster_type: str
    num_clusters: int
    dataset_name: str
    seed: int


@dataclass
class ClusterSubgraphSelection:
    node_indices: List[int]
    cluster_ids: List[int]
    active_node_ratio: float


def _validate_num_clusters(num_nodes: int, num_clusters: int) -> None:
    if num_clusters < 1:
        raise ValueError(f"num_clusters must be >= 1, got {num_clusters}.")
    if num_nodes < num_clusters:
        raise ValueError(
            "num_clusters cannot exceed num_nodes for non-empty balanced clusters, "
            f"got num_nodes={num_nodes}, num_clusters={num_clusters}."
        )


def _assignment_from_clusters(
    clusters: Sequence[Sequence[int]],
    cluster_type: str,
    dataset_name: str,
    seed: int,
) -> ClusterAssignment:
    clusters = [sorted(int(node) for node in cluster) for cluster in clusters]
    num_clusters = len(clusters)
    num_nodes = sum(len(cluster) for cluster in clusters)
    flattened = [node for cluster in clusters for node in cluster]
    if sorted(flattened) != list(range(num_nodes)):
        raise ValueError("Cluster assignment must cover node ids [0, num_nodes) exactly once.")
    labels = [-1] * num_nodes
    for cluster_id, cluster in enumerate(clusters):
        for node in cluster:
            labels[node] = cluster_id
    return ClusterAssignment(
        clusters=clusters,
        labels=labels,
        cluster_type=cluster_type,
        num_clusters=num_clusters,
        dataset_name=str(dataset_name),
        seed=int(seed),
    )


def _split_leaf_by_spread(indices: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    leaf_coords = coords[indices]
    spread = leaf_coords.max(axis=0) - leaf_coords.min(axis=0)
    axis = int(np.argmax(spread))
    order = np.lexsort((indices, leaf_coords[:, 1 - axis], leaf_coords[:, axis]))
    sorted_indices = indices[order]
    midpoint = int(np.ceil(len(sorted_indices) / 2.0))
    return sorted_indices[:midpoint], sorted_indices[midpoint:]


def build_balanced_spatial_kdtree_clusters(
    coords,
    num_clusters: int = 8,
    dataset_name: str = "",
    seed: int = 2023,
) -> ClusterAssignment:
    """Recursively bisect Lat/Lng coordinates into balanced spatial patches."""

    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape [num_nodes, 2], got {coords.shape}.")
    num_nodes = int(coords.shape[0])
    _validate_num_clusters(num_nodes, int(num_clusters))

    leaves = [np.arange(num_nodes, dtype=np.int64)]
    while len(leaves) < int(num_clusters):
        largest_position = max(range(len(leaves)), key=lambda pos: len(leaves[pos]))
        leaf = leaves.pop(largest_position)
        if len(leaf) <= 1:
            raise ValueError("Cannot split singleton spatial leaf while building balanced clusters.")
        left, right = _split_leaf_by_spread(leaf, coords)
        leaves.extend([left, right])

    clusters = [leaf.astype(np.int64).tolist() for leaf in leaves]
    return _assignment_from_clusters(
        clusters=clusters,
        cluster_type="spatial_kdtree",
        dataset_name=dataset_name,
        seed=seed,
    )


def load_spatial_coordinates(meta_csv_path: str):
    """Load Lat/Lng columns from a BasicTS-style meta.csv file."""

    lat_keys = {"lat", "latitude"}
    lng_keys = {"lng", "lon", "long", "longitude"}
    rows = []
    with open(meta_csv_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"meta csv has no header: {meta_csv_path}")
        lower_to_original = {field.lower(): field for field in reader.fieldnames}
        lat_field = next((lower_to_original[key] for key in lat_keys if key in lower_to_original), None)
        lng_field = next((lower_to_original[key] for key in lng_keys if key in lower_to_original), None)
        if lat_field is None or lng_field is None:
            raise ValueError(
                f"meta csv must contain Lat/Lng columns, got columns={reader.fieldnames}."
            )
        for row in reader:
            rows.append([float(row[lat_field]), float(row[lng_field])])
    return np.asarray(rows, dtype=np.float32)


def daily_profile_embeddings(data, steps_per_day: int = 96):
    """Build normalized average daily-profile embeddings from train traffic series."""

    array = np.asarray(data)
    if array.ndim == 3:
        series = array[:, :, 0]
    elif array.ndim == 2:
        series = array
    else:
        raise ValueError(f"data must have shape [T, N] or [T, N, C], got {array.shape}.")

    steps_per_day = int(steps_per_day)
    if steps_per_day <= 0:
        raise ValueError(f"steps_per_day must be positive, got {steps_per_day}.")
    full_days = int(series.shape[0]) // steps_per_day
    if full_days <= 0:
        raise ValueError(
            f"Need at least one full day to build signal clusters, got T={series.shape[0]}, "
            f"steps_per_day={steps_per_day}."
        )

    trimmed = series[: full_days * steps_per_day]
    profile = trimmed.reshape(full_days, steps_per_day, series.shape[1]).mean(axis=0).T
    mean = profile.mean(axis=1, keepdims=True)
    std = profile.std(axis=1, keepdims=True)
    return (profile - mean) / np.maximum(std, 1e-6)


def _init_kmeans_centroids(embeddings: np.ndarray, num_clusters: int, seed: int):
    rng = np.random.default_rng(seed)
    num_nodes = int(embeddings.shape[0])
    first = int(rng.integers(num_nodes))
    selected = [first]
    min_dist = np.sum((embeddings - embeddings[first]) ** 2, axis=1)
    while len(selected) < num_clusters:
        next_index = int(np.argmax(min_dist))
        selected.append(next_index)
        dist = np.sum((embeddings - embeddings[next_index]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, dist)
    return embeddings[selected].copy()


def _run_kmeans(embeddings: np.ndarray, num_clusters: int, seed: int, max_iter: int):
    centroids = _init_kmeans_centroids(embeddings, num_clusters, seed)
    labels = np.zeros(embeddings.shape[0], dtype=np.int64)
    for _ in range(int(max_iter)):
        distances = np.sum((embeddings[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        next_labels = distances.argmin(axis=1)
        if np.array_equal(labels, next_labels):
            break
        labels = next_labels
        for cluster_id in range(num_clusters):
            members = embeddings[labels == cluster_id]
            if len(members) == 0:
                farthest = int(np.argmax(distances.min(axis=1)))
                centroids[cluster_id] = embeddings[farthest]
            else:
                centroids[cluster_id] = members.mean(axis=0)
    return centroids


def _balanced_assign_to_centroids(embeddings: np.ndarray, centroids: np.ndarray):
    num_nodes, num_clusters = int(embeddings.shape[0]), int(centroids.shape[0])
    distances = np.sum((embeddings[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    base_capacity = num_nodes // num_clusters
    extra = num_nodes % num_clusters
    capacities = np.asarray(
        [base_capacity + (1 if cluster_id < extra else 0) for cluster_id in range(num_clusters)],
        dtype=np.int64,
    )
    order_by_distance = np.argsort(distances, axis=1)
    sorted_distances = np.take_along_axis(distances, order_by_distance, axis=1)
    if num_clusters == 1:
        confidence = np.full(num_nodes, np.inf)
    else:
        confidence = sorted_distances[:, 1] - sorted_distances[:, 0]
    node_order = np.lexsort((np.arange(num_nodes), -confidence))

    clusters = [[] for _ in range(num_clusters)]
    remaining = capacities.copy()
    for node in node_order.tolist():
        for cluster_id in order_by_distance[node].tolist():
            if remaining[cluster_id] > 0:
                clusters[cluster_id].append(int(node))
                remaining[cluster_id] -= 1
                break
    return clusters


def build_balanced_signal_kmeans_clusters(
    data,
    num_clusters: int = 8,
    dataset_name: str = "",
    seed: int = 2023,
    steps_per_day: int = 96,
    max_iter: int = 50,
) -> ClusterAssignment:
    """Cluster nodes by normalized average daily profile, then rebalance sizes."""

    embeddings = np.asarray(daily_profile_embeddings(data, steps_per_day=steps_per_day), dtype=np.float32)
    num_nodes = int(embeddings.shape[0])
    _validate_num_clusters(num_nodes, int(num_clusters))
    centroids = _run_kmeans(embeddings, int(num_clusters), int(seed), int(max_iter))
    clusters = _balanced_assign_to_centroids(embeddings, centroids)
    return _assignment_from_clusters(
        clusters=clusters,
        cluster_type="signal_kmeans",
        dataset_name=dataset_name,
        seed=seed,
    )


def build_random_balanced_clusters(
    num_nodes: int,
    num_clusters: int = 8,
    dataset_name: str = "",
    seed: int = 2023,
) -> ClusterAssignment:
    """Seeded random balanced partition for sanity-check ablations."""

    num_nodes = int(num_nodes)
    num_clusters = int(num_clusters)
    _validate_num_clusters(num_nodes, num_clusters)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(num_nodes).tolist()
    base_size = num_nodes // num_clusters
    extra = num_nodes % num_clusters
    clusters = []
    cursor = 0
    for cluster_id in range(num_clusters):
        size = base_size + (1 if cluster_id < extra else 0)
        clusters.append(permutation[cursor: cursor + size])
        cursor += size
    return _assignment_from_clusters(
        clusters=clusters,
        cluster_type="random_balanced_clusters",
        dataset_name=dataset_name,
        seed=seed,
    )


def cluster_assignment_cache_path(
    cache_dir: str,
    dataset_name: str,
    cluster_type: str,
    num_clusters: int,
    seed: int,
) -> str:
    filename = f"{dataset_name}_{cluster_type}_k{int(num_clusters)}_s{int(seed)}.npz"
    return os.path.join(cache_dir, filename)


def save_cluster_assignment(assignment: ClusterAssignment, cache_path: str) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    metadata = {
        "cluster_type": assignment.cluster_type,
        "num_clusters": assignment.num_clusters,
        "dataset_name": assignment.dataset_name,
        "seed": assignment.seed,
        "clusters": assignment.clusters,
    }
    np.savez_compressed(
        cache_path,
        labels=np.asarray(assignment.labels, dtype=np.int64),
        metadata=json.dumps(metadata),
    )


def load_cluster_assignment(cache_path: str) -> ClusterAssignment:
    with np.load(cache_path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"]))
        labels = payload["labels"].astype(np.int64).tolist()
    clusters = metadata["clusters"]
    assignment = _assignment_from_clusters(
        clusters=clusters,
        cluster_type=metadata["cluster_type"],
        dataset_name=metadata["dataset_name"],
        seed=int(metadata["seed"]),
    )
    if assignment.labels != labels:
        raise ValueError(f"Cached cluster labels do not match cached clusters: {cache_path}")
    return assignment


class ClusterSubgraphScheduler:
    """Epoch-local scheduler that visits fixed node clusters evenly."""

    def __init__(self, assignment: ClusterAssignment, num_active_clusters: int = 2, seed: int = 2023):
        if num_active_clusters < 1:
            raise ValueError(f"num_active_clusters must be >= 1, got {num_active_clusters}.")
        if num_active_clusters > assignment.num_clusters:
            raise ValueError(
                "num_active_clusters cannot exceed num_clusters, "
                f"got {num_active_clusters}>{assignment.num_clusters}."
            )
        self.assignment = assignment
        self.num_active_clusters = int(num_active_clusters)
        self.seed = int(seed)
        self.force_full = False
        self._epoch_groups = []

    @property
    def num_nodes(self) -> int:
        return len(self.assignment.labels)

    def begin_epoch(self, epoch: int, force_full: bool = False) -> None:
        self.force_full = bool(force_full)
        if self.force_full:
            self._epoch_groups = [list(range(self.assignment.num_clusters))]
            return
        rng = np.random.default_rng(self.seed + int(epoch))
        order = rng.permutation(self.assignment.num_clusters).tolist()
        groups = []
        for start in range(0, len(order), self.num_active_clusters):
            group = order[start: start + self.num_active_clusters]
            if len(group) < self.num_active_clusters:
                group.extend(order[: self.num_active_clusters - len(group)])
            groups.append(group)
        self._epoch_groups = groups

    def select(self, iter_index: int) -> ClusterSubgraphSelection:
        if not self._epoch_groups:
            self.begin_epoch(epoch=0, force_full=False)
        if self.force_full:
            cluster_ids = list(range(self.assignment.num_clusters))
        else:
            cluster_ids = self._epoch_groups[int(iter_index) % len(self._epoch_groups)]
        node_indices = sorted(
            node
            for cluster_id in cluster_ids
            for node in self.assignment.clusters[int(cluster_id)]
        )
        return ClusterSubgraphSelection(
            node_indices=node_indices,
            cluster_ids=[int(cluster_id) for cluster_id in cluster_ids],
            active_node_ratio=float(len(node_indices)) / float(self.num_nodes),
        )


def reduce_node_scores_to_samples(node_scores: torch.Tensor, node_indices: Iterable[int]) -> torch.Tensor:
    """Reduce [B, N] reference/current scores over the active node subset."""

    if node_scores.ndim != 2:
        raise ValueError(f"node_scores must have shape [B, N], got {tuple(node_scores.shape)}.")
    node_indices = torch.as_tensor(node_indices, dtype=torch.long, device=node_scores.device)
    if node_indices.numel() == 0:
        raise ValueError("node_indices cannot be empty.")
    return node_scores.index_select(1, node_indices).mean(dim=1)
