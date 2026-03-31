import argparse
import json
import pickle
import random
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from tools.graph_sampling_tools import (
    add_one_random_node,
    build_two_hop_projection_graph,
    combine_far_apart,
    get_all_sink_cases,
    get_all_subgraphs,
    get_longest_path,
    select_confounder_samples,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent


def progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an Urban Traffic Benchmark NPZ dataset so it can be consumed by "
            "the CausalRivers benchmark pipeline."
        )
    )
    parser.add_argument("--input-npz", required=True, help="Path to the Urban Traffic Benchmark .npz file.")
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Optional dataset identifier used for output directories. Defaults to the input file stem.",
    )
    parser.add_argument(
        "--product-dir",
        default=str(SCRIPT_DIR / "product"),
        help="Base directory where converted time series and graph assets will be written.",
    )
    parser.add_argument(
        "--labels-dir",
        default=str(SCRIPT_DIR / "datasets"),
        help="Base directory where sampled label pickles will be written.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["debug_set"],
        choices=["debug_set", "random", "1_random", "2_hop", "confounder", "sink", "close", "disjoint", "root_cause"],
        help=(
            "Sampling strategies to export. 'debug_set' is the safest smoke-test option on large traffic graphs."
        ),
    )
    parser.add_argument(
        "--n-vars",
        nargs="+",
        type=int,
        default=[3],
        help="Subgraph sizes to generate.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5,
        help="Maximum number of samples per strategy/size combination. Ignored when <= 0.",
    )
    parser.add_argument(
        "--label-tag",
        default=None,
        help=(
            "Optional suffix appended to each exported label directory, for example "
            "'10k' -> debug_set_3_10k."
        ),
    )
    parser.add_argument(
        "--resolution",
        default=None,
        help=(
            "Optional coarser aggregation resolution for the exported product assets, "
            "for example 15min, 30min, 1h, or 6h. The benchmark resolution should match."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used when downsampling candidate subgraphs.",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=0.05,
        help="Maximum geo_distance used by the 'close' strategy.",
    )
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    return name.replace("-", "_").replace(" ", "_")


def parse_resolution_to_minutes(resolution: str | None) -> int | None:
    if resolution is None:
        return None
    text = str(resolution).strip().lower()
    if not text:
        return None
    if text.isdigit():
        minutes = int(text)
    else:
        text = text.replace("minutes", "min").replace("minute", "min").replace("hours", "h").replace("hour", "h")
        minutes = int(pd.Timedelta(text).total_seconds() // 60)
    if minutes <= 0:
        raise ValueError(f"Resolution must be positive, got {resolution}.")
    return minutes


def resolution_minutes_to_suffix(minutes: int) -> str:
    if minutes <= 0:
        raise ValueError(f"Resolution minutes must be positive, got {minutes}.")
    if minutes % 60 == 0:
        return f"{minutes // 60}h"
    return f"{minutes}min"


def infer_dataset_name(base_name: str, resolution_minutes: int | None, native_minutes: int) -> str:
    if resolution_minutes is None or resolution_minutes == native_minutes:
        return sanitize_name(base_name)
    return sanitize_name(f"{base_name}_{resolution_minutes_to_suffix(resolution_minutes)}")


def maybe_scalar(array_like):
    if np.ndim(array_like) == 0:
        return array_like.item()
    return array_like


def infer_coordinate_indices(data) -> list[int] | None:
    if "subgraph_coordinate_feature_indices" in data:
        indices = np.asarray(data["subgraph_coordinate_feature_indices"]).astype(int).tolist()
        if len(indices) == 4:
            return indices

    if "spatial_node_feature_names" not in data:
        return None

    feature_names = [str(x) for x in maybe_scalar(data["spatial_node_feature_names"])]
    wanted = [
        "x_coordinate_start",
        "y_coordinate_start",
        "x_coordinate_end",
        "y_coordinate_end",
    ]
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    if all(name in name_to_idx for name in wanted):
        return [name_to_idx[name] for name in wanted]
    return None


def midpoint_from_spatial_features(spatial_row: np.ndarray, coordinate_indices: list[int]) -> tuple[float, float]:
    x_start, y_start, x_end, y_end = [float(spatial_row[idx]) for idx in coordinate_indices]
    return ((x_start + x_end) / 2.0, (y_start + y_end) / 2.0)


def build_graph(data) -> tuple[nx.DiGraph, dict]:
    targets = data["targets"]
    num_nodes = int(maybe_scalar(data["num_nodes"])) if "num_nodes" in data else targets.shape[1]
    edges = np.asarray(data["edges"]).astype(int)
    spatial_features = data["spatial_node_features"][0] if "spatial_node_features" in data else None
    coordinate_indices = infer_coordinate_indices(data)

    graph = nx.DiGraph()
    for node_id in progress(range(num_nodes), total=num_nodes, desc="Building nodes"):
        attrs = {"origin": "traffic"}
        if spatial_features is not None and coordinate_indices is not None:
            midpoint = midpoint_from_spatial_features(spatial_features[node_id], coordinate_indices)
            attrs["p"] = midpoint
        graph.add_node(node_id, **attrs)

    for source, target in progress(edges, total=len(edges), desc="Building edges"):
        edge_attrs = {
            "origin": "road_network",
            "h_distance": 0.0,
            "quality_h": 0,
            "quality_geo": 0,
            "geo_distance": None,
        }
        if coordinate_indices is not None and "p" in graph.nodes[source] and "p" in graph.nodes[target]:
            source_p = np.asarray(graph.nodes[source]["p"], dtype=float)
            target_p = np.asarray(graph.nodes[target]["p"], dtype=float)
            edge_attrs["geo_distance"] = float(np.linalg.norm(source_p - target_p))
        graph.add_edge(int(source), int(target), **edge_attrs)

    metadata = {
        "num_nodes": num_nodes,
        "num_edges": int(graph.number_of_edges()),
        "has_coordinates": coordinate_indices is not None,
        "coordinate_indices": coordinate_indices,
    }
    return graph, metadata


def aggregate_targets(targets: np.ndarray, unix_timestamps: np.ndarray, resolution_minutes: int | None) -> tuple[np.ndarray, np.ndarray]:
    if resolution_minutes is None:
        return np.asarray(targets, dtype=np.float32), np.asarray(unix_timestamps, dtype=np.int64)

    dt_index = pd.to_datetime(np.asarray(unix_timestamps, dtype=np.int64), unit="s")
    rule = resolution_minutes_to_suffix(resolution_minutes)
    frame = pd.DataFrame(np.asarray(targets, dtype=np.float32), index=dt_index)
    aggregated = frame.groupby(pd.DatetimeIndex(frame.index.floor(rule)), sort=True).mean()
    aggregated_index = pd.DatetimeIndex(aggregated.index)
    return aggregated.to_numpy(dtype=np.float32), aggregated_index.astype("datetime64[s]").astype(np.int64)


def _limit_candidates(candidates, max_samples: int, seed: int):
    candidates = list(dict.fromkeys(tuple(sorted(sample)) for sample in candidates))
    if max_samples <= 0 or len(candidates) <= max_samples:
        return candidates
    rng = random.Random(seed)
    chosen = rng.sample(candidates, max_samples)
    return sorted(chosen)


def sample_connected_candidates(graph: nx.DiGraph, n_vars: int, max_samples: int, seed: int):
    if max_samples <= 0:
        raise ValueError("sample_connected_candidates requires a positive max_samples.")

    rng = random.Random(seed)
    graph_nodes = list(graph.nodes)
    samples = set()
    max_attempts = max(500, max_samples * 100)
    progress_bar = tqdm(total=max_samples, desc=f"Sampling {n_vars}-node connected sets") if tqdm else None

    for _ in range(max_attempts):
        start = rng.choice(graph_nodes)
        sample = [start]
        sample_set = {start}

        frontier = set(graph.predecessors(start)) | set(graph.successors(start))
        frontier -= sample_set

        while len(sample) < n_vars and frontier:
            next_node = rng.choice(list(frontier))
            sample.append(next_node)
            sample_set.add(next_node)

            frontier |= set(graph.predecessors(next_node)) | set(graph.successors(next_node))
            frontier -= sample_set

        if len(sample) == n_vars:
            previous_count = len(samples)
            samples.add(tuple(sorted(sample)))
            if progress_bar is not None and len(samples) > previous_count:
                progress_bar.update(len(samples) - previous_count)
            if len(samples) >= max_samples:
                break

    if progress_bar is not None:
        progress_bar.close()
    return sorted(samples)


def format_label_dirname(strategy: str, n_vars: int, label_tag: str | None) -> str:
    base = f"{strategy}_{n_vars}"
    if label_tag:
        return f"{base}_{sanitize_name(label_tag)}"
    return base


def _is_root_cause_candidate(graph: nx.DiGraph, sample, n_vars: int):
    subgraph = graph.subgraph(sample)
    return nx.is_directed_acyclic_graph(subgraph) and nx.dag_longest_path_length(subgraph) == (n_vars - 1)


def _safe_root_cause_samples(graph: nx.DiGraph, n_vars: int, max_samples: int, seed: int):
    if max_samples > 0:
        candidates = sample_connected_candidates(
            graph,
            n_vars=n_vars,
            max_samples=max(10, max_samples * 5),
            seed=seed,
        )
        root_like = [sample for sample in candidates if _is_root_cause_candidate(graph, sample, n_vars=n_vars)]
        return _limit_candidates(root_like, max_samples, seed)

    candidates = get_all_subgraphs(graph, n_vars=n_vars)
    return [sample for sample in candidates if _is_root_cause_candidate(graph, sample, n_vars=n_vars)]


def generate_samples(graph: nx.DiGraph, strategy: str, n_vars: int, max_samples: int, seed: int, max_distance: float):
    if strategy == "debug_set":
        debug_samples = max_samples if max_samples > 0 else 5
        return sample_connected_candidates(graph, n_vars=n_vars, max_samples=debug_samples, seed=seed)

    if strategy == "random":
        if max_samples <= 0:
            candidates = get_all_subgraphs(graph, n_vars=n_vars)
            return _limit_candidates(candidates, max_samples, seed)
        return sample_connected_candidates(graph, n_vars=n_vars, max_samples=max_samples, seed=seed)

    if strategy == "1_random":
        isolates = list(nx.isolates(graph))
        if not isolates:
            raise ValueError("The graph has no isolated nodes, so the '1_random' strategy cannot be generated.")
        base_candidates = (
            sample_connected_candidates(graph, n_vars=n_vars - 1, max_samples=max_samples, seed=seed)
            if max_samples > 0
            else get_all_subgraphs(graph, n_vars=n_vars - 1)
        )
        candidates = add_one_random_node(graph, base_candidates)
        return _limit_candidates(candidates, max_samples, seed)

    if strategy == "confounder":
        candidates = select_confounder_samples(graph, n_vars=n_vars)
        return _limit_candidates(candidates, max_samples, seed)

    if strategy == "2_hop":
        if max_samples <= 0:
            candidates = get_all_subgraphs(graph, n_vars=n_vars)
            return _limit_candidates(candidates, max_samples, seed)
        return sample_connected_candidates(graph, n_vars=n_vars, max_samples=max_samples, seed=seed)

    if strategy == "sink":
        candidates = get_all_sink_cases(graph, n_vars=n_vars, restrict=max_samples if max_samples > 0 else 15)
        return _limit_candidates(candidates, max_samples, seed)

    if strategy == "close":
        if any(graph.edges[edge]["geo_distance"] is None for edge in graph.edges):
            raise ValueError(
                "The dataset does not expose coordinates, so the 'close' strategy cannot be generated."
            )
        candidates = get_all_subgraphs(graph, n_vars=n_vars)
        candidates = [
            sample
            for sample in candidates
            if get_longest_path(graph.subgraph(sample), measure="geo_distance") < max_distance
        ]
        return _limit_candidates(candidates, max_samples, seed)

    if strategy == "disjoint":
        if n_vars % 2 != 0:
            raise ValueError("The 'disjoint' strategy requires an even n_vars.")
        if any("p" not in graph.nodes[node] for node in graph.nodes):
            raise ValueError("The dataset does not expose coordinates, so the 'disjoint' strategy cannot be generated.")
        base_candidates = get_all_subgraphs(graph, n_vars=n_vars // 2)
        candidates = combine_far_apart(graph, base_candidates)
        return _limit_candidates(candidates, max_samples, seed)

    if strategy == "root_cause":
        candidates = _safe_root_cause_samples(graph, n_vars=n_vars, max_samples=max_samples, seed=seed)
        return _limit_candidates(candidates, max_samples, seed)

    raise NotImplementedError(f"Unsupported strategy: {strategy}")


def save_product_assets(
    product_dir: Path,
    graph: nx.DiGraph,
    data,
    metadata: dict,
    source_path: Path,
    targets_override: np.ndarray | None = None,
    timestamps_override: np.ndarray | None = None,
):
    product_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing product assets to {product_dir}")
    with open(product_dir / "graph.p", "wb") as handle:
        pickle.dump(graph, handle)

    targets_to_save = np.asarray(targets_override if targets_override is not None else data["targets"], dtype=np.float32)
    timestamps_to_save = np.asarray(
        timestamps_override if timestamps_override is not None else data["unix_timestamps"],
        dtype=np.int64,
    )
    np.save(product_dir / "targets.npy", targets_to_save)
    np.save(product_dir / "unix_timestamps.npy", timestamps_to_save)
    np.save(product_dir / "node_ids.npy", np.arange(graph.number_of_nodes(), dtype=np.int64))

    for split_name in ["train_timestamps", "val_timestamps", "test_timestamps"]:
        if split_name in data:
            np.save(product_dir / f"{split_name}.npy", data[split_name])

    manifest = {
        "source_npz": str(source_path.resolve()),
        "num_timestamps": int(targets_to_save.shape[0]),
        "num_nodes": int(graph.number_of_nodes()),
        "num_edges": int(graph.number_of_edges()),
        "timestamp_frequency_seconds": int(np.median(np.diff(timestamps_to_save))),
        "warning": (
            "Labels generated from this product use road-topology edges as pseudo ground truth. "
            "This is useful for making the pipeline runnable, but it is not equivalent to the "
            "causal ground truth used by the original CausalRivers benchmark."
        ),
        **metadata,
    }
    with open(product_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def _build_label_graph(graph: nx.DiGraph, node_ids, strategy: str):
    if strategy == "2_hop":
        return build_two_hop_projection_graph(graph, node_ids)
    return nx.subgraph(graph, node_ids).copy()


def save_label_samples(graph: nx.DiGraph, samples, output_path: Path, strategy: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subgraphs = [
        _build_label_graph(graph, node_ids, strategy=strategy)
        for node_ids in progress(samples, total=len(samples), desc=f"Saving {output_path.parent.name}")
    ]
    with open(output_path, "wb") as handle:
        pickle.dump(subgraphs, handle)


def main():
    args = parse_args()
    source_path = Path(args.input_npz)
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    print(f"Loading traffic dataset from {source_path}")
    data = np.load(source_path, allow_pickle=True)
    native_minutes = int(round(np.median(np.diff(np.asarray(data["unix_timestamps"], dtype=np.int64))) / 60))
    target_resolution_minutes = parse_resolution_to_minutes(args.resolution) or native_minutes
    if target_resolution_minutes < native_minutes:
        raise ValueError(
            f"Target resolution {target_resolution_minutes}min is finer than native resolution "
            f"{native_minutes}min."
        )
    if target_resolution_minutes % native_minutes != 0:
        raise ValueError(
            f"Target resolution {target_resolution_minutes}min must be an integer multiple of native "
            f"resolution {native_minutes}min."
        )

    dataset_name = sanitize_name(args.dataset_name or infer_dataset_name(source_path.stem, target_resolution_minutes, native_minutes))
    label_filename = f"{dataset_name}.p"

    aggregated_targets, aggregated_timestamps = aggregate_targets(
        data["targets"],
        data["unix_timestamps"],
        None if target_resolution_minutes == native_minutes else target_resolution_minutes,
    )

    print("Building graph representation")
    graph, graph_metadata = build_graph(data)

    product_dir = Path(args.product_dir) / f"traffic_{dataset_name}"
    save_product_assets(
        product_dir,
        graph,
        data,
        graph_metadata,
        source_path,
        targets_override=aggregated_targets,
        timestamps_override=aggregated_timestamps,
    )

    summary = []
    strategy_pairs = [(strategy, n_vars) for strategy in args.strategies for n_vars in args.n_vars]
    for strategy, n_vars in progress(
        strategy_pairs,
        total=len(strategy_pairs),
        desc="Generating label sets",
    ):
        print(f"Sampling strategy={strategy}, n_vars={n_vars}")
        samples = generate_samples(
            graph=graph,
            strategy=strategy,
            n_vars=n_vars,
            max_samples=args.max_samples,
            seed=args.seed,
            max_distance=args.max_distance,
        )
        if not samples:
            raise ValueError(
                f"No samples were generated for strategy={strategy}, n_vars={n_vars}. "
                "Try a different strategy or a smaller sample size."
            )
        label_dirname = format_label_dirname(strategy, n_vars, args.label_tag)
        label_path = Path(args.labels_dir) / f"traffic_{dataset_name}" / label_dirname / label_filename
        save_label_samples(graph, samples, label_path, strategy=strategy)
        summary.append((strategy, n_vars, len(samples), label_path))

    print(f"Prepared product assets in: {product_dir}")
    for strategy, n_vars, count, label_path in summary:
        print(f"Saved {count} samples for {strategy}/{n_vars}: {label_path}")

    print("\nExample benchmark command:")
    example_label_path = summary[0][3]
    print(
        "python benchmark.py "
        f"label_path={example_label_path} "
        f"data_path={product_dir} "
        "method=var "
        "data_preprocess.normalize=False "
        f"data_preprocess.resolution={resolution_minutes_to_suffix(target_resolution_minutes)}"
    )
    print(
        "To use this dataset with benchmark_traffic_multi.yaml, override only the paths and resolution, "
        "for example:"
    )
    print(
        "python benchmark.py --config-name benchmark_traffic_multi "
        f"label_path={example_label_path} "
        f"data_path={product_dir} "
        f"data_preprocess.resolution={resolution_minutes_to_suffix(target_resolution_minutes)}"
    )


if __name__ == "__main__":
    main()
