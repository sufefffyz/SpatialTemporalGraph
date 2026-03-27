"""Split Urban Traffic Benchmark NPZ graph into subgraphs by road-type feature.

Example:
  python scripts/split_graph_by_road_type.py \
      --input /data/yuzhang_fei/Urban_Traffic_Benchmark/city_traffic_l_speed.npz \
      --output-dir /data/yuzhang_fei/Urban_Traffic_Benchmark/subgraphs_l_speed

By default, the script auto-detects a road-type-like feature from
`spatial_node_feature_names`. You can also specify a feature explicitly via
`--road-type-feature`, e.g. category / edge_type / road_type / highway.

By default, saved edges keep all connections where at least one endpoint belongs
to the selected road-type group (edge mode: one-endpoint). You can switch to
induced subgraph behavior via `--edge-mode both-endpoints`.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


DEFAULT_FEATURE_CANDIDATES = [
    "road_type",
    "highway",
    "highway_type",
    "osm_highway",
    "functional_class",
    "road_class",
    "fclass",
    "category",
    "edge_type",
]


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")
    return value or "unknown"


def as_list_of_str(values: Iterable) -> List[str]:
    out = []
    for v in values:
        if isinstance(v, bytes):
            out.append(v.decode("utf-8", errors="replace"))
        else:
            out.append(str(v))
    return out


def get_spatial_matrix(spatial: np.ndarray) -> np.ndarray:
    # Expected shape is [1, num_nodes, num_features] in this benchmark.
    if spatial.ndim == 3:
        return spatial[0]
    if spatial.ndim == 2:
        return spatial
    raise ValueError(f"Unsupported spatial_node_features shape: {spatial.shape}")


def choose_feature_name(
    feature_names: List[str],
    explicit_name: str | None,
) -> str:
    feature_name_lut = {name.lower(): name for name in feature_names}
    if explicit_name:
        key = explicit_name.lower()
        if key not in feature_name_lut:
            raise ValueError(
                f"Requested feature '{explicit_name}' not found in spatial_node_feature_names: {feature_names}"
            )
        return feature_name_lut[key]

    for candidate in DEFAULT_FEATURE_CANDIDATES:
        if candidate in feature_name_lut:
            return feature_name_lut[candidate]

    raise ValueError(
        "Cannot auto-detect road-type feature. "
        f"Please pass --road-type-feature. Available features: {feature_names}"
    )


def extract_group_values(spatial: np.ndarray, feature_names: List[str], feature_name: str) -> np.ndarray:
    feature_idx = feature_names.index(feature_name)
    col = spatial[:, feature_idx]
    values = []
    for v in col:
        if isinstance(v, bytes):
            s = v.decode("utf-8", errors="replace")
        else:
            s = str(v)
        values.append(s)
    return np.array(values, dtype=object)


def find_coordinate_features(feature_names: List[str]) -> Dict[str, int]:
    wanted = [
        "x_coordinate_start",
        "y_coordinate_start",
        "x_coordinate_end",
        "y_coordinate_end",
    ]
    out: Dict[str, int] = {}
    for name in wanted:
        if name in feature_names:
            out[name] = feature_names.index(name)
    return out


def filter_edges_for_nodes(
    edges: np.ndarray,
    anchor_node_ids: np.ndarray,
    edge_mode: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    anchor_node_ids = anchor_node_ids.astype(int)
    anchor_node_set = set(anchor_node_ids.tolist())

    kept_original_edges: List[Tuple[int, int]] = []
    covered_node_set = set(anchor_node_set)

    for src, dst in edges:
        src_i = int(src)
        dst_i = int(dst)

        if edge_mode == "both-endpoints":
            keep = src_i in anchor_node_set and dst_i in anchor_node_set
        elif edge_mode == "one-endpoint":
            keep = src_i in anchor_node_set or dst_i in anchor_node_set
        else:
            raise ValueError(f"Unsupported edge_mode: {edge_mode}")

        if keep:
            kept_original_edges.append((src_i, dst_i))
            covered_node_set.add(src_i)
            covered_node_set.add(dst_i)

    covered_node_ids = np.asarray(sorted(covered_node_set), dtype=np.int64)
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(covered_node_ids.tolist())}

    if kept_original_edges:
        remapped_edges = np.asarray(
            [[old_to_new[src_i], old_to_new[dst_i]] for src_i, dst_i in kept_original_edges],
            dtype=np.int64,
        )
    else:
        remapped_edges = np.empty((0, 2), dtype=np.int64)

    return remapped_edges, covered_node_ids, old_to_new


def subset_dataset_by_nodes(
    data: Dict[str, np.ndarray],
    node_ids: np.ndarray,
    new_edges: np.ndarray,
    anchor_node_ids: np.ndarray,
    feature_name: str,
    feature_value: str,
    edge_mode: str,
    input_npz: str,
    coordinate_features: Dict[str, int],
) -> Dict[str, np.ndarray]:
    out = {}
    node_ids = node_ids.astype(int)

    for key, value in data.items():
        if key == "targets":
            out[key] = value[:, node_ids]
        elif key == "spatial_node_features":
            if value.ndim == 3:
                out[key] = value[:, node_ids, :]
            elif value.ndim == 2:
                out[key] = value[node_ids, :]
            else:
                raise ValueError(f"Unexpected shape for {key}: {value.shape}")
        elif key == "spatiotemporal_node_features":
            if value.ndim == 3:
                out[key] = value[:, node_ids, :]
            else:
                out[key] = value
        elif key == "edges":
            out[key] = new_edges
        elif key in {"num_nodes"}:
            out[key] = np.array([node_ids.shape[0]], dtype=np.int64)
        else:
            out[key] = value

    # Backward-compatible aliases.
    out["node_ids_in_original_graph"] = node_ids
    out["anchor_node_ids_in_original_graph"] = anchor_node_ids.astype(int)

    # Explicit metadata fields for downstream visualization / provenance.
    out["subgraph_node_ids_in_original_graph"] = node_ids
    out["subgraph_anchor_node_ids_in_original_graph"] = anchor_node_ids.astype(int)
    out["subgraph_anchor_node_mask"] = np.isin(node_ids, anchor_node_ids).astype(np.uint8)
    out["subgraph_edge_mode"] = np.array(edge_mode)
    out["subgraph_road_type_feature_name"] = np.array(feature_name)
    out["subgraph_road_type_feature_value"] = np.array(feature_value)
    out["subgraph_source_npz"] = np.array(input_npz)
    out["subgraph_saved_utc"] = np.array(datetime.utcnow().isoformat(timespec="seconds") + "Z")
    out["subgraph_coordinate_feature_names"] = np.array(list(coordinate_features.keys()), dtype=object)
    out["subgraph_coordinate_feature_indices"] = np.array(list(coordinate_features.values()), dtype=np.int64)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Split NPZ graph by road type feature.")
    parser.add_argument("--input", type=Path, required=True, help="Input NPZ path")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store subgraphs")
    parser.add_argument(
        "--road-type-feature",
        type=str,
        default=None,
        help="Feature name in spatial_node_feature_names used to split graph",
    )
    parser.add_argument(
        "--min-nodes",
        type=int,
        default=1,
        help="Skip subgraphs that contain fewer than this number of nodes",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=0,
        help="If > 0, only keep top-k largest groups",
    )
    parser.add_argument(
        "--edge-mode",
        type=str,
        choices=["one-endpoint", "both-endpoints"],
        default="one-endpoint",
        help=(
            "Edge filtering mode. "
            "one-endpoint: keep edges where at least one endpoint belongs to selected type; "
            "both-endpoints: keep only induced edges inside selected type nodes."
        ),
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = dict(np.load(args.input, allow_pickle=True))
    required = ["edges", "spatial_node_features", "spatial_node_feature_names", "targets"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing required keys in NPZ: {missing}")

    edges = np.asarray(data["edges"])
    spatial = get_spatial_matrix(np.asarray(data["spatial_node_features"]))
    feature_names = as_list_of_str(np.asarray(data["spatial_node_feature_names"]).tolist())
    coordinate_features = find_coordinate_features(feature_names)

    feature_name = choose_feature_name(feature_names, args.road_type_feature)
    group_values = extract_group_values(spatial, feature_names, feature_name)

    total_nodes = int(spatial.shape[0])
    total_edges = int(edges.shape[0])

    unique_values, counts = np.unique(group_values, return_counts=True)
    groups = list(zip(unique_values.tolist(), counts.tolist()))
    groups.sort(key=lambda x: x[1], reverse=True)
    if args.max_groups > 0:
        groups = groups[: args.max_groups]

    dataset_stem = args.input.stem
    summary = {
        "input": str(args.input),
        "feature_used": feature_name,
        "edge_mode": args.edge_mode,
        "total_nodes_before": total_nodes,
        "total_edges_before": total_edges,
        "coordinate_features": coordinate_features,
        "num_groups_total": int(len(unique_values)),
        "groups_saved": [],
    }

    print(f"Using feature: {feature_name}")
    print(f"Found {len(unique_values)} groups")
    print(f"Original graph stats: nodes={total_nodes}, edges={total_edges}")
    if coordinate_features:
        print(f"Coordinate features found: {coordinate_features}")
    else:
        print("WARNING: no standard coordinate features found; OSM plotting may need custom feature mapping.")

    for group_value, count in groups:
        anchor_node_ids = np.flatnonzero(group_values == group_value)
        if anchor_node_ids.shape[0] < args.min_nodes:
            continue

        sub_edges, covered_node_ids, _ = filter_edges_for_nodes(
            edges=edges,
            anchor_node_ids=anchor_node_ids,
            edge_mode=args.edge_mode,
        )
        sub_data = subset_dataset_by_nodes(
            data=data,
            node_ids=covered_node_ids,
            new_edges=sub_edges,
            anchor_node_ids=anchor_node_ids,
            feature_name=feature_name,
            feature_value=str(group_value),
            edge_mode=args.edge_mode,
            input_npz=str(args.input),
            coordinate_features=coordinate_features,
        )

        group_slug = slugify(group_value)
        out_file = args.output_dir / f"{dataset_stem}__{feature_name}__{group_slug}.npz"
        np.savez_compressed(out_file, **sub_data)

        info = {
            "group_value": str(group_value),
            "group_slug": group_slug,
            "num_anchor_nodes": int(anchor_node_ids.shape[0]),
            "num_nodes": int(covered_node_ids.shape[0]),
            "num_edges": int(sub_edges.shape[0]),
            "anchor_node_ratio": float(anchor_node_ids.shape[0] / max(total_nodes, 1)),
            "node_ratio": float(covered_node_ids.shape[0] / max(total_nodes, 1)),
            "edge_ratio": float(sub_edges.shape[0] / max(total_edges, 1)),
            "output_file": str(out_file),
        }
        summary["groups_saved"].append(info)
        print(
            f"Saved group '{group_value}' -> anchor_nodes={info['num_anchor_nodes']}, "
            f"nodes={info['num_nodes']}, "
            f"edges={info['num_edges']}, file={out_file.name}"
        )

    summary_path = args.output_dir / f"{dataset_stem}__split_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
