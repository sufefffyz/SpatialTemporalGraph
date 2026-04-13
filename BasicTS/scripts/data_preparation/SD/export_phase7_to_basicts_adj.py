import argparse
import json
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a phase7 sensor graph into a full-node BasicTS SD adjacency."
    )
    parser.add_argument(
        "--phase7-root",
        required=True,
        help="Pipeline output root or the phase7 sensor_graph directory.",
    )
    parser.add_argument(
        "--dataset-dir",
        default="datasets/SD",
        help="Existing BasicTS SD dataset directory used as the canonical node order.",
    )
    parser.add_argument(
        "--output-dataset-dir",
        default=None,
        help="Target BasicTS dataset directory. Defaults to a sibling like datasets/SD_phys.",
    )
    parser.add_argument(
        "--method",
        choices=["local", "osrm"],
        default="local",
        help="Which phase7 graph to export.",
    )
    parser.add_argument(
        "--phase7-adj",
        default=None,
        help="Optional explicit path to phase7 adjacency pickle.",
    )
    parser.add_argument(
        "--phase7-sensor-ids",
        default=None,
        help="Optional explicit path to phase7 sensor_ids.npy.",
    )
    parser.add_argument(
        "--meta-id-column",
        default="ID",
        help="Column name in meta.csv that defines the canonical sensor order.",
    )
    parser.add_argument(
        "--output-adj",
        default=None,
        help="Optional explicit adjacency output path. Defaults to <output-dataset-dir>/adj_mx.pkl.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Optional dataset name written into desc.json. Defaults to <source_name>_phys.",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="Coverage report path. Defaults to <output-dataset-dir>/phase7_<method>_coverage.json.",
    )
    parser.add_argument(
        "--report-csv",
        default=None,
        help="Per-node coverage CSV path. Defaults to <output-dataset-dir>/phase7_<method>_node_coverage.csv.",
    )
    parser.add_argument(
        "--reuse-mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
        help="How to reuse non-graph files from the source dataset.",
    )
    return parser.parse_args()


def normalize_sensor_id(value) -> str:
    text = str(value).strip()
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def resolve_phase7_adj_path(root: Path, method: str) -> Path:
    candidates = [
        root / f"phase7_{method}_adj_matrix.pkl",
        root / "phase7_adj_matrix.pkl" if method == "local" else None,
        root / "sensor_graph" / f"phase7_{method}_adj_matrix.pkl",
        root / "sensor_graph" / "phase7_adj_matrix.pkl" if method == "local" else None,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Unable to locate phase7 adjacency for method={method} under {root}"
    )


def resolve_phase7_sensor_ids_path(root: Path, method: str) -> Path | None:
    candidates = [
        root / method / "sensor_ids.npy",
        root / "sensor_graph" / method / "sensor_ids.npy",
        root / "sensor_ids.npy",
        root / "sensor_graph" / "sensor_ids.npy",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def load_phase7_adjacency(path: Path) -> tuple[np.ndarray, list[str]]:
    with path.open("rb") as fp:
        payload = pickle.load(fp)

    if isinstance(payload, pd.DataFrame):
        adjacency = payload.to_numpy(dtype=np.float32)
        sensor_ids = [normalize_sensor_id(value) for value in payload.index.tolist()]
        return adjacency, sensor_ids

    if isinstance(payload, tuple) and len(payload) == 3:
        sensor_ids = [normalize_sensor_id(value) for value in payload[0]]
        adjacency = np.asarray(payload[2], dtype=np.float32)
        return adjacency, sensor_ids

    adjacency = np.asarray(payload, dtype=np.float32)
    return adjacency, []


def load_phase7_sensor_ids(path: Path | None, fallback_ids: list[str], expected_size: int) -> list[str]:
    if path is None:
        if not fallback_ids:
            raise ValueError("phase7 adjacency does not include sensor ids and no sensor_ids.npy was found.")
        return fallback_ids

    sensor_ids = np.load(path, allow_pickle=True).tolist()
    normalized = [normalize_sensor_id(value) for value in sensor_ids]
    if len(normalized) != expected_size:
        raise ValueError(
            f"phase7 sensor_ids length {len(normalized)} does not match adjacency size {expected_size}."
        )
    return normalized


def load_canonical_sensor_ids(meta_path: Path, id_column: str) -> list[str]:
    meta_df = pd.read_csv(meta_path)
    if id_column not in meta_df.columns:
        raise KeyError(f"Column {id_column!r} not found in {meta_path}")
    canonical_ids = [normalize_sensor_id(value) for value in meta_df[id_column].tolist()]
    if len(canonical_ids) != len(set(canonical_ids)):
        raise ValueError(f"Canonical sensor ids in {meta_path} are not unique.")
    return canonical_ids


def build_full_adjacency(
    canonical_ids: list[str],
    phase7_ids: list[str],
    phase7_adj: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    if phase7_adj.shape[0] != phase7_adj.shape[1]:
        raise ValueError(f"phase7 adjacency must be square, got {phase7_adj.shape}")
    if phase7_adj.shape[0] != len(phase7_ids):
        raise ValueError(
            f"phase7 adjacency size {phase7_adj.shape[0]} does not match sensor id count {len(phase7_ids)}."
        )
    if len(phase7_ids) != len(set(phase7_ids)):
        raise ValueError("phase7 sensor ids are not unique.")

    canonical_index = {sensor_id: idx for idx, sensor_id in enumerate(canonical_ids)}
    full_adj = np.zeros((len(canonical_ids), len(canonical_ids)), dtype=np.float32)

    matched_phase7_indices = []
    matched_canonical_indices = []
    missing_phase7_ids = []

    for phase7_idx, sensor_id in enumerate(phase7_ids):
        canonical_idx = canonical_index.get(sensor_id)
        if canonical_idx is None:
            missing_phase7_ids.append(sensor_id)
            continue
        matched_phase7_indices.append(phase7_idx)
        matched_canonical_indices.append(canonical_idx)

    if matched_phase7_indices:
        matched_phase7_indices_np = np.asarray(matched_phase7_indices, dtype=np.int64)
        matched_canonical_indices_np = np.asarray(matched_canonical_indices, dtype=np.int64)
        sub_adj = phase7_adj[np.ix_(matched_phase7_indices_np, matched_phase7_indices_np)]
        full_adj[np.ix_(matched_canonical_indices_np, matched_canonical_indices_np)] = sub_adj.astype(np.float32)

    matched_set = set(phase7_ids) - set(missing_phase7_ids)
    coverage_df = pd.DataFrame(
        {
            "sensor_id": canonical_ids,
            "matched_in_phase7": [sensor_id in matched_set for sensor_id in canonical_ids],
        }
    )
    return full_adj, coverage_df


def save_pickle_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fp:
        pickle.dump(np.asarray(array, dtype=np.float32), fp, protocol=4)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def replicate_file(source: Path, target: Path, mode: str) -> None:
    if target.exists() or target.is_symlink():
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            raise IsADirectoryError(f"Target exists and is not a file: {target}")

    ensure_parent(target)

    if mode == "symlink":
        rel_source = os.path.relpath(source, start=target.parent)
        target.symlink_to(rel_source)
        return

    if mode == "hardlink":
        os.link(source, target)
        return

    shutil.copy2(source, target)


def copy_desc_with_new_name(source_desc_path: Path, target_desc_path: Path, dataset_name: str) -> None:
    desc = json.loads(source_desc_path.read_text(encoding="utf-8"))
    desc["name"] = dataset_name
    ensure_parent(target_desc_path)
    target_desc_path.write_text(json.dumps(desc, indent=4), encoding="utf-8")


def main() -> None:
    args = parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    phase7_root = Path(args.phase7_root).resolve()
    meta_path = dataset_dir / "meta.csv"
    current_adj_path = dataset_dir / "adj_mx.pkl"
    source_desc_path = dataset_dir / "desc.json"

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.csv not found: {meta_path}")
    if not source_desc_path.exists():
        raise FileNotFoundError(f"desc.json not found: {source_desc_path}")
    if not phase7_root.exists():
        raise FileNotFoundError(f"phase7 root not found: {phase7_root}")

    phase7_adj_path = (
        Path(args.phase7_adj).resolve()
        if args.phase7_adj is not None
        else resolve_phase7_adj_path(phase7_root, args.method)
    )
    phase7_sensor_ids_path = (
        Path(args.phase7_sensor_ids).resolve()
        if args.phase7_sensor_ids is not None
        else resolve_phase7_sensor_ids_path(phase7_root, args.method)
    )

    canonical_ids = load_canonical_sensor_ids(meta_path, args.meta_id_column)
    phase7_adj, fallback_phase7_ids = load_phase7_adjacency(phase7_adj_path)
    phase7_ids = load_phase7_sensor_ids(
        phase7_sensor_ids_path, fallback_phase7_ids, phase7_adj.shape[0]
    )
    full_adj, coverage_df = build_full_adjacency(canonical_ids, phase7_ids, phase7_adj)

    source_desc = json.loads(source_desc_path.read_text(encoding="utf-8"))
    default_dataset_name = f"{source_desc.get('name', dataset_dir.name)}_phys"
    output_dataset_dir = (
        Path(args.output_dataset_dir).resolve()
        if args.output_dataset_dir is not None
        else dataset_dir.parent / default_dataset_name
    )
    output_dataset_name = args.dataset_name or default_dataset_name

    output_dataset_dir.mkdir(parents=True, exist_ok=True)

    data_file = dataset_dir / "data.dat"
    if data_file.exists():
        replicate_file(data_file, output_dataset_dir / "data.dat", args.reuse_mode)

    for optional_name in ["meta.csv", "sensor_ids.npy", "sensor_catalog.csv", "sensor_ids.txt", "temporal_features.npy"]:
        source_file = dataset_dir / optional_name
        if source_file.exists():
            replicate_file(source_file, output_dataset_dir / optional_name, args.reuse_mode)

    copy_desc_with_new_name(source_desc_path, output_dataset_dir / "desc.json", output_dataset_name)

    default_output_adj = output_dataset_dir / "adj_mx.pkl"
    output_adj_path = Path(args.output_adj).resolve() if args.output_adj is not None else default_output_adj
    report_json_path = (
        Path(args.report_json).resolve()
        if args.report_json is not None
        else output_dataset_dir / f"phase7_{args.method}_coverage.json"
    )
    report_csv_path = (
        Path(args.report_csv).resolve()
        if args.report_csv is not None
        else output_dataset_dir / f"phase7_{args.method}_node_coverage.csv"
    )

    save_pickle_array(output_adj_path, full_adj)

    coverage_df.to_csv(report_csv_path, index=False)
    matched_count = int(coverage_df["matched_in_phase7"].sum())
    report_payload = {
        "source_dataset_dir": str(dataset_dir),
        "output_dataset_dir": str(output_dataset_dir),
        "output_dataset_name": output_dataset_name,
        "phase7_root": str(phase7_root),
        "phase7_adj_path": str(phase7_adj_path),
        "phase7_sensor_ids_path": str(phase7_sensor_ids_path) if phase7_sensor_ids_path else None,
        "method": args.method,
        "reuse_mode": args.reuse_mode,
        "canonical_num_nodes": len(canonical_ids),
        "phase7_num_nodes": len(phase7_ids),
        "matched_num_nodes": matched_count,
        "unmatched_num_nodes": int(len(canonical_ids) - matched_count),
        "phase7_nodes_missing_from_dataset": int(len(set(phase7_ids) - set(canonical_ids))),
        "output_adj_path": str(output_adj_path),
        "output_shape": list(full_adj.shape),
        "output_nonzero_edges": int(np.count_nonzero(full_adj)),
        "coverage_csv_path": str(report_csv_path),
    }
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    print(f"Canonical SD nodes      : {len(canonical_ids)}")
    print(f"Phase7 nodes            : {len(phase7_ids)}")
    print(f"Matched nodes           : {matched_count}")
    print(f"Output dataset dir      : {output_dataset_dir}")
    print(f"Output adjacency        : {output_adj_path}")
    print(f"Coverage report (json)  : {report_json_path}")
    print(f"Coverage report (csv)   : {report_csv_path}")


if __name__ == "__main__":
    main()
