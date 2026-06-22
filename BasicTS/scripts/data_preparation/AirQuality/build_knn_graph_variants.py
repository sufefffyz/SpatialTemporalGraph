import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np


def build_directed_knn_adjacency(distance_m: np.ndarray, k: int, weight: str, symmetric: bool) -> np.ndarray:
    distance_m = np.asarray(distance_m, dtype="float64")
    if distance_m.ndim != 2 or distance_m.shape[0] != distance_m.shape[1]:
        raise ValueError(f"distance_m must be a square matrix, got {distance_m.shape}.")
    num_nodes = distance_m.shape[0]
    k = min(int(k), num_nodes - 1)
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")

    work = distance_m.copy()
    work[~np.isfinite(work)] = np.inf
    np.fill_diagonal(work, np.inf)
    knn_idx = np.argpartition(work, kth=k - 1, axis=1)[:, :k]

    rows = np.repeat(np.arange(num_nodes), k)
    cols = knn_idx.reshape(-1)
    selected_dist = work[rows, cols]
    valid = np.isfinite(selected_dist)
    rows = rows[valid]
    cols = cols[valid]
    selected_dist = selected_dist[valid]

    adj = np.zeros((num_nodes, num_nodes), dtype="float32")
    if weight == "binary":
        values = np.ones_like(selected_dist, dtype="float32")
    elif weight == "gaussian":
        sigma = float(np.median(selected_dist[selected_dist > 0]))
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("Cannot infer a positive Gaussian sigma from selected KNN distances.")
        values = np.exp(-np.square(selected_dist / sigma)).astype("float32")
    else:
        raise ValueError(f"Unsupported weight mode: {weight}.")
    adj[rows, cols] = values

    if symmetric:
        adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0.0)
    return adj.astype("float32")


def link_or_copy(src: Path, dst: Path, copy_data: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_data:
        shutil.copy2(src, dst)
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def clear_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in output_dir.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def create_variant(dataset_root: Path, dataset_name: str, k: int, weight: str, symmetric: bool, copy_data: bool, overwrite: bool) -> Path:
    src = dataset_root / dataset_name
    if not src.exists():
        raise FileNotFoundError(f"Dataset directory not found: {src}")
    variant_name = f"{dataset_name}_KNN_K{k:03d}"
    dst = dataset_root / variant_name
    if dst.exists() and not overwrite:
        raise FileExistsError(f"{dst} already exists. Pass --overwrite to rebuild it.")
    clear_output_dir(dst)

    distance_m = np.load(src / "distance_m.npy")
    adj = build_directed_knn_adjacency(distance_m, k=k, weight=weight, symmetric=symmetric)
    with (dst / "adj_mx.pkl").open("wb") as f:
        pickle.dump([None, None, adj], f)

    for name in ("data.dat", "distance_m.npy"):
        link_or_copy(src / name, dst / name, copy_data=copy_data)

    for extra in ("city_info.csv", "station_info.csv"):
        extra_path = src / extra
        if extra_path.exists():
            link_or_copy(extra_path, dst / extra, copy_data=copy_data)

    desc = json.loads((src / "desc.json").read_text(encoding="utf-8"))
    desc["name"] = variant_name
    desc["has_graph"] = True
    desc["graph_construction"] = {
        "type": "directed_knn",
        "source": dataset_name,
        "distance": "distance_m.npy",
        "k": int(k),
        "weight": weight,
        "symmetric": bool(symmetric),
        "num_edges": int(np.count_nonzero(adj)),
        "avg_out_degree": float(np.count_nonzero(adj) / adj.shape[0]),
    }
    (dst / "desc.json").write_text(json.dumps(desc, indent=4, allow_nan=True), encoding="utf-8")

    degrees = np.count_nonzero(adj, axis=1)
    print(
        f"Saved {variant_name}: nodes={adj.shape[0]}, edges={np.count_nonzero(adj)}, "
        f"out_degree[min/mean/max]={degrees.min()}/{degrees.mean():.2f}/{degrees.max()}, "
        f"weight={weight}, symmetric={symmetric}, output={dst}"
    )
    return dst


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["KnowAir", "CCAQ"])
    parser.add_argument("--ks", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--dataset-root", default="datasets")
    parser.add_argument("--weight", choices=["binary", "gaussian"], default="binary")
    parser.add_argument("--symmetric", action="store_true")
    parser.add_argument("--copy-data", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    for dataset_name in args.datasets:
        for k in args.ks:
            create_variant(
                dataset_root=dataset_root,
                dataset_name=dataset_name,
                k=k,
                weight=args.weight,
                symmetric=args.symmetric,
                copy_data=args.copy_data,
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main()
