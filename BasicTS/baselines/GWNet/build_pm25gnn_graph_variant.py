import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np


def lonlat_to_altitude_xy(lon: float, lat: float) -> tuple[int, int]:
    lon_l = 100.0
    lat_u = 48.0
    res = 0.05
    x = np.int64(np.round((lon - lon_l - res / 2) / res))
    y = np.int64(np.round((lat_u + res / 2 - lat) / res))
    return int(x), int(y)


def bresenham(y0: int, x0: int, y1: int, x1: int):
    dy = abs(y1 - y0)
    dx = abs(x1 - x0)
    sy = 1 if y0 < y1 else -1
    sx = 1 if x0 < x1 else -1
    err = dx - dy
    y, x = y0, x0
    while True:
        yield y, x
        if y == y1 and x == x1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def read_official_city_nodes(city_path: Path, altitude: np.ndarray) -> list[dict]:
    nodes = []
    with city_path.open("r", encoding="utf-8") as f:
        for line in f:
            idx, city, lon, lat = line.rstrip("\n").split(" ")
            lon = float(lon)
            lat = float(lat)
            x, y = lonlat_to_altitude_xy(lon, lat)
            nodes.append(
                {
                    "idx": int(idx),
                    "city": city,
                    "lon": lon,
                    "lat": lat,
                    "x": x,
                    "y": y,
                    "altitude": float(altitude[y, x]),
                }
            )
    return nodes


def build_pm25gnn_adjacency(
    nodes: list[dict],
    altitude: np.ndarray,
    dist_thres: float,
    alti_thres: float,
    use_altitude: bool,
    apply_distance_filter: bool,
) -> tuple[np.ndarray, dict]:
    coords = np.array([[node["lon"], node["lat"]] for node in nodes], dtype=np.float64)
    dist = np.sqrt(np.sum(np.square(coords[:, None, :] - coords[None, :, :]), axis=-1))
    if apply_distance_filter:
        candidate = (dist <= float(dist_thres)) & (dist > 0.0)
    else:
        candidate = dist > 0.0

    adj = np.zeros(candidate.shape, dtype=np.float32)
    for src, dest in np.argwhere(candidate):
        if use_altitude:
            src_node = nodes[int(src)]
            dest_node = nodes[int(dest)]
            points = np.asarray(
                list(bresenham(src_node["y"], src_node["x"], dest_node["y"], dest_node["x"])),
                dtype=np.int64,
            )
            altitude_points = altitude[points[:, 0], points[:, 1]]
            altitude_src = altitude[src_node["y"], src_node["x"]]
            altitude_dest = altitude[dest_node["y"], dest_node["x"]]
            blocked_from_src = np.sum(altitude_points - altitude_src > alti_thres) >= 3
            blocked_from_dest = np.sum(altitude_points - altitude_dest > alti_thres) >= 3
            if blocked_from_src or blocked_from_dest:
                continue
        adj[int(src), int(dest)] = 1.0

    degrees = np.count_nonzero(adj, axis=1)
    candidate_degrees = np.count_nonzero(candidate, axis=1)
    stats = {
        "candidate_num_edges": int(np.count_nonzero(candidate)),
        "candidate_avg_out_degree": float(np.count_nonzero(candidate) / candidate.shape[0]),
        "candidate_out_degree_min": int(candidate_degrees.min()),
        "candidate_out_degree_median": float(np.median(candidate_degrees)),
        "candidate_out_degree_max": int(candidate_degrees.max()),
        "num_edges": int(np.count_nonzero(adj)),
        "avg_out_degree": float(np.count_nonzero(adj) / adj.shape[0]),
        "out_degree_min": int(degrees.min()),
        "out_degree_median": float(np.median(degrees)),
        "out_degree_max": int(degrees.max()),
        "asymmetric_edges": int(sum(1 for i, j in np.argwhere(adj > 0) if adj[int(j), int(i)] <= 0)),
    }
    return adj, stats


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


def create_variant(
    dataset_root: Path,
    source_dataset: str,
    raw_root: Path,
    output_name: str,
    dist_thres: float,
    alti_thres: float,
    use_altitude: bool,
    apply_distance_filter: bool,
    copy_data: bool,
    overwrite: bool,
) -> Path:
    src = dataset_root / source_dataset
    if not src.exists():
        raise FileNotFoundError(f"Source dataset not found: {src}")
    dst = dataset_root / output_name
    if dst.exists() and not overwrite:
        raise FileExistsError(f"{dst} already exists. Pass --overwrite to rebuild it.")
    clear_output_dir(dst)

    altitude = np.load(raw_root / "altitude.npy")
    nodes = read_official_city_nodes(raw_root / "city.txt", altitude)
    desc = json.loads((src / "desc.json").read_text(encoding="utf-8"))
    if int(desc["num_nodes"]) != len(nodes):
        raise ValueError(f"Node count mismatch: desc={desc['num_nodes']} city.txt={len(nodes)}")

    adj, stats = build_pm25gnn_adjacency(
        nodes=nodes,
        altitude=altitude,
        dist_thres=dist_thres,
        alti_thres=alti_thres,
        use_altitude=use_altitude,
        apply_distance_filter=apply_distance_filter,
    )
    with (dst / "adj_mx.pkl").open("wb") as f:
        pickle.dump([None, None, adj], f)

    for name in ("data.dat", "distance_m.npy"):
        source_path = src / name
        if source_path.exists():
            link_or_copy(source_path, dst / name, copy_data=copy_data)

    for extra in ("city_info.csv", "station_info.csv", "desc.json"):
        source_path = src / extra
        if source_path.exists() and extra != "desc.json":
            link_or_copy(source_path, dst / extra, copy_data=copy_data)

    desc["name"] = output_name
    desc["has_graph"] = True
    desc["graph_construction"] = {
        "type": "pm25gnn_official_style" if apply_distance_filter else "pm25gnn_altitude_candidate",
        "source": source_dataset,
        "raw_city_file": str(raw_root / "city.txt"),
        "raw_altitude_file": str(raw_root / "altitude.npy"),
        "distance_metric": "euclidean_lon_lat_degrees",
        "dist_thres": float(dist_thres) if apply_distance_filter else None,
        "apply_distance_filter": bool(apply_distance_filter),
        "alti_thres_m": float(alti_thres),
        "use_altitude": bool(use_altitude),
        "weight": "binary",
        "symmetric_result": stats["asymmetric_edges"] == 0,
        **stats,
    }
    (dst / "desc.json").write_text(json.dumps(desc, indent=4, allow_nan=True), encoding="utf-8")

    print(
        f"Saved {output_name}: nodes={adj.shape[0]}, edges={stats['num_edges']}, "
        f"candidate_edges={stats['candidate_num_edges']}, "
        f"out_degree[min/mean/median/max]={stats['out_degree_min']}/"
        f"{stats['avg_out_degree']:.2f}/{stats['out_degree_median']}/"
        f"{stats['out_degree_max']}, output={dst}"
    )
    return dst


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="datasets")
    parser.add_argument("--source-dataset", default="KnowAir_MAGE13")
    parser.add_argument("--output-name", default="KnowAir_MAGE13_PM25GNN_GRAPH")
    parser.add_argument("--raw-root", default="datasets/raw_data/KnowAir_official")
    parser.add_argument("--dist-thres", type=float, default=3.0)
    parser.add_argument("--alti-thres", type=float, default=1200.0)
    parser.add_argument("--altitude-candidate-only", action="store_true")
    parser.add_argument("--no-altitude", action="store_true")
    parser.add_argument("--copy-data", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    create_variant(
        dataset_root=Path(args.dataset_root),
        source_dataset=args.source_dataset,
        output_name=args.output_name,
        raw_root=Path(args.raw_root),
        dist_thres=args.dist_thres,
        alti_thres=args.alti_thres,
        use_altitude=not args.no_altitude,
        apply_distance_filter=not args.altitude_candidate_only,
        copy_data=args.copy_data,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
