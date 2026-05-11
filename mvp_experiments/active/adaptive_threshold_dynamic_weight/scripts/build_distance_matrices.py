#!/usr/bin/env python3
"""Build straight-line and OSRM road-distance matrices from sensor metadata.

The script is intentionally independent from BasicTS internals so it can be run
from the MVP experiment folder against SD/GBA/GLA/CA metadata files.

Examples:
  python build_distance_matrices.py \
    --dataset SD \
    --meta BasicTS/datasets/SD_5min_full/meta.csv \
    --output-dir mvp_experiments/active/adaptive_threshold_dynamic_weight/outputs/distances/SD \
    --compute straight

  python build_distance_matrices.py \
    --dataset SD \
    --meta BasicTS/datasets/SD_5min_full/meta.csv \
    --output-dir mvp_experiments/active/adaptive_threshold_dynamic_weight/outputs/distances/SD \
    --compute straight,osrm \
    --osrm-url http://127.0.0.1:5000 \
    --osrm-block-size 64 \
    --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LAT_CANDIDATES = ("Lat", "lat", "latitude", "Latitude")
LON_CANDIDATES = ("Lng", "Lon", "lng", "lon", "longitude", "Longitude")
ID_CANDIDATES = ("ID", "id", "sensor_id", "SensorID", "station_id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset label, e.g. SD, GBA, GLA, CA.")
    parser.add_argument("--meta", required=True, type=Path, help="CSV metadata with Lat/Lng columns.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for matrix outputs.")
    parser.add_argument(
        "--compute",
        default="straight,osrm",
        help="Comma-separated outputs to compute: straight, osrm, or straight,osrm.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional first-N node limit for smoke tests.")
    parser.add_argument("--lat-col", default=None, help="Latitude column override.")
    parser.add_argument("--lon-col", default=None, help="Longitude column override.")
    parser.add_argument("--id-col", default=None, help="Node ID column override.")
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument("--straight-block-size", type=int, default=2048)
    parser.add_argument("--osrm-url", default="http://127.0.0.1:5000")
    parser.add_argument(
        "--osrm-block-size",
        type=int,
        default=64,
        help="Source and destination block size for OSRM table queries.",
    )
    parser.add_argument("--osrm-timeout", type=float, default=120.0)
    parser.add_argument("--osrm-retries", type=int, default=2)
    parser.add_argument("--osrm-sleep", type=float, default=0.0)
    parser.add_argument("--probe-only", action="store_true", help="Only test OSRM route/table API with first nodes.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output matrices.")
    parser.add_argument("--resume", action="store_true", help="Open existing OSRM matrix and continue writing blocks.")
    return parser.parse_args()


def choose_column(df: pd.DataFrame, override: str | None, candidates: Iterable[str], kind: str) -> str:
    if override:
        if override not in df.columns:
            raise ValueError(f"{kind} column {override!r} not found. Columns: {list(df.columns)}")
        return override
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Could not infer {kind} column. Columns: {list(df.columns)}")


def load_meta(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(args.meta)
    lat_col = choose_column(df, args.lat_col, LAT_CANDIDATES, "latitude")
    lon_col = choose_column(df, args.lon_col, LON_CANDIDATES, "longitude")
    id_col = choose_column(df, args.id_col, ID_CANDIDATES, "node id")

    keep_cols = [id_col, lat_col, lon_col]
    meta = df.loc[:, keep_cols].copy()
    meta = meta.dropna(subset=[lat_col, lon_col]).reset_index(drop=True)
    if args.limit is not None:
        meta = meta.iloc[: args.limit].copy()
    if meta.empty:
        raise ValueError("No valid metadata rows after dropping missing coordinates.")

    lats = meta[lat_col].astype(float).to_numpy()
    lons = meta[lon_col].astype(float).to_numpy()
    node_ids = meta[id_col].astype(str).tolist()
    return meta, lats, lons, node_ids


def write_node_ids(path: Path, node_ids: list[str], lats: np.ndarray, lons: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "node_id", "lat", "lon"])
        for i, node_id in enumerate(node_ids):
            writer.writerow([i, node_id, float(lats[i]), float(lons[i])])


def haversine_block_m(
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    dst_lat: np.ndarray,
    dst_lon: np.ndarray,
) -> np.ndarray:
    radius_m = 6_371_008.8
    lat1 = np.radians(src_lat)[:, None]
    lon1 = np.radians(src_lon)[:, None]
    lat2 = np.radians(dst_lat)[None, :]
    lon2 = np.radians(dst_lon)[None, :]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return (2.0 * radius_m * np.arcsin(np.sqrt(np.minimum(1.0, a)))).astype(np.float64)


def create_matrix(path: Path, shape: tuple[int, int], dtype: str, fill: float, overwrite: bool) -> np.memmap:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists. Use --overwrite or --resume.")
        path.unlink()
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    arr[:] = fill
    arr.flush()
    return arr


def open_or_create_osrm_matrix(
    path: Path,
    shape: tuple[int, int],
    dtype: str,
    overwrite: bool,
    resume: bool,
) -> np.memmap:
    if path.exists() and resume and not overwrite:
        arr = np.load(path, mmap_mode="r+")
        if arr.shape != shape:
            raise ValueError(f"Existing {path} has shape {arr.shape}, expected {shape}.")
        return arr
    return create_matrix(path, shape, dtype, np.inf, overwrite=overwrite)


def build_straight_matrix(
    out_path: Path,
    lats: np.ndarray,
    lons: np.ndarray,
    dtype: str,
    block_size: int,
    overwrite: bool,
) -> dict:
    n = len(lats)
    out = create_matrix(out_path, (n, n), dtype, 0.0, overwrite=overwrite)
    started = time.time()
    for i0 in range(0, n, block_size):
        i1 = min(n, i0 + block_size)
        block = haversine_block_m(lats[i0:i1], lons[i0:i1], lats, lons).astype(dtype, copy=False)
        out[i0:i1, :] = block
        out.flush()
        print(f"[straight] rows {i0}:{i1} / {n}", flush=True)
    np.fill_diagonal(out, 0.0)
    out.flush()
    return summarize_matrix(out_path, "straight_distance_m", started)


def osrm_request_json(url: str, timeout: float, retries: int) -> dict:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "stg-adaptive-threshold/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.0 + attempt)
    raise RuntimeError(f"OSRM request failed after {retries + 1} attempts: {last_error}")


def osrm_table_block(
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    dst_lat: np.ndarray,
    dst_lon: np.ndarray,
    base_url: str,
    timeout: float,
    retries: int,
) -> np.ndarray:
    coords = [f"{lon:.8f},{lat:.8f}" for lat, lon in zip(src_lat, src_lon)]
    coords.extend(f"{lon:.8f},{lat:.8f}" for lat, lon in zip(dst_lat, dst_lon))
    source_idx = ";".join(str(i) for i in range(len(src_lat)))
    dest_offset = len(src_lat)
    dest_idx = ";".join(str(dest_offset + i) for i in range(len(dst_lat)))
    query = urllib.parse.urlencode(
        {
            "sources": source_idx,
            "destinations": dest_idx,
            "annotations": "distance",
        }
    )
    url = f"{base_url.rstrip('/')}/table/v1/driving/{';'.join(coords)}?{query}"
    payload = osrm_request_json(url, timeout=timeout, retries=retries)
    if payload.get("code") != "Ok":
        raise RuntimeError(f"OSRM table returned {payload.get('code')}: {payload.get('message')}")
    distances = payload.get("distances")
    if distances is None:
        raise RuntimeError("OSRM table response did not contain distances; check OSRM version/profile.")
    block = np.array([[np.inf if v is None else float(v) for v in row] for row in distances], dtype=np.float64)
    # OSRM can return tiny negative distances for nearly co-located snapped points.
    block[np.isfinite(block) & (block < 0.0)] = 0.0
    return block


def probe_osrm(base_url: str, lats: np.ndarray, lons: np.ndarray, timeout: float, retries: int) -> None:
    if len(lats) < 2:
        raise ValueError("Need at least two nodes for OSRM probe.")
    route_coords = f"{lons[0]:.8f},{lats[0]:.8f};{lons[1]:.8f},{lats[1]:.8f}"
    route_url = f"{base_url.rstrip('/')}/route/v1/driving/{route_coords}?overview=false"
    route_payload = osrm_request_json(route_url, timeout=timeout, retries=retries)
    if route_payload.get("code") != "Ok":
        raise RuntimeError(f"OSRM route probe failed: {route_payload}")
    distance = route_payload["routes"][0]["distance"]
    duration = route_payload["routes"][0]["duration"]
    table = osrm_table_block(lats[:2], lons[:2], lats[:2], lons[:2], base_url, timeout, retries)
    print(json.dumps({"route_distance_m": distance, "route_duration_s": duration}, indent=2))
    print("table_distance_m:")
    print(table)


def build_osrm_matrix(
    out_path: Path,
    lats: np.ndarray,
    lons: np.ndarray,
    dtype: str,
    block_size: int,
    base_url: str,
    timeout: float,
    retries: int,
    sleep_s: float,
    overwrite: bool,
    resume: bool,
    progress_path: Path,
) -> dict:
    n = len(lats)
    out = open_or_create_osrm_matrix(out_path, (n, n), dtype, overwrite=overwrite, resume=resume)
    started = time.time()
    with progress_path.open("a") as progress:
        for i0 in range(0, n, block_size):
            i1 = min(n, i0 + block_size)
            for j0 in range(0, n, block_size):
                j1 = min(n, j0 + block_size)
                block = osrm_table_block(
                    lats[i0:i1],
                    lons[i0:i1],
                    lats[j0:j1],
                    lons[j0:j1],
                    base_url,
                    timeout,
                    retries,
                ).astype(dtype, copy=False)
                out[i0:i1, j0:j1] = block
                out.flush()
                record = {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "rows": [i0, i1],
                    "cols": [j0, j1],
                    "finite": int(np.isfinite(block).sum()),
                }
                progress.write(json.dumps(record) + "\n")
                progress.flush()
                print(f"[osrm] rows {i0}:{i1} cols {j0}:{j1} / {n}", flush=True)
                if sleep_s > 0:
                    time.sleep(sleep_s)
    np.fill_diagonal(out, 0.0)
    out.flush()
    return summarize_matrix(out_path, "osrm_shortest_distance_m", started)


def summarize_matrix(path: Path, name: str, started: float) -> dict:
    arr = np.load(path, mmap_mode="r")
    finite = np.isfinite(arr)
    diag = np.diag(arr)
    finite_values = arr[finite]
    positive = finite & (arr > 0)
    negative = finite & (arr < 0)
    positive_values = arr[positive]
    summary = {
        "name": name,
        "path": str(path),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "finite_count": int(finite.sum()),
        "missing_or_inf_count": int(arr.size - finite.sum()),
        "negative_count": int(negative.sum()),
        "positive_count": int(positive.sum()),
        "diag_min": float(np.nanmin(diag)),
        "diag_max": float(np.nanmax(diag)),
        "elapsed_s": round(time.time() - started, 3),
    }
    if finite_values.size:
        summary.update(
            {
                "finite_min": float(np.nanmin(finite_values)),
                "finite_max": float(np.nanmax(finite_values)),
            }
        )
    if positive_values.size:
        qs = np.nanquantile(positive_values, [0.25, 0.5, 0.75, 0.95])
        summary.update(
            {
                "positive_min": float(np.nanmin(positive_values)),
                "positive_q25": float(qs[0]),
                "positive_median": float(qs[1]),
                "positive_q75": float(qs[2]),
                "positive_q95": float(qs[3]),
                "positive_max": float(np.nanmax(positive_values)),
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    compute = {item.strip().lower() for item in args.compute.split(",") if item.strip()}
    unknown = compute - {"straight", "osrm"}
    if unknown:
        raise ValueError(f"Unknown --compute values: {sorted(unknown)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta, lats, lons, node_ids = load_meta(args)
    dataset = args.dataset
    write_node_ids(args.output_dir / f"{dataset}_node_ids.csv", node_ids, lats, lons)

    if args.probe_only:
        probe_osrm(args.osrm_url, lats, lons, args.osrm_timeout, args.osrm_retries)
        return

    summaries = {
        "dataset": dataset,
        "meta": str(args.meta),
        "node_count": len(meta),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "osrm_url": args.osrm_url,
        "outputs": [],
    }

    if "straight" in compute:
        straight_summary = build_straight_matrix(
            args.output_dir / f"{dataset}_straight_distance_m.npy",
            lats,
            lons,
            args.dtype,
            args.straight_block_size,
            overwrite=args.overwrite,
        )
        summaries["outputs"].append(straight_summary)

    if "osrm" in compute:
        osrm_summary = build_osrm_matrix(
            args.output_dir / f"{dataset}_osrm_shortest_distance_m.npy",
            lats,
            lons,
            args.dtype,
            args.osrm_block_size,
            args.osrm_url,
            args.osrm_timeout,
            args.osrm_retries,
            args.osrm_sleep,
            overwrite=args.overwrite,
            resume=args.resume,
            progress_path=args.output_dir / f"{dataset}_osrm_progress.jsonl",
        )
        summaries["outputs"].append(osrm_summary)

    summary_path = args.output_dir / f"{dataset}_distance_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
