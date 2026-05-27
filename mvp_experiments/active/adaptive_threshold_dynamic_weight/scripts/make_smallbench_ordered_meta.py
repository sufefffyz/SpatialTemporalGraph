#!/usr/bin/env python3
"""Create ordered BasicTS metadata for small traffic benchmarks.

The output is a CSV with columns ID,Lat,Lng in the exact node order used by the
BasicTS dataset. This file is then fed to OSRM distance generation.
"""

from __future__ import annotations

import argparse
import ast
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("METR-LA", "PEMS04"))
    parser.add_argument("--basicts-adj", required=True, type=Path, help="BasicTS adj_mx.pkl for node order.")
    parser.add_argument("--source", required=True, type=Path, help="Source coordinate file.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-index-order", action="store_true", help="Allow PEMS04 .geo rows 0..N-1 as node order.")
    return parser.parse_args()


def load_basicts_ids(path: Path) -> list[str] | None:
    with path.open("rb") as fp:
        obj: Any = pickle.load(fp)
    if isinstance(obj, (tuple, list)) and len(obj) >= 2 and isinstance(obj[0], (list, tuple)):
        return [str(item) for item in obj[0]]
    return None


def find_col(df: pd.DataFrame, names: tuple[str, ...], label: str) -> str:
    lowered = {col.lower(): col for col in df.columns}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    raise ValueError(f"Could not find {label} column in {list(df.columns)}")


def parse_geo_coordinates(value: object) -> tuple[float, float]:
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
    else:
        parsed = value
    if not isinstance(parsed, (list, tuple)) or len(parsed) < 2:
        raise ValueError(f"Bad coordinates value: {value!r}")
    lon = float(parsed[0])
    lat = float(parsed[1])
    return lat, lon


def build_metr_la(source: Path, node_ids: list[str] | None) -> pd.DataFrame:
    if node_ids is None:
        raise ValueError("METR-LA BasicTS adj_mx.pkl must contain sensor IDs.")
    df = pd.read_csv(source)
    id_col = find_col(df, ("sensor_id", "id", "ID", "node_id"), "sensor id")
    lat_col = find_col(df, ("latitude", "lat", "Lat"), "latitude")
    lon_col = find_col(df, ("longitude", "lng", "lon", "Lng", "Lon"), "longitude")
    df = df.assign(_id=df[id_col].astype(str)).set_index("_id", drop=False)
    missing = [sensor_id for sensor_id in node_ids if sensor_id not in df.index]
    if missing:
        raise ValueError(f"METR-LA source is missing {len(missing)} BasicTS sensors, e.g. {missing[:5]}")
    ordered = df.loc[node_ids].copy()
    return pd.DataFrame(
        {
            "ID": node_ids,
            "Lat": ordered[lat_col].astype(float).to_numpy(),
            "Lng": ordered[lon_col].astype(float).to_numpy(),
        }
    )


def build_pems04(source: Path, node_ids: list[str] | None, allow_index_order: bool) -> pd.DataFrame:
    df = pd.read_csv(source)
    id_col = find_col(df, ("geo_id", "sensor_id", "id", "ID", "node_id"), "geo id")
    if "coordinates" in df.columns:
        coords = df["coordinates"].apply(parse_geo_coordinates)
        df = df.assign(Lat=[lat for lat, _ in coords], Lng=[lon for _, lon in coords])
    else:
        lat_col = find_col(df, ("latitude", "lat", "Lat"), "latitude")
        lon_col = find_col(df, ("longitude", "lng", "lon", "Lng", "Lon"), "longitude")
        df = df.assign(Lat=df[lat_col].astype(float), Lng=df[lon_col].astype(float))

    if node_ids is not None:
        df = df.assign(_id=df[id_col].astype(str)).set_index("_id", drop=False)
        missing = [sensor_id for sensor_id in node_ids if sensor_id not in df.index]
        if missing:
            raise ValueError(f"PEMS04 source is missing {len(missing)} BasicTS IDs, e.g. {missing[:5]}")
        ordered = df.loc[node_ids].copy()
        ids = node_ids
    else:
        geo_ids = pd.to_numeric(df[id_col], errors="coerce")
        if not allow_index_order or geo_ids.isna().any():
            raise ValueError("PEMS04 BasicTS adj has no IDs. Pass --allow-index-order only after verifying geo_id 0..N-1 alignment.")
        order = geo_ids.astype(int).to_numpy()
        expected = list(range(len(df)))
        if sorted(order.tolist()) != expected:
            raise ValueError("PEMS04 geo_id is not exactly 0..N-1; cannot infer BasicTS node order.")
        ordered = df.assign(_order=order).sort_values("_order").copy()
        ids = [str(i) for i in expected]

    return pd.DataFrame({"ID": ids, "Lat": ordered["Lat"].astype(float).to_numpy(), "Lng": ordered["Lng"].astype(float).to_numpy()})


def main() -> None:
    args = parse_args()
    node_ids = load_basicts_ids(args.basicts_adj)
    if args.dataset == "METR-LA":
        out = build_metr_la(args.source, node_ids)
    else:
        out = build_pems04(args.source, node_ids, args.allow_index_order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"wrote {args.output} rows={len(out)}")


if __name__ == "__main__":
    main()
