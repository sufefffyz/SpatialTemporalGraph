#!/usr/bin/env python3

import argparse
import json
import pickle
from pathlib import Path

import folium
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a BasicTS graph dataset on an interactive map."
    )
    parser.add_argument(
        "--dataset-dir",
        default="datasets/SD",
        help="BasicTS dataset directory containing meta.csv and adj_mx.pkl.",
    )
    parser.add_argument(
        "--output-html",
        default=None,
        help="Output HTML path. Defaults to <dataset-dir>/graph_map.html.",
    )
    parser.add_argument(
        "--id-column",
        default="ID",
        help="Sensor id column in meta.csv. Default: ID",
    )
    parser.add_argument(
        "--lat-column",
        default="Lat",
        help="Latitude column in meta.csv. Default: Lat",
    )
    parser.add_argument(
        "--lon-column",
        default="Lng",
        help="Longitude column in meta.csv. Default: Lng",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=20000,
        help="Maximum number of edges to render. Default: 20000",
    )
    parser.add_argument(
        "--edge-weight-threshold",
        type=float,
        default=0.0,
        help="Render only edges with weight strictly greater than this threshold.",
    )
    parser.add_argument(
        "--node-radius",
        type=float,
        default=3.0,
        help="Folium circle marker radius for nodes. Default: 3",
    )
    parser.add_argument(
        "--zoom-start",
        type=int,
        default=10,
        help="Initial map zoom. Default: 10",
    )
    return parser.parse_args()


def normalize_sensor_id(value) -> str:
    text = str(value).strip()
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def load_adjacency(path: Path) -> tuple[np.ndarray, list[str] | None]:
    with path.open("rb") as fp:
        payload = pickle.load(fp)

    if isinstance(payload, tuple) and len(payload) == 3:
        sensor_ids = [normalize_sensor_id(value) for value in payload[0]]
        adjacency = np.asarray(payload[2], dtype=np.float32)
        return adjacency, sensor_ids

    adjacency = np.asarray(payload, dtype=np.float32)
    return adjacency, None


def build_meta_frame(meta_path: Path, id_column: str, lat_column: str, lon_column: str) -> pd.DataFrame:
    meta_df = pd.read_csv(meta_path)
    missing = [column for column in [id_column, lat_column, lon_column] if column not in meta_df.columns]
    if missing:
        raise KeyError(f"meta.csv missing required columns: {missing}")

    frame = meta_df.copy()
    frame[id_column] = frame[id_column].map(normalize_sensor_id)
    frame[lat_column] = pd.to_numeric(frame[lat_column], errors="coerce")
    frame[lon_column] = pd.to_numeric(frame[lon_column], errors="coerce")
    frame = frame.dropna(subset=[lat_column, lon_column]).copy()
    frame = frame.drop_duplicates(subset=[id_column], keep="first")
    return frame


def resolve_node_order(meta_df: pd.DataFrame, adjacency: np.ndarray, sensor_ids: list[str] | None, id_column: str) -> tuple[list[str], pd.DataFrame]:
    if sensor_ids is None:
        ordered_ids = meta_df[id_column].tolist()
    else:
        ordered_ids = sensor_ids

    if len(ordered_ids) != adjacency.shape[0]:
        raise ValueError(
            f"Node count mismatch: resolved {len(ordered_ids)} ids but adjacency has shape {adjacency.shape}."
        )

    lookup_df = meta_df.set_index(id_column, drop=False)
    aligned_df = lookup_df.reindex(ordered_ids).reset_index(drop=True)
    missing_rows = aligned_df[aligned_df[id_column].isna()]
    if not missing_rows.empty:
        missing_count = int(missing_rows.shape[0])
        raise ValueError(f"meta.csv is missing {missing_count} sensor ids needed by adjacency.")
    return ordered_ids, aligned_df


def color_for_sensor_type(sensor_type: str) -> str:
    palette = {
        "MAINLINE": "#1f77b4",
        "ML": "#1f77b4",
        "ON-RAMP": "#2ca02c",
        "OR": "#2ca02c",
        "OFF-RAMP": "#d62728",
        "FR": "#d62728",
        "FF": "#9467bd",
    }
    key = str(sensor_type).strip().upper()
    return palette.get(key, "#555555")


def render_map(
    aligned_df: pd.DataFrame,
    adjacency: np.ndarray,
    ordered_ids: list[str],
    args: argparse.Namespace,
    output_html: Path,
) -> dict:
    center_lat = float(aligned_df[args.lat_column].mean())
    center_lon = float(aligned_df[args.lon_column].mean())
    graph_map = folium.Map(location=[center_lat, center_lon], zoom_start=args.zoom_start)

    edge_indices = np.argwhere(adjacency > args.edge_weight_threshold)
    original_edge_count = int(edge_indices.shape[0])
    truncated = False
    if args.max_edges > 0 and original_edge_count > args.max_edges:
        edge_indices = edge_indices[: args.max_edges]
        truncated = True

    edge_layer = folium.FeatureGroup(name=f"edges ({len(edge_indices)})", show=True)
    for source_idx, target_idx in edge_indices.tolist():
        source_row = aligned_df.iloc[int(source_idx)]
        target_row = aligned_df.iloc[int(target_idx)]
        weight = float(adjacency[int(source_idx), int(target_idx)])
        coords = [
            [float(source_row[args.lat_column]), float(source_row[args.lon_column])],
            [float(target_row[args.lat_column]), float(target_row[args.lon_column])],
        ]
        popup = (
            f"<b>{ordered_ids[int(source_idx)]} → {ordered_ids[int(target_idx)]}</b><br>"
            f"weight: {weight:.4f}"
        )
        folium.PolyLine(
            locations=coords,
            color="#cc5500",
            weight=1.4,
            opacity=0.35,
            popup=folium.Popup(popup, max_width=240),
        ).add_to(edge_layer)
    edge_layer.add_to(graph_map)

    node_layer = folium.FeatureGroup(name=f"nodes ({len(aligned_df)})", show=True)
    type_column = None
    for candidate in ["Type", "type", "sensor_type"]:
        if candidate in aligned_df.columns:
            type_column = candidate
            break

    for row_idx, row in aligned_df.iterrows():
        sensor_type = row[type_column] if type_column is not None else "UNKNOWN"
        popup_parts = [
            f"<b>{ordered_ids[row_idx]}</b>",
            f"lat={float(row[args.lat_column]):.6f}",
            f"lon={float(row[args.lon_column]):.6f}",
        ]
        for candidate in [type_column, "Fwy", "Direction", "District", "County", "ID2"]:
            if candidate and candidate in row.index and pd.notna(row[candidate]):
                popup_parts.append(f"{candidate}: {row[candidate]}")
        folium.CircleMarker(
            location=[float(row[args.lat_column]), float(row[args.lon_column])],
            radius=args.node_radius,
            color="#111111",
            weight=0.6,
            fill=True,
            fill_color=color_for_sensor_type(sensor_type),
            fill_opacity=0.95,
            popup=folium.Popup("<br>".join(popup_parts), max_width=260),
        ).add_to(node_layer)
    node_layer.add_to(graph_map)

    folium.LayerControl(collapsed=False).add_to(graph_map)

    title_html = (
        "<div style='position: fixed; top: 12px; left: 56px; z-index: 9999; "
        "background: white; padding: 8px 10px; border: 1px solid #ccc; font-size: 14px;'>"
        f"<b>{output_html.parent.name}</b><br>"
        f"nodes={len(aligned_df)} edges={original_edge_count}"
        + (" (truncated)" if truncated else "")
        + "</div>"
    )
    graph_map.get_root().html.add_child(folium.Element(title_html))

    output_html.parent.mkdir(parents=True, exist_ok=True)
    graph_map.save(str(output_html))

    return {
        "num_nodes": int(len(aligned_df)),
        "num_edges": int(original_edge_count),
        "rendered_edges": int(len(edge_indices)),
        "truncated": bool(truncated),
        "output_html": str(output_html),
    }


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    meta_path = dataset_dir / "meta.csv"
    adj_path = dataset_dir / "adj_mx.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.csv not found: {meta_path}")
    if not adj_path.exists():
        raise FileNotFoundError(f"adj_mx.pkl not found: {adj_path}")

    output_html = (
        Path(args.output_html).resolve()
        if args.output_html is not None
        else dataset_dir / "graph_map.html"
    )

    adjacency, sensor_ids = load_adjacency(adj_path)
    meta_df = build_meta_frame(meta_path, args.id_column, args.lat_column, args.lon_column)
    ordered_ids, aligned_df = resolve_node_order(meta_df, adjacency, sensor_ids, args.id_column)
    summary = render_map(aligned_df, adjacency, ordered_ids, args, output_html)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
