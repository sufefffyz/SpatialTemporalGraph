#!/usr/bin/env python3
"""Render a standalone planar traffic-flow heatmap animation.

The script reads BasicTS-style datasets:

    datasets/<DATASET>/meta.csv
    datasets/<DATASET>/data.dat
    datasets/<DATASET>/desc.json

It writes an HTML file with a canvas heatmap, OSM-compatible basemap tiles,
time slider, and playback controls. It does not load external JavaScript, and
the basemap can be disabled with --tile-url "" for a pure planar view.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_ID_COLUMNS = ["ID", "id", "sensor_id", "SensorID", "sensor", "node_id", "ID2"]
DEFAULT_LAT_COLUMNS = ["Lat", "lat", "latitude", "Latitude", "LAT", "y", "Y"]
DEFAULT_LON_COLUMNS = [
    "Lng",
    "lng",
    "Lon",
    "lon",
    "longitude",
    "Longitude",
    "LON",
    "long",
    "Long",
    "x",
    "X",
]
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a standalone planar traffic-flow heatmap animation HTML."
    )
    parser.add_argument("--dataset", default="SD", help="Dataset name under datasets/<DATASET>.")
    parser.add_argument("--dataset-dir", default=None, help="Explicit dataset directory.")
    parser.add_argument("--data-file", default=None, help="Explicit data.dat path.")
    parser.add_argument("--data-shape", default=None, help="Override data shape as T,N or T,N,C.")
    parser.add_argument("--num-features", type=int, default=None, help="Feature count for shape inference.")
    parser.add_argument("--channel", type=int, default=0, help="Feature channel to visualize.")
    parser.add_argument("--start-index", type=int, default=None, help="First global time index.")
    parser.add_argument("--start-datetime", default=None, help="Start time such as '2019-07-01 00:00'.")
    parser.add_argument(
        "--dataset-start-datetime",
        default=None,
        help="Datetime corresponding to data index 0; enables absolute --start-datetime lookup.",
    )
    parser.add_argument("--num-steps", type=int, default=96, help="Rendered time steps after time stride.")
    parser.add_argument("--time-stride", type=int, default=1, help="Render every Nth source time step.")
    parser.add_argument("--step-minutes", type=int, default=None, help="Minutes per source step.")
    parser.add_argument("--output-html", default=None, help="Output HTML path.")
    parser.add_argument("--id-column", default=None, help="Sensor id column in meta.csv.")
    parser.add_argument("--lat-column", default=None, help="Latitude column in meta.csv.")
    parser.add_argument("--lon-column", default=None, help="Longitude column in meta.csv.")
    parser.add_argument("--max-nodes", type=int, default=0, help="Maximum sensors to render; 0 means all.")
    parser.add_argument("--sample-stride", type=int, default=1, help="Render every Nth sensor.")
    parser.add_argument("--canvas-width", type=int, default=1180, help="Canvas CSS width in pixels.")
    parser.add_argument("--canvas-height", type=int, default=780, help="Canvas CSS height in pixels.")
    parser.add_argument("--heat-radius", type=float, default=34.0, help="Heat kernel radius in pixels.")
    parser.add_argument("--point-radius", type=float, default=2.8, help="Sensor point radius in pixels.")
    parser.add_argument("--low-percentile", type=float, default=5.0, help="Low clipping percentile.")
    parser.add_argument("--high-percentile", type=float, default=97.0, help="High clipping percentile.")
    parser.add_argument("--gamma", type=float, default=0.55, help="Contrast gamma; lower means stronger mids.")
    parser.add_argument("--value-precision", type=int, default=2, help="Value decimals stored in HTML.")
    parser.add_argument("--play-interval-ms", type=int, default=140, help="Playback interval.")
    parser.add_argument("--hide-points", action="store_true", help="Hide sensor point overlay.")
    parser.add_argument(
        "--tile-url",
        default="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        help="OSM-compatible tile URL template. Use an empty string to disable the basemap.",
    )
    parser.add_argument("--tile-zoom", type=int, default=11, help="Basemap tile zoom level.")
    parser.add_argument("--basemap-opacity", type=float, default=0.82, help="OSM basemap opacity.")
    parser.add_argument("--basemap-dim", type=float, default=0.20, help="Dark overlay over basemap for heat contrast.")
    return parser.parse_args()


def script_basicts_root() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_sensor_id(value: Any) -> str:
    text = str(value).strip()
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def resolve_dataset_dir(dataset: str, dataset_dir: str | None) -> Path:
    if dataset_dir is not None:
        return Path(dataset_dir).expanduser().resolve()
    dataset_path = Path(dataset).expanduser()
    if dataset_path.exists():
        return dataset_path.resolve()
    root = script_basicts_root()
    cwd = Path.cwd().resolve()
    candidates = [
        cwd / "datasets" / dataset,
        root / "datasets" / dataset,
        cwd / "SpatialTemporalGraph" / "BasicTS" / "datasets" / dataset,
    ]
    for path in candidates:
        if (path / "meta.csv").exists() and (path / "data.dat").exists():
            return path.resolve()
    return (root / "datasets" / dataset).resolve()


def read_description(dataset_dir: Path) -> dict[str, Any]:
    desc_path = dataset_dir / "desc.json"
    if desc_path.exists():
        with desc_path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    return {}


def parse_shape(shape_text: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(part.strip()) for part in shape_text.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid --data-shape {shape_text!r}; expected T,N or T,N,C.") from exc
    if len(shape) not in (2, 3) or any(dim <= 0 for dim in shape):
        raise ValueError(f"Invalid --data-shape {shape_text!r}; expected positive T,N or T,N,C.")
    return shape


def infer_shape_from_file(data_path: Path, num_nodes: int, num_features: int | None) -> tuple[int, ...]:
    num_float32 = data_path.stat().st_size // np.dtype(np.float32).itemsize
    if num_features is not None:
        denominator = num_nodes * num_features
        if denominator <= 0 or num_float32 % denominator != 0:
            raise ValueError("Cannot infer data shape from --num-features.")
        return (num_float32 // denominator, num_nodes, num_features)

    for feature_count in (3, 1):
        denominator = num_nodes * feature_count
        if denominator > 0 and num_float32 % denominator == 0:
            time_steps = num_float32 // denominator
            return (time_steps, num_nodes, feature_count) if feature_count > 1 else (time_steps, num_nodes)
    raise ValueError("Cannot infer shape. Provide desc.json, --data-shape, or --num-features.")


def resolve_data_shape(
    desc: dict[str, Any], data_path: Path, meta_rows: int, args: argparse.Namespace
) -> tuple[int, ...]:
    if args.data_shape is not None:
        return parse_shape(args.data_shape)
    if "shape" in desc:
        shape = tuple(int(dim) for dim in desc["shape"])
        if len(shape) not in (2, 3):
            raise ValueError(f"Unsupported desc.json shape: {shape}")
        return shape
    return infer_shape_from_file(data_path, meta_rows, args.num_features)


def detect_column(columns: list[str], requested: str | None, candidates: list[str], role: str) -> str:
    if requested is not None:
        if requested not in columns:
            raise KeyError(f"Requested {role} column {requested!r} not found in meta.csv.")
        return requested
    lower_lookup = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        if candidate.lower() in lower_lookup:
            return lower_lookup[candidate.lower()]
    raise KeyError(f"Could not auto-detect {role} column in meta.csv columns: {columns}")


def build_meta_frame(
    meta_path: Path,
    id_column: str | None,
    lat_column: str | None,
    lon_column: str | None,
    num_nodes: int,
    sample_stride: int,
    max_nodes: int,
) -> tuple[pd.DataFrame, str, str, str, int]:
    meta_df = pd.read_csv(meta_path)
    if meta_df.empty:
        raise ValueError(f"meta.csv is empty: {meta_path}")
    columns = meta_df.columns.tolist()
    resolved_id = detect_column(columns, id_column, DEFAULT_ID_COLUMNS, "sensor id")
    resolved_lat = detect_column(columns, lat_column, DEFAULT_LAT_COLUMNS, "latitude")
    resolved_lon = detect_column(columns, lon_column, DEFAULT_LON_COLUMNS, "longitude")

    frame = meta_df.reset_index().rename(columns={"index": "node_index"}).copy()
    frame[resolved_id] = frame[resolved_id].map(normalize_sensor_id)
    frame[resolved_lat] = pd.to_numeric(frame[resolved_lat], errors="coerce")
    frame[resolved_lon] = pd.to_numeric(frame[resolved_lon], errors="coerce")
    frame = frame.dropna(subset=[resolved_lat, resolved_lon]).copy()
    frame = frame[frame["node_index"] < num_nodes].copy()
    frame = frame.drop_duplicates(subset=[resolved_id], keep="first")
    valid_node_count = int(len(frame))

    if sample_stride <= 0:
        raise ValueError("--sample-stride must be positive.")
    if sample_stride > 1:
        frame = frame.iloc[::sample_stride].copy()
    if max_nodes > 0 and len(frame) > max_nodes:
        selected_positions = np.linspace(0, len(frame) - 1, max_nodes).round().astype(int)
        frame = frame.iloc[selected_positions].copy()
    if frame.empty:
        raise ValueError("No valid sensors remain after coordinate validation.")
    return frame.reset_index(drop=True), resolved_id, resolved_lat, resolved_lon, valid_node_count


def resolve_step_minutes(desc: dict[str, Any], requested: int | None) -> int:
    if requested is not None:
        if requested <= 0:
            raise ValueError("--step-minutes must be positive.")
        return requested
    for key in ("frequency (minutes)", "frequency_minutes", "freq_minutes"):
        if key in desc:
            value = int(desc[key])
            if value > 0:
                return value
    return 5


def parse_datetime(text: str, arg_name: str) -> datetime:
    try:
        timestamp = pd.to_datetime(text)
    except Exception as exc:
        raise ValueError(f"Could not parse {arg_name}={text!r} as a datetime.") from exc
    if pd.isna(timestamp):
        raise ValueError(f"Could not parse {arg_name}={text!r} as a datetime.")
    return timestamp.to_pydatetime().replace(tzinfo=None)


def feature_descriptions(desc: dict[str, Any]) -> list[str]:
    values = desc.get("feature_description", [])
    if not isinstance(values, list):
        return []
    return [str(value).lower() for value in values]


def find_feature_channel(desc: dict[str, Any], phrases: tuple[str, ...]) -> int | None:
    for index, description in enumerate(feature_descriptions(desc)):
        if any(phrase in description for phrase in phrases):
            return index
    return None


def start_index_from_datetime_features(
    data: np.memmap,
    desc: dict[str, Any],
    start_datetime: datetime,
    step_minutes: int,
) -> int | None:
    if data.ndim != 3:
        return None
    tod_channel = find_feature_channel(desc, ("time of day", "timeofday", "tod"))
    dow_channel = find_feature_channel(desc, ("day of week", "dayofweek", "dow"))
    if tod_channel is None or dow_channel is None:
        return None
    if tod_channel >= data.shape[2] or dow_channel >= data.shape[2]:
        return None
    target_minutes = start_datetime.hour * 60 + start_datetime.minute
    target_tod = (target_minutes % 1440.0) / 1440.0
    target_dow = start_datetime.weekday() / 7.0
    tod_values = np.asarray(data[:, 0, tod_channel], dtype=np.float64)
    dow_values = np.asarray(data[:, 0, dow_channel], dtype=np.float64)
    tod_diff = np.abs(((tod_values - target_tod + 0.5) % 1.0) - 0.5)
    dow_diff = np.abs(dow_values - target_dow)
    tod_tolerance = max(step_minutes / 1440.0 / 2.0, 1e-4)
    matches = np.where((tod_diff <= tod_tolerance) & (dow_diff <= 1e-4))[0]
    if matches.size == 0:
        return None
    return int(matches[0])


def resolve_start_index(
    args: argparse.Namespace,
    data: np.memmap,
    desc: dict[str, Any],
    step_minutes: int,
) -> tuple[int, str]:
    if args.start_index is not None and args.start_datetime is not None:
        raise ValueError("Use either --start-index or --start-datetime, not both.")
    if args.start_index is not None:
        return int(args.start_index), "explicit-index"
    if args.start_datetime is None:
        return 0, "default-zero"
    start_datetime = parse_datetime(args.start_datetime, "--start-datetime")
    if args.dataset_start_datetime is not None:
        dataset_start = parse_datetime(args.dataset_start_datetime, "--dataset-start-datetime")
        delta_minutes = (start_datetime - dataset_start).total_seconds() / 60.0
        return int(round(delta_minutes / step_minutes)), "dataset-start-datetime"
    feature_index = start_index_from_datetime_features(data, desc, start_datetime, step_minutes)
    if feature_index is not None:
        return feature_index, "time-features-cycle"
    steps_per_day = max(1, int(round(1440 / step_minutes)))
    synthetic_index = start_datetime.weekday() * steps_per_day
    synthetic_index += int(round((start_datetime.hour * 60 + start_datetime.minute) / step_minutes))
    return synthetic_index, "synthetic-week"


def validate_window(start_index: int, num_steps: int, time_stride: int, total_steps: int) -> None:
    if start_index < 0:
        raise ValueError(f"Start index must be non-negative, got {start_index}.")
    if num_steps <= 0:
        raise ValueError(f"--num-steps must be positive, got {num_steps}.")
    if time_stride <= 0:
        raise ValueError(f"--time-stride must be positive, got {time_stride}.")
    end_index = start_index + (num_steps - 1) * time_stride + 1
    if start_index >= total_steps or end_index > total_steps:
        raise ValueError(
            f"Requested window [{start_index}, {end_index}) exceeds data length {total_steps}."
        )


def load_data_memmap(data_path: Path, shape: tuple[int, ...]) -> np.memmap:
    try:
        return np.memmap(data_path, dtype=np.float32, mode="r", shape=shape)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(f"Error loading data file {data_path} with shape {shape}.") from exc


def extract_window(
    data: np.memmap,
    start_index: int,
    num_steps: int,
    time_stride: int,
    node_indices: np.ndarray,
    channel: int,
) -> np.ndarray:
    indices = start_index + np.arange(num_steps, dtype=int) * time_stride
    if data.ndim == 2:
        if channel != 0:
            raise ValueError(f"--channel {channel} requested but data has shape {data.shape}.")
        return np.asarray(data[indices, :][:, node_indices], dtype=np.float32)
    if data.ndim == 3:
        if channel < 0 or channel >= data.shape[2]:
            raise ValueError(f"--channel {channel} is outside data feature dimension {data.shape[2]}.")
        return np.asarray(data[indices, :, channel][:, node_indices], dtype=np.float32)
    raise ValueError(f"Unsupported data shape: {data.shape}")


def build_time_labels(
    data: np.memmap,
    desc: dict[str, Any],
    start_index: int,
    num_steps: int,
    time_stride: int,
    step_minutes: int,
    dataset_start_datetime: str | None,
) -> list[str]:
    if dataset_start_datetime is not None:
        start_dt = parse_datetime(dataset_start_datetime, "--dataset-start-datetime")
        return [
            (start_dt + timedelta(minutes=(start_index + offset * time_stride) * step_minutes)).strftime(
                "%Y-%m-%d %H:%M"
            )
            for offset in range(num_steps)
        ]
    if data.ndim == 3:
        tod_channel = find_feature_channel(desc, ("time of day", "timeofday", "tod"))
        dow_channel = find_feature_channel(desc, ("day of week", "dayofweek", "dow"))
        if tod_channel is not None and dow_channel is not None:
            indices = start_index + np.arange(num_steps, dtype=int) * time_stride
            tod_values = np.asarray(data[indices, 0, tod_channel], dtype=np.float64)
            dow_values = np.asarray(data[indices, 0, dow_channel], dtype=np.float64)
            labels = []
            for tod_value, dow_value in zip(tod_values, dow_values):
                day_index = int(round(float(dow_value) * 7.0)) % 7
                minutes = int(round(float(tod_value) * 1440.0 / step_minutes) * step_minutes) % 1440
                labels.append(f"{DAY_NAMES[day_index]} {minutes // 60:02d}:{minutes % 60:02d}")
            return labels
    return [f"step {start_index + offset * time_stride}" for offset in range(num_steps)]


def mercator_y(lat: np.ndarray) -> np.ndarray:
    clipped = np.clip(lat, -85.05112878, 85.05112878)
    radians = np.deg2rad(clipped)
    return np.log(np.tan(np.pi / 4.0 + radians / 2.0))


def build_sensor_payload(
    meta_df: pd.DataFrame,
    values: np.ndarray,
    id_column: str,
    lat_column: str,
    lon_column: str,
    value_precision: int,
) -> list[dict[str, Any]]:
    lon = meta_df[lon_column].to_numpy(dtype=np.float64)
    lat = meta_df[lat_column].to_numpy(dtype=np.float64)
    x_span = max(float(lon.max() - lon.min()), 1e-9)
    y_raw = mercator_y(lat)
    y_span = max(float(y_raw.max() - y_raw.min()), 1e-9)
    x_norm = (lon - lon.min()) / x_span
    y_norm = 1.0 - (y_raw - y_raw.min()) / y_span

    sensors = []
    rounded = np.round(np.nan_to_num(values.T, nan=0.0, posinf=0.0, neginf=0.0), value_precision)
    for row_idx, row in meta_df.iterrows():
        sensors.append(
            {
                "id": str(row[id_column]),
                "x": round(float(x_norm[row_idx]), 7),
                "y": round(float(y_norm[row_idx]), 7),
                "lat": round(float(row[lat_column]), 7),
                "lon": round(float(row[lon_column]), 7),
                "v": rounded[row_idx].tolist(),
            }
        )
    return sensors


def compute_contrast(values: np.ndarray, low_percentile: float, high_percentile: float) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    positive = finite[finite > 0]
    if positive.size == 0:
        return 0.0, 1.0
    low_p = min(max(low_percentile, 0.0), 99.9)
    high_p = min(max(high_percentile, low_p + 0.1), 100.0)
    low = float(np.percentile(positive, low_p))
    high = float(np.percentile(positive, high_p))
    if not math.isfinite(low):
        low = 0.0
    if not math.isfinite(high) or high <= low:
        high = float(np.max(positive))
    if high <= low:
        high = low + 1.0
    return low, high


def html_escape_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def render_html(
    sensors: list[dict[str, Any]],
    time_labels: list[str],
    args: argparse.Namespace,
    output_html: Path,
    summary: dict[str, Any],
) -> None:
    sensors_json = html_escape_json(sensors)
    labels_json = html_escape_json(time_labels)
    summary_json = html_escape_json(summary)
    title = f"{summary['dataset']} planar heatmap"
    show_points = "false" if args.hide_points else "true"
    tile_url = args.tile_url or ""

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
  html, body {{
    height: 100%;
    margin: 0;
    overflow: hidden;
    background: #020617;
    color: #e2e8f0;
    font-family: "Avenir Next", "Gill Sans", sans-serif;
  }}
  #heatmap-wrap {{
    height: 100vh;
    width: 100vw;
    display: grid;
    place-items: center;
    background:
      radial-gradient(circle at 22% 18%, rgba(59, 130, 246, 0.22), transparent 30%),
      radial-gradient(circle at 78% 82%, rgba(248, 113, 113, 0.13), transparent 32%),
      #020617;
  }}
  #heatmap-canvas {{
    width: min(100vw, {int(args.canvas_width)}px);
    height: min(100vh, {int(args.canvas_height)}px);
    max-width: 100vw;
    max-height: 100vh;
    border: 1px solid rgba(15, 23, 42, 0.32);
    border-radius: 12px;
    box-shadow: 0 18px 55px rgba(0, 0, 0, 0.45);
  }}
  .panel {{
    position: fixed;
    left: 18px;
    top: 18px;
    width: 330px;
    z-index: 10;
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid rgba(148, 163, 184, 0.24);
    border-radius: 14px;
    box-shadow: 0 14px 34px rgba(0, 0, 0, 0.34);
    padding: 13px 15px;
    backdrop-filter: blur(8px);
  }}
  .title {{
    font-size: 15px;
    font-weight: 800;
    margin-bottom: 5px;
  }}
  .time {{
    color: #bae6fd;
    font-size: 13px;
    margin-bottom: 9px;
  }}
  .slider {{
    width: 100%;
    accent-color: #f97316;
  }}
  .row {{
    align-items: center;
    display: flex;
    gap: 10px;
    margin-top: 8px;
  }}
  .button {{
    background: #f97316;
    border: 0;
    border-radius: 999px;
    color: white;
    cursor: pointer;
    font-size: 12px;
    font-weight: 800;
    padding: 6px 15px;
  }}
  .small {{
    color: #cbd5e1;
    font-size: 11px;
    line-height: 1.35;
    margin-top: 8px;
  }}
  .legend {{
    position: fixed;
    right: 18px;
    bottom: 18px;
    width: 280px;
    z-index: 10;
    background: rgba(15, 23, 42, 0.88);
    border: 1px solid rgba(148, 163, 184, 0.24);
    border-radius: 14px;
    box-shadow: 0 14px 34px rgba(0, 0, 0, 0.34);
    padding: 11px 13px;
  }}
  .legend-bar {{
    height: 12px;
    border-radius: 999px;
    background: linear-gradient(90deg, #050816, #1e1b4b, #1d4ed8, #06b6d4, #22c55e, #fde047, #f97316, #ef4444, #ffffff);
    margin-bottom: 7px;
  }}
  .legend-labels {{
    display: flex;
    justify-content: space-between;
    color: #cbd5e1;
    font-size: 11px;
  }}
  </style>
</head>
<body>
  <div id="heatmap-wrap">
    <canvas id="heatmap-canvas"></canvas>
  </div>
  <div class="panel">
    <div class="title">{html.escape(title)}</div>
    <div class="time" id="time-label">loading...</div>
    <input class="slider" id="slider" type="range" min="0" max="{len(time_labels) - 1}" value="0" step="1" />
    <div class="row">
      <button class="button" id="play-button" type="button">Play</button>
      <span id="step-label">1 / {len(time_labels)}</span>
    </div>
    <div class="small" id="summary-label"></div>
  </div>
  <div class="legend">
    <div class="legend-bar"></div>
    <div class="legend-labels">
      <span>{summary['contrast_low']:.2f}</span>
      <span>OSM + clipped flow</span>
      <span>{summary['contrast_high']:.2f}</span>
    </div>
  </div>
  <script>
  (function() {{
    const sensors = {sensors_json};
    const labels = {labels_json};
    const summary = {summary_json};
    const canvas = document.getElementById("heatmap-canvas");
    const ctx = canvas.getContext("2d");
    const slider = document.getElementById("slider");
    const playButton = document.getElementById("play-button");
    const timeLabel = document.getElementById("time-label");
    const stepLabel = document.getElementById("step-label");
    const summaryLabel = document.getElementById("summary-label");
    const heatRadius = {float(args.heat_radius):.6f};
    const pointRadius = {float(args.point_radius):.6f};
    const gamma = {float(args.gamma):.6f};
    const showPoints = {show_points};
    const tileUrlTemplate = {json.dumps(tile_url)};
    const tileZoom = {int(args.tile_zoom)};
    const basemapOpacity = {float(args.basemap_opacity):.6f};
    const basemapDim = {float(args.basemap_dim):.6f};
    const low = summary.contrast_low;
    const high = summary.contrast_high;
    const playIntervalMs = {int(args.play_interval_ms)};
    const tileCache = new Map();
    let currentStep = 0;
    let timer = null;

    function resizeCanvas() {{
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      canvas.width = Math.max(320, Math.round(rect.width * dpr));
      canvas.height = Math.max(240, Math.round(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }}

    function colorRamp(t) {{
      const stops = [
        [0.00, [5, 8, 22]],
        [0.14, [30, 27, 75]],
        [0.28, [29, 78, 216]],
        [0.42, [6, 182, 212]],
        [0.56, [34, 197, 94]],
        [0.70, [253, 224, 71]],
        [0.84, [249, 115, 22]],
        [0.94, [239, 68, 68]],
        [1.00, [255, 255, 255]]
      ];
      const x = Math.max(0, Math.min(1, t));
      for (let i = 0; i < stops.length - 1; i += 1) {{
        const a = stops[i];
        const b = stops[i + 1];
        if (x >= a[0] && x <= b[0]) {{
          const u = (x - a[0]) / Math.max(1e-9, b[0] - a[0]);
          const r = Math.round(a[1][0] + (b[1][0] - a[1][0]) * u);
          const g = Math.round(a[1][1] + (b[1][1] - a[1][1]) * u);
          const bl = Math.round(a[1][2] + (b[1][2] - a[1][2]) * u);
          return [r, g, bl];
        }}
      }}
      return stops[stops.length - 1][1];
    }}

    function normalize(value) {{
      const raw = (Number(value) - low) / Math.max(1e-9, high - low);
      return Math.pow(Math.max(0, Math.min(1, raw)), gamma);
    }}

    function clampLat(lat) {{
      return Math.max(-85.05112878, Math.min(85.05112878, lat));
    }}

    function mercatorNorm(lat, lon) {{
      const clampedLat = clampLat(lat);
      const sinLat = Math.sin((clampedLat * Math.PI) / 180);
      const x = (lon + 180) / 360;
      const y = 0.5 - Math.log((1 + sinLat) / (1 - sinLat)) / (4 * Math.PI);
      return [x, y];
    }}

    function mercatorBounds() {{
      const nw = mercatorNorm(summary.max_lat, summary.min_lon);
      const se = mercatorNorm(summary.min_lat, summary.max_lon);
      return {{
        minX: Math.min(nw[0], se[0]),
        maxX: Math.max(nw[0], se[0]),
        minY: Math.min(nw[1], se[1]),
        maxY: Math.max(nw[1], se[1])
      }};
    }}

    function projectNorm(normX, normY, width, height) {{
      const pad = Math.max(44, Math.min(width, height) * 0.065);
      const bounds = mercatorBounds();
      const xRange = Math.max(1e-12, bounds.maxX - bounds.minX);
      const yRange = Math.max(1e-12, bounds.maxY - bounds.minY);
      const x = pad + ((normX - bounds.minX) / xRange) * Math.max(1, width - 2 * pad);
      const y = pad + ((normY - bounds.minY) / yRange) * Math.max(1, height - 2 * pad);
      return [x, y];
    }}

    function projectGeo(lat, lon, width, height) {{
      const p = mercatorNorm(lat, lon);
      return projectNorm(p[0], p[1], width, height);
    }}

    function tileUrl(x, y, z) {{
      const subdomains = ["a", "b", "c", "d"];
      const s = subdomains[Math.abs(x + y) % subdomains.length];
      return tileUrlTemplate
        .replaceAll("{{z}}", String(z))
        .replaceAll("{{x}}", String(x))
        .replaceAll("{{y}}", String(y))
        .replaceAll("{{s}}", s);
    }}

    function getTileImage(url) {{
      if (tileCache.has(url)) {{
        return tileCache.get(url);
      }}
      const image = new Image();
      image.referrerPolicy = "no-referrer";
      image.onload = function() {{
        drawFrame(currentStep);
      }};
      image.src = url;
      tileCache.set(url, image);
      return image;
    }}

    function drawBasemap(width, height) {{
      if (!tileUrlTemplate) {{
        return false;
      }}
      ctx.save();
      ctx.globalAlpha = Math.max(0, Math.min(1, basemapOpacity));
      ctx.fillStyle = "#dbeafe";
      ctx.fillRect(0, 0, width, height);
      const z = Math.max(0, Math.min(20, Number(tileZoom) || 11));
      const n = Math.pow(2, z);
      const bounds = mercatorBounds();
      const xStart = Math.floor(bounds.minX * n) - 1;
      const xEnd = Math.floor(bounds.maxX * n) + 1;
      const yStart = Math.floor(bounds.minY * n) - 1;
      const yEnd = Math.floor(bounds.maxY * n) + 1;
      for (let x = xStart; x <= xEnd; x += 1) {{
        for (let y = yStart; y <= yEnd; y += 1) {{
          if (y < 0 || y >= n) {{
            continue;
          }}
          const wrappedX = ((x % n) + n) % n;
          const p0 = projectNorm(x / n, y / n, width, height);
          const p1 = projectNorm((x + 1) / n, (y + 1) / n, width, height);
          const url = tileUrl(wrappedX, y, z);
          const image = getTileImage(url);
          const dx = Math.min(p0[0], p1[0]);
          const dy = Math.min(p0[1], p1[1]);
          const dw = Math.abs(p1[0] - p0[0]) + 1;
          const dh = Math.abs(p1[1] - p0[1]) + 1;
          if (image.complete && image.naturalWidth > 0) {{
            ctx.drawImage(image, dx, dy, dw, dh);
          }} else {{
            ctx.fillStyle = "#cbd5e1";
            ctx.fillRect(dx, dy, dw, dh);
          }}
        }}
      }}
      ctx.restore();
      ctx.fillStyle = `rgba(2, 6, 23, ${{Math.max(0, Math.min(0.85, basemapDim))}})`;
      ctx.fillRect(0, 0, width, height);
      return true;
    }}

    function drawBackground(width, height) {{
      const hasBasemap = drawBasemap(width, height);
      if (hasBasemap) {{
        ctx.strokeStyle = "rgba(15, 23, 42, 0.16)";
        ctx.lineWidth = 1;
        return;
      }}
      const gradient = ctx.createLinearGradient(0, 0, width, height);
      gradient.addColorStop(0, "#020617");
      gradient.addColorStop(0.55, "#0f172a");
      gradient.addColorStop(1, "#111827");
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = "rgba(148, 163, 184, 0.16)";
      ctx.lineWidth = 1;
      for (let i = 1; i < 6; i += 1) {{
        const x = (width * i) / 6;
        const y = (height * i) / 6;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }}
    }}

    function sensorXY(sensor, width, height) {{
      return projectGeo(sensor.lat, sensor.lon, width, height);
    }}

    function drawFrame(step) {{
      const rect = canvas.getBoundingClientRect();
      const width = rect.width;
      const height = rect.height;
      ctx.clearRect(0, 0, width, height);
      drawBackground(width, height);
      ctx.globalCompositeOperation = "lighter";
      for (const sensor of sensors) {{
        const level = normalize(sensor.v[step]);
        if (level <= 0) {{
          continue;
        }}
        const xy = sensorXY(sensor, width, height);
        const radius = heatRadius * (0.55 + 0.85 * level);
        const rgb = colorRamp(level);
        const grad = ctx.createRadialGradient(xy[0], xy[1], 0, xy[0], xy[1], radius);
        grad.addColorStop(0, `rgba(${{rgb[0]}},${{rgb[1]}},${{rgb[2]}},${{0.82 * level + 0.18}})`);
        grad.addColorStop(0.42, `rgba(${{rgb[0]}},${{rgb[1]}},${{rgb[2]}},${{0.42 * level}})`);
        grad.addColorStop(1, "rgba(0,0,0,0)");
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(xy[0], xy[1], radius, 0, Math.PI * 2);
        ctx.fill();
      }}
      ctx.globalCompositeOperation = "source-over";
      if (showPoints) {{
        for (const sensor of sensors) {{
          const level = normalize(sensor.v[step]);
          const xy = sensorXY(sensor, width, height);
          const rgb = colorRamp(level);
          ctx.fillStyle = `rgba(${{rgb[0]}},${{rgb[1]}},${{rgb[2]}},0.94)`;
          ctx.strokeStyle = "rgba(255,255,255,0.62)";
          ctx.lineWidth = 0.8;
          ctx.beginPath();
          ctx.arc(xy[0], xy[1], pointRadius, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        }}
      }}
    }}

    function setStep(step) {{
      currentStep = Math.max(0, Math.min(labels.length - 1, Number(step)));
      slider.value = currentStep;
      timeLabel.textContent = labels[currentStep];
      stepLabel.textContent = `${{currentStep + 1}} / ${{labels.length}}`;
      drawFrame(currentStep);
    }}

    function stopPlayback() {{
      if (timer !== null) {{
        clearInterval(timer);
        timer = null;
      }}
      playButton.textContent = "Play";
    }}

    function startPlayback() {{
      stopPlayback();
      timer = setInterval(function() {{
        const nextStep = currentStep + 1 >= labels.length ? 0 : currentStep + 1;
        setStep(nextStep);
      }}, playIntervalMs);
      playButton.textContent = "Pause";
    }}

    slider.addEventListener("input", function(event) {{
      stopPlayback();
      setStep(event.target.value);
    }});
    playButton.addEventListener("click", function() {{
      if (timer === null) {{
        startPlayback();
      }} else {{
        stopPlayback();
      }}
    }});
    window.addEventListener("resize", function() {{
      resizeCanvas();
      setStep(currentStep);
    }});

    summaryLabel.textContent =
      `nodes=${{summary.rendered_nodes}}, steps=${{summary.num_steps}}, stride=${{summary.time_stride}}, ` +
      `gamma=${{summary.gamma}}, clip=${{summary.contrast_low.toFixed(1)}}-${{summary.contrast_high.toFixed(1)}}`;
    resizeCanvas();
    setStep(0);
  }})();
  </script>
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(page, encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_dir = resolve_dataset_dir(args.dataset, args.dataset_dir)
    meta_path = dataset_dir / "meta.csv"
    data_path = Path(args.data_file).expanduser().resolve() if args.data_file else dataset_dir / "data.dat"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.csv not found: {meta_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"data.dat not found: {data_path}")

    desc = read_description(dataset_dir)
    raw_meta_rows = int(pd.read_csv(meta_path, usecols=[0]).shape[0])
    shape = resolve_data_shape(desc, data_path, raw_meta_rows or 1, args)
    data = load_data_memmap(data_path, shape)
    total_steps = int(data.shape[0])
    num_nodes = int(data.shape[1])
    step_minutes = resolve_step_minutes(desc, args.step_minutes)
    start_index, start_mode = resolve_start_index(args, data, desc, step_minutes)
    validate_window(start_index, args.num_steps, args.time_stride, total_steps)

    meta_df, id_column, lat_column, lon_column, valid_node_count = build_meta_frame(
        meta_path,
        args.id_column,
        args.lat_column,
        args.lon_column,
        num_nodes,
        args.sample_stride,
        args.max_nodes,
    )
    node_indices = meta_df["node_index"].to_numpy(dtype=int)
    values = extract_window(data, start_index, args.num_steps, args.time_stride, node_indices, args.channel)
    low, high = compute_contrast(values, args.low_percentile, args.high_percentile)
    sensors = build_sensor_payload(meta_df, values, id_column, lat_column, lon_column, args.value_precision)
    time_labels = build_time_labels(
        data,
        desc,
        start_index,
        args.num_steps,
        args.time_stride,
        step_minutes,
        args.dataset_start_datetime,
    )
    output_html = (
        Path(args.output_html).expanduser().resolve()
        if args.output_html is not None
        else dataset_dir / "traffic_plane_heatmap_animation.html"
    )
    summary = {
        "dataset": args.dataset,
        "dataset_dir": str(dataset_dir),
        "data_path": str(data_path),
        "data_shape": list(shape),
        "channel": int(args.channel),
        "start_index": int(start_index),
        "start_mode": start_mode,
        "num_steps": int(args.num_steps),
        "time_stride": int(args.time_stride),
        "step_minutes": int(step_minutes),
        "effective_step_minutes": int(step_minutes * args.time_stride),
        "valid_nodes": int(valid_node_count),
        "rendered_nodes": int(len(sensors)),
        "contrast_low": float(low),
        "contrast_high": float(high),
        "gamma": float(args.gamma),
        "heat_radius": float(args.heat_radius),
        "tile_url": args.tile_url,
        "tile_zoom": int(args.tile_zoom),
        "basemap_opacity": float(args.basemap_opacity),
        "basemap_dim": float(args.basemap_dim),
        "min_lat": float(meta_df[lat_column].min()),
        "max_lat": float(meta_df[lat_column].max()),
        "min_lon": float(meta_df[lon_column].min()),
        "max_lon": float(meta_df[lon_column].max()),
        "output_html": str(output_html),
        "id_column": id_column,
        "lat_column": lat_column,
        "lon_column": lon_column,
    }
    render_html(sensors, time_labels, args, output_html, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
