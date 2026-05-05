#!/usr/bin/env python3
"""Render an interactive traffic-flow bar animation on a Leaflet map.

The script targets BasicTS-style traffic datasets:

    datasets/<DATASET>/meta.csv
    datasets/<DATASET>/data.dat
    datasets/<DATASET>/desc.json

Each sensor is drawn as a vertical bar whose height changes with the selected
time step. Folium is used when available; otherwise the script writes a pure
Leaflet HTML page.
"""

import argparse
import html
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import folium
except ModuleNotFoundError:
    folium = None


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
        description="Create an interactive HTML traffic map with animated per-sensor bars."
    )
    parser.add_argument(
        "--dataset",
        default="SD",
        help="Dataset name under datasets/<DATASET>. Default: SD",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Explicit dataset directory containing meta.csv, data.dat, and desc.json.",
    )
    parser.add_argument(
        "--data-file",
        default=None,
        help="Explicit data memmap path. Defaults to <dataset-dir>/data.dat.",
    )
    parser.add_argument(
        "--data-shape",
        default=None,
        help="Override data shape as T,N or T,N,C. Useful when desc.json is absent.",
    )
    parser.add_argument(
        "--num-features",
        type=int,
        default=None,
        help="Feature count used only to infer shape when desc.json and --data-shape are absent.",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Feature channel to visualize. Default: 0 (traffic flow).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="First global time index to render.",
    )
    parser.add_argument(
        "--start-datetime",
        default=None,
        help=(
            "Start time such as '2024-01-02 08:00'. If --dataset-start-datetime is "
            "not provided, this matches the first time-of-day/day-of-week feature cycle."
        ),
    )
    parser.add_argument(
        "--dataset-start-datetime",
        default=None,
        help="Datetime corresponding to data index 0; enables absolute --start-datetime lookup.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=96,
        help="Number of rendered time steps to include after applying --time-stride. Default: 96.",
    )
    parser.add_argument(
        "--time-stride",
        type=int,
        default=1,
        help="Render every Nth time step from the source data. Use 6 on 5min data for 30min frames.",
    )
    parser.add_argument(
        "--step-minutes",
        type=int,
        default=None,
        help="Minutes per step. Defaults to desc.json 'frequency (minutes)' when available.",
    )
    parser.add_argument(
        "--output-html",
        default=None,
        help="Output HTML path. Defaults to <dataset-dir>/traffic_map_bar_animation.html.",
    )
    parser.add_argument(
        "--value-scale",
        type=float,
        default=0.0,
        help="Pixels per traffic-flow unit. Use <=0 for percentile-based automatic scaling.",
    )
    parser.add_argument(
        "--scale-percentile",
        type=float,
        default=95.0,
        help="Percentile used when --value-scale <= 0. Default: 95.",
    )
    parser.add_argument(
        "--max-bar-height",
        type=float,
        default=52.0,
        help="Maximum rendered bar height in pixels. Default: 52.",
    )
    parser.add_argument(
        "--min-bar-height",
        type=float,
        default=2.0,
        help="Minimum positive bar height in pixels. Default: 2.",
    )
    parser.add_argument(
        "--bar-width",
        type=float,
        default=8.0,
        help="Rendered bar width in pixels. Default: 8.",
    )
    parser.add_argument(
        "--value-precision",
        type=int,
        default=2,
        help="Decimal places stored for popup values. Default: 2.",
    )
    parser.add_argument(
        "--id-column",
        default=None,
        help="Sensor id column in meta.csv. Auto-detected by default.",
    )
    parser.add_argument(
        "--lat-column",
        default=None,
        help="Latitude column in meta.csv. Auto-detected by default.",
    )
    parser.add_argument(
        "--lon-column",
        default=None,
        help="Longitude column in meta.csv. Auto-detected by default.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=0,
        help="Maximum number of sensors to render. Use 0 for all valid sensors.",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=1,
        help="Render every Nth sensor after metadata validation. Default: 1.",
    )
    parser.add_argument(
        "--zoom-start",
        type=int,
        default=10,
        help="Initial Folium map zoom. Default: 10.",
    )
    parser.add_argument(
        "--tiles",
        default="CartoDB positron",
        help="Folium tile layer. Default: CartoDB positron.",
    )
    parser.add_argument(
        "--play-interval-ms",
        type=int,
        default=450,
        help="Playback interval in milliseconds. Default: 450.",
    )
    parser.add_argument(
        "--no-fit-bounds",
        action="store_true",
        help="Do not automatically fit the map to rendered sensors.",
    )
    parser.add_argument(
        "--renderer",
        choices=["auto", "leaflet", "offline"],
        default="auto",
        help=(
            "Rendering backend. 'auto' uses Folium/Leaflet when available; "
            "'offline' writes a self-contained SVG map with no external resources."
        ),
    )
    return parser.parse_args()


def normalize_sensor_id(value: Any) -> str:
    text = str(value).strip()
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def script_basicts_root() -> Path:
    return Path(__file__).resolve().parents[2]


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

    complete_candidates = [
        path for path in candidates if (path / "meta.csv").exists() and (path / "data.dat").exists()
    ]
    if complete_candidates:
        return complete_candidates[0].resolve()

    existing_candidates = [path for path in candidates if path.exists()]
    if existing_candidates:
        return existing_candidates[0].resolve()

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
            raise ValueError(
                f"Cannot infer data shape: file has {num_float32} float32 values, "
                f"num_nodes={num_nodes}, num_features={num_features}."
            )
        return (num_float32 // denominator, num_nodes, num_features)

    for feature_count in (3, 1):
        denominator = num_nodes * feature_count
        if denominator > 0 and num_float32 % denominator == 0:
            time_steps = num_float32 // denominator
            return (time_steps, num_nodes, feature_count) if feature_count > 1 else (time_steps, num_nodes)

    raise ValueError(
        "Cannot infer data shape from data.dat. Provide desc.json, --data-shape, or --num-features."
    )


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
            raise KeyError(f"Requested {role} column {requested!r} not found in meta.csv columns: {columns}")
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
        raise ValueError(
            "No valid sensors remain after coordinate validation and node-count alignment. "
            f"meta rows={len(meta_df)}, data nodes={num_nodes}."
        )

    frame = frame.reset_index(drop=True)
    return frame, resolved_id, resolved_lat, resolved_lon, valid_node_count


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

    target_minutes = (
        start_datetime.hour * 60
        + start_datetime.minute
        + start_datetime.second / 60.0
        + start_datetime.microsecond / 60_000_000.0
    )
    target_tod = (target_minutes % 1440.0) / 1440.0
    target_dow = start_datetime.weekday() / 7.0
    tod_values = np.asarray(data[:, 0, tod_channel], dtype=np.float64)
    dow_values = np.asarray(data[:, 0, dow_channel], dtype=np.float64)
    tod_diff = np.abs(((tod_values - target_tod + 0.5) % 1.0) - 0.5)
    dow_diff = np.abs(dow_values - target_dow)
    tod_tolerance = max(step_minutes / 1440.0 / 2.0, 1e-4)
    dow_tolerance = 1e-4
    matches = np.where((tod_diff <= tod_tolerance) & (dow_diff <= dow_tolerance))[0]
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
        return args.start_index, "explicit-index"
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
    synthetic_index = start_datetime.weekday() * steps_per_day + int(
        round((start_datetime.hour * 60 + start_datetime.minute) / step_minutes)
    )
    return synthetic_index, "synthetic-week"


def validate_window(start_index: int, num_steps: int, time_stride: int, total_steps: int) -> None:
    if start_index < 0:
        raise ValueError(f"Start index must be non-negative, got {start_index}.")
    if num_steps <= 0:
        raise ValueError(f"--num-steps must be positive, got {num_steps}.")
    if time_stride <= 0:
        raise ValueError(f"--time-stride must be positive, got {time_stride}.")
    if start_index >= total_steps:
        raise ValueError(f"Start index {start_index} is outside data with {total_steps} time steps.")
    end_index = start_index + (num_steps - 1) * time_stride + 1
    if end_index > total_steps:
        raise ValueError(
            f"Requested window [{start_index}, {end_index}) with --num-steps={num_steps} "
            f"and --time-stride={time_stride} exceeds data length {total_steps}."
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
        values = np.asarray(data[indices, :][:, node_indices], dtype=np.float32)
    elif data.ndim == 3:
        if channel < 0 or channel >= data.shape[2]:
            raise ValueError(f"--channel {channel} is outside data feature dimension {data.shape[2]}.")
        values = np.asarray(data[indices, :, channel][:, node_indices], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")
    return values


def compute_scale(values: np.ndarray, args: argparse.Namespace) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    positive = finite[finite > 0]
    if positive.size == 0:
        reference = 1.0
    else:
        percentile = min(max(args.scale_percentile, 1.0), 100.0)
        reference = float(np.percentile(positive, percentile))
        if not math.isfinite(reference) or reference <= 0:
            reference = float(np.max(positive))
    if args.value_scale > 0:
        return float(args.value_scale), reference
    return float(args.max_bar_height / max(reference, 1e-6)), reference


def prepare_bar_arrays(
    values: np.ndarray,
    value_scale: float,
    min_bar_height: float,
    max_bar_height: float,
    value_precision: int,
) -> tuple[list[list[float]], list[list[int]]]:
    safe_values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    positive_values = np.clip(safe_values, 0.0, None)
    heights = positive_values * value_scale
    heights = np.where(positive_values > 0, np.maximum(heights, min_bar_height), 0.0)
    heights = np.clip(heights, 0.0, max_bar_height)
    values_by_node = np.round(safe_values.T, value_precision).tolist()
    heights_by_node = np.rint(heights.T).astype(int).tolist()
    return values_by_node, heights_by_node


def format_minutes_from_fraction(value: float, step_minutes: int) -> str:
    minutes = int(round(float(value) * 1440.0 / step_minutes) * step_minutes) % 1440
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


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
        if (
            tod_channel is not None
            and dow_channel is not None
            and tod_channel < data.shape[2]
            and dow_channel < data.shape[2]
        ):
            indices = start_index + np.arange(num_steps, dtype=int) * time_stride
            tod_values = np.asarray(data[indices, 0, tod_channel], dtype=np.float64)
            dow_values = np.asarray(data[indices, 0, dow_channel], dtype=np.float64)
            labels = []
            for tod_value, dow_value in zip(tod_values, dow_values, strict=True):
                day_index = int(round(float(dow_value) * 7.0)) % 7
                labels.append(f"{DAY_NAMES[day_index]} {format_minutes_from_fraction(tod_value, step_minutes)}")
            return labels

    return [f"step {start_index + offset * time_stride}" for offset in range(num_steps)]


def optional_meta_value(row: pd.Series, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in row.index and pd.notna(row[candidate]):
            return str(row[candidate])
    return None


def build_sensor_payload(
    meta_df: pd.DataFrame,
    values_by_node: list[list[float]],
    heights_by_node: list[list[int]],
    id_column: str,
    lat_column: str,
    lon_column: str,
) -> list[dict[str, Any]]:
    sensors: list[dict[str, Any]] = []
    for row_idx, row in meta_df.iterrows():
        metadata_parts = []
        for candidate in ["Type", "type", "Fwy", "Direction", "District", "County", "Lanes", "ID2"]:
            value = optional_meta_value(row, [candidate])
            if value is not None:
                metadata_parts.append(f"{candidate}: {value}")
        sensors.append(
            {
                "id": str(row[id_column]),
                "lat": round(float(row[lat_column]), 7),
                "lon": round(float(row[lon_column]), 7),
                "meta": " | ".join(metadata_parts),
                "v": values_by_node[row_idx],
                "h": heights_by_node[row_idx],
            }
        )
    return sensors


def html_escape_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def build_control_html(map_id: str, title: str, num_steps: int) -> str:
    safe_title = html.escape(title)
    return f"""
<div id="{map_id}-traffic-panel" class="traffic-panel">
  <div class="traffic-panel-title">{safe_title}</div>
  <div class="traffic-panel-time" id="{map_id}-time-label">loading...</div>
  <input id="{map_id}-slider" class="traffic-slider" type="range" min="0" max="{num_steps - 1}" value="0" step="1" />
  <div class="traffic-panel-row">
    <button id="{map_id}-play" class="traffic-play-button" type="button">Play</button>
    <span id="{map_id}-step-label" class="traffic-step-label">1 / {num_steps}</span>
  </div>
  <div id="{map_id}-summary" class="traffic-panel-summary"></div>
  <div class="traffic-legend">
    <span>low</span><span class="traffic-gradient"></span><span>high</span>
  </div>
</div>
"""


def build_css(max_bar_height: float, bar_width: float) -> str:
    return f"""
<style>
.traffic-panel {{
  position: fixed;
  top: 16px;
  left: 56px;
  z-index: 9999;
  width: 300px;
  background: rgba(255, 255, 255, 0.94);
  border: 1px solid rgba(23, 37, 84, 0.16);
  border-radius: 14px;
  box-shadow: 0 12px 32px rgba(15, 23, 42, 0.18);
  color: #172554;
  font-family: "Avenir Next", "Gill Sans", sans-serif;
  padding: 12px 14px;
}}
.traffic-panel-title {{
  font-size: 15px;
  font-weight: 700;
  letter-spacing: 0.01em;
  margin-bottom: 4px;
}}
.traffic-panel-time {{
  color: #475569;
  font-size: 13px;
  margin-bottom: 8px;
}}
.traffic-slider {{
  width: 100%;
  accent-color: #0f766e;
}}
.traffic-panel-row {{
  align-items: center;
  display: flex;
  gap: 10px;
  margin-top: 8px;
}}
.traffic-play-button {{
  background: #0f766e;
  border: 0;
  border-radius: 999px;
  color: white;
  cursor: pointer;
  font-size: 12px;
  font-weight: 700;
  padding: 6px 14px;
}}
.traffic-step-label {{
  color: #334155;
  font-size: 12px;
}}
.traffic-panel-summary {{
  color: #64748b;
  font-size: 11px;
  line-height: 1.35;
  margin-top: 8px;
}}
.traffic-legend {{
  align-items: center;
  color: #64748b;
  display: grid;
  font-size: 10px;
  gap: 6px;
  grid-template-columns: auto 1fr auto;
  margin-top: 9px;
}}
.traffic-gradient {{
  background: linear-gradient(90deg, #22c55e, #f59e0b, #ef4444);
  border-radius: 999px;
  height: 8px;
}}
.traffic-bar-icon {{
  background: transparent;
  border: 0;
}}
.traffic-bar-shell {{
  height: {max_bar_height + 10:.0f}px;
  pointer-events: auto;
  position: relative;
  width: {bar_width + 10:.0f}px;
}}
.traffic-bar-fill {{
  border: 1px solid rgba(15, 23, 42, 0.32);
  border-radius: 999px 999px 3px 3px;
  bottom: 5px;
  box-shadow: 0 2px 7px rgba(15, 23, 42, 0.3);
  left: 50%;
  min-height: 0;
  position: absolute;
  transform: translateX(-50%);
  transition: height 120ms linear, background-color 120ms linear;
  width: {bar_width:.0f}px;
}}
.traffic-bar-dot {{
  background: #0f172a;
  border: 1px solid white;
  border-radius: 999px;
  bottom: 1px;
  box-shadow: 0 1px 4px rgba(15, 23, 42, 0.35);
  height: 6px;
  left: 50%;
  position: absolute;
  transform: translateX(-50%);
  width: 6px;
}}
@media (max-width: 640px) {{
  .traffic-panel {{
    left: 12px;
    right: 12px;
    top: 12px;
    width: auto;
  }}
}}
</style>
"""


def build_icon_html(initial_height: int, initial_color: str) -> str:
    return (
        '<div class="traffic-bar-shell">'
        f'<div class="traffic-bar-fill" style="height:{initial_height}px;background:{initial_color};"></div>'
        '<div class="traffic-bar-dot"></div>'
        "</div>"
    )


def build_animation_js(
    map_name: str,
    map_id: str,
    sensors: list[dict[str, Any]],
    time_labels: list[str],
    args: argparse.Namespace,
    summary: dict[str, Any],
    include_script_tag: bool = True,
) -> str:
    sensor_json = html_escape_json(sensors)
    labels_json = html_escape_json(time_labels)
    summary_json = html_escape_json(summary)
    icon_width = int(round(args.bar_width + 10))
    icon_height = int(round(args.max_bar_height + 10))
    icon_anchor_x = int(round(icon_width / 2))
    icon_anchor_y = int(round(args.max_bar_height + 7))
    initial_html = build_icon_html(0, "#22c55e").replace("`", "\\`")

    script = f"""
(function() {{
  const map = {map_name};
  const sensors = {sensor_json};
  const timeLabels = {labels_json};
  const summary = {summary_json};
  const maxHeight = {float(args.max_bar_height):.6f};
  const iconSize = [{icon_width}, {icon_height}];
  const iconAnchor = [{icon_anchor_x}, {icon_anchor_y}];
  const playIntervalMs = {int(args.play_interval_ms)};
  const initialHtml = `{initial_html}`;
  const layer = L.layerGroup().addTo(map);
  const markers = [];
  let currentStep = 0;
  let timer = null;

  const slider = document.getElementById("{map_id}-slider");
  const playButton = document.getElementById("{map_id}-play");
  const timeLabel = document.getElementById("{map_id}-time-label");
  const stepLabel = document.getElementById("{map_id}-step-label");
  const summaryLabel = document.getElementById("{map_id}-summary");

  function colorForHeight(height) {{
    const ratio = Math.max(0, Math.min(1, height / Math.max(maxHeight, 1)));
    if (ratio < 0.5) {{
      const t = ratio / 0.5;
      return mixColor([34, 197, 94], [245, 158, 11], t);
    }}
    return mixColor([245, 158, 11], [239, 68, 68], (ratio - 0.5) / 0.5);
  }}

  function mixColor(a, b, t) {{
    const r = Math.round(a[0] + (b[0] - a[0]) * t);
    const g = Math.round(a[1] + (b[1] - a[1]) * t);
    const bl = Math.round(a[2] + (b[2] - a[2]) * t);
    return `rgb(${{r}}, ${{g}}, ${{bl}})`;
  }}

  function makeIcon() {{
    return L.divIcon({{
      className: "traffic-bar-icon",
      html: initialHtml,
      iconAnchor: iconAnchor,
      iconSize: iconSize
    }});
  }}

  function valueText(sensor, step) {{
    const value = sensor.v[step];
    if (value === null || value === undefined || Number.isNaN(Number(value))) {{
      return "value: n/a";
    }}
    return `value: ${{value}}`;
  }}

  function popupHtml(sensor, step) {{
    const metaLine = sensor.meta ? `<br><span>${{sensor.meta}}</span>` : "";
    return `<b>${{sensor.id}}</b><br>${{timeLabels[step]}}<br>${{valueText(sensor, step)}}${{metaLine}}`;
  }}

  function updateMarker(marker, sensor, step) {{
    const height = sensor.h[step] || 0;
    const element = marker.getElement();
    if (!element) {{
      return;
    }}
    const fill = element.querySelector(".traffic-bar-fill");
    if (fill) {{
      fill.style.height = `${{height}}px`;
      fill.style.background = colorForHeight(height);
      fill.title = `${{sensor.id}} | ${{timeLabels[step]}} | ${{valueText(sensor, step)}}`;
    }}
    if (marker.isPopupOpen()) {{
      marker.setPopupContent(popupHtml(sensor, step));
    }}
  }}

  function setStep(step) {{
    currentStep = Math.max(0, Math.min(timeLabels.length - 1, Number(step)));
    slider.value = currentStep;
    timeLabel.textContent = timeLabels[currentStep];
    stepLabel.textContent = `${{currentStep + 1}} / ${{timeLabels.length}}`;
    for (let index = 0; index < markers.length; index += 1) {{
      updateMarker(markers[index], sensors[index], currentStep);
    }}
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
      const nextStep = currentStep + 1 >= timeLabels.length ? 0 : currentStep + 1;
      setStep(nextStep);
    }}, playIntervalMs);
    playButton.textContent = "Pause";
  }}

  for (const sensor of sensors) {{
    const marker = L.marker([sensor.lat, sensor.lon], {{
      icon: makeIcon(),
      keyboard: false
    }}).addTo(layer);
    marker.bindPopup(popupHtml(sensor, 0), {{ maxWidth: 320 }});
    marker.bindTooltip(sensor.id, {{ direction: "top", opacity: 0.88 }});
    marker.on("mouseover", function() {{
      marker.setTooltipContent(`${{sensor.id}} | ${{timeLabels[currentStep]}} | ${{valueText(sensor, currentStep)}}`);
    }});
    markers.push(marker);
  }}

  L.control.layers(null, {{ "Traffic bars": layer }}, {{ collapsed: false }}).addTo(map);
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
  summaryLabel.textContent = `nodes=${{summary.rendered_nodes}}, steps=${{summary.num_steps}}, start=${{summary.start_index}}, scale=${{summary.value_scale.toFixed(4)}} px/unit`;
  setStep(0);
}})();
"""
    if include_script_tag:
        return f"<script>\n{script}</script>\n"
    return script


def tile_layer_config(tiles: str) -> tuple[str, str]:
    normalized = tiles.strip().lower()
    if "{z}" in tiles and "{x}" in tiles and "{y}" in tiles:
        return tiles, ""
    if "openstreetmap" in normalized or normalized in {"osm", "openstreetmap"}:
        return (
            "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        )
    return (
        "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors '
        '&copy; <a href="https://carto.com/attributions">CARTO</a>',
    )


def build_standalone_page_css(max_bar_height: float, bar_width: float) -> str:
    base_css = build_css(max_bar_height, bar_width).replace("</style>", "")
    return (
        base_css
        + """
html, body {
  height: 100%;
  margin: 0;
}
#traffic-map-canvas {
  height: 100vh;
  width: 100vw;
}
</style>
"""
    )


def render_standalone_leaflet_map(
    sensors: list[dict[str, Any]],
    time_labels: list[str],
    meta_df: pd.DataFrame,
    lat_column: str,
    lon_column: str,
    args: argparse.Namespace,
    output_html: Path,
    summary: dict[str, Any],
) -> None:
    center_lat = float(meta_df[lat_column].mean())
    center_lon = float(meta_df[lon_column].mean())
    min_lat = float(meta_df[lat_column].min())
    min_lon = float(meta_df[lon_column].min())
    max_lat = float(meta_df[lat_column].max())
    max_lon = float(meta_df[lon_column].max())
    tile_url, attribution = tile_layer_config(args.tiles)
    map_id = "traffic-map"
    title = f"{summary['dataset']} traffic flow"
    fit_bounds_js = ""
    if not args.no_fit_bounds:
        fit_bounds_js = (
            f"trafficMap.fitBounds([[{min_lat:.7f}, {min_lon:.7f}], "
            f"[{max_lat:.7f}, {max_lon:.7f}]], {{ padding: [18, 18] }});"
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
  {build_standalone_page_css(args.max_bar_height, args.bar_width)}
</head>
<body>
  <div id="traffic-map-canvas"></div>
  {build_control_html(map_id, title, len(time_labels))}
  <script>
  const trafficMap = L.map("traffic-map-canvas").setView([{center_lat:.7f}, {center_lon:.7f}], {int(args.zoom_start)});
  L.tileLayer({json.dumps(tile_url)}, {{
    attribution: {json.dumps(attribution)},
    maxZoom: 19,
    subdomains: "abcd"
  }}).addTo(trafficMap);
  {fit_bounds_js}
  </script>
  {build_animation_js("trafficMap", map_id, sensors, time_labels, args, summary)}
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(page, encoding="utf-8")


def render_offline_svg_map(
    sensors: list[dict[str, Any]],
    time_labels: list[str],
    meta_df: pd.DataFrame,
    lat_column: str,
    lon_column: str,
    args: argparse.Namespace,
    output_html: Path,
    summary: dict[str, Any],
) -> None:
    min_lat = float(meta_df[lat_column].min())
    min_lon = float(meta_df[lon_column].min())
    max_lat = float(meta_df[lat_column].max())
    max_lon = float(meta_df[lon_column].max())
    map_id = "traffic-map"
    title = f"{summary['dataset']} traffic flow"
    sensor_json = html_escape_json(sensors)
    labels_json = html_escape_json(time_labels)
    summary_json = html_escape_json(summary)
    bar_width = float(args.bar_width)
    max_height = float(args.max_bar_height)

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  {build_css(args.max_bar_height, args.bar_width)}
  <style>
  html, body {{
    height: 100%;
    margin: 0;
    overflow: hidden;
    background: #e5edf4;
  }}
  #offline-map {{
    display: block;
    height: 100vh;
    width: 100vw;
    background:
      linear-gradient(180deg, rgba(255,255,255,0.78), rgba(226,232,240,0.72)),
      radial-gradient(circle at 20% 25%, rgba(45, 212, 191, 0.18), transparent 32%),
      #dbeafe;
  }}
  .offline-frame {{
    fill: rgba(255, 255, 255, 0.44);
    stroke: rgba(51, 65, 85, 0.2);
    stroke-width: 1;
  }}
  .offline-grid-line {{
    stroke: rgba(51, 65, 85, 0.16);
    stroke-width: 1;
  }}
  .offline-label {{
    fill: #475569;
    font-family: "Avenir Next", "Gill Sans", sans-serif;
    font-size: 11px;
  }}
  .offline-sensor-dot {{
    fill: #0f172a;
    opacity: 0.8;
    stroke: white;
    stroke-width: 1.2;
  }}
  .offline-sensor-bar {{
    filter: drop-shadow(0 2px 3px rgba(15, 23, 42, 0.35));
    stroke: rgba(15, 23, 42, 0.35);
    stroke-width: 0.7;
  }}
  </style>
</head>
<body>
  <svg id="offline-map" width="100%" height="100%" preserveAspectRatio="none" aria-label="{html.escape(title)}">
    <g id="offline-grid"></g>
    <g id="offline-sensor-layer"></g>
  </svg>
  {build_control_html(map_id, title, len(time_labels))}
  <script>
  (function() {{
    const sensors = {sensor_json};
    const timeLabels = {labels_json};
    const summary = {summary_json};
    const minLat = {min_lat:.9f};
    const maxLat = {max_lat:.9f};
    const minLon = {min_lon:.9f};
    const maxLon = {max_lon:.9f};
    const barWidth = {bar_width:.6f};
    const maxHeight = {max_height:.6f};
    const svg = document.getElementById("offline-map");
    const gridLayer = document.getElementById("offline-grid");
    const sensorLayer = document.getElementById("offline-sensor-layer");
    const slider = document.getElementById("{map_id}-slider");
    const playButton = document.getElementById("{map_id}-play");
    const timeLabel = document.getElementById("{map_id}-time-label");
    const stepLabel = document.getElementById("{map_id}-step-label");
    const summaryLabel = document.getElementById("{map_id}-summary");
    const svgNS = "http://www.w3.org/2000/svg";
    const nodes = [];
    let currentStep = 0;
    let timer = null;

    function syncSvgViewport() {{
      const rect = svg.getBoundingClientRect();
      const width = Math.max(320, Math.round(rect.width || window.innerWidth || 960));
      const height = Math.max(240, Math.round(rect.height || window.innerHeight || 640));
      svg.setAttribute("width", width);
      svg.setAttribute("height", height);
      svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
      return [width, height];
    }}

    function colorForHeight(height) {{
      const ratio = Math.max(0, Math.min(1, height / Math.max(maxHeight, 1)));
      if (ratio < 0.5) {{
        return mixColor([34, 197, 94], [245, 158, 11], ratio / 0.5);
      }}
      return mixColor([245, 158, 11], [239, 68, 68], (ratio - 0.5) / 0.5);
    }}

    function mixColor(a, b, t) {{
      const r = Math.round(a[0] + (b[0] - a[0]) * t);
      const g = Math.round(a[1] + (b[1] - a[1]) * t);
      const bl = Math.round(a[2] + (b[2] - a[2]) * t);
      return `rgb(${{r}}, ${{g}}, ${{bl}})`;
    }}

    function project(lat, lon) {{
      const size = syncSvgViewport();
      const width = size[0];
      const height = size[1];
      const pad = Math.max(58, Math.min(width, height) * 0.07);
      const lonRange = Math.max(1e-9, maxLon - minLon);
      const latRange = Math.max(1e-9, maxLat - minLat);
      const x = pad + ((lon - minLon) / lonRange) * Math.max(1, width - 2 * pad);
      const y = pad + ((maxLat - lat) / latRange) * Math.max(1, height - 2 * pad);
      return [x, y, width, height, pad];
    }}

    function drawGrid() {{
      while (gridLayer.firstChild) {{
        gridLayer.removeChild(gridLayer.firstChild);
      }}
      const topLeft = project(maxLat, minLon);
      const bottomRight = project(minLat, maxLon);
      const x0 = topLeft[0];
      const y0 = topLeft[1];
      const x1 = bottomRight[0];
      const y1 = bottomRight[1];
      const frame = document.createElementNS(svgNS, "rect");
      frame.setAttribute("class", "offline-frame");
      frame.setAttribute("x", x0);
      frame.setAttribute("y", y0);
      frame.setAttribute("width", Math.max(1, x1 - x0));
      frame.setAttribute("height", Math.max(1, y1 - y0));
      frame.setAttribute("rx", "10");
      gridLayer.appendChild(frame);

      for (let i = 1; i < 5; i += 1) {{
        const tx = x0 + ((x1 - x0) * i) / 5;
        const ty = y0 + ((y1 - y0) * i) / 5;
        const v = document.createElementNS(svgNS, "line");
        v.setAttribute("class", "offline-grid-line");
        v.setAttribute("x1", tx);
        v.setAttribute("x2", tx);
        v.setAttribute("y1", y0);
        v.setAttribute("y2", y1);
        gridLayer.appendChild(v);
        const h = document.createElementNS(svgNS, "line");
        h.setAttribute("class", "offline-grid-line");
        h.setAttribute("x1", x0);
        h.setAttribute("x2", x1);
        h.setAttribute("y1", ty);
        h.setAttribute("y2", ty);
        gridLayer.appendChild(h);
      }}
    }}

    function valueText(sensor, step) {{
      const value = sensor.v[step];
      if (value === null || value === undefined || Number.isNaN(Number(value))) {{
        return "value: n/a";
      }}
      return `value: ${{value}}`;
    }}

    function updatePositions() {{
      syncSvgViewport();
      drawGrid();
      for (const item of nodes) {{
        const p = project(item.sensor.lat, item.sensor.lon);
        item.group.setAttribute("transform", `translate(${{p[0]}},${{p[1]}})`);
      }}
    }}

    function setStep(step) {{
      currentStep = Math.max(0, Math.min(timeLabels.length - 1, Number(step)));
      slider.value = currentStep;
      timeLabel.textContent = timeLabels[currentStep];
      stepLabel.textContent = `${{currentStep + 1}} / ${{timeLabels.length}}`;
      for (const item of nodes) {{
        const height = item.sensor.h[currentStep] || 0;
        item.rect.setAttribute("height", height);
        item.rect.setAttribute("y", -height - 3);
        item.rect.setAttribute("fill", colorForHeight(height));
        item.title.textContent = `${{item.sensor.id}} | ${{timeLabels[currentStep]}} | ${{valueText(item.sensor, currentStep)}}`;
      }}
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
        const nextStep = currentStep + 1 >= timeLabels.length ? 0 : currentStep + 1;
        setStep(nextStep);
      }}, {int(args.play_interval_ms)});
      playButton.textContent = "Pause";
    }}

    for (const sensor of sensors) {{
      const group = document.createElementNS(svgNS, "g");
      group.setAttribute("class", "offline-sensor");
      const rect = document.createElementNS(svgNS, "rect");
      rect.setAttribute("class", "offline-sensor-bar");
      rect.setAttribute("x", -barWidth / 2);
      rect.setAttribute("width", barWidth);
      rect.setAttribute("rx", Math.min(4, barWidth / 2));
      const dot = document.createElementNS(svgNS, "circle");
      dot.setAttribute("class", "offline-sensor-dot");
      dot.setAttribute("r", 3.2);
      const title = document.createElementNS(svgNS, "title");
      group.appendChild(title);
      group.appendChild(rect);
      group.appendChild(dot);
      sensorLayer.appendChild(group);
      nodes.push({{ sensor, group, rect, title }});
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
      updatePositions();
      setStep(currentStep);
    }});
    summaryLabel.textContent = `renderer=offline-svg, nodes=${{summary.rendered_nodes}}, steps=${{summary.num_steps}}, stride=${{summary.time_stride || 1}}`;
    updatePositions();
    setStep(0);
  }})();
  </script>
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(page, encoding="utf-8")


def render_folium_map(
    sensors: list[dict[str, Any]],
    time_labels: list[str],
    meta_df: pd.DataFrame,
    lat_column: str,
    lon_column: str,
    args: argparse.Namespace,
    output_html: Path,
    summary: dict[str, Any],
) -> None:
    if folium is None:
        raise RuntimeError("folium renderer requested but folium is not installed.")

    center_lat = float(meta_df[lat_column].mean())
    center_lon = float(meta_df[lon_column].mean())
    traffic_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=args.zoom_start,
        tiles=args.tiles,
        control_scale=True,
    )
    map_name = traffic_map.get_name()
    map_id = map_name.replace("_", "-")

    if not args.no_fit_bounds:
        bounds = [
            [float(meta_df[lat_column].min()), float(meta_df[lon_column].min())],
            [float(meta_df[lat_column].max()), float(meta_df[lon_column].max())],
        ]
        traffic_map.fit_bounds(bounds, padding=(18, 18))

    title = f"{summary['dataset']} traffic flow"
    traffic_map.get_root().header.add_child(folium.Element(build_css(args.max_bar_height, args.bar_width)))
    traffic_map.get_root().html.add_child(folium.Element(build_control_html(map_id, title, len(time_labels))))
    traffic_map.get_root().script.add_child(
        folium.Element(
            build_animation_js(
                map_name, map_id, sensors, time_labels, args, summary, include_script_tag=False
            )
        )
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    traffic_map.save(str(output_html))


def render_map(
    sensors: list[dict[str, Any]],
    time_labels: list[str],
    meta_df: pd.DataFrame,
    lat_column: str,
    lon_column: str,
    args: argparse.Namespace,
    output_html: Path,
    summary: dict[str, Any],
) -> None:
    if args.renderer == "offline":
        summary["renderer"] = "offline-svg"
        render_offline_svg_map(
            sensors, time_labels, meta_df, lat_column, lon_column, args, output_html, summary
        )
        return

    if args.renderer == "leaflet" or folium is None:
        summary["renderer"] = "standalone-leaflet"
        render_standalone_leaflet_map(
            sensors, time_labels, meta_df, lat_column, lon_column, args, output_html, summary
        )
        return

    summary["renderer"] = "folium"
    render_folium_map(sensors, time_labels, meta_df, lat_column, lon_column, args, output_html, summary)


def build_summary(
    args: argparse.Namespace,
    dataset_dir: Path,
    data_path: Path,
    shape: tuple[int, ...],
    start_index: int,
    start_mode: str,
    step_minutes: int,
    rendered_nodes: int,
    valid_nodes: int,
    value_scale: float,
    reference_value: float,
    output_html: Path,
) -> dict[str, Any]:
    return {
        "dataset": args.dataset,
        "dataset_dir": str(dataset_dir),
        "data_path": str(data_path),
        "data_shape": list(shape),
        "channel": int(args.channel),
        "start_index": int(start_index),
        "start_mode": start_mode,
        "num_steps": int(args.num_steps),
        "step_minutes": int(step_minutes),
        "valid_nodes": int(valid_nodes),
        "rendered_nodes": int(rendered_nodes),
        "value_scale": float(value_scale),
        "scale_reference_value": float(reference_value),
        "max_bar_height": float(args.max_bar_height),
        "output_html": str(output_html),
    }


def main() -> None:
    args = parse_args()
    dataset_dir = resolve_dataset_dir(args.dataset, args.dataset_dir)
    meta_path = dataset_dir / "meta.csv"
    data_path = Path(args.data_file).expanduser().resolve() if args.data_file else dataset_dir / "data.dat"
    desc = read_description(dataset_dir)

    if not meta_path.exists():
        raise FileNotFoundError(f"meta.csv not found: {meta_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"data.dat not found: {data_path}")

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
    value_scale, reference_value = compute_scale(values, args)
    values_by_node, heights_by_node = prepare_bar_arrays(
        values,
        value_scale,
        args.min_bar_height,
        args.max_bar_height,
        args.value_precision,
    )
    sensors = build_sensor_payload(
        meta_df,
        values_by_node,
        heights_by_node,
        id_column,
        lat_column,
        lon_column,
    )
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
        else dataset_dir / "traffic_map_bar_animation.html"
    )
    summary = build_summary(
        args,
        dataset_dir,
        data_path,
        shape,
        start_index,
        start_mode,
        step_minutes,
        len(sensors),
        valid_node_count,
        value_scale,
        reference_value,
        output_html,
    )
    summary["id_column"] = id_column
    summary["lat_column"] = lat_column
    summary["lon_column"] = lon_column
    summary["time_stride"] = int(args.time_stride)
    summary["effective_step_minutes"] = int(step_minutes * args.time_stride)

    render_map(sensors, time_labels, meta_df, lat_column, lon_column, args, output_html, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
