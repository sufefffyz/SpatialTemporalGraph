"""Plot road segments on an OSM basemap and color them by speed for a given day.

This script is designed for the Urban Traffic Benchmark raw files:
  - <city_prefix>_static_features.parquet
  - <city_prefix>_raw_speed.parquet
  - city_traffic_<l|m>_volume.npz (for unix_timestamps)

Example:
  python scripts/plot_speed_on_osm.py \
      --data-dir /data/yuzhang_fei/Urban_Traffic_Benchmark \
      --city-prefix city_M \
      --day 2019-07-01 \
      --output-html scripts/city_M_speed_2019-07-01.html
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CoordColumns:
    x_start: str
    y_start: str
    x_end: str
    y_end: str


@dataclass
class OsrmStats:
    requests: int = 0
    cache_hits: int = 0
    misses: int = 0
    failures: int = 0


def _iter_chunks(items: list[str], chunk_size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def _find_first_by_keywords(columns: list[str], include_all: list[str]) -> str | None:
    for col in columns:
        low = col.lower()
        if all(k in low for k in include_all):
            return col
    return None


def detect_coordinate_columns(df: pd.DataFrame) -> CoordColumns:
    cols = df.columns.tolist()

    # Try robust keyword-based detection first.
    x_start = _find_first_by_keywords(cols, ["x", "start"]) or _find_first_by_keywords(cols, ["lon", "start"])
    y_start = _find_first_by_keywords(cols, ["y", "start"]) or _find_first_by_keywords(cols, ["lat", "start"])
    x_end = _find_first_by_keywords(cols, ["x", "end"]) or _find_first_by_keywords(cols, ["lon", "end"])
    y_end = _find_first_by_keywords(cols, ["y", "end"]) or _find_first_by_keywords(cols, ["lat", "end"])

    if all(v is not None for v in [x_start, y_start, x_end, y_end]):
        return CoordColumns(x_start=x_start, y_start=y_start, x_end=x_end, y_end=y_end)

    # Fallback: according to the dataset conversion script, the last 4 fields are
    # [x_start, y_start, x_end, y_end].
    if len(cols) >= 4:
        return CoordColumns(x_start=cols[-4], y_start=cols[-3], x_end=cols[-2], y_end=cols[-1])

    raise ValueError("Unable to detect coordinate columns from static features.")


def get_city_short_name(city_prefix: str) -> str:
    low = city_prefix.lower()
    if low.endswith("_l"):
        return "l"
    if low.endswith("_m"):
        return "m"
    raise ValueError("city_prefix must end with _L or _M, e.g. city_L or city_M.")


def load_unix_timestamps(data_dir: Path, city_prefix: str) -> np.ndarray:
    city_short = get_city_short_name(city_prefix)

    candidates = [
        data_dir / f"city_traffic_{city_short}_volume.npz",
        data_dir / f"city_traffic_{city_short}_speed.npz",
    ]

    for path in candidates:
        if not path.exists():
            continue
        try:
            with np.load(path, allow_pickle=False) as data:
                return data["unix_timestamps"]
        except Exception as exc:
            print(f"Warning: failed to read unix_timestamps from {path.name}: {exc}")

    raise FileNotFoundError("Could not load unix_timestamps from companion NPZ files.")


def compute_daily_speed_means(
    speed_parquet_path: Path,
    unix_timestamps: np.ndarray,
    day: str,
    node_ids: np.ndarray,
    chunk_cols: int = 512,
) -> dict[int, float]:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    day_start = pd.Timestamp(day)
    day_end = day_start + pd.Timedelta(days=1)

    ts = pd.to_datetime(unix_timestamps, unit="s", utc=False)
    mask = (ts >= day_start) & (ts < day_end)
    mask_count = int(mask.sum())

    if mask_count == 0:
        raise ValueError(f"No timestamps found for day={day}.")

    print(f"Matched {mask_count} timestamps for day {day}.")

    pf = pq.ParquetFile(speed_parquet_path)
    available_cols = set(pf.schema_arrow.names)

    node_cols = [f"node_{int(i)}" for i in node_ids if f"node_{int(i)}" in available_cols]
    if not node_cols:
        raise ValueError("No matching node_* columns found in speed parquet for the static node ids.")

    print(f"Computing daily means for {len(node_cols)} road segments...")

    mask_pa = pa.array(mask.tolist())
    means: dict[int, float] = {}

    done = 0
    for chunk in _iter_chunks(node_cols, chunk_cols):
        table = pq.read_table(speed_parquet_path, columns=chunk)

        for col in chunk:
            filtered = pc.filter(table[col], mask_pa)
            mean_val = pc.mean(filtered).as_py()
            if mean_val is not None:
                means[int(col.split("_")[1])] = float(mean_val)

        done += len(chunk)
        print(f"  processed {done}/{len(node_cols)} columns")

    return means


def load_osrm_cache(cache_file: Path) -> dict[str, list[list[float]]]:
    if not cache_file.exists():
        return {}
    try:
        with cache_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as exc:
        print(f"Warning: failed to read OSRM cache {cache_file}: {exc}")
    return {}


def save_osrm_cache(cache_file: Path, cache: dict[str, list[list[float]]]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("w", encoding="utf-8") as f:
        json.dump(cache, f)


def _osrm_key(start_lat: float, start_lon: float, end_lat: float, end_lon: float) -> str:
    return f"{start_lat:.6f},{start_lon:.6f}->{end_lat:.6f},{end_lon:.6f}"


def fetch_osrm_geometry(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    osrm_url: str,
    timeout_sec: float,
    cache: dict[str, list[list[float]]],
    stats: OsrmStats,
) -> list[list[float]] | None:
    key = _osrm_key(start_lat, start_lon, end_lat, end_lon)
    if key in cache:
        stats.cache_hits += 1
        return cache[key]

    params = urlencode(
        {
            "overview": "full",
            "geometries": "geojson",
            "steps": "false",
            "alternatives": "false",
        }
    )
    base = osrm_url.rstrip("/")
    route = f"{base}/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?{params}"

    try:
        with urlopen(route, timeout=timeout_sec) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        routes = payload.get("routes") or []
        if not routes:
            stats.failures += 1
            return None
        coords = routes[0]["geometry"]["coordinates"]
        latlon = [[float(lat), float(lon)] for lon, lat in coords]
        cache[key] = latlon
        stats.requests += 1
        stats.misses += 1
        return latlon
    except Exception:
        stats.failures += 1
        return None


def build_map(
    static_df: pd.DataFrame,
    coords: CoordColumns,
    speed_mean_by_node: dict[int, float],
    output_html: Path,
    max_segments: int,
    line_weight: float,
    endpoint_markers: bool,
    endpoint_marker_radius: float,
    endpoint_marker_opacity: float,
    max_endpoint_markers: int,
    use_osrm_geometry: bool,
    osrm_url: str,
    osrm_timeout_sec: float,
    max_osrm_requests: int,
    osrm_cache_file: Path,
    allow_straight_fallback: bool,
) -> None:
    import folium
    from branca.colormap import LinearColormap

    static_df = static_df.copy()
    static_df["node_id"] = np.arange(len(static_df), dtype=int)
    static_df["speed_mean"] = static_df["node_id"].map(speed_mean_by_node)
    static_df = static_df[static_df["speed_mean"].notna()]

    if static_df.empty:
        raise ValueError("No segments have speed values after matching.")

    if len(static_df) > max_segments:
        static_df = static_df.sample(n=max_segments, random_state=42)
        print(f"Sampling {max_segments} segments for visualization.")

    for col in [coords.x_start, coords.y_start, coords.x_end, coords.y_end]:
        static_df[col] = pd.to_numeric(static_df[col], errors="coerce")

    static_df = static_df.dropna(subset=[coords.x_start, coords.y_start, coords.x_end, coords.y_end])

    if static_df.empty:
        raise ValueError("No valid geometries after coordinate cleaning.")

    center_lat = float(((static_df[coords.y_start] + static_df[coords.y_end]) / 2).mean())
    center_lon = float(((static_df[coords.x_start] + static_df[coords.x_end]) / 2).mean())

    vmin = float(static_df["speed_mean"].quantile(0.02))
    vmax = float(static_df["speed_mean"].quantile(0.98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(static_df["speed_mean"].min())
        vmax = float(static_df["speed_mean"].max())

    cmap = LinearColormap(
        colors=["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"],
        vmin=vmin,
        vmax=vmax,
        caption="Average speed",
    )

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="OpenStreetMap", control_scale=True)

    osrm_cache = load_osrm_cache(osrm_cache_file) if use_osrm_geometry else {}
    osrm_stats = OsrmStats()

    for i, (_, row) in enumerate(static_df.iterrows(), start=1):
        val = float(row["speed_mean"])
        start_lat = float(row[coords.y_start])
        start_lon = float(row[coords.x_start])
        end_lat = float(row[coords.y_end])
        end_lon = float(row[coords.x_end])
        color = cmap(val)

        route_locations: list[list[float]] | None = None
        if use_osrm_geometry and osrm_stats.requests < max_osrm_requests:
            route_locations = fetch_osrm_geometry(
                start_lat=start_lat,
                start_lon=start_lon,
                end_lat=end_lat,
                end_lon=end_lon,
                osrm_url=osrm_url,
                timeout_sec=osrm_timeout_sec,
                cache=osrm_cache,
                stats=osrm_stats,
            )

        if route_locations is None:
            if not allow_straight_fallback and use_osrm_geometry:
                continue
            route_locations = [[start_lat, start_lon], [end_lat, end_lon]]

        folium.PolyLine(
            locations=route_locations,
            color=color,
            weight=line_weight,
            opacity=0.85,
            tooltip=f"node_{int(row['node_id'])}: speed={val:.2f}",
        ).add_to(fmap)

        if use_osrm_geometry and i % 500 == 0:
            print(
                f"  OSRM progress {i}/{len(static_df)} | "
                f"requests={osrm_stats.requests}, cache_hits={osrm_stats.cache_hits}, failures={osrm_stats.failures}"
            )

    if endpoint_markers:
        plotted_points = 0
        for _, row in static_df.iterrows():
            if plotted_points >= max_endpoint_markers:
                break

            val = float(row["speed_mean"])
            color = cmap(val)
            endpoints = [
                (float(row[coords.y_start]), float(row[coords.x_start])),
                (float(row[coords.y_end]), float(row[coords.x_end])),
            ]

            for lat, lon in endpoints:
                if plotted_points >= max_endpoint_markers:
                    break

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=endpoint_marker_radius,
                    color="#111111",
                    weight=0.6,
                    opacity=endpoint_marker_opacity,
                    fill=True,
                    fill_color=color,
                    fill_opacity=endpoint_marker_opacity,
                ).add_to(fmap)
                plotted_points += 1

        print(f"Plotted endpoint markers: {plotted_points}")

    if use_osrm_geometry:
        save_osrm_cache(osrm_cache_file, osrm_cache)
        print(
            f"OSRM summary: requests={osrm_stats.requests}, cache_hits={osrm_stats.cache_hits}, "
            f"failures={osrm_stats.failures}, cache_size={len(osrm_cache)}"
        )
        print(f"Saved OSRM cache: {osrm_cache_file}")

    cmap.add_to(fmap)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(output_html))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot daily speed on OSM road segments.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory that stores benchmark files.")
    parser.add_argument("--city-prefix", type=str, required=True, help="city_L or city_M")
    parser.add_argument("--day", type=str, required=True, help="Day in YYYY-MM-DD")
    parser.add_argument("--speed-parquet", type=Path, default=None, help="Override speed parquet path")
    parser.add_argument("--static-parquet", type=Path, default=None, help="Override static features parquet path")
    parser.add_argument("--output-html", type=Path, default=None, help="Output html map path")
    parser.add_argument("--max-segments", type=int, default=25000, help="Maximum segments drawn")
    parser.add_argument("--chunk-cols", type=int, default=512, help="How many node columns to process per chunk")
    parser.add_argument("--line-weight", type=float, default=2.0, help="Polyline width")
    parser.add_argument(
        "--use-osrm-geometry",
        action="store_true",
        help="Use OSRM route geometry instead of straight line segments.",
    )
    parser.add_argument(
        "--osrm-url",
        type=str,
        default="http://127.0.0.1:5000",
        help="OSRM HTTP service base URL.",
    )
    parser.add_argument(
        "--osrm-timeout-sec",
        type=float,
        default=2.5,
        help="Timeout in seconds for each OSRM route request.",
    )
    parser.add_argument(
        "--max-osrm-requests",
        type=int,
        default=20000,
        help="Maximum number of OSRM route requests per run.",
    )
    parser.add_argument(
        "--osrm-cache-file",
        type=Path,
        default=None,
        help="Path to JSON cache for route geometries.",
    )
    parser.add_argument(
        "--no-straight-fallback",
        action="store_true",
        help="If set, skip segments when OSRM route is unavailable instead of drawing straight fallback.",
    )
    parser.add_argument(
        "--endpoint-markers",
        action="store_true",
        help="Draw a circle marker at both endpoints of each road segment.",
    )
    parser.add_argument(
        "--endpoint-marker-radius",
        type=float,
        default=1.6,
        help="Circle marker radius for endpoints.",
    )
    parser.add_argument(
        "--endpoint-marker-opacity",
        type=float,
        default=0.8,
        help="Opacity for endpoint markers (0 to 1).",
    )
    parser.add_argument(
        "--max-endpoint-markers",
        type=int,
        default=100000,
        help="Maximum number of endpoint markers to draw.",
    )
    args = parser.parse_args()

    speed_parquet = args.speed_parquet or (args.data_dir / f"{args.city_prefix}_raw_speed.parquet")
    static_parquet = args.static_parquet or (args.data_dir / f"{args.city_prefix}_static_features.parquet")

    if args.output_html is None:
        output_html = Path(__file__).resolve().parent / f"{args.city_prefix}_speed_{args.day}.html"
    else:
        output_html = args.output_html

    if args.osrm_cache_file is None:
        osrm_cache_file = Path(__file__).resolve().parent / f"{args.city_prefix}_osrm_cache.json"
    else:
        osrm_cache_file = args.osrm_cache_file

    if not speed_parquet.exists():
        raise FileNotFoundError(f"Speed parquet not found: {speed_parquet}")
    if not static_parquet.exists():
        raise FileNotFoundError(f"Static parquet not found: {static_parquet}")

    print(f"Reading static features: {static_parquet}")
    static_df = pd.read_parquet(static_parquet)
    coords = detect_coordinate_columns(static_df)
    print(f"Using coordinate columns: {coords}")

    print("Loading unix_timestamps...")
    unix_timestamps = load_unix_timestamps(args.data_dir, args.city_prefix)

    node_ids = np.arange(len(static_df), dtype=int)

    print(f"Reading speed parquet in chunks: {speed_parquet}")
    speed_means = compute_daily_speed_means(
        speed_parquet_path=speed_parquet,
        unix_timestamps=unix_timestamps,
        day=args.day,
        node_ids=node_ids,
        chunk_cols=args.chunk_cols,
    )

    print("Building OSM map...")
    build_map(
        static_df=static_df,
        coords=coords,
        speed_mean_by_node=speed_means,
        output_html=output_html,
        max_segments=args.max_segments,
        line_weight=args.line_weight,
        endpoint_markers=args.endpoint_markers,
        endpoint_marker_radius=args.endpoint_marker_radius,
        endpoint_marker_opacity=args.endpoint_marker_opacity,
        max_endpoint_markers=args.max_endpoint_markers,
        use_osrm_geometry=args.use_osrm_geometry,
        osrm_url=args.osrm_url,
        osrm_timeout_sec=args.osrm_timeout_sec,
        max_osrm_requests=args.max_osrm_requests,
        osrm_cache_file=osrm_cache_file,
        allow_straight_fallback=not args.no_straight_fallback,
    )

    print(f"Done. Saved map to: {output_html}")


if __name__ == "__main__":
    main()
