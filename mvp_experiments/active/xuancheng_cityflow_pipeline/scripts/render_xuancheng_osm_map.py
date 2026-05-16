#!/usr/bin/env python3
"""Render Xuancheng CityFlow roads on an OpenStreetMap Leaflet basemap."""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


DEFAULT_OSM_TILES = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
DEFAULT_OSM_ATTRIBUTION = (
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roadnet", required=True, help="CityFlow roadnet JSON.")
    parser.add_argument("--output-html", required=True, help="Output interactive HTML path.")
    parser.add_argument(
        "--sumo-net",
        default=None,
        help=(
            "Optional SUMO net XML. Required for Xuancheng because the CityFlow roadnet "
            "uses local SUMO coordinates rather than lon/lat."
        ),
    )
    parser.add_argument(
        "--npz",
        default=None,
        help="Optional road aggregation NPZ from run_cityflow_*_road_aggregation.py.",
    )
    parser.add_argument(
        "--feature",
        default="entered_veh",
        help="Feature to visualize when --npz is provided. Default: entered_veh.",
    )
    parser.add_argument(
        "--time-agg",
        choices=["auto", "sum", "mean", "max", "bucket"],
        default="auto",
        help=(
            "How to collapse time when --npz is provided. Auto uses sum for entered/exited "
            "counts and mean otherwise. Default: auto."
        ),
    )
    parser.add_argument(
        "--bucket-index",
        type=int,
        default=0,
        help="Bucket to visualize when --time-agg=bucket. Default: 0.",
    )
    parser.add_argument(
        "--hide-zero",
        action="store_true",
        help="Hide roads whose selected value is zero or missing.",
    )
    parser.add_argument(
        "--coord-mode",
        choices=["auto", "lonlat", "latlon", "sumo"],
        default="auto",
        help="Interpret CityFlow point x/y coordinates. Default: auto.",
    )
    parser.add_argument("--line-weight-min", type=float, default=1.4)
    parser.add_argument("--line-weight-max", type=float, default=7.0)
    parser.add_argument("--line-opacity", type=float, default=0.78)
    parser.add_argument("--zoom-start", type=int, default=12)
    parser.add_argument("--tiles", default=DEFAULT_OSM_TILES, help="Leaflet tile URL template.")
    parser.add_argument("--title", default="Xuancheng CityFlow roads on OpenStreetMap")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def road_length_m(points: list[dict[str, Any]]) -> float:
    total = 0.0
    for p0, p1 in zip(points, points[1:]):
        total += math.hypot(float(p0["x"]) - float(p1["x"]), float(p0["y"]) - float(p1["y"]))
    return total


def infer_coord_mode(roads: list[dict[str, Any]]) -> str:
    xs: list[float] = []
    ys: list[float] = []
    for road in roads:
        for point in road.get("points", []):
            try:
                xs.append(float(point["x"]))
                ys.append(float(point["y"]))
            except (KeyError, TypeError, ValueError):
                continue
    if not xs or not ys:
        return "lonlat"
    x_abs = max(abs(value) for value in xs)
    y_abs = max(abs(value) for value in ys)
    if x_abs <= 90.0 and y_abs <= 180.0:
        return "latlon"
    if x_abs <= 180.0 and y_abs <= 90.0:
        return "lonlat"
    return "sumo"


def parse_pair(text: str) -> tuple[float, float]:
    left, right = text.split(",", maxsplit=1)
    return float(left), float(right)


def parse_proj_parameter(proj: str) -> dict[str, str | bool]:
    values: dict[str, str | bool] = {}
    for part in proj.split():
        if not part.startswith("+"):
            continue
        body = part[1:]
        if "=" in body:
            key, value = body.split("=", maxsplit=1)
            values[key] = value
        else:
            values[body] = True
    return values


def utm_to_lonlat(
    easting: float,
    northing: float,
    zone: int,
    northern: bool = True,
) -> tuple[float, float]:
    """Convert WGS84 UTM meters to lon/lat degrees.

    This keeps the renderer dependency-light on the server. It is used only for
    the released Xuancheng SUMO projection (+proj=utm +zone=50 +datum=WGS84).
    """

    if not northern:
        northing -= 10_000_000.0

    a = 6378137.0
    f = 1 / 298.257223563
    k0 = 0.9996
    e = math.sqrt(f * (2 - f))
    e1sq = e * e / (1 - e * e)
    x = easting - 500000.0
    y = northing

    m = y / k0
    mu = m / (a * (1 - e**2 / 4 - 3 * e**4 / 64 - 5 * e**6 / 256))
    e1 = (1 - math.sqrt(1 - e * e)) / (1 + math.sqrt(1 - e * e))

    j1 = 3 * e1 / 2 - 27 * e1**3 / 32
    j2 = 21 * e1**2 / 16 - 55 * e1**4 / 32
    j3 = 151 * e1**3 / 96
    j4 = 1097 * e1**4 / 512
    fp = mu + j1 * math.sin(2 * mu) + j2 * math.sin(4 * mu) + j3 * math.sin(6 * mu) + j4 * math.sin(8 * mu)

    sin_fp = math.sin(fp)
    cos_fp = math.cos(fp)
    tan_fp = math.tan(fp)
    c1 = e1sq * cos_fp**2
    t1 = tan_fp**2
    n1 = a / math.sqrt(1 - e**2 * sin_fp**2)
    r1 = a * (1 - e**2) / (1 - e**2 * sin_fp**2) ** 1.5
    d = x / (n1 * k0)

    lat = fp - (n1 * tan_fp / r1) * (
        d**2 / 2
        - (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * e1sq) * d**4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 252 * e1sq - 3 * c1**2) * d**6 / 720
    )
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    lon = lon0 + (
        d
        - (1 + 2 * t1 + c1) * d**3 / 6
        + (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * e1sq + 24 * t1**2) * d**5 / 120
    ) / cos_fp

    return math.degrees(lon), math.degrees(lat)


class CoordinateTransformer:
    def __init__(self, mode: str, sumo_net: Path | None = None) -> None:
        self.mode = mode
        self.net_offset: tuple[float, float] | None = None
        self.utm_zone: int | None = None
        self.utm_northern = True
        if mode == "sumo":
            if sumo_net is None:
                raise SystemExit(
                    "local SUMO coordinates detected; pass --sumo-net /path/to/xuancheng.net.xml "
                    "so coordinates can be converted to lon/lat for OpenStreetMap"
                )
            self._load_sumo_location(sumo_net)

    def _load_sumo_location(self, sumo_net: Path) -> None:
        root = ET.parse(sumo_net).getroot()
        location = root.find("location")
        if location is None:
            raise SystemExit(f"{sumo_net} does not contain a SUMO <location> element")
        net_offset = location.attrib.get("netOffset")
        proj_parameter = location.attrib.get("projParameter", "")
        if not net_offset:
            raise SystemExit(f"{sumo_net} location is missing netOffset")
        proj_values = parse_proj_parameter(proj_parameter)
        if proj_values.get("proj") != "utm" or "zone" not in proj_values:
            raise SystemExit(
                f"unsupported SUMO projection {proj_parameter!r}; install pyproj support or "
                "pre-convert the roadnet to lon/lat"
            )
        self.net_offset = parse_pair(net_offset)
        self.utm_zone = int(str(proj_values["zone"]))
        self.utm_northern = "south" not in proj_values

    def point_to_lonlat(self, point: dict[str, Any]) -> tuple[float, float]:
        x = float(point["x"])
        y = float(point["y"])
        if self.mode == "latlon":
            return y, x
        if self.mode == "lonlat":
            return x, y
        if self.net_offset is None or self.utm_zone is None:
            raise AssertionError("SUMO transformer is not initialized")
        # SUMO internal coordinate = projected coordinate + netOffset.
        easting = x - self.net_offset[0]
        northing = y - self.net_offset[1]
        return utm_to_lonlat(easting, northing, self.utm_zone, self.utm_northern)


def load_npz_values(
    npz_path: Path,
    feature: str,
    time_agg: str,
    bucket_index: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise SystemExit("numpy is required when --npz is provided") from exc

    archive = np.load(npz_path, allow_pickle=False)
    required = {"data", "road_ids", "feature_names"}
    missing = required.difference(archive.files)
    if missing:
        raise SystemExit(f"{npz_path} is missing required keys: {sorted(missing)}")

    data = archive["data"]
    if data.ndim != 3:
        raise SystemExit(f"expected NPZ data with shape T,N,F; got {data.shape}")

    road_ids = [str(value) for value in archive["road_ids"].tolist()]
    feature_names = [str(value) for value in archive["feature_names"].tolist()]
    if feature not in feature_names:
        raise SystemExit(f"feature {feature!r} not found; available features: {feature_names}")
    feature_index = feature_names.index(feature)

    resolved_agg = time_agg
    if resolved_agg == "auto":
        resolved_agg = "sum" if feature in {"entered_veh", "exited_veh"} else "mean"

    feature_tensor = data[:, :, feature_index].astype("float64")
    if resolved_agg == "bucket":
        if bucket_index < 0 or bucket_index >= feature_tensor.shape[0]:
            raise SystemExit(
                f"--bucket-index {bucket_index} is outside NPZ time dimension {feature_tensor.shape[0]}"
            )
        values = feature_tensor[bucket_index]
        time_label = f"bucket {bucket_index}"
    elif resolved_agg == "sum":
        values = np.nansum(feature_tensor, axis=0)
        time_label = "all buckets, sum"
    elif resolved_agg == "mean":
        values = np.nanmean(feature_tensor, axis=0)
        time_label = "all buckets, mean"
    elif resolved_agg == "max":
        values = np.nanmax(feature_tensor, axis=0)
        time_label = "all buckets, max"
    else:
        raise AssertionError(resolved_agg)

    bucket_start = archive["bucket_start_s"].tolist() if "bucket_start_s" in archive.files else []
    bucket_end = archive["bucket_end_s"].tolist() if "bucket_end_s" in archive.files else []
    values_by_road = {road_id: float(value) for road_id, value in zip(road_ids, values, strict=True)}
    metadata = {
        "npz": str(npz_path),
        "shape": list(data.shape),
        "feature": feature,
        "time_agg": resolved_agg,
        "time_label": time_label,
        "bucket_index": bucket_index if resolved_agg == "bucket" else None,
        "bucket_start_s": bucket_start[bucket_index] if resolved_agg == "bucket" and bucket_start else None,
        "bucket_end_s": bucket_end[bucket_index] if resolved_agg == "bucket" and bucket_end else None,
    }
    return values_by_road, metadata


def finite_positive(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value) and value > 0]


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    q = min(max(q, 0.0), 100.0)
    pos = (len(sorted_values) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def build_color_breaks(values: list[float]) -> list[float]:
    positives = sorted(finite_positive(values))
    if not positives:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    return [percentile(positives, q) for q in (20.0, 40.0, 60.0, 80.0, 95.0)]


def color_for_value(value: float | None, breaks: list[float]) -> str:
    if value is None or not math.isfinite(value) or value <= 0:
        return "#64748b"
    palette = ["#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51", "#b91c1c"]
    for index, threshold in enumerate(breaks):
        if value <= threshold:
            return palette[index]
    return palette[-1]


def weight_for_value(value: float | None, max_value: float, min_weight: float, max_weight: float) -> float:
    if value is None or not math.isfinite(value) or value <= 0 or max_value <= 0:
        return min_weight
    ratio = min(max(value / max_value, 0.0), 1.0)
    return min_weight + (max_weight - min_weight) * math.sqrt(ratio)


def build_geojson(
    roadnet: dict[str, Any],
    values_by_road: dict[str, float] | None,
    coord_mode: str,
    sumo_net: Path | None,
    hide_zero: bool,
    min_weight: float,
    max_weight: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    roads = roadnet.get("roads", [])
    resolved_mode = infer_coord_mode(roads) if coord_mode == "auto" else coord_mode
    transformer = CoordinateTransformer(resolved_mode, sumo_net)
    all_values = list(values_by_road.values()) if values_by_road else []
    max_value = max(finite_positive(all_values), default=0.0)
    breaks = build_color_breaks(all_values)
    features = []
    bounds: list[tuple[float, float]] = []

    for road in roads:
        road_id = str(road["id"])
        points = road.get("points", [])
        if len(points) < 2:
            continue
        value = values_by_road.get(road_id) if values_by_road is not None else None
        if hide_zero and (value is None or not math.isfinite(value) or value <= 0):
            continue

        coordinates: list[list[float]] = []
        for point in points:
            try:
                lon, lat = transformer.point_to_lonlat(point)
            except (KeyError, TypeError, ValueError):
                continue
            if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
                continue
            coordinates.append([round(lon, 7), round(lat, 7)])
            bounds.append((lat, lon))
        if len(coordinates) < 2:
            continue

        lanes = road.get("lanes", [])
        speed_limit_kmh = 0.0
        if lanes:
            speed_limit_kmh = sum(float(lane.get("maxSpeed", 0.0)) for lane in lanes) / len(lanes) * 3.6
        properties = {
            "road_id": road_id,
            "value": round(value, 6) if value is not None and math.isfinite(value) else None,
            "color": color_for_value(value, breaks) if values_by_road is not None else "#2563eb",
            "weight": round(weight_for_value(value, max_value, min_weight, max_weight), 3)
            if values_by_road is not None
            else min_weight,
            "lane_count": len(lanes),
            "length_m": round(road_length_m(points), 3),
            "speed_limit_kmh": round(speed_limit_kmh, 3),
        }
        features.append(
            {
                "type": "Feature",
                "properties": properties,
                "geometry": {"type": "LineString", "coordinates": coordinates},
            }
        )

    if not bounds:
        raise SystemExit("no road geometries with valid lon/lat coordinates were found")

    lats = [lat for lat, _ in bounds]
    lons = [lon for _, lon in bounds]
    summary = {
        "roads_in_roadnet": len(roads),
        "roads_rendered": len(features),
        "coord_mode": resolved_mode,
        "sumo_net": str(sumo_net) if sumo_net is not None else None,
        "bounds": [[min(lats), min(lons)], [max(lats), max(lons)]],
        "value_breaks": breaks,
        "max_value": max_value,
    }
    return {"type": "FeatureCollection", "features": features}, summary


def html_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def build_html(
    title: str,
    geojson: dict[str, Any],
    summary: dict[str, Any],
    value_metadata: dict[str, Any] | None,
    tiles: str,
    zoom_start: int,
    line_opacity: float,
) -> str:
    safe_title = html.escape(title)
    map_summary = dict(summary)
    if value_metadata is not None:
        map_summary.update(value_metadata)
    legend_breaks = summary.get("value_breaks", [])
    legend_text = " | ".join(f"{value:.2f}" for value in legend_breaks if math.isfinite(float(value)))
    if not legend_text:
        legend_text = "roadnet only"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{safe_title}</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    html, body, #map {{
      height: 100%;
      margin: 0;
      width: 100%;
    }}
    body {{
      color: #172554;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .panel {{
      background: rgba(255, 255, 255, 0.94);
      border: 1px solid rgba(15, 23, 42, 0.15);
      border-radius: 8px;
      box-shadow: 0 10px 28px rgba(15, 23, 42, 0.17);
      left: 14px;
      max-width: 370px;
      padding: 12px 14px;
      position: fixed;
      top: 14px;
      z-index: 1000;
    }}
    .panel h1 {{
      font-size: 16px;
      line-height: 1.25;
      margin: 0 0 6px;
    }}
    .panel .meta {{
      color: #475569;
      font-size: 12px;
      line-height: 1.45;
    }}
    .legend {{
      bottom: 24px;
      left: 14px;
      position: fixed;
      z-index: 1000;
    }}
    .legend-inner {{
      background: rgba(255, 255, 255, 0.94);
      border: 1px solid rgba(15, 23, 42, 0.15);
      border-radius: 8px;
      box-shadow: 0 8px 22px rgba(15, 23, 42, 0.14);
      color: #334155;
      font-size: 12px;
      padding: 10px 12px;
    }}
    .swatches {{
      display: grid;
      gap: 3px;
      grid-template-columns: repeat(6, 28px);
      margin: 7px 0 5px;
    }}
    .swatches span {{
      height: 8px;
    }}
    @media (max-width: 680px) {{
      .panel {{
        left: 10px;
        right: 10px;
        max-width: none;
      }}
      .legend {{
        left: 10px;
        right: 10px;
      }}
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="panel">
    <h1>{safe_title}</h1>
    <div class="meta" id="map-meta">loading...</div>
  </div>
  <div class="legend">
    <div class="legend-inner">
      <div><b>{html.escape(str(map_summary.get("feature", "road geometry")))}</b></div>
      <div class="swatches">
        <span style="background:#2a9d8f"></span>
        <span style="background:#8ab17d"></span>
        <span style="background:#e9c46a"></span>
        <span style="background:#f4a261"></span>
        <span style="background:#e76f51"></span>
        <span style="background:#b91c1c"></span>
      </div>
      <div>breaks: {html.escape(legend_text)}</div>
    </div>
  </div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const roadData = {html_json(geojson)};
    const summary = {html_json(map_summary)};
    const map = L.map("map", {{ zoomControl: true }});
    L.tileLayer("{tiles}", {{
      maxZoom: 19,
      attribution: `{DEFAULT_OSM_ATTRIBUTION}`
    }}).addTo(map);

    function popupHtml(properties) {{
      const value = properties.value === null || properties.value === undefined ? "n/a" : properties.value;
      return `
        <b>${{properties.road_id}}</b><br>
        value: ${{value}}<br>
        lanes: ${{properties.lane_count}}<br>
        length: ${{properties.length_m}} m<br>
        speed limit: ${{properties.speed_limit_kmh}} km/h
      `;
    }}

    const roads = L.geoJSON(roadData, {{
      style: function(feature) {{
        return {{
          color: feature.properties.color,
          opacity: {line_opacity:.4f},
          weight: feature.properties.weight,
          lineCap: "round",
          lineJoin: "round"
        }};
      }},
      onEachFeature: function(feature, layer) {{
        layer.bindPopup(popupHtml(feature.properties), {{ maxWidth: 260 }});
        layer.bindTooltip(feature.properties.road_id, {{ sticky: true, opacity: 0.88 }});
        layer.on("mouseover", function() {{
          layer.setStyle({{ weight: Math.max(feature.properties.weight + 2, 4), opacity: 1.0 }});
        }});
        layer.on("mouseout", function() {{
          roads.resetStyle(layer);
        }});
      }}
    }}).addTo(map);

    const bounds = L.latLngBounds(summary.bounds);
    map.fitBounds(bounds, {{ padding: [22, 22], maxZoom: {int(zoom_start)} }});
    document.getElementById("map-meta").textContent =
      `rendered=${{summary.roads_rendered}}/${{summary.roads_in_roadnet}} roads, ` +
      `coord=${{summary.coord_mode}}, ` +
      `feature=${{summary.feature || "none"}}, time=${{summary.time_label || "roadnet only"}}`;
  </script>
</body>
</html>
"""


def write_summary(output_html: Path, summary: dict[str, Any], value_metadata: dict[str, Any] | None) -> Path:
    summary_path = output_html.with_suffix(output_html.suffix + ".summary.json")
    payload = dict(summary)
    if value_metadata is not None:
        payload.update(value_metadata)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return summary_path


def main() -> int:
    args = parse_args()
    roadnet_path = Path(args.roadnet).expanduser().resolve()
    sumo_net_path = Path(args.sumo_net).expanduser().resolve() if args.sumo_net else None
    output_html = Path(args.output_html).expanduser().resolve()
    roadnet = load_json(roadnet_path)

    values_by_road = None
    value_metadata = None
    if args.npz is not None:
        values_by_road, value_metadata = load_npz_values(
            Path(args.npz).expanduser().resolve(),
            args.feature,
            args.time_agg,
            args.bucket_index,
        )

    geojson, summary = build_geojson(
        roadnet,
        values_by_road,
        args.coord_mode,
        sumo_net_path,
        args.hide_zero,
        args.line_weight_min,
        args.line_weight_max,
    )
    summary.update({"roadnet": str(roadnet_path), "output_html": str(output_html)})

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(
        build_html(
            args.title,
            geojson,
            summary,
            value_metadata,
            args.tiles,
            args.zoom_start,
            args.line_opacity,
        ),
        encoding="utf-8",
    )
    summary_path = write_summary(output_html, summary, value_metadata)

    print(f"[done] wrote {output_html}")
    print(f"[done] wrote {summary_path}")
    print(
        f"[summary] rendered={summary['roads_rendered']}/{summary['roads_in_roadnet']} "
        f"coord_mode={summary['coord_mode']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
