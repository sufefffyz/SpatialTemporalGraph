#!/usr/bin/env python3
"""Render per-road prediction errors on the Xuancheng OSM basemap."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from pathlib import Path
from typing import Any

from render_xuancheng_osm_map import (
    CoordinateTransformer,
    TILE_PRESETS,
    infer_coord_mode,
    load_json,
    road_length_m,
)


PALETTE = ["#38bdf8", "#22c55e", "#facc15", "#fb923c", "#ef4444", "#b91c1c"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roadnet", required=True, help="CityFlow roadnet JSON.")
    parser.add_argument("--sumo-net", required=True, help="SUMO net XML for coordinate conversion.")
    parser.add_argument(
        "--error-dir",
        type=Path,
        required=True,
        help="Directory containing xcheng_5min_*_per_road_errors.csv.",
    )
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, default=None)
    parser.add_argument("--title", default="Xuancheng SparseGWNet 5min Road Error Map")
    parser.add_argument(
        "--tile-preset",
        choices=sorted(TILE_PRESETS),
        default="carto-dark",
    )
    parser.add_argument("--coord-mode", choices=["auto", "lonlat", "latlon", "sumo"], default="auto")
    parser.add_argument("--line-opacity", type=float, default=0.9)
    parser.add_argument("--halo-opacity", type=float, default=0.7)
    parser.add_argument("--halo-extra", type=float, default=2.4)
    parser.add_argument("--zoom-start", type=int, default=13)
    return parser.parse_args()


def load_error_csv(path: Path) -> dict[str, dict[str, float | str]]:
    rows: dict[str, dict[str, float | str]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            road_id = str(row["road_id"])
            parsed: dict[str, float | str] = {"road_id": road_id}
            for key, value in row.items():
                if key == "road_id":
                    continue
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows[road_id] = parsed
    return rows


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
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def layer_breaks(values: list[float]) -> list[float]:
    positives = sorted(finite_positive(values))
    if not positives:
        return [0.0] * 6
    return [percentile(positives, q) for q in (50, 75, 90, 95, 99, 100)]


def load_layers(error_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, float | str]]]]:
    specs = [
        ("flow_mae", "5min Flow MAE", "XCHENG_5MIN_FLOW", "mae", "xcheng_5min_flow_per_road_errors.csv"),
        ("flow_rmse", "5min Flow RMSE", "XCHENG_5MIN_FLOW", "rmse", "xcheng_5min_flow_per_road_errors.csv"),
        ("stock_mae", "5min Stock MAE", "XCHENG_5MIN_STOCK", "mae", "xcheng_5min_stock_per_road_errors.csv"),
        ("stock_rmse", "5min Stock RMSE", "XCHENG_5MIN_STOCK", "rmse", "xcheng_5min_stock_per_road_errors.csv"),
    ]
    tables: dict[str, dict[str, dict[str, float | str]]] = {}
    layers: dict[str, Any] = {}
    for layer_id, label, dataset, metric, filename in specs:
        path = error_dir / filename
        if dataset not in tables:
            tables[dataset] = load_error_csv(path)
        table = tables[dataset]
        values = [float(row[metric]) for row in table.values()]
        top = sorted(
            (
                {
                    "road_id": road_id,
                    "value": float(row[metric]),
                    "mae": float(row["mae"]),
                    "rmse": float(row["rmse"]),
                    "target_mean": float(row["target_mean"]),
                    "target_max": float(row["target_max"]),
                    "length_m": float(row["length_m"]),
                }
                for road_id, row in table.items()
            ),
            key=lambda item: item["value"],
            reverse=True,
        )[:10]
        layers[layer_id] = {
            "label": label,
            "dataset": dataset,
            "metric": metric,
            "breaks": layer_breaks(values),
            "max": max(values) if values else 0.0,
            "p99": percentile(sorted(finite_positive(values)), 99) if finite_positive(values) else 0.0,
            "top": top,
        }
    return layers, tables


def feature_values_for_road(
    road_id: str,
    tables: dict[str, dict[str, dict[str, float | str]]],
) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    mapping = {
        "flow_mae": ("XCHENG_5MIN_FLOW", "mae"),
        "flow_rmse": ("XCHENG_5MIN_FLOW", "rmse"),
        "stock_mae": ("XCHENG_5MIN_STOCK", "mae"),
        "stock_rmse": ("XCHENG_5MIN_STOCK", "rmse"),
    }
    for layer_id, (dataset, metric) in mapping.items():
        row = tables.get(dataset, {}).get(road_id)
        values[layer_id] = float(row[metric]) if row is not None else None
    return values


def build_geojson(
    roadnet: dict[str, Any],
    sumo_net: Path,
    coord_mode: str,
    tables: dict[str, dict[str, dict[str, float | str]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    roads = roadnet.get("roads", [])
    resolved_mode = infer_coord_mode(roads) if coord_mode == "auto" else coord_mode
    transformer = CoordinateTransformer(resolved_mode, sumo_net)
    features = []
    bounds: list[tuple[float, float]] = []

    for road in roads:
        road_id = str(road["id"])
        points = road.get("points", [])
        if len(points) < 2:
            continue
        coordinates: list[list[float]] = []
        for point in points:
            try:
                lon, lat = transformer.point_to_lonlat(point)
            except (KeyError, TypeError, ValueError):
                continue
            if -180 <= lon <= 180 and -90 <= lat <= 90:
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
            "lane_count": len(lanes),
            "length_m": round(road_length_m(points), 3),
            "speed_limit_kmh": round(speed_limit_kmh, 3),
            "values": feature_values_for_road(road_id, tables),
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
        "bounds": [[min(lats), min(lons)], [max(lats), max(lons)]],
    }
    return {"type": "FeatureCollection", "features": features}, summary


def html_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def build_html(
    title: str,
    geojson: dict[str, Any],
    layers: dict[str, Any],
    summary: dict[str, Any],
    tiles: str,
    attribution: str,
    zoom_start: int,
    line_opacity: float,
    halo_opacity: float,
    halo_extra: float,
) -> str:
    safe_title = html.escape(title)
    palette_json = html_json(PALETTE)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{safe_title}</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    html, body, #map {{ height: 100%; margin: 0; width: 100%; }}
    body {{ color: #e5e7eb; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .panel, .legend, .topbox {{
      background: rgba(15, 23, 42, 0.88);
      border: 1px solid rgba(226, 232, 240, 0.18);
      border-radius: 8px;
      box-shadow: 0 12px 30px rgba(0,0,0,0.35);
      color: #e5e7eb;
      position: fixed;
      z-index: 1000;
    }}
    .panel {{ left: 14px; top: 14px; max-width: 420px; padding: 12px 14px; }}
    .panel h1 {{ font-size: 16px; margin: 0 0 8px; }}
    .panel .meta {{ color: #cbd5e1; font-size: 12px; line-height: 1.45; }}
    .control-row {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 9px; }}
    .layer-btn {{
      background: rgba(30, 41, 59, 0.95);
      border: 1px solid rgba(148, 163, 184, 0.45);
      border-radius: 5px;
      color: #e5e7eb;
      cursor: pointer;
      font-size: 12px;
      padding: 5px 7px;
    }}
    .layer-btn.active {{ background: #f97316; border-color: #fed7aa; color: #111827; }}
    .legend {{ bottom: 22px; left: 14px; padding: 10px 12px; width: 360px; }}
    .swatches {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 3px; margin: 7px 0 5px; }}
    .swatches span {{ height: 9px; }}
    .legend-text {{ color: #cbd5e1; font-size: 12px; line-height: 1.45; }}
    .topbox {{ bottom: 22px; right: 14px; max-height: 330px; overflow: auto; padding: 10px 12px; width: 390px; }}
    .topbox h2 {{ font-size: 13px; margin: 0 0 6px; }}
    .topbox table {{ border-collapse: collapse; font-size: 11px; width: 100%; }}
    .topbox td, .topbox th {{ border-bottom: 1px solid rgba(148, 163, 184, 0.22); padding: 3px 4px; text-align: right; }}
    .topbox td:first-child, .topbox th:first-child {{ text-align: left; }}
    @media (max-width: 760px) {{
      .panel {{ left: 10px; right: 10px; max-width: none; }}
      .legend {{ left: 10px; right: 10px; width: auto; }}
      .topbox {{ display: none; }}
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="panel">
    <h1>{safe_title}</h1>
    <div class="meta" id="meta"></div>
    <div class="control-row" id="layer-buttons"></div>
  </div>
  <div class="legend">
    <div><b id="legend-title"></b></div>
    <div class="swatches" id="swatches"></div>
    <div class="legend-text" id="legend-text"></div>
  </div>
  <div class="topbox">
    <h2 id="top-title"></h2>
    <div id="top-table"></div>
  </div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const roadData = {html_json(geojson)};
    const layerSpecs = {html_json(layers)};
    const summary = {html_json(summary)};
    const palette = {palette_json};
    let selectedLayer = "stock_rmse";
    const map = L.map("map", {{ zoomControl: true }});
    L.tileLayer("{tiles}", {{
      maxZoom: 19,
      attribution: {html_json(attribution)}
    }}).addTo(map);

    function fmt(value) {{
      if (value === null || value === undefined || !Number.isFinite(value)) return "n/a";
      if (Math.abs(value) >= 100) return value.toFixed(1);
      if (Math.abs(value) >= 10) return value.toFixed(2);
      if (Math.abs(value) >= 1) return value.toFixed(3);
      return value.toFixed(4);
    }}

    function colorFor(value, breaks) {{
      if (value === null || value === undefined || !Number.isFinite(value)) return "#475569";
      if (value <= 0) return "#64748b";
      for (let i = 0; i < breaks.length; i++) {{
        if (value <= breaks[i]) return palette[i];
      }}
      return palette[palette.length - 1];
    }}

    function weightFor(value, spec) {{
      if (value === null || value === undefined || !Number.isFinite(value) || value <= 0) return 1.3;
      const denom = Math.max(spec.p99 || spec.max || 1, 1e-9);
      const ratio = Math.min(Math.sqrt(value / denom), 1);
      return 1.4 + ratio * 6.0;
    }}

    function styleFor(feature) {{
      const spec = layerSpecs[selectedLayer];
      const value = feature.properties.values[selectedLayer];
      return {{
        color: colorFor(value, spec.breaks),
        opacity: {line_opacity:.4f},
        weight: weightFor(value, spec),
        lineCap: "round",
        lineJoin: "round"
      }};
    }}

    function haloStyle(feature) {{
      const spec = layerSpecs[selectedLayer];
      const value = feature.properties.values[selectedLayer];
      return {{
        color: "#020617",
        opacity: {halo_opacity:.4f},
        weight: weightFor(value, spec) + {halo_extra:.3f},
        lineCap: "round",
        lineJoin: "round"
      }};
    }}

    function popupHtml(p) {{
      return `<b>${{p.road_id}}</b><br>
        flow MAE: ${{fmt(p.values.flow_mae)}}<br>
        flow RMSE: ${{fmt(p.values.flow_rmse)}}<br>
        stock MAE: ${{fmt(p.values.stock_mae)}}<br>
        stock RMSE: ${{fmt(p.values.stock_rmse)}}<br>
        lanes: ${{p.lane_count}}, length: ${{fmt(p.length_m)}} m`;
    }}

    const halo = L.geoJSON(roadData, {{ interactive: false, style: haloStyle }}).addTo(map);
    const roads = L.geoJSON(roadData, {{
      style: styleFor,
      onEachFeature: function(feature, layer) {{
        layer.bindPopup(popupHtml(feature.properties), {{ maxWidth: 280 }});
        layer.bindTooltip(feature.properties.road_id, {{ sticky: true, opacity: 0.88 }});
        layer.on("mouseover", function() {{
          layer.setStyle({{ weight: styleFor(feature).weight + 2.0, opacity: 1.0 }});
        }});
        layer.on("mouseout", function() {{ roads.resetStyle(layer); }});
      }}
    }}).addTo(map);

    function updateButtons() {{
      const box = document.getElementById("layer-buttons");
      box.innerHTML = "";
      for (const [id, spec] of Object.entries(layerSpecs)) {{
        const btn = document.createElement("button");
        btn.className = "layer-btn" + (id === selectedLayer ? " active" : "");
        btn.textContent = spec.label;
        btn.onclick = () => {{ selectedLayer = id; updateView(); }};
        box.appendChild(btn);
      }}
    }}

    function updateLegend() {{
      const spec = layerSpecs[selectedLayer];
      document.getElementById("legend-title").textContent = spec.label;
      const swatches = document.getElementById("swatches");
      swatches.innerHTML = "";
      for (const color of palette) {{
        const span = document.createElement("span");
        span.style.background = color;
        swatches.appendChild(span);
      }}
      document.getElementById("legend-text").textContent =
        "Quantile breaks: " + spec.breaks.map(fmt).join(" | ") +
        " ; line width clipped at p99=" + fmt(spec.p99);
    }}

    function updateTopTable() {{
      const spec = layerSpecs[selectedLayer];
      document.getElementById("top-title").textContent = "Top roads by " + spec.label;
      let html = "<table><thead><tr><th>road</th><th>value</th><th>MAE</th><th>RMSE</th><th>target max</th></tr></thead><tbody>";
      for (const row of spec.top) {{
        html += `<tr><td>${{row.road_id}}</td><td>${{fmt(row.value)}}</td><td>${{fmt(row.mae)}}</td><td>${{fmt(row.rmse)}}</td><td>${{fmt(row.target_max)}}</td></tr>`;
      }}
      html += "</tbody></table>";
      document.getElementById("top-table").innerHTML = html;
    }}

    function updateView() {{
      halo.setStyle(haloStyle);
      roads.setStyle(styleFor);
      updateButtons();
      updateLegend();
      updateTopTable();
      const spec = layerSpecs[selectedLayer];
      document.getElementById("meta").textContent =
        `roads=${{summary.roads_rendered}}/${{summary.roads_in_roadnet}}, selected=${{spec.label}}, ` +
        `metric=${{spec.metric}}, dataset=${{spec.dataset}}`;
    }}

    map.fitBounds(L.latLngBounds(summary.bounds), {{ padding: [22, 22], maxZoom: {int(zoom_start)} }});
    updateView();
  </script>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    roadnet_path = Path(args.roadnet).expanduser().resolve()
    sumo_net_path = Path(args.sumo_net).expanduser().resolve()
    error_dir = args.error_dir.expanduser().resolve()
    output_html = args.output_html.expanduser().resolve()
    output_summary = args.output_summary.expanduser().resolve() if args.output_summary else output_html.with_suffix(".summary.json")
    tile = TILE_PRESETS[args.tile_preset]

    layers, tables = load_layers(error_dir)
    geojson, summary = build_geojson(load_json(roadnet_path), sumo_net_path, args.coord_mode, tables)
    summary.update(
        {
            "roadnet": str(roadnet_path),
            "sumo_net": str(sumo_net_path),
            "error_dir": str(error_dir),
            "output_html": str(output_html),
            "tile_preset": args.tile_preset,
            "layers": layers,
        }
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(
        build_html(
            args.title,
            geojson,
            layers,
            summary,
            tile["tiles"],
            tile["attribution"],
            args.zoom_start,
            args.line_opacity,
            args.halo_opacity,
            args.halo_extra,
        ),
        encoding="utf-8",
    )
    output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[done] wrote {output_html}")
    print(f"[done] wrote {output_summary}")
    print(f"[summary] rendered={summary['roads_rendered']}/{summary['roads_in_roadnet']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
