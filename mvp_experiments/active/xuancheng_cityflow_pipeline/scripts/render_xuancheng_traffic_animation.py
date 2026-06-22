#!/usr/bin/env python3
"""Render an animated Xuancheng road-level traffic evolution map.

The animation uses road aggregation NPZ files produced by
``run_cityflow_road_aggregation.py`` or
``run_cityflow_paper_mp_road_aggregation.py``. It visualizes the time evolution
of per-road stock, entered flow, exited flow, and speed on a Leaflet map.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from render_xuancheng_osm_map import (
    CoordinateTransformer,
    TILE_PRESETS,
    infer_coord_mode,
    load_json,
    road_length_m,
)


FEATURE_ALIASES = {
    "entered": "entered_veh",
    "exited": "exited_veh",
    "stock": "mean_active_veh",
    "speed": "mean_speed_kmh",
}

PALETTE = ["#1d4ed8", "#0891b2", "#16a34a", "#facc15", "#f97316", "#dc2626", "#7f1d1d"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roadnet", required=True, help="CityFlow roadnet JSON.")
    parser.add_argument("--sumo-net", required=True, help="SUMO net XML for coordinate conversion.")
    parser.add_argument("--npz", required=True, help="Road aggregation NPZ.")
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--output-summary", default=None)
    parser.add_argument("--title", default="Xuancheng traffic evolution")
    parser.add_argument("--date", default="2023-04-03")
    parser.add_argument(
        "--aggregate-factor",
        type=int,
        default=5,
        help="Aggregate source buckets into animation frames. For 60s source buckets, 5 gives 5min frames.",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument(
        "--focus-roads",
        default="34180205108,34180205109_1,34180205109,34180200419",
        help="Comma-separated road IDs highlighted in the side panel.",
    )
    parser.add_argument(
        "--tile-preset",
        choices=sorted(TILE_PRESETS),
        default="carto-dark",
    )
    parser.add_argument("--coord-mode", choices=["auto", "lonlat", "latlon", "sumo"], default="auto")
    parser.add_argument("--line-opacity", type=float, default=0.88)
    parser.add_argument("--halo-opacity", type=float, default=0.55)
    parser.add_argument("--zoom-start", type=int, default=12)
    parser.add_argument("--round-decimals", type=int, default=3)
    return parser.parse_args()


def html_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def percentile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    finite = finite[finite > 0]
    if finite.size == 0:
        return 0.0
    return float(np.percentile(finite, q))


def aggregate_npz(
    npz_path: Path,
    aggregate_factor: int,
    start_frame: int,
    end_frame: int | None,
    round_decimals: int,
) -> dict[str, Any]:
    archive = np.load(npz_path, allow_pickle=False)
    required = {"data", "road_ids", "feature_names"}
    missing = required.difference(archive.files)
    if missing:
        raise SystemExit(f"{npz_path} is missing required keys: {sorted(missing)}")

    data = np.asarray(archive["data"], dtype=np.float32)
    road_ids = [str(value) for value in archive["road_ids"].tolist()]
    feature_names = [str(value) for value in archive["feature_names"].tolist()]
    feature_index = {name: idx for idx, name in enumerate(feature_names)}
    for feature in FEATURE_ALIASES.values():
        if feature not in feature_index:
            raise SystemExit(f"{feature!r} not found in {npz_path}; available: {feature_names}")

    if aggregate_factor < 1:
        raise SystemExit("--aggregate-factor must be >= 1")
    usable = (data.shape[0] // aggregate_factor) * aggregate_factor
    if usable <= 0:
        raise SystemExit(f"not enough source buckets {data.shape[0]} for aggregate factor {aggregate_factor}")
    data = data[:usable]
    grouped = data.reshape(usable // aggregate_factor, aggregate_factor, data.shape[1], data.shape[2])

    entered = grouped[:, :, :, feature_index["entered_veh"]].sum(axis=1)
    exited = grouped[:, :, :, feature_index["exited_veh"]].sum(axis=1)
    stock = grouped[:, :, :, feature_index["mean_active_veh"]].mean(axis=1)
    speed = grouped[:, :, :, feature_index["mean_speed_kmh"]].mean(axis=1)

    bucket_start = archive["bucket_start_s"][:usable] if "bucket_start_s" in archive.files else None
    bucket_end = archive["bucket_end_s"][:usable] if "bucket_end_s" in archive.files else None
    if bucket_start is not None and bucket_end is not None:
        frame_start = bucket_start.reshape(-1, aggregate_factor)[:, 0]
        frame_end = bucket_end.reshape(-1, aggregate_factor)[:, -1]
    else:
        frame_start = np.arange(stock.shape[0], dtype=np.float32) * aggregate_factor
        frame_end = frame_start + aggregate_factor

    total_frames = stock.shape[0]
    start_frame = max(start_frame, 0)
    end_frame = total_frames if end_frame is None else min(end_frame, total_frames)
    if start_frame >= end_frame:
        raise SystemExit(f"empty frame range: start={start_frame}, end={end_frame}, total={total_frames}")

    sl = slice(start_frame, end_frame)
    arrays = {
        "stock": stock[sl],
        "entered": entered[sl],
        "exited": exited[sl],
        "speed": speed[sl],
    }
    rounded_arrays = {
        key: np.round(value.astype(np.float32), round_decimals).tolist() for key, value in arrays.items()
    }
    frame_start = frame_start[sl]
    frame_end = frame_end[sl]
    frame_labels = [seconds_to_label(float(value)) for value in frame_start]

    net_entered = entered[sl].sum(axis=1)
    net_exited = exited[sl].sum(axis=1)
    net_stock = stock[sl].sum(axis=1)
    net_speed = np.divide(
        (speed[sl] * np.maximum(stock[sl], 0)).sum(axis=1),
        np.maximum(stock[sl].sum(axis=1), 1e-6),
    )

    metric_breaks = {
        "stock": [percentile(stock[sl], q) for q in (20, 40, 60, 80, 95, 99)],
        "entered": [percentile(entered[sl], q) for q in (20, 40, 60, 80, 95, 99)],
        "exited": [percentile(exited[sl], q) for q in (20, 40, 60, 80, 95, 99)],
        "speed": [percentile(speed[sl], q) for q in (20, 40, 60, 80, 95, 99)],
    }
    max_values = {key: float(np.nanmax(value)) for key, value in arrays.items()}

    return {
        "road_ids": road_ids,
        "frames": {
            "start_s": np.round(frame_start.astype(np.float32), 3).tolist(),
            "end_s": np.round(frame_end.astype(np.float32), 3).tolist(),
            "labels": frame_labels,
        },
        "series": rounded_arrays,
        "network": {
            "entered": np.round(net_entered.astype(np.float32), round_decimals).tolist(),
            "exited": np.round(net_exited.astype(np.float32), round_decimals).tolist(),
            "stock": np.round(net_stock.astype(np.float32), round_decimals).tolist(),
            "speed": np.round(net_speed.astype(np.float32), round_decimals).tolist(),
        },
        "metric_breaks": metric_breaks,
        "max_values": max_values,
        "source": {
            "npz": str(npz_path),
            "shape": list(archive["data"].shape),
            "feature_names": feature_names,
            "aggregate_factor": aggregate_factor,
            "start_frame": start_frame,
            "end_frame": end_frame,
        },
    }


def seconds_to_label(seconds: float) -> str:
    seconds = max(seconds, 0.0)
    hour = int(seconds // 3600)
    minute = int((seconds % 3600) // 60)
    return f"{hour:02d}:{minute:02d}"


def build_geojson(
    roadnet: dict[str, Any],
    sumo_net: Path,
    coord_mode: str,
    road_id_to_index: dict[str, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    roads = roadnet.get("roads", [])
    resolved_mode = infer_coord_mode(roads) if coord_mode == "auto" else coord_mode
    transformer = CoordinateTransformer(resolved_mode, sumo_net)
    features = []
    bounds: list[tuple[float, float]] = []

    for road in roads:
        road_id = str(road["id"])
        data_index = road_id_to_index.get(road_id)
        if data_index is None:
            continue
        coordinates: list[list[float]] = []
        for point in road.get("points", []):
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
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "road_id": road_id,
                    "data_index": data_index,
                    "lane_count": len(lanes),
                    "length_m": round(road_length_m(road.get("points", [])), 3),
                    "speed_limit_kmh": round(speed_limit_kmh, 3),
                },
                "geometry": {"type": "LineString", "coordinates": coordinates},
            }
        )

    if not bounds:
        raise SystemExit("no matching road geometries with valid lon/lat coordinates were found")
    lats = [lat for lat, _ in bounds]
    lons = [lon for _, lon in bounds]
    summary = {
        "roads_in_roadnet": len(roads),
        "roads_rendered": len(features),
        "coord_mode": resolved_mode,
        "bounds": [[min(lats), min(lons)], [max(lats), max(lons)]],
    }
    return {"type": "FeatureCollection", "features": features}, summary


def build_html(
    title: str,
    geojson: dict[str, Any],
    animation_data: dict[str, Any],
    summary: dict[str, Any],
    focus_roads: list[str],
    tile_preset: str,
    line_opacity: float,
    halo_opacity: float,
    zoom_start: int,
) -> str:
    safe_title = html.escape(title)
    tiles = TILE_PRESETS[tile_preset]["tiles"]
    attribution = TILE_PRESETS[tile_preset]["attribution"]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{safe_title}</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    html, body, #map {{ height: 100%; margin: 0; width: 100%; }}
    body {{ background: #080b12; color: #e5e7eb; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    #map {{ background: #0f172a; }}
    .panel {{
      position: absolute; z-index: 1000; left: 16px; top: 16px; width: min(420px, calc(100vw - 32px));
      background: rgba(8, 13, 24, 0.88); border: 1px solid rgba(148, 163, 184, 0.34);
      border-radius: 8px; box-shadow: 0 18px 50px rgba(0,0,0,.34); backdrop-filter: blur(8px);
    }}
    .panel header {{ padding: 14px 16px 10px; border-bottom: 1px solid rgba(148, 163, 184, .2); }}
    .panel h1 {{ font-size: 16px; line-height: 1.25; margin: 0; font-weight: 700; }}
    .panel .sub {{ color: #94a3b8; font-size: 12px; margin-top: 4px; }}
    .controls {{ padding: 12px 16px; display: grid; gap: 10px; }}
    .row {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
    button, select {{
      color: #e5e7eb; background: #111827; border: 1px solid #334155; border-radius: 6px;
      padding: 7px 10px; font-size: 12px;
    }}
    button:hover, select:hover {{ border-color: #60a5fa; }}
    input[type="range"] {{ width: 100%; accent-color: #38bdf8; }}
    .time {{ font-variant-numeric: tabular-nums; min-width: 56px; font-weight: 700; }}
    .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }}
    .stat {{ background: rgba(15, 23, 42, .72); border: 1px solid rgba(148, 163, 184, .18); border-radius: 6px; padding: 8px; }}
    .stat label {{ display: block; color: #94a3b8; font-size: 10px; text-transform: uppercase; letter-spacing: .05em; }}
    .stat strong {{ display: block; font-size: 13px; margin-top: 4px; font-variant-numeric: tabular-nums; }}
    #chart {{ width: 100%; height: 90px; background: rgba(15, 23, 42, .44); border: 1px solid rgba(148, 163, 184, .18); border-radius: 6px; }}
    .focus {{ display: grid; gap: 5px; font-size: 12px; }}
    .focus-row {{ display: grid; grid-template-columns: 104px 1fr; gap: 8px; color: #cbd5e1; }}
    .focus-row b {{ color: #f8fafc; font-variant-numeric: tabular-nums; }}
    .legend {{ color: #cbd5e1; font-size: 11px; line-height: 1.45; }}
    .swatches {{ display: flex; gap: 3px; margin-top: 5px; }}
    .swatches span {{ height: 8px; flex: 1; border-radius: 999px; }}
    @media (max-width: 720px) {{ .panel {{ left: 8px; right: 8px; top: 8px; width: auto; }} .stats {{ grid-template-columns: repeat(2, 1fr); }} }}
  </style>
</head>
<body>
  <div id="map"></div>
  <section class="panel">
    <header>
      <h1>{safe_title}</h1>
      <div class="sub">Road-level animation from CityFlow aggregation. Color/width update by frame.</div>
    </header>
    <div class="controls">
      <div class="row">
        <button id="play">Pause</button>
        <select id="metric">
          <option value="stock">stock</option>
          <option value="entered">entered</option>
          <option value="exited">exited</option>
          <option value="speed">speed</option>
        </select>
        <select id="speed">
          <option value="900">slow</option>
          <option value="450" selected>normal</option>
          <option value="170">fast</option>
        </select>
        <span class="time" id="timeLabel">00:00</span>
      </div>
      <input id="slider" type="range" min="0" max="0" value="0" />
      <canvas id="chart" width="760" height="170"></canvas>
      <div class="stats">
        <div class="stat"><label>entered</label><strong id="netEntered">0</strong></div>
        <div class="stat"><label>exited</label><strong id="netExited">0</strong></div>
        <div class="stat"><label>stock</label><strong id="netStock">0</strong></div>
        <div class="stat"><label>speed</label><strong id="netSpeed">0</strong></div>
      </div>
      <div class="focus" id="focus"></div>
      <div class="legend">
        <div id="legendText"></div>
        <div class="swatches" id="swatches"></div>
      </div>
    </div>
  </section>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const GEOJSON = {html_json(geojson)};
    const ANIM = {html_json(animation_data)};
    const SUMMARY = {html_json(summary)};
    const FOCUS_ROADS = {html_json(focus_roads)};
    const PALETTE = {html_json(PALETTE)};
    const lineOpacity = {line_opacity:.3f};
    const haloOpacity = {halo_opacity:.3f};
    const map = L.map('map', {{ preferCanvas: true, zoomControl: true }});
    L.tileLayer({html_json(tiles)}, {{ maxZoom: 19, attribution: {html_json(attribution)} }}).addTo(map);
    map.fitBounds(SUMMARY.bounds, {{ padding: [28, 28] }});
    map.setZoom(Math.max(map.getZoom(), {zoom_start}));

    const roads = [];
    const focusIndex = new Map();
    ANIM.road_ids.forEach((id, idx) => focusIndex.set(id, idx));

    function finite(v) {{ return Number.isFinite(v); }}
    function breaksFor(metric) {{ return ANIM.metric_breaks[metric] || [0,0,0,0,0,0]; }}
    function maxFor(metric) {{ return Math.max(ANIM.max_values[metric] || 0, 1e-6); }}
    function colorFor(value, metric) {{
      if (!finite(value) || value <= 0) return '#334155';
      const b = breaksFor(metric);
      for (let i = 0; i < b.length; i++) if (value <= b[i]) return PALETTE[i];
      return PALETTE[PALETTE.length - 1];
    }}
    function widthFor(value, metric) {{
      if (!finite(value) || value <= 0) return 1.1;
      const ratio = Math.min(Math.max(value / maxFor(metric), 0), 1);
      return 1.2 + 8.2 * Math.sqrt(ratio);
    }}
    function fmt(value, digits=2) {{
      if (!finite(value)) return 'NA';
      if (Math.abs(value) >= 1000) return value.toFixed(0);
      if (Math.abs(value) >= 10) return value.toFixed(1);
      return value.toFixed(digits);
    }}

    L.geoJSON(GEOJSON, {{
      renderer: L.canvas({{ padding: 0.5 }}),
      style: () => ({{ color: '#020617', weight: 4.0, opacity: haloOpacity }}),
      interactive: false
    }}).addTo(map);

    L.geoJSON(GEOJSON, {{
      renderer: L.canvas({{ padding: 0.5 }}),
      style: feature => ({{ color: '#334155', weight: 1.1, opacity: 0.35 }}),
      onEachFeature: (feature, layer) => {{
        layer.feature = feature;
        roads.push(layer);
        const p = feature.properties;
        layer.bindTooltip('', {{ sticky: true, opacity: 0.94 }});
      }}
    }}).addTo(map);

    const slider = document.getElementById('slider');
    const play = document.getElementById('play');
    const metricSelect = document.getElementById('metric');
    const speedSelect = document.getElementById('speed');
    const timeLabel = document.getElementById('timeLabel');
    const chart = document.getElementById('chart');
    const ctx = chart.getContext('2d');
    let frame = 0;
    let metric = 'stock';
    let playing = true;
    let timer = null;
    slider.max = String(ANIM.frames.labels.length - 1);

    function valueAt(metric, frame, dataIndex) {{
      return ANIM.series[metric][frame][dataIndex];
    }}
    function updateLayer(layer) {{
      const p = layer.feature.properties;
      const value = valueAt(metric, frame, p.data_index);
      const highlight = FOCUS_ROADS.includes(p.road_id);
      layer.setStyle({{
        color: colorFor(value, metric),
        weight: widthFor(value, metric) + (highlight ? 2.8 : 0),
        opacity: highlight ? 1.0 : lineOpacity
      }});
      const stock = valueAt('stock', frame, p.data_index);
      const entered = valueAt('entered', frame, p.data_index);
      const exited = valueAt('exited', frame, p.data_index);
      const speed = valueAt('speed', frame, p.data_index);
      layer.setTooltipContent(
        `<b>${{p.road_id}}</b><br>` +
        `stock: ${{fmt(stock)}}<br>entered: ${{fmt(entered)}} | exited: ${{fmt(exited)}}<br>` +
        `speed: ${{fmt(speed)}} km/h<br>len: ${{fmt(p.length_m, 0)}}m, lanes: ${{p.lane_count}}`
      );
    }}
    function updateFocus() {{
      const box = document.getElementById('focus');
      box.innerHTML = FOCUS_ROADS.map(id => {{
        const idx = focusIndex.get(id);
        if (idx === undefined) return `<div class="focus-row"><b>${{id}}</b><span>not in tensor</span></div>`;
        const s = valueAt('stock', frame, idx);
        const en = valueAt('entered', frame, idx);
        const ex = valueAt('exited', frame, idx);
        const sp = valueAt('speed', frame, idx);
        return `<div class="focus-row"><b>${{id}}</b><span>stock ${{fmt(s)}} | in ${{fmt(en)}} / out ${{fmt(ex)}} | ${{fmt(sp)}} km/h</span></div>`;
      }}).join('');
    }}
    function drawChart() {{
      const w = chart.width, h = chart.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, w, h);
      const values = ANIM.network[metric];
      const maxV = Math.max(...values, 1e-6);
      ctx.strokeStyle = '#334155';
      ctx.lineWidth = 1;
      for (let i = 1; i < 4; i++) {{
        const y = (h - 20) * i / 4 + 8;
        ctx.beginPath(); ctx.moveTo(10, y); ctx.lineTo(w - 10, y); ctx.stroke();
      }}
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 3;
      ctx.beginPath();
      values.forEach((v, i) => {{
        const x = 12 + i * (w - 24) / Math.max(values.length - 1, 1);
        const y = h - 14 - (v / maxV) * (h - 28);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }});
      ctx.stroke();
      const x = 12 + frame * (w - 24) / Math.max(values.length - 1, 1);
      ctx.strokeStyle = '#f97316';
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(x, 8); ctx.lineTo(x, h - 10); ctx.stroke();
    }}
    function updateLegend() {{
      const breaks = breaksFor(metric);
      document.getElementById('legendText').textContent = `${{metric}} breaks: ` + breaks.map(v => fmt(v)).join(' / ');
      document.getElementById('swatches').innerHTML = PALETTE.map(c => `<span style="background:${{c}}"></span>`).join('');
    }}
    function updateFrame(nextFrame) {{
      frame = Math.max(0, Math.min(nextFrame, ANIM.frames.labels.length - 1));
      slider.value = String(frame);
      timeLabel.textContent = ANIM.frames.labels[frame];
      roads.forEach(updateLayer);
      document.getElementById('netEntered').textContent = fmt(ANIM.network.entered[frame]);
      document.getElementById('netExited').textContent = fmt(ANIM.network.exited[frame]);
      document.getElementById('netStock').textContent = fmt(ANIM.network.stock[frame]);
      document.getElementById('netSpeed').textContent = fmt(ANIM.network.speed[frame]);
      updateFocus();
      drawChart();
    }}
    function tick() {{
      updateFrame(frame + 1 >= ANIM.frames.labels.length ? 0 : frame + 1);
    }}
    function restartTimer() {{
      if (timer) clearInterval(timer);
      if (playing) timer = setInterval(tick, Number(speedSelect.value));
      play.textContent = playing ? 'Pause' : 'Play';
    }}
    slider.addEventListener('input', e => updateFrame(Number(e.target.value)));
    play.addEventListener('click', () => {{ playing = !playing; restartTimer(); }});
    speedSelect.addEventListener('change', restartTimer);
    metricSelect.addEventListener('change', e => {{ metric = e.target.value; updateLegend(); updateFrame(frame); }});
    updateLegend();
    updateFrame(0);
    restartTimer();
  </script>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    roadnet_path = Path(args.roadnet).expanduser().resolve()
    sumo_net_path = Path(args.sumo_net).expanduser().resolve()
    npz_path = Path(args.npz).expanduser().resolve()
    output_html = Path(args.output_html).expanduser().resolve()
    output_summary = (
        Path(args.output_summary).expanduser().resolve()
        if args.output_summary
        else output_html.with_suffix(output_html.suffix + ".summary.json")
    )

    animation_data = aggregate_npz(
        npz_path,
        args.aggregate_factor,
        args.start_frame,
        args.end_frame,
        args.round_decimals,
    )
    road_id_to_index = {road_id: idx for idx, road_id in enumerate(animation_data["road_ids"])}
    roadnet = load_json(roadnet_path)
    geojson, geometry_summary = build_geojson(
        roadnet,
        sumo_net_path,
        args.coord_mode,
        road_id_to_index,
    )
    focus_roads = [road.strip() for road in args.focus_roads.split(",") if road.strip()]
    summary = {
        **geometry_summary,
        "title": args.title,
        "date": args.date,
        "frames": len(animation_data["frames"]["labels"]),
        "focus_roads": focus_roads,
        "tile_preset": args.tile_preset,
        "source": animation_data["source"],
    }
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(
        build_html(
            args.title,
            geojson,
            animation_data,
            summary,
            focus_roads,
            args.tile_preset,
            args.line_opacity,
            args.halo_opacity,
            args.zoom_start,
        ),
        encoding="utf-8",
    )
    output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
