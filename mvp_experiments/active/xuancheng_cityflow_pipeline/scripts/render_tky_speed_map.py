#!/usr/bin/env python3
"""Render one day of EXPY-TKY/Tokyo road-link speeds on a Leaflet map."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


TILE_PRESETS = {
    "carto-dark": {
        "tiles": "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        "attribution": (
            '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
            'contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        ),
    },
    "carto-positron": {
        "tiles": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        "attribution": (
            '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
            'contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        ),
    },
    "osm": {
        "tiles": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attribution": '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    },
}

SPEED_COLORS = ["#b91c1c", "#ef4444", "#f97316", "#facc15", "#84cc16", "#22c55e", "#00e5ff"]
MISSING_COLOR = "#6b7280"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed-day-npz", required=True, help="NPZ with speed shaped [144, 1843].")
    parser.add_argument("--link-info", required=True, help="MegaCRN expy-tky_link_info_add_end_latlon.csv.")
    parser.add_argument("--link-index", required=True, help="MegaCRN tokyo_link_idx.csv for the 1843 paper links.")
    parser.add_argument("--output-html", required=True, help="Output HTML file.")
    parser.add_argument("--date", default=None, help="Date label. Defaults to NPZ metadata if available.")
    parser.add_argument("--tile-preset", choices=sorted(TILE_PRESETS), default="carto-dark")
    parser.add_argument("--title", default="EXPY-TKY road-link speeds")
    parser.add_argument("--line-weight-min", type=float, default=2.0)
    parser.add_argument("--line-weight-max", type=float, default=7.0)
    parser.add_argument("--line-opacity", type=float, default=0.92)
    return parser.parse_args()


def positive_mean(values: np.ndarray) -> np.ndarray:
    positive = np.where(values > 0, values, np.nan)
    return np.nanmean(positive, axis=0)


def quantile_breaks(values: np.ndarray) -> list[float]:
    valid = values[np.isfinite(values) & (values > 0)]
    if valid.size == 0:
        return [0.0 for _ in SPEED_COLORS[:-1]]
    qs = np.quantile(valid, [0.05, 0.20, 0.40, 0.60, 0.80, 0.95])
    breaks: list[float] = []
    last = -math.inf
    for value in qs:
        value = float(value)
        if value <= last:
            value = last + 1e-6
        breaks.append(value)
        last = value
    return breaks


def color_index(value: float, breaks: list[float]) -> int:
    if not math.isfinite(value) or value <= 0:
        return -1
    for idx, cut in enumerate(breaks):
        if value <= cut:
            return idx
    return len(SPEED_COLORS) - 1


def line_weight(value: float, valid_min: float, valid_max: float, min_weight: float, max_weight: float) -> float:
    if not math.isfinite(value) or value <= 0 or valid_max <= valid_min:
        return min_weight
    ratio = (value - valid_min) / (valid_max - valid_min)
    ratio = max(0.0, min(1.0, ratio))
    return min_weight + (max_weight - min_weight) * math.sqrt(ratio)


def load_tky_links(link_info_path: Path, link_index_path: Path) -> pd.DataFrame:
    link_info = pd.read_csv(link_info_path)
    link_indices = pd.read_csv(link_index_path, header=None).iloc[:, 0].astype(int).to_numpy()
    if link_indices.size == link_info.shape[0]:
        return link_info.iloc[link_indices].reset_index(drop=True)
    if link_indices.max(initial=-1) >= len(link_info):
        raise ValueError(
            f"Link index file references row {link_indices.max()}, but link info has only {len(link_info)} rows."
        )
    return link_info.iloc[link_indices].reset_index(drop=True)


def build_feature_collection(
    links: pd.DataFrame,
    speeds: np.ndarray,
    day_mean: np.ndarray,
    breaks: list[float],
    min_weight: float,
    max_weight: float,
) -> dict:
    valid = day_mean[np.isfinite(day_mean) & (day_mean > 0)]
    valid_min = float(valid.min()) if valid.size else 0.0
    valid_max = float(valid.max()) if valid.size else 1.0
    features = []
    for node_idx, row in links.iterrows():
        values = speeds[:, node_idx].astype(float)
        mean_value = float(day_mean[node_idx]) if math.isfinite(float(day_mean[node_idx])) else 0.0
        color_idx = color_index(mean_value, breaks)
        color = MISSING_COLOR if color_idx < 0 else SPEED_COLORS[color_idx]
        coords = [
            [float(row["start_lon"]), float(row["start_lat"])],
            [float(row["end_lon"]), float(row["end_lat"])],
        ]
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    "node_index": int(node_idx),
                    "link_id": int(row["link_id"]),
                    "roadname": str(row.get("roadname", "")),
                    "day_mean_speed": round(mean_value, 3),
                    "values": [None if value <= 0 else round(float(value), 3) for value in values],
                    "color": color,
                    "weight": round(line_weight(mean_value, valid_min, valid_max, min_weight, max_weight), 3),
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def render_html(
    output_path: Path,
    title: str,
    date_label: str,
    feature_collection: dict,
    breaks: list[float],
    tile_preset: str,
    line_opacity: float,
) -> None:
    tile = TILE_PRESETS[tile_preset]
    escaped_title = html.escape(title)
    legend_rows = []
    lower = "missing/0"
    for idx, color in enumerate([MISSING_COLOR] + SPEED_COLORS):
        if idx == 0:
            label = lower
        else:
            color_idx = idx - 1
            if color_idx == 0:
                label = f"0-{breaks[0]:.1f} km/h"
            elif color_idx < len(breaks):
                label = f"{breaks[color_idx - 1]:.1f}-{breaks[color_idx]:.1f} km/h"
            else:
                label = f"> {breaks[-1]:.1f} km/h"
        legend_rows.append(f'<div><span style="background:{color}"></span>{html.escape(label)}</div>')
    feature_json = json.dumps(feature_collection, ensure_ascii=False, separators=(",", ":"))
    breaks_json = json.dumps(breaks)
    colors_json = json.dumps(SPEED_COLORS)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escaped_title}</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    html, body, #map {{ height: 100%; margin: 0; background: #0f172a; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .panel {{ position: absolute; z-index: 500; left: 16px; top: 16px; max-width: 390px; color: #f8fafc;
      background: rgba(15, 23, 42, 0.86); border: 1px solid rgba(148, 163, 184, 0.34);
      border-radius: 8px; padding: 12px 14px; box-shadow: 0 10px 28px rgba(0, 0, 0, 0.34); }}
    .panel h1 {{ margin: 0 0 8px; font-size: 16px; font-weight: 700; letter-spacing: 0; }}
    .panel .meta {{ font-size: 12px; color: #cbd5e1; line-height: 1.45; }}
    .slider-row {{ margin-top: 10px; display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: center; }}
    input[type="range"] {{ width: 100%; accent-color: #22d3ee; }}
    .time-label {{ min-width: 74px; text-align: right; font-variant-numeric: tabular-nums; font-size: 13px; color: #f8fafc; }}
    .button-row {{ margin-top: 8px; display: flex; gap: 8px; }}
    button {{ color: #e2e8f0; background: #1e293b; border: 1px solid #475569; border-radius: 6px; padding: 5px 8px; cursor: pointer; }}
    button.active {{ background: #0e7490; border-color: #22d3ee; color: white; }}
    .legend {{ margin-top: 10px; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 5px 12px; font-size: 12px; color: #dbeafe; }}
    .legend span {{ display: inline-block; width: 18px; height: 8px; margin-right: 6px; border-radius: 999px; border: 1px solid rgba(255,255,255,0.35); }}
    .leaflet-tooltip {{ background: rgba(15, 23, 42, 0.9); color: #f8fafc; border: 1px solid rgba(148,163,184,0.45); }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="panel">
    <h1>{escaped_title}</h1>
    <div class="meta">Date: {html.escape(date_label)} · 1843 Shuto Expressway road links · speed channel, 10-minute slots</div>
    <div class="slider-row">
      <input id="slot" type="range" min="0" max="143" step="1" value="0" />
      <div class="time-label" id="timeLabel">mean</div>
    </div>
    <div class="button-row">
      <button id="meanBtn" class="active">day mean</button>
      <button id="slotBtn">slot</button>
    </div>
    <div class="legend">{''.join(legend_rows)}</div>
  </div>
  <script>
    const featureCollection = {feature_json};
    const breaks = {breaks_json};
    const colors = {colors_json};
    const missingColor = "{MISSING_COLOR}";
    const lineOpacity = {line_opacity:.3f};
    let mode = "mean";

    const map = L.map("map", {{ preferCanvas: true }});
    L.tileLayer("{tile['tiles']}", {{
      maxZoom: 19,
      attribution: "{tile['attribution']}"
    }}).addTo(map);

    function colorFor(value) {{
      if (value === null || value === undefined || Number.isNaN(value) || value <= 0) return missingColor;
      for (let i = 0; i < breaks.length; i++) {{
        if (value <= breaks[i]) return colors[i];
      }}
      return colors[colors.length - 1];
    }}

    function slotLabel(slot) {{
      const minutes = slot * 10;
      const hh = Math.floor(minutes / 60).toString().padStart(2, "0");
      const mm = (minutes % 60).toString().padStart(2, "0");
      return `${{hh}}:${{mm}}`;
    }}

    function styleFeature(feature) {{
      const value = mode === "mean" ? feature.properties.day_mean_speed : feature.properties.values[Number(slotInput.value)];
      return {{
        color: colorFor(value),
        weight: mode === "mean" ? feature.properties.weight : Math.max(2.0, feature.properties.weight - 0.4),
        opacity: lineOpacity,
        lineCap: "round",
        lineJoin: "round"
      }};
    }}

    const layer = L.geoJSON(featureCollection, {{
      style: styleFeature,
      onEachFeature: (feature, line) => {{
        line.bindTooltip("", {{ sticky: true }});
        line.on("mouseover", () => {{
          const value = mode === "mean" ? feature.properties.day_mean_speed : feature.properties.values[Number(slotInput.value)];
          const valueText = value === null || value === undefined ? "missing/0" : `${{Number(value).toFixed(1)}} km/h`;
          line.setTooltipContent(
            `<b>${{feature.properties.roadname}}</b><br/>node ${{feature.properties.node_index}} · link ${{feature.properties.link_id}}<br/>${{mode === "mean" ? "day mean" : slotLabel(Number(slotInput.value))}}: ${{valueText}}`
          );
        }});
      }}
    }}).addTo(map);
    map.fitBounds(layer.getBounds().pad(0.06));

    const slotInput = document.getElementById("slot");
    const timeLabel = document.getElementById("timeLabel");
    const meanBtn = document.getElementById("meanBtn");
    const slotBtn = document.getElementById("slotBtn");

    function refresh() {{
      layer.setStyle(styleFeature);
      if (mode === "mean") {{
        timeLabel.textContent = "mean";
      }} else {{
        timeLabel.textContent = slotLabel(Number(slotInput.value));
      }}
      meanBtn.classList.toggle("active", mode === "mean");
      slotBtn.classList.toggle("active", mode === "slot");
    }}
    slotInput.addEventListener("input", () => {{
      mode = "slot";
      refresh();
    }});
    meanBtn.addEventListener("click", () => {{
      mode = "mean";
      refresh();
    }});
    slotBtn.addEventListener("click", () => {{
      mode = "slot";
      refresh();
    }});
    refresh();
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    speed_npz = np.load(args.speed_day_npz)
    speeds = speed_npz["speed"].astype(float)
    if speeds.shape != (144, 1843):
        raise ValueError(f"Expected speed shape (144, 1843), got {speeds.shape}.")
    date_label = args.date
    if date_label is None:
        date_value = speed_npz.get("date")
        date_label = str(date_value.tolist() if hasattr(date_value, "tolist") else date_value)
    links = load_tky_links(Path(args.link_info), Path(args.link_index))
    if len(links) != speeds.shape[1]:
        raise ValueError(f"Link count {len(links)} does not match speed node count {speeds.shape[1]}.")
    day_mean = positive_mean(speeds)
    breaks = quantile_breaks(day_mean)
    feature_collection = build_feature_collection(
        links,
        speeds,
        day_mean,
        breaks,
        args.line_weight_min,
        args.line_weight_max,
    )
    render_html(
        Path(args.output_html),
        args.title,
        date_label or "unknown",
        feature_collection,
        breaks,
        args.tile_preset,
        args.line_opacity,
    )
    valid = day_mean[np.isfinite(day_mean) & (day_mean > 0)]
    print(
        json.dumps(
            {
                "output_html": args.output_html,
                "date": date_label,
                "speed_shape": list(speeds.shape),
                "rendered_links": len(feature_collection["features"]),
                "positive_day_mean_links": int(valid.size),
                "day_mean_min": float(valid.min()) if valid.size else None,
                "day_mean_mean": float(valid.mean()) if valid.size else None,
                "day_mean_max": float(valid.max()) if valid.size else None,
                "breaks": breaks,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
