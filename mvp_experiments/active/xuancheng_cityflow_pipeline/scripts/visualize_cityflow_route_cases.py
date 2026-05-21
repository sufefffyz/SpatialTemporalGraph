#!/usr/bin/env python3
"""Visualize Xuancheng CityFlow route-consistency cases on a map.

This is a diagnostic helper for official raw flow files. It does not modify
routes. It reads a CityFlow warning log, pulls the corresponding flow routes,
checks each route against the released roadnet, and renders representative
cases as a Leaflet HTML map.
"""

from __future__ import annotations

import argparse
import collections
import html
import json
import re
from pathlib import Path
from typing import Any

from render_xuancheng_osm_map import (
    CoordinateTransformer,
    TILE_PRESETS,
    html_json,
    infer_coord_mode,
    road_length_m,
)


INVALID_RE = re.compile(r"Invalid route 'flow_(\d+)'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roadnet", required=True, help="CityFlow roadnet JSON.")
    parser.add_argument("--flow", required=True, help="CityFlow daily flow JSON.")
    parser.add_argument("--log", required=True, help="Log containing Invalid route warnings.")
    parser.add_argument("--sumo-net", required=True, help="SUMO net XML for coordinate conversion.")
    parser.add_argument("--output-html", required=True, help="Output Leaflet HTML.")
    parser.add_argument("--max-cases", type=int, default=8, help="Number of invalid-route cases to render.")
    parser.add_argument("--max-expanded-roads", type=int, default=80, help="Cap inferred path roads per case.")
    parser.add_argument("--tile-preset", choices=sorted(TILE_PRESETS), default="carto-dark")
    parser.add_argument("--title", default="Xuancheng CityFlow route consistency cases")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_invalid_flow_indices(log_path: Path) -> list[int]:
    seen: set[int] = set()
    indices: list[int] = []
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for match in INVALID_RE.finditer(text):
        idx = int(match.group(1))
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
    return indices


def build_road_graph(roadnet: dict[str, Any]) -> tuple[set[str], dict[str, list[str]], dict[tuple[str, str], str]]:
    road_ids = {str(road["id"]) for road in roadnet.get("roads", [])}
    adjacency: dict[str, list[str]] = collections.defaultdict(list)
    turn_type: dict[tuple[str, str], str] = {}
    for inter in roadnet.get("intersections", []):
        for road_link in inter.get("roadLinks", []):
            start = str(road_link.get("startRoad"))
            end = str(road_link.get("endRoad"))
            if start in road_ids and end in road_ids:
                adjacency[start].append(end)
                turn_type[(start, end)] = str(road_link.get("type", ""))
    for road_id in road_ids:
        adjacency.setdefault(road_id, [])
    return road_ids, adjacency, turn_type


def shortest_path(adjacency: dict[str, list[str]], start: str, goal: str) -> list[str] | None:
    if start == goal:
        return [start]
    queue: collections.deque[str] = collections.deque([start])
    parent: dict[str, str | None] = {start: None}
    while queue:
        cur = queue.popleft()
        for nxt in adjacency.get(cur, []):
            if nxt in parent:
                continue
            parent[nxt] = cur
            if nxt == goal:
                path = [goal]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])  # type: ignore[arg-type]
                path.reverse()
                return path
            queue.append(nxt)
    return None


def road_geometries(
    roadnet: dict[str, Any],
    sumo_net: Path,
) -> tuple[dict[str, list[list[float]]], dict[str, list[float]], list[dict[str, Any]], list[list[float]]]:
    roads = roadnet.get("roads", [])
    transformer = CoordinateTransformer(infer_coord_mode(roads), sumo_net)
    geometries: dict[str, list[list[float]]] = {}
    centroids: dict[str, list[float]] = {}
    base_roads: list[dict[str, Any]] = []
    bounds: list[list[float]] = []

    for road in roads:
        road_id = str(road["id"])
        coords: list[list[float]] = []
        for point in road.get("points", []):
            try:
                lon, lat = transformer.point_to_lonlat(point)
            except (KeyError, TypeError, ValueError):
                continue
            if -180 <= lon <= 180 and -90 <= lat <= 90:
                coords.append([round(lat, 7), round(lon, 7)])
                bounds.append([lat, lon])
        if len(coords) < 2:
            continue
        geometries[road_id] = coords
        centroids[road_id] = [
            sum(lat for lat, _ in coords) / len(coords),
            sum(lon for _, lon in coords) / len(coords),
        ]
        base_roads.append(
            {
                "id": road_id,
                "coords": coords,
                "lanes": len(road.get("lanes", [])),
                "length_m": round(road_length_m(road.get("points", [])), 2),
            }
        )
    return geometries, centroids, base_roads, bounds


def analyze_route(
    flow_index: int,
    flow: dict[str, Any],
    road_ids: set[str],
    adjacency: dict[str, list[str]],
    turn_type: dict[tuple[str, str], str],
    max_expanded_roads: int,
) -> dict[str, Any]:
    route = [str(road_id) for road_id in flow.get("route", [])]
    missing = [road_id for road_id in route if road_id not in road_ids]
    direct_pairs = 0
    expanded_pairs: list[dict[str, Any]] = []
    unreachable_pairs: list[dict[str, Any]] = []
    inferred_roads: list[str] = []

    if not missing:
        for start, end in zip(route, route[1:]):
            if (start, end) in turn_type:
                direct_pairs += 1
                continue
            path = shortest_path(adjacency, start, end)
            if path is None:
                unreachable_pairs.append({"start": start, "end": end})
            else:
                expanded_pairs.append(
                    {
                        "start": start,
                        "end": end,
                        "path_len": len(path),
                        "inserted_roads": max(0, len(path) - 2),
                    }
                )
                for road_id in path[1:-1]:
                    if len(inferred_roads) < max_expanded_roads:
                        inferred_roads.append(road_id)

    if missing:
        status = "missing_road"
    elif unreachable_pairs:
        status = "unreachable_anchor_pair"
    elif expanded_pairs:
        status = "requires_shortest_path_expansion"
    else:
        status = "direct_route_but_cityflow_rejected"

    return {
        "flow_id": f"flow_{flow_index}",
        "flow_index": flow_index,
        "status": status,
        "start_time": flow.get("startTime"),
        "end_time": flow.get("endTime"),
        "route_len": len(route),
        "route": route,
        "missing_roads": missing,
        "direct_pairs": direct_pairs,
        "expanded_pairs": expanded_pairs,
        "unreachable_pairs": unreachable_pairs,
        "inferred_roads": inferred_roads,
    }


def select_cases(cases: list[dict[str, Any]], max_cases: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for status in (
        "missing_road",
        "unreachable_anchor_pair",
        "requires_shortest_path_expansion",
        "direct_route_but_cityflow_rejected",
    ):
        for case in cases:
            if case["status"] == status:
                selected.append(case)
                break
    for case in cases:
        if len(selected) >= max_cases:
            break
        if case not in selected:
            selected.append(case)
    return selected[:max_cases]


def case_map_payload(
    case: dict[str, Any],
    geometries: dict[str, list[list[float]]],
    centroids: dict[str, list[float]],
) -> dict[str, Any]:
    route_roads = []
    for order, road_id in enumerate(case["route"]):
        coords = geometries.get(road_id)
        if coords:
            route_roads.append({"road_id": road_id, "order": order, "coords": coords})

    inferred_roads = []
    for road_id in case["inferred_roads"]:
        coords = geometries.get(road_id)
        if coords:
            inferred_roads.append({"road_id": road_id, "coords": coords})

    bad_links = []
    for pair in case["unreachable_pairs"]:
        start = pair["start"]
        end = pair["end"]
        if start in centroids and end in centroids:
            bad_links.append({"start": start, "end": end, "coords": [centroids[start], centroids[end]]})

    return {
        **case,
        "route_roads": route_roads,
        "inferred_roads_render": inferred_roads,
        "bad_links": bad_links,
    }


def build_html(
    title: str,
    base_roads: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    bounds: list[list[float]],
    tiles: str,
    attribution: str,
) -> str:
    safe_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{safe_title}</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    html, body, #map {{ height: 100%; margin: 0; width: 100%; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .panel {{
      background: rgba(255,255,255,.95);
      border: 1px solid rgba(15,23,42,.18);
      border-radius: 8px;
      box-shadow: 0 10px 28px rgba(15,23,42,.20);
      left: 14px;
      max-width: 470px;
      padding: 12px 14px;
      position: fixed;
      top: 14px;
      z-index: 1000;
    }}
    .panel h1 {{ font-size: 16px; line-height: 1.25; margin: 0 0 8px; }}
    .panel select {{ box-sizing: border-box; font-size: 13px; margin-bottom: 8px; width: 100%; }}
    .meta {{ color: #334155; font-size: 12px; line-height: 1.45; max-height: 220px; overflow: auto; }}
    .legend {{
      background: rgba(255,255,255,.95);
      border: 1px solid rgba(15,23,42,.18);
      border-radius: 8px;
      bottom: 22px;
      box-shadow: 0 8px 22px rgba(15,23,42,.16);
      color: #334155;
      font-size: 12px;
      left: 14px;
      padding: 10px 12px;
      position: fixed;
      z-index: 1000;
    }}
    .sw {{ display: inline-block; height: 10px; margin-right: 6px; width: 28px; }}
    @media (max-width: 700px) {{ .panel {{ left: 10px; right: 10px; max-width: none; }} }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="panel">
    <h1>{safe_title}</h1>
    <select id="case-select"></select>
    <div class="meta" id="case-meta"></div>
  </div>
  <div class="legend">
    <div><span class="sw" style="background:#64748b"></span>roadnet background</div>
    <div><span class="sw" style="background:#38bdf8"></span>route anchor roads</div>
    <div><span class="sw" style="background:#facc15"></span>inferred shortest-path roads</div>
    <div><span class="sw" style="background:#ef4444"></span>unreachable anchor jump</div>
  </div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const baseRoads = {html_json(base_roads)};
    const cases = {html_json(cases)};
    const bounds = {html_json(bounds)};
    const map = L.map("map");
    L.tileLayer("{tiles}", {{ maxZoom: 19, attribution: {html_json(attribution)} }}).addTo(map);
    if (bounds.length) {{
      map.fitBounds(bounds, {{ padding: [28, 28] }});
    }}

    const baseLayer = L.layerGroup();
    for (const road of baseRoads) {{
      L.polyline(road.coords, {{
        color: "#64748b",
        weight: 1.1,
        opacity: 0.28
      }}).bindPopup(`<b>${{road.id}}</b><br>lanes: ${{road.lanes}}<br>length: ${{road.length_m}} m`).addTo(baseLayer);
    }}
    baseLayer.addTo(map);

    let overlay = L.layerGroup().addTo(map);
    const select = document.getElementById("case-select");
    const meta = document.getElementById("case-meta");
    for (let i = 0; i < cases.length; i++) {{
      const c = cases[i];
      const opt = document.createElement("option");
      opt.value = String(i);
      opt.textContent = `${{c.flow_id}} | ${{c.status}} | route_len=${{c.route_len}}`;
      select.appendChild(opt);
    }}

    function renderCase(index) {{
      overlay.clearLayers();
      const c = cases[index];
      const caseBounds = [];
      for (const road of c.route_roads) {{
        const line = L.polyline(road.coords, {{
          color: "#38bdf8",
          weight: 4,
          opacity: 0.92
        }}).bindPopup(`<b>${{c.flow_id}}</b><br>anchor #${{road.order}}<br>${{road.road_id}}`);
        line.addTo(overlay);
        for (const p of road.coords) caseBounds.push(p);
      }}
      for (const road of c.inferred_roads_render) {{
        const line = L.polyline(road.coords, {{
          color: "#facc15",
          weight: 3,
          opacity: 0.86
        }}).bindPopup(`<b>${{c.flow_id}}</b><br>inferred road<br>${{road.road_id}}`);
        line.addTo(overlay);
        for (const p of road.coords) caseBounds.push(p);
      }}
      for (const link of c.bad_links) {{
        const line = L.polyline(link.coords, {{
          color: "#ef4444",
          weight: 5,
          opacity: 0.95,
          dashArray: "8 7"
        }}).bindPopup(`<b>unreachable anchor pair</b><br>${{link.start}} -> ${{link.end}}`);
        line.addTo(overlay);
        for (const p of link.coords) caseBounds.push(p);
      }}
      if (caseBounds.length) map.fitBounds(caseBounds, {{ padding: [40, 40] }});
      const expandedPreview = c.expanded_pairs.slice(0, 4).map(p =>
        `${{p.start}} -> ${{p.end}} (path_len=${{p.path_len}}, inserted=${{p.inserted_roads}})`
      ).join("<br>");
      const unreachablePreview = c.unreachable_pairs.slice(0, 6).map(p => `${{p.start}} -> ${{p.end}}`).join("<br>");
      const missingPreview = c.missing_roads.slice(0, 8).join("<br>");
      meta.innerHTML = `
        <b>${{c.flow_id}}</b><br>
        status: <b>${{c.status}}</b><br>
        start/end time: ${{c.start_time}} / ${{c.end_time}}<br>
        route length: ${{c.route_len}}; direct pairs: ${{c.direct_pairs}}<br>
        missing roads: ${{c.missing_roads.length}}<br>
        unreachable pairs: ${{c.unreachable_pairs.length}}<br>
        expanded pairs: ${{c.expanded_pairs.length}}<br>
        ${{missingPreview ? "<hr><b>missing</b><br>" + missingPreview : ""}}
        ${{unreachablePreview ? "<hr><b>unreachable</b><br>" + unreachablePreview : ""}}
        ${{expandedPreview ? "<hr><b>expansion examples</b><br>" + expandedPreview : ""}}
      `;
    }}
    select.addEventListener("change", () => renderCase(Number(select.value)));
    if (cases.length) renderCase(0);
  </script>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    roadnet_path = Path(args.roadnet).expanduser().resolve()
    flow_path = Path(args.flow).expanduser().resolve()
    log_path = Path(args.log).expanduser().resolve()
    sumo_net_path = Path(args.sumo_net).expanduser().resolve()
    output_html = Path(args.output_html).expanduser().resolve()

    roadnet = load_json(roadnet_path)
    flow_data = load_json(flow_path)
    road_ids, adjacency, turn_type = build_road_graph(roadnet)
    invalid_indices = parse_invalid_flow_indices(log_path)
    if not invalid_indices:
        raise SystemExit(f"no invalid route warnings found in {log_path}")

    cases: list[dict[str, Any]] = []
    for flow_index in invalid_indices:
        if 0 <= flow_index < len(flow_data):
            cases.append(
                analyze_route(
                    flow_index,
                    flow_data[flow_index],
                    road_ids,
                    adjacency,
                    turn_type,
                    args.max_expanded_roads,
                )
            )

    selected = select_cases(cases, args.max_cases)
    geometries, centroids, base_roads, bounds = road_geometries(roadnet, sumo_net_path)
    payload_cases = [case_map_payload(case, geometries, centroids) for case in selected]
    tile_info = TILE_PRESETS[args.tile_preset]
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(
        build_html(
            args.title,
            base_roads,
            payload_cases,
            bounds,
            tile_info["tiles"],
            tile_info["attribution"],
        ),
        encoding="utf-8",
    )

    status_counts = collections.Counter(case["status"] for case in cases)
    summary_path = output_html.with_suffix(".summary.json")
    summary = {
        "roadnet": str(roadnet_path),
        "flow": str(flow_path),
        "log": str(log_path),
        "output_html": str(output_html),
        "invalid_warning_count": len(invalid_indices),
        "analyzed_count": len(cases),
        "rendered_count": len(payload_cases),
        "status_counts": dict(status_counts),
        "rendered_cases": [
            {
                key: case[key]
                for key in (
                    "flow_id",
                    "status",
                    "route_len",
                    "missing_roads",
                    "unreachable_pairs",
                    "expanded_pairs",
                )
            }
            for case in selected
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[done] wrote {output_html}")
    print(f"[done] wrote {summary_path}")
    print(f"[summary] invalid={len(invalid_indices)} analyzed={len(cases)} status_counts={dict(status_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
