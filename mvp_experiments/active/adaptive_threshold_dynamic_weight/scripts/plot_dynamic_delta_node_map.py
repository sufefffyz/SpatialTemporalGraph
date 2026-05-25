#!/usr/bin/env python3
"""Draw red/green node maps for DynamicThreshold delta MAE."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _color_for(value: float, limit: float) -> str:
    def lerp(a: int, b: int, t: float) -> int:
        return int(round(a + (b - a) * t))

    x = max(-1.0, min(1.0, value / limit))
    if x < 0:
        t = x + 1.0
        c0, c1 = (28, 143, 58), (247, 247, 214)
    else:
        t = x
        c0, c1 = (247, 247, 214), (190, 35, 35)
    return "#%02x%02x%02x" % tuple(lerp(c0[i], c1[i], t) for i in range(3))


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _make_delta_bins(
    df: pd.DataFrame, bin_count: int, bin_mode: str
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    if bin_count < 2:
        raise ValueError("--bin-count must be at least 2")
    values = df["delta_dynamic_minus_original"].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("delta_dynamic_minus_original has no finite values")

    if bin_mode == "uniform":
        edges = np.linspace(float(finite.min()), float(finite.max()), bin_count + 1)
    elif bin_mode == "quantile":
        edges = np.nanquantile(finite, np.linspace(0.0, 1.0, bin_count + 1))
        edges = np.unique(edges)
        if edges.size <= 2:
            edges = np.linspace(float(finite.min()), float(finite.max()), bin_count + 1)
    else:
        raise ValueError(f"Unknown bin mode: {bin_mode}")

    if edges[0] == edges[-1]:
        edges = np.array([edges[0] - 1e-6, edges[-1] + 1e-6])

    bins = np.searchsorted(edges[1:-1], values, side="right")
    out = df.copy()
    out["_delta_bin"] = bins

    bin_meta = []
    for idx in range(len(edges) - 1):
        lo = float(edges[idx])
        hi = float(edges[idx + 1])
        count = int((bins == idx).sum())
        label = f"B{idx + 1}: [{lo:.3f}, {hi:.3f}]"
        bin_meta.append({"index": idx, "lo": lo, "hi": hi, "count": count, "label": label})
    return out, bin_meta


def _plot_png(df: pd.DataFrame, out_path: Path, limit: float) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    delta = df["delta_dynamic_minus_original"].to_numpy(float)
    clipped = np.clip(delta, -limit, limit)
    size = 22 + 55 * (np.abs(clipped) / limit) ** 0.7
    fig, ax = plt.subplots(figsize=(9, 8))
    sc = ax.scatter(
        df["Lng"],
        df["Lat"],
        c=clipped,
        s=size,
        cmap="RdYlGn_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        alpha=0.88,
        linewidth=0.35,
        edgecolor="black",
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("SD nodes: DynamicThreshold MAE - Original Adaptive MAE")
    ax.text(
        0.01,
        0.01,
        "marker size = |delta MAE|, clipped at the same percentile as color",
        transform=ax.transAxes,
        fontsize=8,
        color="#334155",
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.85},
    )
    ax.grid(True, alpha=0.18)
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label(f"Delta MAE clipped to +/-{limit:.2f}; red=worse, green=better")
    examples = pd.concat(
        [
            df.nlargest(5, "delta_dynamic_minus_original"),
            df.nsmallest(5, "delta_dynamic_minus_original"),
        ]
    )
    for _, row in examples.iterrows():
        ax.annotate(
            str(int(row["node_index"])),
            (row["Lng"], row["Lat"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="black",
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _write_html(
    df: pd.DataFrame,
    out_path: Path,
    limit: float,
    bin_count: int,
    bin_mode: str,
) -> None:
    width, height = 1050, 880
    margin = 70
    df, bin_meta = _make_delta_bins(df, bin_count=bin_count, bin_mode=bin_mode)
    lng = df["Lng"].to_numpy(float)
    lat = df["Lat"].to_numpy(float)
    lng_min, lng_max = float(lng.min()), float(lng.max())
    lat_min, lat_max = float(lat.min()), float(lat.max())
    lng_pad = (lng_max - lng_min) * 0.04
    lat_pad = (lat_max - lat_min) * 0.04
    lng_min -= lng_pad
    lng_max += lng_pad
    lat_min -= lat_pad
    lat_max += lat_pad

    def sx(x: float) -> float:
        return margin + (x - lng_min) / (lng_max - lng_min) * (width - 2 * margin)

    def sy(y: float) -> float:
        return height - margin - (y - lat_min) / (lat_max - lat_min) * (height - 2 * margin)

    circles = []
    for _, row in df.iterrows():
        delta = float(row["delta_dynamic_minus_original"])
        clipped = max(-limit, min(limit, delta))
        radius = 3.2 + 8.5 * (abs(clipped) / limit) ** 0.65
        state = "better" if delta < 0 else "worse" if delta > 0 else "tie"
        bin_index = int(row["_delta_bin"])
        bin_label = str(bin_meta[bin_index]["label"])
        info = {
            "node_index": _fmt(row["node_index"], 0),
            "sensor_id": _fmt(row.get("ID", "")),
            "freeway": _fmt(row.get("Fwy", "")),
            "direction": _fmt(row.get("Direction", "")),
            "lanes": _fmt(row.get("Lanes", ""), 0),
            "lat": _fmt(row.get("Lat", ""), 6),
            "lng": _fmt(row.get("Lng", ""), 6),
            "delta_dynamic_minus_original": _fmt(delta),
            "original_adaptive_mae": _fmt(row["original_adaptive_mae"]),
            "dynamic_threshold_mae": _fmt(row["dynamic_threshold_mae"]),
            "original_out_degree": _fmt(row.get("original_out_degree", ""), 2),
            "dynamic_out_degree_mean": _fmt(row.get("dynamic_out_degree_mean", ""), 2),
            "degree_delta_dynamic_minus_original": _fmt(
                row.get("degree_delta_dynamic_minus_original", ""), 2
            ),
            "dynamic_same_fwy_share": _fmt(row.get("dynamic_same_fwy_share", ""), 3),
            "dynamic_overlap_precision": _fmt(row.get("dynamic_overlap_precision", ""), 3),
            "dynamic_added_osrm_mean_m": _fmt(row.get("dynamic_added_osrm_mean_m", ""), 1),
            "test_valid_mean": _fmt(row.get("test_valid_mean", ""), 4),
            "test_p95": _fmt(row.get("test_p95", ""), 4),
            "delta_bin": bin_label,
        }
        info_attr = html.escape(json.dumps(info, ensure_ascii=True), quote=True)
        tooltip = (
            f"node={int(row['node_index'])}\n"
            f"ID={row.get('ID', '')}\n"
            f"Fwy={row.get('Fwy', '')} {row.get('Direction', '')}, lanes={row.get('Lanes', '')}\n"
            f"bin={bin_label}\n"
            f"delta={delta:.4f} (red worse, green better)\n"
            f"original_mae={row['original_adaptive_mae']:.4f}\n"
            f"dynamic_mae={row['dynamic_threshold_mae']:.4f}\n"
            f"dynamic_degree={row['dynamic_out_degree_mean']:.2f}\n"
            f"degree_delta={row['degree_delta_dynamic_minus_original']:.2f}\n"
            f"same_fwy_share={row.get('dynamic_same_fwy_share', float('nan')):.3f}\n"
            f"overlap_precision={row.get('dynamic_overlap_precision', float('nan')):.3f}"
        )
        circles.append(
            f'<circle class="node" cx="{sx(float(row["Lng"])):.2f}" '
            f'cy="{sy(float(row["Lat"])):.2f}" r="{radius:.2f}" '
            f'fill="{_color_for(delta, limit)}" stroke="#1f2937" '
            f'stroke-width="0.7" fill-opacity="0.88" tabindex="0" role="button" '
            f'aria-label="node {int(row["node_index"])} delta {delta:.4f}" '
            f'data-bin="{bin_index}" data-state="{state}" data-delta="{delta:.8f}" '
            f'data-info="{info_attr}">'
            f"<title>{html.escape(tooltip)}</title></circle>"
        )

    controls = [
        f'<button class="filter-btn active" data-filter="all">All ({len(df)})</button>',
        f'<button class="filter-btn" data-filter="better">Better / green ({int((df["delta_dynamic_minus_original"] < 0).sum())})</button>',
        f'<button class="filter-btn" data-filter="worse">Worse / red ({int((df["delta_dynamic_minus_original"] > 0).sum())})</button>',
    ]
    for meta in bin_meta:
        controls.append(
            '<button class="filter-btn bucket" '
            f'data-filter="bin" data-bin="{meta["index"]}">'
            f'{html.escape(str(meta["label"]))} '
            f'({int(meta["count"])})</button>'
        )

    legend = []
    for i in range(101):
        value = -limit + 2 * limit * i / 100
        x = margin + i * 3.3
        legend.append(
            f'<rect x="{x:.1f}" y="34" width="3.5" height="16" '
            f'fill="{_color_for(value, limit)}"/>'
        )

    ticks = []
    for value in np.linspace(lng_min, lng_max, 6):
        x = sx(float(value))
        ticks.append(
            f'<line x1="{x:.1f}" y1="{margin}" x2="{x:.1f}" '
            f'y2="{height-margin}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        ticks.append(
            f'<text x="{x:.1f}" y="{height-margin+24}" text-anchor="middle" '
            f'font-size="12">{value:.2f}</text>'
        )
    for value in np.linspace(lat_min, lat_max, 6):
        y = sy(float(value))
        ticks.append(
            f'<line x1="{margin}" y1="{y:.1f}" x2="{width-margin}" '
            f'y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        ticks.append(
            f'<text x="{margin-12}" y="{y+4:.1f}" text-anchor="end" '
            f'font-size="12">{value:.2f}</text>'
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>SD DynamicThreshold Delta Node Map</title>
<style>
  body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f8fafc; color: #0f172a; }}
  .wrap {{ padding: 20px 24px; }}
  h1 {{ margin: 0 0 6px; font-size: 22px; }}
  p {{ margin: 4px 0 14px; color: #475569; }}
  .controls {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0 16px; max-width: 1120px; }}
  .filter-btn {{ border: 1px solid #cbd5e1; background: #fff; color: #0f172a; border-radius: 6px; padding: 6px 9px; font-size: 12px; cursor: pointer; }}
  .filter-btn:hover {{ border-color: #64748b; }}
  .filter-btn.active {{ background: #0f172a; color: #fff; border-color: #0f172a; }}
  .panel-grid {{ display: grid; grid-template-columns: minmax(0, 1050px) minmax(260px, 360px); gap: 16px; align-items: start; }}
  svg {{ background: white; border: 1px solid #cbd5e1; border-radius: 8px; max-width: 100%; height: auto; }}
  .node:hover {{ stroke: #000; stroke-width: 2; fill-opacity: 1; }}
  .node {{ cursor: pointer; transition: fill-opacity 120ms ease, stroke-opacity 120ms ease, stroke-width 120ms ease; }}
  .node.muted {{ fill: #e2e8f0 !important; fill-opacity: 0.08 !important; stroke: #cbd5e1 !important; stroke-opacity: 0.22 !important; }}
  .node.selected {{ stroke: #020617 !important; stroke-width: 3 !important; fill-opacity: 1 !important; }}
  .axis-label {{ font-size: 13px; fill: #334155; }}
  .note {{ font-size: 13px; max-width: 960px; line-height: 1.55; }}
  .detail {{ background: #fff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px; position: sticky; top: 12px; }}
  .detail h2 {{ margin: 0 0 10px; font-size: 16px; }}
  .detail table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
  .detail th {{ text-align: left; color: #475569; font-weight: 600; width: 48%; padding: 5px 6px 5px 0; border-bottom: 1px solid #e2e8f0; vertical-align: top; }}
  .detail td {{ padding: 5px 0; border-bottom: 1px solid #e2e8f0; word-break: break-word; }}
  .status {{ color: #475569; font-size: 13px; margin-bottom: 8px; }}
  .size-dot {{ fill: #94a3b8; stroke: #334155; stroke-width: 0.6; fill-opacity: 0.55; }}
  @media (max-width: 1180px) {{ .panel-grid {{ grid-template-columns: 1fr; }} .detail {{ position: static; max-width: 1050px; }} }}
</style>
</head>
<body>
<div class="wrap">
<h1>SD node delta map</h1>
<p>Red means DynamicThreshold is worse than original adaptive GWNet; green means it is better. Color is clipped to +/-{limit:.2f} at p98 absolute delta MAE. Node size also encodes |delta MAE| after the same clipping, not node degree. Click a node for details.</p>
<div class="controls" aria-label="delta filters">
  {''.join(controls)}
</div>
<div class="status" id="filter-status">Showing all nodes.</div>
<div class="panel-grid">
<svg viewBox="0 0 {width} {height}" role="img" aria-label="SD nodes colored by delta MAE">
  <rect x="{margin}" y="{margin}" width="{width-2*margin}" height="{height-2*margin}" fill="#ffffff" stroke="#94a3b8" />
  {''.join(ticks)}
  <text x="{width/2:.1f}" y="{height-20}" text-anchor="middle" class="axis-label">Longitude</text>
  <text x="18" y="{height/2:.1f}" transform="rotate(-90 18 {height/2:.1f})" text-anchor="middle" class="axis-label">Latitude</text>
  {''.join(circles)}
  <g>
    <text x="{margin}" y="24" font-size="13" fill="#334155">green better</text>
    {''.join(legend)}
    <text x="{margin + 101*3.3 + 10:.1f}" y="47" font-size="13" fill="#334155">red worse</text>
    <text x="{margin}" y="68" font-size="12" fill="#64748b">delta range shown: -{limit:.2f} to +{limit:.2f}</text>
  </g>
  <g transform="translate({width - margin - 190}, 28)">
    <text x="0" y="0" font-size="13" fill="#334155">size = |delta MAE|</text>
    <circle class="size-dot" cx="16" cy="28" r="3.2" />
    <circle class="size-dot" cx="62" cy="28" r="7.4" />
    <circle class="size-dot" cx="122" cy="28" r="11.7" />
    <text x="0" y="54" font-size="11" fill="#64748b">small</text>
    <text x="104" y="54" font-size="11" fill="#64748b">large</text>
  </g>
</svg>
<aside class="detail" id="node-detail">
  <h2>Node details</h2>
  <p class="note">Click a node to inspect MAE delta, degrees, freeway metadata, and graph-overlap statistics.</p>
</aside>
</div>
<p class="note">Node 238 is the strongest red outlier; node 240 is the strongest green outlier. This is an offline longitude-latitude map, so it does not need online map tiles.</p>
</div>
<script>
const nodes = Array.from(document.querySelectorAll(".node"));
const buttons = Array.from(document.querySelectorAll(".filter-btn"));
const detail = document.getElementById("node-detail");
const statusBox = document.getElementById("filter-status");

function escapeHtml(value) {{
  const text = String(value ?? "");
  const chars = {{"&": "&amp;", "<": "&lt;", ">": "&gt;", "\\"": "&quot;", "'": "&#039;"}};
  return text.replace(/[&<>"']/g, function(ch) {{ return chars[ch]; }});
}}

function showNode(node) {{
  nodes.forEach(function(item) {{ item.classList.remove("selected"); }});
  node.classList.add("selected");
  const info = JSON.parse(node.dataset.info);
  const rows = Object.keys(info).map(function(key) {{
    return "<tr><th>" + escapeHtml(key) + "</th><td>" + escapeHtml(info[key]) + "</td></tr>";
  }}).join("");
  detail.innerHTML = "<h2>Node " + escapeHtml(info.node_index) + "</h2><table>" + rows + "</table>";
}}

function matchesFilter(node, button) {{
  const filter = button.dataset.filter;
  if (filter === "all") return true;
  if (filter === "better") return node.dataset.state === "better";
  if (filter === "worse") return node.dataset.state === "worse";
  if (filter === "bin") return node.dataset.bin === button.dataset.bin;
  return true;
}}

function applyFilter(button) {{
  buttons.forEach(function(item) {{ item.classList.remove("active"); }});
  button.classList.add("active");
  let count = 0;
  nodes.forEach(function(node) {{
    const match = matchesFilter(node, button);
    node.classList.toggle("muted", !match);
    if (match) count += 1;
  }});
  statusBox.textContent = "Showing " + count + " highlighted nodes; other nodes are kept as faint gray spatial context.";
}}

buttons.forEach(function(button) {{
  button.addEventListener("click", function() {{ applyFilter(button); }});
}});
nodes.forEach(function(node) {{
  node.addEventListener("click", function() {{ showNode(node); }});
  node.addEventListener("keydown", function(event) {{
    if (event.key === "Enter" || event.key === " ") {{
      event.preventDefault();
      showNode(node);
    }}
  }});
}});
</script>
</body>
</html>
"""
    out_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="node_feature_table.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="sd_delta_node_map_red_green")
    parser.add_argument("--clip-percentile", type=float, default=98.0)
    parser.add_argument("--bin-count", type=int, default=10)
    parser.add_argument("--bin-mode", choices=("quantile", "uniform"), default="quantile")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    limit = float(np.nanpercentile(np.abs(df["delta_dynamic_minus_original"]), args.clip_percentile))
    limit = max(limit, 1e-6)
    png_path = out_dir / f"{args.prefix}.png"
    html_path = out_dir / f"{args.prefix}.html"
    _plot_png(df, png_path, limit)
    _write_html(df, html_path, limit, bin_count=args.bin_count, bin_mode=args.bin_mode)
    print(
        json.dumps(
            {
                "png": str(png_path),
                "html": str(html_path),
                "clip_abs_delta": limit,
                "bin_count": args.bin_count,
                "bin_mode": args.bin_mode,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
