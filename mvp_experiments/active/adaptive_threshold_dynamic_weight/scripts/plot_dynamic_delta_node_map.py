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


def _write_html(df: pd.DataFrame, out_path: Path, limit: float) -> None:
    width, height = 1050, 880
    margin = 70
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
        tooltip = (
            f"node={int(row['node_index'])}\n"
            f"ID={row.get('ID', '')}\n"
            f"Fwy={row.get('Fwy', '')} {row.get('Direction', '')}, lanes={row.get('Lanes', '')}\n"
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
            f'stroke-width="0.7" fill-opacity="0.88">'
            f"<title>{html.escape(tooltip)}</title></circle>"
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
  svg {{ background: white; border: 1px solid #cbd5e1; border-radius: 8px; max-width: 100%; height: auto; }}
  .node:hover {{ stroke: #000; stroke-width: 2; fill-opacity: 1; }}
  .axis-label {{ font-size: 13px; fill: #334155; }}
  .note {{ font-size: 13px; max-width: 960px; line-height: 1.55; }}
</style>
</head>
<body>
<div class="wrap">
<h1>SD node delta map</h1>
<p>Red means DynamicThreshold is worse than original adaptive GWNet; green means it is better. Color is clipped to +/-{limit:.2f} at p98 absolute delta MAE. Hover a node for details.</p>
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
</svg>
<p class="note">Node 238 is the strongest red outlier; node 240 is the strongest green outlier. This is an offline longitude-latitude map, so it does not need online map tiles.</p>
</div>
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
    _write_html(df, html_path, limit)
    print(json.dumps({"png": str(png_path), "html": str(html_path), "clip_abs_delta": limit}, indent=2))


if __name__ == "__main__":
    main()
