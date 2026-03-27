"""Plot correct/incorrect segments on a map with layer toggles.

Input files:
  - validation CSV from validate_osrm_lengths.py
  - static parquet with segment endpoint coordinates

Example:
  python scripts/plot_length_validation_layers.py \
      --data-dir /data/yuzhang_fei/Urban_Traffic_Benchmark \
      --city-prefix city_M \
      --validation-csv scripts/city_M_osrm_length_validation.csv \
      --output-html scripts/city_M_validation_layers.html
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import folium
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CoordColumns:
    x_start: str
    y_start: str
    x_end: str
    y_end: str


def detect_coordinate_columns(df: pd.DataFrame) -> CoordColumns:
    cols = df.columns.tolist()

    def first_match(words: list[str]) -> str | None:
        for c in cols:
            cl = c.lower()
            if all(w in cl for w in words):
                return c
        return None

    x_start = first_match(["x", "start"]) or first_match(["lon", "start"])
    y_start = first_match(["y", "start"]) or first_match(["lat", "start"])
    x_end = first_match(["x", "end"]) or first_match(["lon", "end"])
    y_end = first_match(["y", "end"]) or first_match(["lat", "end"])

    if all(v is not None for v in [x_start, y_start, x_end, y_end]):
        return CoordColumns(x_start=x_start, y_start=y_start, x_end=x_end, y_end=y_end)

    if len(cols) >= 4:
        return CoordColumns(x_start=cols[-4], y_start=cols[-3], x_end=cols[-2], y_end=cols[-1])

    raise ValueError("Unable to detect coordinate columns in static parquet.")


def normalize_pass_column(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    as_str = s.astype(str).str.strip().str.lower()
    return as_str.isin(["true", "1", "yes", "y"])


def add_segment_with_endpoints(
    group: folium.FeatureGroup,
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    line_color: str,
    endpoint_color: str,
    line_weight: float,
    endpoint_radius: float,
    endpoint_opacity: float,
    popup_text: str,
) -> None:
    folium.PolyLine(
        locations=[[start_lat, start_lon], [end_lat, end_lon]],
        color=line_color,
        weight=line_weight,
        opacity=0.9,
        popup=popup_text,
    ).add_to(group)

    # Visible endpoint markers for easier connectivity inspection.
    for lat, lon in [(start_lat, start_lon), (end_lat, end_lon)]:
        folium.CircleMarker(
            location=[lat, lon],
            radius=endpoint_radius,
            color="#000000",
            weight=1.0,
            opacity=endpoint_opacity,
            fill=True,
            fill_color=endpoint_color,
            fill_opacity=endpoint_opacity,
        ).add_to(group)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot correct/incorrect length-validation segments with layer toggle.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--city-prefix", type=str, required=True, help="city_M or city_L")
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, default=None)
    parser.add_argument("--line-weight", type=float, default=2.4)
    parser.add_argument("--endpoint-radius", type=float, default=3.4)
    parser.add_argument("--endpoint-opacity", type=float, default=0.95)
    parser.add_argument("--max-per-layer", type=int, default=0,
                        help="Limit number of segments per layer (0 means all).")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--only-incorrect",
        action="store_true",
        help="Only draw incorrect segments.",
    )
    mode_group.add_argument(
        "--only-correct",
        action="store_true",
        help="Only draw correct segments.",
    )
    args = parser.parse_args()

    static_parquet = args.data_dir / f"{args.city_prefix}_static_features.parquet"
    if not static_parquet.exists():
        raise FileNotFoundError(f"Missing static parquet: {static_parquet}")
    if not args.validation_csv.exists():
        raise FileNotFoundError(f"Missing validation CSV: {args.validation_csv}")

    if args.output_html is None:
        output_html = Path(__file__).resolve().parent / f"{args.city_prefix}_length_validation_layers.html"
    else:
        output_html = args.output_html

    print(f"Reading static parquet: {static_parquet}")
    static_df = pd.read_parquet(static_parquet)
    coords = detect_coordinate_columns(static_df)

    static_df = static_df[[coords.x_start, coords.y_start, coords.x_end, coords.y_end]].copy()
    static_df["node_id"] = np.arange(len(static_df), dtype=int)
    for col in [coords.x_start, coords.y_start, coords.x_end, coords.y_end]:
        static_df[col] = pd.to_numeric(static_df[col], errors="coerce")

    print(f"Reading validation CSV: {args.validation_csv}")
    val_df = pd.read_csv(args.validation_csv)
    if "node_id" not in val_df.columns or "pass" not in val_df.columns:
        raise ValueError("Validation CSV must contain node_id and pass columns.")

    val_df = val_df.copy()
    val_df["pass"] = normalize_pass_column(val_df["pass"])
    if "relative_error" not in val_df.columns:
        val_df["relative_error"] = np.nan

    merged = val_df.merge(static_df, on="node_id", how="inner")
    merged = merged.dropna(subset=[coords.x_start, coords.y_start, coords.x_end, coords.y_end])

    if merged.empty:
        raise ValueError("No rows available after merging validation CSV with static coordinates.")

    correct_df = merged[merged["pass"]].copy()
    incorrect_df = merged[~merged["pass"]].copy()

    if args.max_per_layer > 0:
        if len(correct_df) > args.max_per_layer:
            correct_df = correct_df.sample(n=args.max_per_layer, random_state=42)
        if len(incorrect_df) > args.max_per_layer:
            incorrect_df = incorrect_df.sample(n=args.max_per_layer, random_state=42)

    center_lat = float(((merged[coords.y_start] + merged[coords.y_end]) / 2).mean())
    center_lon = float(((merged[coords.x_start] + merged[coords.x_end]) / 2).mean())

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="OpenStreetMap", control_scale=True)

    correct_group = folium.FeatureGroup(name=f"Correct segments ({len(correct_df)})", show=True)
    incorrect_group = folium.FeatureGroup(name=f"Incorrect segments ({len(incorrect_df)})", show=True)

    if not args.only_incorrect:
        print(f"Drawing correct segments: {len(correct_df)}")
        for row in correct_df.itertuples(index=False):
            popup = (
                f"node_id={int(row.node_id)}<br>"
                f"relative_error={getattr(row, 'relative_error', np.nan):.6g}<br>"
                f"status=correct"
            )
            add_segment_with_endpoints(
                group=correct_group,
                start_lat=float(getattr(row, coords.y_start)),
                start_lon=float(getattr(row, coords.x_start)),
                end_lat=float(getattr(row, coords.y_end)),
                end_lon=float(getattr(row, coords.x_end)),
                line_color="#16a34a",
                endpoint_color="#22c55e",
                line_weight=args.line_weight,
                endpoint_radius=args.endpoint_radius,
                endpoint_opacity=args.endpoint_opacity,
                popup_text=popup,
            )

    if not args.only_correct:
        print(f"Drawing incorrect segments: {len(incorrect_df)}")
        for row in incorrect_df.itertuples(index=False):
            popup = (
                f"node_id={int(row.node_id)}<br>"
                f"relative_error={getattr(row, 'relative_error', np.nan):.6g}<br>"
                f"status=incorrect"
            )
            add_segment_with_endpoints(
                group=incorrect_group,
                start_lat=float(getattr(row, coords.y_start)),
                start_lon=float(getattr(row, coords.x_start)),
                end_lat=float(getattr(row, coords.y_end)),
                end_lon=float(getattr(row, coords.x_end)),
                line_color="#dc2626",
                endpoint_color="#f97316",
                line_weight=args.line_weight,
                endpoint_radius=args.endpoint_radius,
                endpoint_opacity=args.endpoint_opacity,
                popup_text=popup,
            )

    groups_added = 0
    if not args.only_incorrect:
        correct_group.add_to(fmap)
        groups_added += 1
    if not args.only_correct:
        incorrect_group.add_to(fmap)
        groups_added += 1

    if groups_added > 1:
        folium.LayerControl(collapsed=False).add_to(fmap)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(output_html))

    print(f"Saved layer map to: {output_html}")


if __name__ == "__main__":
    main()
