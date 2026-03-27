"""Validate whether OSRM route length matches dataset `length` within a threshold.

Example:
  python scripts/validate_osrm_lengths.py \
      --data-dir /data/yuzhang_fei/Urban_Traffic_Benchmark \
      --city-prefix city_M \
      --osrm-url http://127.0.0.1:5000 \
      --threshold 0.05 \
      --sample-size 5000
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
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
        # Fallback used by this dataset conversion flow.
        return CoordColumns(x_start=cols[-4], y_start=cols[-3], x_end=cols[-2], y_end=cols[-1])

    raise ValueError("Unable to detect coordinate columns.")


def load_cache(cache_file: Path) -> dict[str, float]:
    if not cache_file.exists():
        return {}
    try:
        with cache_file.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return {k: float(v) for k, v in obj.items()}
    except Exception as exc:
        print(f"Warning: failed to load cache {cache_file}: {exc}")
    return {}


def save_cache(cache_file: Path, cache: dict[str, float]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("w", encoding="utf-8") as f:
        json.dump(cache, f)


def make_key(start_lat: float, start_lon: float, end_lat: float, end_lon: float) -> str:
    return f"{start_lat:.6f},{start_lon:.6f}->{end_lat:.6f},{end_lon:.6f}"


def fetch_osrm_distance(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    osrm_url: str,
    timeout_sec: float,
    cache: dict[str, float],
) -> float | None:
    key = make_key(start_lat, start_lon, end_lat, end_lon)
    if key in cache:
        return cache[key]

    params = urlencode(
        {
            "overview": "false",
            "steps": "false",
            "alternatives": "false",
        }
    )
    url = f"{osrm_url.rstrip('/')}/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?{params}"

    try:
        with urlopen(url, timeout=timeout_sec) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        routes = payload.get("routes") or []
        if not routes:
            return None
        dist = float(routes[0]["distance"])
        cache[key] = dist
        return dist
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate OSRM length against static `length` field.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--city-prefix", type=str, required=True, help="city_M or city_L")
    parser.add_argument("--osrm-url", type=str, default="http://127.0.0.1:5000")
    parser.add_argument("--threshold", type=float, default=0.05, help="Relative error threshold, e.g. 0.05")
    parser.add_argument("--sample-size", type=int, default=5000, help="Number of segments to validate")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--timeout-sec", type=float, default=2.5)
    parser.add_argument("--max-requests", type=int, default=20000)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--cache-file", type=Path, default=None)
    args = parser.parse_args()

    static_parquet = args.data_dir / f"{args.city_prefix}_static_features.parquet"
    if not static_parquet.exists():
        raise FileNotFoundError(f"Missing file: {static_parquet}")

    if args.output_csv is None:
        output_csv = Path(__file__).resolve().parent / f"{args.city_prefix}_osrm_length_validation.csv"
    else:
        output_csv = args.output_csv

    if args.output_json is None:
        output_json = Path(__file__).resolve().parent / f"{args.city_prefix}_osrm_length_validation_summary.json"
    else:
        output_json = args.output_json

    if args.cache_file is None:
        cache_file = Path(__file__).resolve().parent / f"{args.city_prefix}_osrm_distance_cache.json"
    else:
        cache_file = args.cache_file

    print(f"Reading: {static_parquet}")
    df = pd.read_parquet(static_parquet)
    if "length" not in df.columns:
        raise ValueError("Column `length` not found in static features.")

    coords = detect_coordinate_columns(df)
    needed_cols = [coords.x_start, coords.y_start, coords.x_end, coords.y_end, "length"]
    df = df[needed_cols].copy()
    df["node_id"] = np.arange(len(df), dtype=int)

    for col in needed_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=needed_cols)
    df = df[df["length"] > 0]

    if len(df) == 0:
        raise ValueError("No valid segments with positive length found.")

    sample_size = min(args.sample_size, len(df))
    if sample_size < len(df):
        df = df.sample(n=sample_size, random_state=args.random_seed)

    print(f"Validating {len(df)} segments with threshold={args.threshold:.2%}")

    cache = load_cache(cache_file)
    rows: list[dict[str, float | int | bool | str | None]] = []

    req_count = 0
    ok_count = 0
    fail_osrm = 0

    for i, row in enumerate(df.itertuples(index=False), start=1):
        start_lon = float(getattr(row, coords.x_start))
        start_lat = float(getattr(row, coords.y_start))
        end_lon = float(getattr(row, coords.x_end))
        end_lat = float(getattr(row, coords.y_end))
        length_ref = float(getattr(row, "length"))
        node_id = int(getattr(row, "node_id"))

        dist = fetch_osrm_distance(
            start_lat=start_lat,
            start_lon=start_lon,
            end_lat=end_lat,
            end_lon=end_lon,
            osrm_url=args.osrm_url,
            timeout_sec=args.timeout_sec,
            cache=cache,
        )

        if dist is None:
            fail_osrm += 1
            rows.append(
                {
                    "node_id": node_id,
                    "length_ref": length_ref,
                    "osrm_distance": None,
                    "relative_error": None,
                    "pass": False,
                    "status": "osrm_failed",
                }
            )
        else:
            req_count += 1
            rel_err = abs(dist - length_ref) / max(length_ref, 1e-9)
            passed = rel_err <= args.threshold
            if passed:
                ok_count += 1
            rows.append(
                {
                    "node_id": node_id,
                    "length_ref": length_ref,
                    "osrm_distance": dist,
                    "relative_error": rel_err,
                    "pass": passed,
                    "status": "ok",
                }
            )

        if i % 200 == 0:
            print(f"  processed {i}/{len(df)}")

        if req_count >= args.max_requests:
            print("Reached max requests limit, stopping early.")
            break

    save_cache(cache_file, cache)

    result_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False)

    ok_df = result_df[result_df["status"] == "ok"].copy()
    pass_rate = float(ok_df["pass"].mean()) if len(ok_df) else 0.0

    summary = {
        "city_prefix": args.city_prefix,
        "threshold": args.threshold,
        "validated_total": int(len(result_df)),
        "ok_count": int(len(ok_df)),
        "osrm_failed_count": int((result_df["status"] == "osrm_failed").sum()),
        "pass_count": int(ok_df["pass"].sum()) if len(ok_df) else 0,
        "pass_rate_on_ok": pass_rate,
        "mean_relative_error_on_ok": float(ok_df["relative_error"].mean()) if len(ok_df) else None,
        "median_relative_error_on_ok": float(ok_df["relative_error"].median()) if len(ok_df) else None,
        "p95_relative_error_on_ok": float(ok_df["relative_error"].quantile(0.95)) if len(ok_df) else None,
        "output_csv": str(output_csv),
        "cache_file": str(cache_file),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nValidation summary:")
    for k, v in summary.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
