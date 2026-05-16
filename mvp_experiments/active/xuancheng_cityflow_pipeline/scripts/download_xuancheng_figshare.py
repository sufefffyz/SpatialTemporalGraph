#!/usr/bin/env python3
"""Download the Xuancheng CityFlow files from the official Figshare release."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ARTICLE_ID = 29925824
ARTICLE_DOI = "10.6084/m9.figshare.29925824.v5"
BASE_URL = "https://ndownloader.figshare.com/files"

DAILY_FILE_INFO = {
    "2023-04-01": (57238469, 200210220),
    "2023-04-02": (57238472, 198598785),
    "2023-04-03": (57238481, 213008752),
    "2023-04-04": (57238478, 213003967),
    "2023-04-05": (57238415, 157667232),
    "2023-04-06": (57238487, 207191063),
    "2023-04-07": (57238484, 200835137),
    "2023-04-08": (57238421, 183346329),
    "2023-04-09": (57238424, 183346329),
    "2023-04-10": (57238442, 185896032),
    "2023-04-11": (57238436, 182439202),
    "2023-04-12": (57238430, 182492671),
    "2023-04-13": (57238460, 187626778),
    "2023-04-14": (57238493, 206410636),
    "2023-04-15": (57238433, 183803420),
    "2023-04-16": (57238448, 182276005),
    "2023-04-17": (57238445, 182419815),
    "2023-04-18": (57238451, 183357370),
    "2023-04-19": (57238439, 181466139),
    "2023-04-20": (57238427, 177821055),
    "2023-04-21": (57238466, 189829723),
    "2023-04-22": (57238463, 185105699),
    "2023-04-23": (57238490, 203079882),
    "2023-04-24": (57238496, 200078843),
    "2023-04-25": (57238475, 192088174),
    "2023-04-26": (57238454, 181749771),
    "2023-04-27": (57238457, 181080564),
    "2023-04-28": (57238502, 235129894),
    "2023-04-29": (57238505, 237073449),
    "2023-04-30": (57238499, 205586359),
}

SUPPORT_FILE_INFO = {
    "roadnet_xuancheng250319.json": (57238370, 6583764),
    "config_xuancheng_test.json": (57238352, 323),
    "config_xuancheng_test_save.json": (57238355, 427),
}

SUMO_FILE_INFO = {
    "xuancheng.net.xml": (62215847, 5216503),
}

DOC_FILE_INFO = {
    "Documentation.pdf": (61742701, 5134531),
}


def daily_name(date_text: str) -> str:
    return f"data_{date_text.replace('-', '_')}_type_filtered.json"


def parse_date(text: str) -> str:
    try:
        parsed = dt.date.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date {text!r}, expected YYYY-MM-DD") from exc
    normalized = parsed.isoformat()
    if normalized not in DAILY_FILE_INFO:
        start = min(DAILY_FILE_INFO)
        end = max(DAILY_FILE_INFO)
        raise argparse.ArgumentTypeError(
            f"{normalized} is not in the Xuancheng release range {start}..{end}"
        )
    return normalized


def human_size(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} GiB"


def download_one(url: str, dest: Path, expected_size: int, force: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        actual_size = dest.stat().st_size
        if actual_size == expected_size:
            print(f"[skip] {dest} already exists ({human_size(actual_size)})")
            return
        print(
            f"[warn] {dest} exists with {human_size(actual_size)}, "
            f"expected {human_size(expected_size)}; re-downloading"
        )

    part = dest.with_suffix(dest.suffix + ".part")
    if part.exists():
        part.unlink()

    print(f"[download] {url} -> {dest} ({human_size(expected_size)})")
    req = urllib.request.Request(url, headers={"User-Agent": "xuancheng-cityflow-pipeline/1.0"})
    started = time.time()
    last_report = started
    bytes_done = 0
    try:
        with urllib.request.urlopen(req, timeout=120) as response, part.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                bytes_done += len(chunk)
                now = time.time()
                if now - last_report >= 10:
                    pct = 100.0 * bytes_done / expected_size if expected_size else 0.0
                    print(f"  {human_size(bytes_done)} / {human_size(expected_size)} ({pct:.1f}%)")
                    last_report = now
    except urllib.error.URLError as exc:
        if part.exists():
            part.unlink()
        raise SystemExit(f"download failed for {url}: {exc}") from exc

    actual_size = part.stat().st_size
    if expected_size and actual_size != expected_size:
        part.unlink()
        raise SystemExit(
            f"downloaded size mismatch for {dest.name}: "
            f"got {actual_size}, expected {expected_size}"
        )
    part.replace(dest)
    elapsed = max(time.time() - started, 1e-6)
    print(f"[done] {dest.name}: {human_size(actual_size)} in {elapsed:.1f}s")


def build_manifest(args: argparse.Namespace) -> list[tuple[str, str, int, Path]]:
    data_root = Path(args.data_root).expanduser().resolve()
    raw_dir = data_root / "raw"
    manifest: list[tuple[str, str, int, Path]] = []

    selected_dates = sorted(DAILY_FILE_INFO) if args.all_days else args.days
    for date_text in selected_dates:
        file_id, size = DAILY_FILE_INFO[date_text]
        name = daily_name(date_text)
        manifest.append((name, f"{BASE_URL}/{file_id}", size, raw_dir / name))

    for name, (file_id, size) in SUPPORT_FILE_INFO.items():
        manifest.append((name, f"{BASE_URL}/{file_id}", size, raw_dir / name))

    if args.include_sumo:
        for name, (file_id, size) in SUMO_FILE_INFO.items():
            manifest.append((name, f"{BASE_URL}/{file_id}", size, raw_dir / name))

    if args.include_doc:
        for name, (file_id, size) in DOC_FILE_INFO.items():
            manifest.append((name, f"{BASE_URL}/{file_id}", size, raw_dir / name))

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Destination root for Xuancheng data.")
    parser.add_argument(
        "--days",
        nargs="+",
        type=parse_date,
        default=["2023-04-03"],
        help="Daily flow dates to download. Default: 2023-04-03.",
    )
    parser.add_argument("--all-days", action="store_true", help="Download all 30 released days.")
    parser.add_argument("--include-sumo", action="store_true", help="Also download xuancheng.net.xml.")
    parser.add_argument("--include-doc", action="store_true", help="Also download release Documentation.pdf.")
    parser.add_argument("--force", action="store_true", help="Re-download even if files already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print files and sizes without downloading.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_manifest(args)
    total_size = sum(size for _, _, size, _ in manifest)

    print(f"Figshare article: {ARTICLE_ID} ({ARTICLE_DOI})")
    print(f"Files selected: {len(manifest)}")
    print(f"Total selected size: {human_size(total_size)}")
    for name, url, size, dest in manifest:
        print(f"  {name}: {human_size(size)} -> {dest}")
        if args.dry_run:
            continue
        download_one(url, dest, size, args.force)

    return 0


if __name__ == "__main__":
    sys.exit(main())
