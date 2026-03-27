"""Inspect file structures for downloaded Urban Traffic Benchmark data.

Usage:
  python scripts/inspect_downloaded_data.py \
      --data-dir /data/yuzhang_fei/Urban_Traffic_Benchmark
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path
from typing import Iterable
from datetime import datetime

import numpy as np
from numpy.lib import format as npformat


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _read_npy_header_from_npz(npz_file: zipfile.ZipFile, member_name: str):
    with npz_file.open(member_name, "r") as f:
        version = npformat.read_magic(f)
        if version == (1, 0):
            shape, fortran_order, dtype = npformat.read_array_header_1_0(f)
        elif version in ((2, 0), (3, 0)):
            shape, fortran_order, dtype = npformat.read_array_header_2_0(f)
        else:
            raise ValueError(f"Unsupported .npy format version: {version}")

    return shape, fortran_order, dtype


def summarize_npz(file_path: Path, max_items: int, deep: bool = False) -> list[str]:
    lines: list[str] = []

    # Fast path: only inspect .npy headers in the zip archive, without materializing arrays.
    with zipfile.ZipFile(file_path, mode="r") as npz_zip:
        member_names = [name for name in npz_zip.namelist() if name.endswith(".npy")]

    keys = [Path(name).stem for name in member_names]
    lines.append(f"  keys ({len(keys)}): {keys}")

    with zipfile.ZipFile(file_path, mode="r") as npz_zip:
        for member_name in member_names[:max_items]:
            key = Path(member_name).stem
            try:
                shape, fortran_order, dtype = _read_npy_header_from_npz(npz_zip, member_name)
                lines.append(
                    f"  - {key}: shape={shape}, dtype={dtype}, fortran_order={fortran_order}"
                )
            except Exception as exc:
                lines.append(f"  - {key}: header parse failed: {exc}")

    if deep:
        # Optional slow path: materialize arrays and compute value ranges.
        try:
            data = np.load(file_path, allow_pickle=False)
            for key in keys[:max_items]:
                try:
                    arr = data[key]
                    lines.append(f"    {key} min={np.nanmin(arr):.6g}, max={np.nanmax(arr):.6g}")
                except Exception:
                    lines.append(f"    {key} min/max unavailable")
        except Exception as exc:
            lines.append(f"  deep inspection failed: {exc}")

    if len(keys) > max_items:
        lines.append(f"  ... {len(keys) - max_items} more arrays")

    return lines


def summarize_parquet(file_path: Path, max_items: int) -> list[str]:
    lines: list[str] = []

    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:  # pragma: no cover
        lines.append(f"  failed to import pyarrow: {exc}")
        lines.append("  install pyarrow to inspect parquet schema")
        return lines

    pq_file = pq.ParquetFile(file_path)
    schema = pq_file.schema_arrow
    lines.append(f"  row_groups={pq_file.num_row_groups}, rows={pq_file.metadata.num_rows}")
    lines.append(f"  columns={len(schema.names)}")

    for name, field in list(zip(schema.names, schema.types))[:max_items]:
        lines.append(f"  - {name}: {field}")

    if len(schema.names) > max_items:
        lines.append(f"  ... {len(schema.names) - max_items} more columns")

    return lines


def summarize_text(file_path: Path, max_lines: int) -> list[str]:
    lines: list[str] = []
    try:
        with file_path.open("r", encoding="utf-8", errors="replace") as f:
            for idx, line in enumerate(f):
                if idx >= max_lines:
                    break
                lines.append(f"  L{idx + 1}: {line.rstrip()}")
    except Exception as exc:
        lines.append(f"  failed to read text: {exc}")
    return lines


def iter_files(data_dir: Path, recursive: bool = True) -> Iterable[Path]:
    if recursive:
        for root, _, files in os.walk(data_dir):
            for name in sorted(files):
                yield Path(root) / name
    else:
        for p in sorted(data_dir.iterdir()):
            if p.is_file():
                yield p


def inspect_file(file_path: Path, max_items: int, max_text_lines: int, npz_deep: bool) -> list[str]:
    suffix = file_path.suffix.lower()

    if suffix == ".npz":
        return summarize_npz(file_path, max_items=max_items, deep=npz_deep)
    if suffix == ".parquet":
        return summarize_parquet(file_path, max_items=max_items)
    if suffix in {".log", ".txt", ".md", ".csv", ".aria2"}:
        return summarize_text(file_path, max_lines=max_text_lines)

    return ["  unsupported format for deep inspection (showing metadata only)"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect downloaded data files and print their structure.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with downloaded data files.")
    parser.add_argument("--max-items", type=int, default=10, help="Max arrays/columns to print per file.")
    parser.add_argument("--max-text-lines", type=int, default=5, help="Max preview lines for text-like files.")
    parser.add_argument("--non-recursive", action="store_true", help="Only inspect top-level files.")
    parser.add_argument(
        "--npz-deep",
        action="store_true",
        help="Deep NPZ check (computes per-array min/max). This can be slow for large arrays.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help=(
            "Where to save the report. "
            "Default: scripts/inspect_downloaded_data_report_<timestamp>.txt"
        ),
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"data directory does not exist or is not a directory: {data_dir}")

    if args.output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(__file__).resolve().parent / f"inspect_downloaded_data_report_{timestamp}.txt"
    else:
        output_file = args.output_file

    output_file.parent.mkdir(parents=True, exist_ok=True)

    report_lines: list[str] = []

    def emit(line: str = "") -> None:
        print(line)
        report_lines.append(line)

    emit(f"Inspecting: {data_dir}")
    emit()

    files = sorted(iter_files(data_dir, recursive=not args.non_recursive))
    if not files:
        emit("No files found.")
        output_file.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        emit(f"Saved report to: {output_file}")
        return

    for file_path in files:
        stat = file_path.stat()
        rel = safe_relpath(file_path, data_dir)
        emit(f"[FILE] {rel}")
        emit(f"  size={format_size(stat.st_size)}")
        emit(f"  ext={file_path.suffix.lower() or '<none>'}")

        try:
            details = inspect_file(
                file_path,
                max_items=args.max_items,
                max_text_lines=args.max_text_lines,
                npz_deep=args.npz_deep,
            )
            for line in details:
                emit(line)
        except Exception as exc:
            emit(f"  inspect failed: {exc}")

        emit()

    output_file.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    emit(f"Saved report to: {output_file}")


if __name__ == "__main__":
    main()
