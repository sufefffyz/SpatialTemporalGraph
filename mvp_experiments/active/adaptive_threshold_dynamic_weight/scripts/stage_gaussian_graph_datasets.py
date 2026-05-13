#!/usr/bin/env python3
"""Stage Gaussian-threshold graph variants as BasicTS dataset directories."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path, help="Graph summary JSON from build_gaussian_threshold_graphs.py.")
    parser.add_argument("--base-dataset", required=True, type=Path, help="Existing BasicTS dataset directory, e.g. BasicTS/datasets/SD.")
    parser.add_argument("--output-root", required=True, type=Path, help="BasicTS datasets directory.")
    parser.add_argument("--prefix", default="SD_OSRMGG", help="Prefix for staged dataset names.")
    parser.add_argument("--copy-adj", action="store_true", help="Copy adj_mx.pkl instead of symlinking.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def beta_suffix(label: str) -> str:
    match = re.match(r"beta(\d+)p(\d+)", label)
    if not match:
        return re.sub(r"[^A-Za-z0-9]+", "_", label).upper()
    whole, frac = match.groups()
    return f"B{whole}{frac}"


def replace_path(path: Path, target: Path, copy: bool = False) -> None:
    if path.exists() or path.is_symlink():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    if copy:
        shutil.copy2(target, path)
    else:
        path.symlink_to(target.resolve())


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=True)

    base_desc = json.loads((args.base_dataset / "desc.json").read_text(encoding="utf-8"))
    staged = []
    for graph in summary["graphs"]:
        label = graph["label"]
        dataset_name = f"{args.prefix}_{beta_suffix(label)}"
        dataset_dir = args.output_root / dataset_name
        if dataset_dir.exists() and not args.overwrite:
            raise FileExistsError(f"{dataset_dir} exists. Use --overwrite.")
        dataset_dir.mkdir(parents=True, exist_ok=True)

        desc = dict(base_desc)
        desc["name"] = dataset_name
        desc["graph_variant"] = {
            "source": "osrm_gaussian_global",
            "label": label,
            "target_avg_degree": graph["target_avg_degree"],
            "actual_avg_degree": graph["actual_avg_degree"],
            "actual_edge_count": graph["actual_edge_count"],
            "threshold_similarity": graph["threshold_similarity"],
            "threshold_distance_m": graph["threshold_distance_m"],
            "source_graph_path": graph["path"],
        }
        (dataset_dir / "desc.json").write_text(json.dumps(desc, indent=2), encoding="utf-8")

        for filename in ("data.dat", "meta.csv"):
            source = args.base_dataset / filename
            if source.exists():
                replace_path(dataset_dir / filename, source)

        graph_path = Path(graph["path"])
        replace_path(dataset_dir / "adj_mx.pkl", graph_path, copy=args.copy_adj)
        staged.append(
            {
                "dataset_name": dataset_name,
                "label": label,
                "path": str(dataset_dir),
                "adj_mx": str((dataset_dir / "adj_mx.pkl").resolve()),
            }
        )

    manifest = {
        "summary": str(args.summary),
        "base_dataset": str(args.base_dataset),
        "output_root": str(args.output_root),
        "staged": staged,
    }
    manifest_path = args.output_root / f"{args.prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
