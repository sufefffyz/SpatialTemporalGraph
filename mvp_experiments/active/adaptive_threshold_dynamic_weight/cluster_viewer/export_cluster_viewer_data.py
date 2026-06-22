#!/usr/bin/env python3
"""Export SD/GLA/GBA daily signal-cluster assignments for the static viewer."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DATASETS = {
    "SD": {
        "meta": "PatchSTG/data/SD/sd_meta.csv",
        "assign_dir": "mvp_experiments/active/adaptive_threshold_dynamic_weight/outputs/signal_kmeans_clusters_1day_month/SD",
    },
    "GLA": {
        "meta": "PatchSTG/data/GLA/gla_meta.csv",
        "assign_dir": "mvp_experiments/active/adaptive_threshold_dynamic_weight/outputs/signal_kmeans_clusters_1day_month/GLA",
    },
    "GBA": {
        "meta": "PatchSTG/data/GBA/gba_meta.csv",
        "assign_dir": "mvp_experiments/active/adaptive_threshold_dynamic_weight/outputs/signal_kmeans_clusters_1day_month/GBA",
    },
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[3]
    return argparse.ArgumentParser(description=__doc__).parse_args(namespace=argparse.Namespace(
        repo_root=repo_root,
        output=script_dir / "data" / "cluster_data.js",
        embeddings=["time", "freq"],
        num_clusters=8,
    ))


def load_nodes(meta_path: Path) -> list[dict[str, object]]:
    with meta_path.open("r", encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        raise ValueError(f"{meta_path} is empty")
    columns = set(rows[0].keys())
    required = {"ID", "Lat", "Lng"}
    missing = required - columns
    if missing:
        raise ValueError(f"{meta_path} missing required columns: {sorted(missing)}")
    nodes = []
    for node_index, row in enumerate(rows):
        nodes.append(
            {
                "index": int(node_index),
                "id": str(row["ID"]),
                "lat": float(row["Lat"]),
                "lon": float(row["Lng"]),
                "fwy": "" if "Fwy" not in columns else str(row.get("Fwy", "")),
                "direction": ""
                if "Direction" not in columns
                else str(row["Direction"]),
                "county": "" if "County" not in columns else str(row.get("County", "")),
            }
        )
    return nodes


def load_assignment_matrix(path: Path, num_nodes: int) -> tuple[list[dict[str, object]], list[list[int]]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        raise ValueError(f"{path} is empty")
    columns = set(rows[0].keys())
    required = {"window_index", "start_step", "end_step", "node_index", "cluster"}
    missing = required - columns
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    days = []
    labels_by_day = []
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(int(row["window_index"]), []).append(row)
    for window_index in sorted(grouped):
        group = sorted(grouped[window_index], key=lambda row: int(row["node_index"]))
        if len(group) != num_nodes:
            raise ValueError(f"{path} window {window_index}: expected {num_nodes} rows, got {len(group)}")
        node_index = [int(row["node_index"]) for row in group]
        if node_index != list(range(num_nodes)):
            raise ValueError(f"{path} window {window_index}: node_index is not contiguous 0..N-1")
        labels = [int(row["cluster"]) for row in group]
        start_step = int(group[0]["start_step"])
        end_step = int(group[0]["end_step"])
        days.append(
            {
                "windowIndex": int(window_index),
                "label": f"Day {int(window_index) + 1:02d}",
                "startStep": start_step,
                "endStep": end_step,
            }
        )
        labels_by_day.append(labels)
    return days, labels_by_day


def build_payload(repo_root: Path, embeddings: list[str], num_clusters: int) -> dict[str, object]:
    datasets = {}
    for dataset, spec in DATASETS.items():
        meta_path = repo_root / spec["meta"]
        nodes = load_nodes(meta_path)
        assignment_payload = {}
        for embedding in embeddings:
            assignment_path = repo_root / spec["assign_dir"] / f"{dataset}_{embedding}_k{num_clusters}_assignments.csv"
            days, labels = load_assignment_matrix(assignment_path, len(nodes))
            assignment_payload[embedding] = {
                "days": days,
                "labels": labels,
            }
        datasets[dataset] = {
            "nodes": nodes,
            "numClusters": int(num_clusters),
            "embeddings": assignment_payload,
        }
    return {
        "schemaVersion": 1,
        "description": "Daily KMeans cluster assignments for SD/GLA/GBA signal diagnostics.",
        "datasets": datasets,
    }


def main() -> None:
    args = parse_args()
    payload = build_payload(args.repo_root, args.embeddings, args.num_clusters)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = "window.CLUSTER_VIEWER_DATA = "
    text += json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    text += ";\n"
    args.output.write_text(text, encoding="utf-8")
    for name, dataset in payload["datasets"].items():
        num_nodes = len(dataset["nodes"])
        num_days = len(dataset["embeddings"]["time"]["days"])
        print(f"{name}: nodes={num_nodes}, days={num_days}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
