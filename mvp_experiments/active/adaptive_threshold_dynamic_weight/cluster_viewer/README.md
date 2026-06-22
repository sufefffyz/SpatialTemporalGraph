# Signal Cluster Viewer

Interactive static viewer for the 30-day SD/GLA/GBA signal clustering diagnostics.

It is intentionally separate from training code. The viewer consumes existing
`*_assignments.csv` files and metadata CSVs, then renders a canvas map with:

- dataset switch: SD / GLA / GBA
- embedding switch: time / freq
- day slider and play/pause animation
- cluster filter: show all or highlight one cluster while fading other nodes
- mouse hover tooltip
- wheel zoom, drag pan, and fit reset

## Build Data

From the repository root:

```bash
python mvp_experiments/active/adaptive_threshold_dynamic_weight/cluster_viewer/export_cluster_viewer_data.py
```

The script writes:

```text
mvp_experiments/active/adaptive_threshold_dynamic_weight/cluster_viewer/data/cluster_data.js
```

## Open

Open `index.html` directly in a browser:

```text
mvp_experiments/active/adaptive_threshold_dynamic_weight/cluster_viewer/index.html
```

The generated data is a JavaScript assignment rather than JSON fetch, so the
viewer works from `file://` without a local server.
