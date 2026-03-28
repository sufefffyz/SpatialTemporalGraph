#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _patch_deepknockoffs_version_lookup():
    import pkg_resources

    original_get_distribution = pkg_resources.get_distribution

    class _DummyDistribution:
        version = "0.1.0-local"

    def _patched_get_distribution(name):
        if str(name) == "DeepKnockoffs":
            return _DummyDistribution()
        return original_get_distribution(name)

    pkg_resources.get_distribution = _patched_get_distribution


def _load_cdmi(repo_path: Path):
    src_path = repo_path / "src"
    if not repo_path.exists() or not src_path.exists():
        raise FileNotFoundError(
            f"Expected a deepCausality checkout with a src/ directory, got: {repo_path}"
        )

    _patch_deepknockoffs_version_lookup()
    sys.path.insert(0, str(repo_path))
    sys.path.insert(0, str(src_path))
    import cdmi  # noqa: WPS433

    return cdmi


def main():
    parser = argparse.ArgumentParser(description="Run official CDMI on a pickled pandas DataFrame.")
    parser.add_argument("--repo-path", required=True)
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--params-json", required=True)
    parser.add_argument("--output-npy", required=True)
    args = parser.parse_args()

    repo_path = Path(args.repo_path).expanduser().resolve()
    input_path = Path(args.input_pkl).resolve()
    params_path = Path(args.params_json).resolve()
    output_path = Path(args.output_npy).resolve()

    frame = pd.read_pickle(input_path)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame in {input_path}")

    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index)

    with params_path.open("r", encoding="utf-8") as handle:
        params = json.load(handle)

    cdmi = _load_cdmi(repo_path)
    _, predicted_graph, _ = cdmi.causal_graph(frame, params)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.asarray(predicted_graph, dtype=float))


if __name__ == "__main__":
    main()
