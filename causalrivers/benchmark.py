import hydra
from omegaconf import DictConfig, OmegaConf
import datetime
import time
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import pandas as pd

from tools.tools import (
    iter_joint_samples,
    load_label_graphs,
    load_joint_samples_eager,
    load_timeseries_source,
    prepare_single_sample,
    standard_preprocessing,
    save_run,
    save_multi_run,
)
from tools.scoring_tools import score
from tools.baseline_methods import get_baseline_method

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None


def _progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def resolve_method_cfgs(cfg: DictConfig):
    if cfg.get("methods"):
        return list(cfg.methods)
    return [cfg.method]


def resolve_load_mode(cfg: DictConfig):
    return str(cfg.get("load_mode", "streaming")).lower()


def chunk_sequence(items, chunk_size):
    for start_idx in range(0, len(items), chunk_size):
        yield start_idx // chunk_size, items[start_idx : start_idx + chunk_size]


def resolve_mp_context():
    preferred_method = "fork" if sys.platform.startswith("linux") else "spawn"
    return mp.get_context(preferred_method)


def _run_method_set(sample_data, method_cfgs):
    runtimes = {method_cfg.name: 0.0 for method_cfg in method_cfgs}
    preds = {method_cfg.name: [] for method_cfg in method_cfgs}
    method_fns = {
        method_cfg.name: get_baseline_method(method_cfg.name)
        for method_cfg in method_cfgs
    }

    for method_cfg in method_cfgs:
        method_start = time.perf_counter()
        preds[method_cfg.name].append(
            method_fns[method_cfg.name](sample_data.copy(), method_cfg)
        )
        runtimes[method_cfg.name] += time.perf_counter() - method_start
    return preds, runtimes


def _run_single_method_serial(samples, method_cfg):
    method_name = method_cfg.name
    method_fn = get_baseline_method(method_name)
    preds = []
    runtime = 0.0
    for sample_data in _progress(samples, total=len(samples), desc=f"{method_name.upper()} samples"):
        method_start = time.perf_counter()
        preds.append(method_fn(sample_data.copy(), method_cfg))
        runtime += time.perf_counter() - method_start
    return preds, runtime


def run_serial_benchmark(cfg, method_cfgs):
    runtimes = {method_cfg.name: 0.0 for method_cfg in method_cfgs}
    preds_by_method = {method_cfg.name: [] for method_cfg in method_cfgs}
    test_labels = []

    for sample_data, sample_labels in iter_joint_samples(
        cfg,
        preprocessing=standard_preprocessing if cfg.dt_preprocess else None,
    ):
        test_labels.append(sample_labels)
        preds, sample_runtimes = _run_method_set(sample_data, method_cfgs)
        for method_cfg in method_cfgs:
            method_name = method_cfg.name
            preds_by_method[method_name].extend(preds[method_name])
            runtimes[method_name] += sample_runtimes[method_name]

    return test_labels, preds_by_method, runtimes


def run_eager_benchmark(cfg, method_cfgs):
    print("Loading data eagerly into memory...")
    test_data, test_labels = load_joint_samples_eager(
        cfg,
        preprocessing=standard_preprocessing if cfg.dt_preprocess else None,
        human_readable_labels=False,
    )
    print(f"Eager mode loaded {len(test_data)} samples")

    preds_by_method = {}
    runtimes = {}
    for method_cfg in method_cfgs:
        preds_by_method[method_cfg.name], runtimes[method_cfg.name] = _run_single_method_serial(
            test_data,
            method_cfg,
        )
    return test_labels, preds_by_method, runtimes


def _parallel_worker(chunk_index, sample_graphs, cfg_dict, method_cfg_dicts):
    cfg = OmegaConf.create(cfg_dict)
    method_cfgs = [OmegaConf.create(method_cfg) for method_cfg in method_cfg_dicts]
    source = load_timeseries_source(cfg.data_path, index_col="datetime")
    preprocessing = standard_preprocessing if cfg.dt_preprocess else None

    chunk_labels = []
    preds_by_method = {method_cfg.name: [] for method_cfg in method_cfgs}
    runtimes = {method_cfg.name: 0.0 for method_cfg in method_cfgs}

    for sample_graph in sample_graphs:
        sample_data, sample_labels = prepare_single_sample(
            sample_graph,
            source,
            cfg,
            preprocessing=preprocessing,
            human_readable_labels=False,
        )
        chunk_labels.append(sample_labels)
        preds, sample_runtimes = _run_method_set(sample_data, method_cfgs)
        for method_cfg in method_cfgs:
            method_name = method_cfg.name
            preds_by_method[method_name].extend(preds[method_name])
            runtimes[method_name] += sample_runtimes[method_name]

    return chunk_index, chunk_labels, preds_by_method, runtimes


def run_parallel_benchmark(cfg, method_cfgs, sample_graphs):
    n_jobs = int(cfg.n_jobs)
    chunk_size = max(1, int(cfg.chunk_size))
    method_names = [method_cfg.name for method_cfg in method_cfgs]
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    method_cfg_dicts = [
        OmegaConf.to_container(method_cfg, resolve=True) for method_cfg in method_cfgs
    ]
    chunk_results = {}
    chunks = list(chunk_sequence(sample_graphs, chunk_size))
    max_workers = min(n_jobs, len(chunks))

    def collect_results(executor):
        future_to_chunk = {
            executor.submit(
                _parallel_worker,
                chunk_index,
                chunk_graphs,
                cfg_dict,
                method_cfg_dicts,
            ): chunk_index
            for chunk_index, chunk_graphs in chunks
        }

        progress_bar = tqdm(total=len(sample_graphs), desc="Parallel samples") if tqdm else None
        for future in as_completed(future_to_chunk):
            chunk_index, chunk_labels, chunk_preds, chunk_runtimes = future.result()
            chunk_results[chunk_index] = {
                "labels": chunk_labels,
                "preds": chunk_preds,
                "runtimes": chunk_runtimes,
            }
            if progress_bar is not None:
                progress_bar.update(len(chunk_labels))
        if progress_bar is not None:
            progress_bar.close()

    print(f"Running in parallel with {max_workers} workers and chunk_size={chunk_size}")
    try:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=resolve_mp_context(),
        ) as executor:
            collect_results(executor)
    except (PermissionError, OSError) as exc:
        print(f"Process workers unavailable ({exc}); falling back to thread workers")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            collect_results(executor)

    test_labels = []
    preds_by_method = {method_name: [] for method_name in method_names}
    runtimes = {method_name: 0.0 for method_name in method_names}
    for chunk_index in sorted(chunk_results):
        chunk_data = chunk_results[chunk_index]
        test_labels.extend(chunk_data["labels"])
        for method_name in method_names:
            preds_by_method[method_name].extend(chunk_data["preds"][method_name])
            runtimes[method_name] += chunk_data["runtimes"][method_name]

    return test_labels, preds_by_method, runtimes


# Example script to benchmark causal discovery methods.
@hydra.main(version_base=None, config_path="config", config_name="benchmark.yaml")
def main(cfg: DictConfig):
    start = datetime.datetime.now()
    method_cfgs = resolve_method_cfgs(cfg)
    method_names = [method_cfg.name for method_cfg in method_cfgs]
    if len(set(method_names)) != len(method_names):
        raise ValueError(f"Method names must be unique when running multiple baselines: {method_names}")

    print(cfg)
    print("Baselines:", ", ".join(method_names))
    load_mode = resolve_load_mode(cfg)
    n_jobs = int(cfg.get("n_jobs", 1))
    if load_mode == "eager":
        if n_jobs > 1:
            print("Eager mode keeps the original all-at-once loading path; sample-level prediction parallelism is disabled in this mode.")
        test_labels, preds_by_method, runtime_seconds = run_eager_benchmark(
            cfg,
            method_cfgs,
        )
    elif n_jobs > 1:
        sample_graphs = load_label_graphs(cfg)
        print(f"Loaded {len(sample_graphs)} label graphs for parallel execution")
        if len(sample_graphs) <= 1:
            print("Only one sample selected; falling back to serial execution")
            test_labels, preds_by_method, runtime_seconds = run_serial_benchmark(
                cfg,
                method_cfgs,
            )
        else:
            test_labels, preds_by_method, runtime_seconds = run_parallel_benchmark(
                cfg,
                method_cfgs,
                sample_graphs,
            )
    else:
        print("Loading data and streaming samples...")
        test_labels, preds_by_method, runtime_seconds = run_serial_benchmark(
            cfg,
            method_cfgs,
        )

    print(f"Scoring {len(test_labels)} samples...")
    score_tables = []
    for method_cfg in method_cfgs:
        method_name = method_cfg.name
        score_tables.append(
            score(
                preds_by_method[method_name],
                test_labels,
                remove_autoregressive=cfg.remove_diagonal,
                name=method_name,
                n_jobs=int(cfg.get("score_n_jobs", 1)),
                chunk_size=int(cfg.get("score_chunk_size", 512)),
            )
        )

    out = pd.concat(score_tables, axis=1)
    stop_time = datetime.datetime.now() - start
    runtimes = {
        method_name: datetime.timedelta(seconds=runtime_seconds[method_name])
        for method_name in method_names
    }
    print(out)

    if cfg.save_full_out:
        print("Saving...")
        if len(method_names) == 1:
            cfg.method = method_cfgs[0]
            save_run(out, stop_time, preds_by_method[method_names[0]], cfg)
        else:
            save_multi_run(out, runtimes, stop_time, preds_by_method, cfg, method_names)
    print("Done", stop_time)


if __name__ == "__main__":
    main()
