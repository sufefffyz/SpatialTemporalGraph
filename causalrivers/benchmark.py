import hydra
from omegaconf import DictConfig
import datetime
import pandas as pd

from tools.tools import (
    iter_joint_samples,
    standard_preprocessing,
    save_run,
    save_multi_run,
)
from tools.scoring_tools import score
from tools.baseline_methods import get_baseline_method


def resolve_method_cfgs(cfg: DictConfig):
    if cfg.get("methods"):
        return list(cfg.methods)
    return [cfg.method]


# Example script to benchmark causal discovery methods.
@hydra.main(version_base=None, config_path="config", config_name="benchmark.yaml")
def main(cfg: DictConfig):
    start = datetime.datetime.now()
    method_cfgs = resolve_method_cfgs(cfg)
    method_names = [method_cfg.name for method_cfg in method_cfgs]
    if len(set(method_names)) != len(method_names):
        raise ValueError(f"Method names must be unique when running multiple baselines: {method_names}")

    method_fns = {method_name: get_baseline_method(method_name) for method_name in method_names}
    runtimes = {method_name: datetime.timedelta() for method_name in method_names}
    preds_by_method = {method_name: [] for method_name in method_names}
    test_labels = []

    print(cfg)
    print("Baselines:", ", ".join(method_names))
    print("Loading data and streaming samples...")
    for sample_data, sample_labels in iter_joint_samples(
        cfg,
        preprocessing=standard_preprocessing if cfg.dt_preprocess else None,
    ):
        test_labels.append(sample_labels)
        for method_cfg in method_cfgs:
            method_name = method_cfg.name
            method_start = datetime.datetime.now()
            preds_by_method[method_name].append(
                method_fns[method_name](sample_data.copy(), method_cfg)
            )
            runtimes[method_name] += datetime.datetime.now() - method_start

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
            )
        )

    out = pd.concat(score_tables, axis=1)
    stop_time = datetime.datetime.now() - start
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
