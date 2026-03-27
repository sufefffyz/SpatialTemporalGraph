import hydra
from omegaconf import DictConfig
import datetime

from tools.tools import (
    load_joint_samples,
    benchmarking,
    standard_preprocessing,
    save_run,
)
from tools.scoring_tools import score


# Example script to benchmark causal discovery methods.
@hydra.main(version_base=None, config_path="config", config_name="benchmark.yaml")
def main(cfg: DictConfig):
    
    
    if cfg.method.name == "var":
        from tools.baseline_methods import var_baseline as cd_method
    else:
        print("SPECIFY AND LOAD YOUR OWN METHOD HERE")

    start = datetime.datetime.now()
    print(cfg)
    print("Loading data...")
    # First, we load the full dataset from path and preprocess according to config..
    test_data, test_labels = load_joint_samples(
        cfg, preprocessing=standard_preprocessing if cfg.dt_preprocess else None
    )

    # Then we simply run the specified method on all samples.
    print("Performing Causal Discovery...")
    preds = benchmarking(test_data, cfg, cd_method)
    out = score(
        preds,
        test_labels,
        remove_autoregressive=cfg.remove_diagonal,
        name=cfg.method.name,
    )
    stop_time = datetime.datetime.now() - start
    print(out)

    if cfg.save_full_out:
        print("Saving...")
        save_run(out, stop_time, preds, cfg)
    print("Done", stop_time)


if __name__ == "__main__":
    main()
