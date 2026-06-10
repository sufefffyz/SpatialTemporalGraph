import os

from easydict import EasyDict

from basicts.data.coreset_tsf_dataset import CoresetTimeSeriesForecastingDataset


def _ratio_tag(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _append_ckpt_tag(cfg: EasyDict, tag: str) -> None:
    root, name = os.path.split(cfg.TRAIN.CKPT_SAVE_DIR)
    if tag not in name:
        cfg.TRAIN.CKPT_SAVE_DIR = os.path.join(root, f"{name}_{tag}")


def apply_coreset_cfg(cfg: EasyDict, backbone_tag: str) -> EasyDict:
    """Attach train-only coreset selection to an existing BasicTS config.

    Runtime controls:
    - CORESET_STRATEGY: full, random, temporal, kcenter, temporal_kcenter,
      temporal_kcenter_difficulty
    - CORESET_RATIO: selected train-window ratio, e.g. 0.1
    - CORESET_SEED: selection seed
    - CORESET_TEMPORAL_PERIOD: samples per day, 96 for LargeST-SD 15min
    - CORESET_CACHE_DIR: cache path for selected train-window indices
    """

    strategy = os.environ.get("CORESET_STRATEGY", "random").strip().lower()
    ratio = float(os.environ.get("CORESET_RATIO", "0.1"))
    seed = int(os.environ.get("CORESET_SEED", os.environ.get("BASICTS_SEED", "2023")))
    temporal_period = int(os.environ.get("CORESET_TEMPORAL_PERIOD", "96"))
    cache_dir = os.environ.get(
        "CORESET_CACHE_DIR",
        os.path.join("datasets", cfg.DATASET.NAME, "coreset_indices"),
    )

    cfg.DATASET.TYPE = CoresetTimeSeriesForecastingDataset
    cfg.DATASET.PARAM.update(
        {
            "coreset_ratio": ratio,
            "coreset_strategy": strategy,
            "coreset_seed": seed,
            "coreset_temporal_period": temporal_period,
            "coreset_cache_dir": cache_dir,
        }
    )

    _append_ckpt_tag(cfg, f"coreset_{backbone_tag}_{strategy}_r{_ratio_tag(ratio)}_s{seed}")
    cfg.DESCRIPTION = f"{cfg.DESCRIPTION} [coreset:{strategy}, ratio={ratio}, seed={seed}]"
    return cfg
