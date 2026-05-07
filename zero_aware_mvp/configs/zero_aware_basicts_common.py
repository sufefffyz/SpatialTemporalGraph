from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BASICTS_ROOT = REPO_ROOT / "BasicTS"
if str(BASICTS_ROOT) not in sys.path:
    sys.path.append(str(BASICTS_ROOT))

from stgraph_ext.config_utils import build_agcrn_cfg, build_gwnet_cfg


def _apply_common_overrides(cfg, model_tag: str):
    dataset_name = os.getenv("ZA_DATASET_NAME", cfg.DATASET.NAME)
    num_epochs = int(os.getenv("ZA_NUM_EPOCHS", str(cfg.TRAIN.NUM_EPOCHS)))
    seed = int(os.getenv("ZA_SEED", "42"))
    batch_size = int(os.getenv("ZA_BATCH_SIZE", str(cfg.TRAIN.DATA.BATCH_SIZE)))

    cfg.DESCRIPTION = f"Zero-aware MVP BasicTS config for {model_tag} on {dataset_name}"
    cfg.ENV.SEED = seed
    cfg.DATASET.NAME = dataset_name
    cfg.DATASET.PARAM.dataset_name = dataset_name
    cfg.SCALER.PARAM.dataset_name = dataset_name

    cfg.TRAIN.NUM_EPOCHS = num_epochs
    cfg.TRAIN.DATA.BATCH_SIZE = batch_size
    cfg.VAL.DATA.BATCH_SIZE = batch_size
    cfg.TEST.DATA.BATCH_SIZE = batch_size
    cfg.TRAIN.EARLY_STOPPING_PATIENCE = int(os.getenv("ZA_PATIENCE", "5"))

    # Critical for this benchmark: zero is an observed value, not missing data.
    cfg.METRICS.NULL_VAL = np.nan
    if "MAPE" in cfg.METRICS.FUNCS:
        cfg.METRICS.FUNCS.pop("MAPE")
    cfg.METRICS.TARGET = "MAE"

    cfg.EVAL.SAVE_RESULTS = True
    cfg.EVAL.USE_GPU = os.getenv("ZA_EVAL_USE_GPU", "1") != "0"

    input_len = cfg.DATASET.PARAM.input_len
    output_len = cfg.DATASET.PARAM.output_len
    cfg.TRAIN.CKPT_SAVE_DIR = os.path.join(
        "checkpoints",
        "zero_aware_mvp",
        model_tag,
        f"{dataset_name}_{num_epochs}_{input_len}_{output_len}_seed{seed}",
    )
    if "GRAPH_SNAPSHOT" in cfg:
        cfg.GRAPH_SNAPSHOT.ENABLED = os.getenv("ZA_GRAPH_SNAPSHOT", "0") == "1"
    return cfg


def build_zero_aware_agcrn_cfg():
    cfg = build_agcrn_cfg(
        os.getenv("ZA_DATASET_NAME", "TRAFFIC_VOLUME_FULL_5MIN"),
        num_epochs=int(os.getenv("ZA_NUM_EPOCHS", "3")),
    )
    return _apply_common_overrides(cfg, "AGCRN")


def build_zero_aware_gwnet_cfg():
    cfg = build_gwnet_cfg(
        os.getenv("ZA_DATASET_NAME", "TRAFFIC_VOLUME_FULL_5MIN"),
        num_epochs=int(os.getenv("ZA_NUM_EPOCHS", "3")),
    )
    return _apply_common_overrides(cfg, "GWNet")
