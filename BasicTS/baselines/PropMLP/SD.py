import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + "/../../.."))

from basicts.data import TimeSeriesForecastingDataset
from basicts.utils.adjacent_matrix_norm import calculate_transition_matrix
from basicts.metrics import masked_mae, masked_mape, masked_rmse, masked_wape
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils.serialization import load_pkl

from .arch import PropMLP

DATA_NAME = "SD"
DESC_PATH = Path("datasets") / DATA_NAME / "desc.json"
DESC = json.loads(DESC_PATH.read_text(encoding="utf-8"))
REGULAR_SETTINGS = DESC["regular_settings"]
INPUT_LEN = REGULAR_SETTINGS["INPUT_LEN"]
OUTPUT_LEN = REGULAR_SETTINGS["OUTPUT_LEN"]
TRAIN_VAL_TEST_RATIO = REGULAR_SETTINGS["TRAIN_VAL_TEST_RATIO"]
NORM_EACH_CHANNEL = REGULAR_SETTINGS["NORM_EACH_CHANNEL"]
RESCALE = REGULAR_SETTINGS["RESCALE"]
NULL_VAL = REGULAR_SETTINGS["NULL_VAL"]

GRAPH_VARIANT = os.environ.get("PROP_GRAPH_VARIANT", "distthre")
FEATURE_MODE = os.environ.get("PROP_FEATURE_MODE", "x_pfpb")
MAX_ORDER = int(os.environ.get("PROP_MAX_ORDER", "1"))
HIDDEN_DIM = int(os.environ.get("PROP_HIDDEN_DIM", "128"))
DROPOUT = float(os.environ.get("PROP_DROPOUT", "0.1"))
SEED = int(os.environ.get("PROP_SEED", "42"))
ADD_SELF_LOOP = os.environ.get("PROP_ADD_SELF_LOOP", "0").lower() in {"1", "true", "yes", "on"}


def _graph_path() -> str:
    if GRAPH_VARIANT == "distthre":
        return "datasets/SD/adj_mx.pkl"
    if GRAPH_VARIANT == "physical_dir":
        return "datasets/SD_phys/adj_mx.pkl"
    raise ValueError(f"Unsupported PROP_GRAPH_VARIANT: {GRAPH_VARIANT}")


def _unwrap_adj_payload(raw_adj):
    if isinstance(raw_adj, (list, tuple)):
        if len(raw_adj) == 3:
            raw_adj = raw_adj[2]
        elif len(raw_adj) == 1:
            raw_adj = raw_adj[0]
    return np.asarray(raw_adj, dtype=np.float32)


def _supports():
    raw_adj = _unwrap_adj_payload(load_pkl(_graph_path()))
    if ADD_SELF_LOOP:
        raw_adj = raw_adj + np.eye(raw_adj.shape[0], dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        support_forward = np.asarray(calculate_transition_matrix(raw_adj).T, dtype=np.float32)
        support_backward = np.asarray(calculate_transition_matrix(raw_adj.T).T, dtype=np.float32)
    return [
        torch.tensor(support_forward, dtype=torch.float32),
        torch.tensor(support_backward, dtype=torch.float32),
    ]


SUPPORTS = _supports()
if FEATURE_MODE == "x_only":
    MODEL_SUPPORTS = []
    INCLUDE_ORIGINAL = True
elif FEATURE_MODE == "pfpb":
    MODEL_SUPPORTS = SUPPORTS
    INCLUDE_ORIGINAL = False
elif FEATURE_MODE == "x_pfpb":
    MODEL_SUPPORTS = SUPPORTS
    INCLUDE_ORIGINAL = True
else:
    raise ValueError(f"Unsupported PROP_FEATURE_MODE: {FEATURE_MODE}")

MODEL_ARCH = PropMLP
MODEL_PARAM = {
    "num_nodes": int(DESC["num_nodes"]),
    "seq_len": INPUT_LEN,
    "pred_len": OUTPUT_LEN,
    "supports": MODEL_SUPPORTS,
    "max_order": MAX_ORDER,
    "include_original": INCLUDE_ORIGINAL,
    "hidden_dim": HIDDEN_DIM,
    "dropout": DROPOUT,
}
NUM_EPOCHS = 100

CFG = EasyDict()
CFG.DESCRIPTION = (
    f"PropMLP on old SD: graph={GRAPH_VARIANT}, mode={FEATURE_MODE}, "
    f"order={MAX_ORDER}, self_loop={ADD_SELF_LOOP}"
)
CFG.GPU_NUM = 1
CFG.RUNNER = SimpleTimeSeriesForecastingRunner

CFG.ENV = EasyDict()
CFG.ENV.SEED = SEED
CFG.ENV.DETERMINISTIC = True
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True
CFG.ENV.CUDNN.BENCHMARK = True
CFG.ENV.CUDNN.DETERMINISTIC = True

CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict(
    {
        "dataset_name": DATA_NAME,
        "train_val_test_ratio": TRAIN_VAL_TEST_RATIO,
        "input_len": INPUT_LEN,
        "output_len": OUTPUT_LEN,
    }
)

CFG.SCALER = EasyDict()
CFG.SCALER.TYPE = ZScoreScaler
CFG.SCALER.PARAM = EasyDict(
    {
        "dataset_name": DATA_NAME,
        "train_ratio": TRAIN_VAL_TEST_RATIO[0],
        "norm_each_channel": NORM_EACH_CHANNEL,
        "rescale": RESCALE,
    }
)

CFG.MODEL = EasyDict()
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0]
CFG.MODEL.TARGET_FEATURES = [0]

CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict(
    {
        "MAE": masked_mae,
        "MAPE": masked_mape,
        "RMSE": masked_rmse,
        "WAPE": masked_wape,
    }
)
CFG.METRICS.TARGET = "MAE"
CFG.METRICS.NULL_VAL = NULL_VAL

CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    MODEL_ARCH.__name__,
    "_".join(
        [
            DATA_NAME,
            GRAPH_VARIANT,
            FEATURE_MODE,
            f"k{MAX_ORDER}",
            "selfloop" if ADD_SELF_LOOP else "noselfloop",
            str(NUM_EPOCHS),
            str(INPUT_LEN),
            str(OUTPUT_LEN),
        ]
    ),
)
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.001,
    "weight_decay": 0.0001,
}
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 64
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.EARLY_STOPPING_PATIENCE = 15

CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 64

CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 64

CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [3, 6, 12]
CFG.EVAL.USE_GPU = True
