import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + "/../../.."))

from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import masked_mae, masked_mape, masked_rmse, masked_wape
from basicts.runners import WandBTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils.adjacent_matrix_norm import calculate_transition_matrix
from basicts.utils.serialization import load_pkl

from .arch import GraphWaveNet


DATA_NAME = os.environ.get("BASICTS_DATA_NAME", "SD_OSRMGG_B100")
GRAPH_TAG = os.environ.get("BASICTS_GRAPH_TAG", DATA_NAME.replace("SD_OSRMGG_", "beta").lower())
DESC_PATH = Path("datasets") / DATA_NAME / "desc.json"
ADJ_PATH = Path("datasets") / DATA_NAME / "adj_mx.pkl"
DESC = json.loads(DESC_PATH.read_text(encoding="utf-8"))
REGULAR_SETTINGS = DESC["regular_settings"]
INPUT_LEN = REGULAR_SETTINGS["INPUT_LEN"]
OUTPUT_LEN = REGULAR_SETTINGS["OUTPUT_LEN"]
TRAIN_VAL_TEST_RATIO = REGULAR_SETTINGS["TRAIN_VAL_TEST_RATIO"]
NORM_EACH_CHANNEL = REGULAR_SETTINGS["NORM_EACH_CHANNEL"]
RESCALE = REGULAR_SETTINGS["RESCALE"]
NULL_VAL = REGULAR_SETTINGS["NULL_VAL"]


raw_adj = load_pkl(str(ADJ_PATH))
if isinstance(raw_adj, (list, tuple)):
    if len(raw_adj) == 3:
        raw_adj = raw_adj[2]
    elif len(raw_adj) == 1:
        raw_adj = raw_adj[0]
raw_adj = np.asarray(raw_adj, dtype=np.float32)

support_forward = calculate_transition_matrix(raw_adj).T
support_backward = calculate_transition_matrix(raw_adj.T).T

MODEL_ARCH = GraphWaveNet
MODEL_PARAM = {
    "num_nodes": int(DESC["num_nodes"]),
    "supports": [
        torch.tensor(support_forward, dtype=torch.float32),
        torch.tensor(support_backward, dtype=torch.float32),
    ],
    "dropout": 0.3,
    "gcn_bool": True,
    "addaptadj": False,
    "aptinit": None,
    "in_dim": 2,
    "out_dim": OUTPUT_LEN,
    "residual_channels": 32,
    "dilation_channels": 32,
    "skip_channels": 256,
    "end_channels": 512,
    "kernel_size": 2,
    "blocks": 4,
    "layers": 2,
}
NUM_EPOCHS = int(os.environ.get("BASICTS_NUM_EPOCHS", "100"))
RUN_TAG = os.environ.get("BASICTS_RUN_TAG", "").strip()

CFG = EasyDict()
CFG.DESCRIPTION = f"GraphWaveNet fixed graph on {DATA_NAME} ({GRAPH_TAG})"
CFG.GPU_NUM = 1
CFG.RUNNER = WandBTimeSeriesForecastingRunner

CFG.ENV = EasyDict()
CFG.ENV.SEED = int(os.environ.get("BASICTS_SEED", "2023"))
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
CFG.MODEL.FORWARD_FEATURES = [0, 1]
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
CKPT_NAME_PARTS = [DATA_NAME, "osrm_gaussian_global_fixed", str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)]
if RUN_TAG:
    CKPT_NAME_PARTS.append(RUN_TAG)
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    MODEL_ARCH.__name__,
    "_".join(CKPT_NAME_PARTS),
)
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.002,
    "weight_decay": 0.0001,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [1, 50],
    "gamma": 0.5,
}
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 64
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}
CFG.TRAIN.EARLY_STOPPING_PATIENCE = 30

CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 64

CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = NUM_EPOCHS
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 64

CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [3, 6, 12]
CFG.EVAL.USE_GPU = True

CFG.WANDB = EasyDict()
CFG.WANDB.PROJECT = os.environ.get("WANDB_PROJECT", "adaptive_threshold_dynamic_weight")
CFG.WANDB.MODE = os.environ.get("WANDB_MODE", "online")
CFG.WANDB.RUN_NAME = os.environ.get("WANDB_NAME", f"{MODEL_ARCH.__name__}_{DATA_NAME}_{GRAPH_TAG}")
CFG.WANDB.GROUP = os.environ.get("WANDB_RUN_GROUP", "sd_osrm_gaussian_global")
CFG.WANDB.TAGS = ["adaptive-threshold", "sd", "osrm-gaussian-global", "gwnet", GRAPH_TAG]
