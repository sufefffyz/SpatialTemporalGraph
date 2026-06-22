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

from .arch import STGCN


def _unwrap_adj_payload(raw_adj):
    if isinstance(raw_adj, (list, tuple)):
        if len(raw_adj) == 3:
            raw_adj = raw_adj[2]
        elif len(raw_adj) == 1:
            raw_adj = raw_adj[0]
    return np.asarray(raw_adj, dtype=np.float32)


DATA_NAME = os.environ.get("BASICTS_DATA_NAME", "SD_GSPTOPK_K064")
GRAPH_TAG = os.environ.get("BASICTS_GRAPH_TAG", DATA_NAME.replace("SD_", "").lower())
DIFFUSION_STEPS = int(os.environ.get("STGCN_DIFFUSION_STEPS", "2"))
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

RAW_ADJ = _unwrap_adj_payload(load_pkl(str(ADJ_PATH)))
RAW_ADJ = RAW_ADJ - np.diag(np.diag(RAW_ADJ))
SUPPORT_FORWARD = calculate_transition_matrix(RAW_ADJ).T
SUPPORT_BACKWARD = calculate_transition_matrix(RAW_ADJ.T).T

MODEL_ARCH = STGCN
MODEL_PARAM = {
    "Ks": DIFFUSION_STEPS,
    "Kt": 3,
    "blocks": [[1], [64, 16, 64], [64, 16, 64], [128, 128], [OUTPUT_LEN]],
    "T": INPUT_LEN,
    "n_vertex": int(DESC["num_nodes"]),
    "act_func": "glu",
    "graph_conv_type": "diffusion_graph_conv",
    "gso": [
        torch.tensor(SUPPORT_FORWARD, dtype=torch.float32),
        torch.tensor(SUPPORT_BACKWARD, dtype=torch.float32),
    ],
    "bias": True,
    "droprate": 0.5,
}
NUM_EPOCHS = int(os.environ.get("BASICTS_NUM_EPOCHS", "100"))
RUN_TAG = os.environ.get("BASICTS_RUN_TAG", "").strip()

CFG = EasyDict()
CFG.DESCRIPTION = f"STGCN-Diff on {DATA_NAME} with predefined top-k diffusion support ({GRAPH_TAG})"
CFG.GPU_NUM = 1
CFG.RUNNER = WandBTimeSeriesForecastingRunner

CFG.ENV = EasyDict()
CFG.ENV.SEED = int(os.environ.get("BASICTS_SEED", "2023"))
CFG.ENV.DETERMINISTIC = False
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True
CFG.ENV.CUDNN.BENCHMARK = True
CFG.ENV.CUDNN.DETERMINISTIC = False

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
CFG.MODEL.NAME = "STGCN_Diff"
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
ckpt_name_parts = [DATA_NAME, "topk_prior_diffusion", f"k{DIFFUSION_STEPS}", str(NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)]
if RUN_TAG:
    ckpt_name_parts.append(RUN_TAG)
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints", CFG.MODEL.NAME, "_".join(ckpt_name_parts))
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.0004,
    "weight_decay": 0.0003,
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
CFG.WANDB.RUN_NAME = os.environ.get("WANDB_NAME", f"{CFG.MODEL.NAME}_{DATA_NAME}_{GRAPH_TAG}_topk_prior_diffusion")
CFG.WANDB.GROUP = os.environ.get("WANDB_RUN_GROUP", "sd_topk_prior_stgcn_diffusion")
CFG.WANDB.TAGS = ["adaptive-threshold", "sd", "topk-prior", "stgcn-diffusion", GRAPH_TAG]
