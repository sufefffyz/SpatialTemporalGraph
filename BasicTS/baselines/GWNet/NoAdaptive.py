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
from basicts.scaler import MultiChannelZScoreScaler, ZScoreScaler
from basicts.utils import load_adj

from .arch import GraphWaveNet


class GraphWaveNetLastStep(GraphWaveNet):
    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        return output[..., -1:].contiguous()


def _env_int_list(name: str, default: list[int]) -> list[int]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _default_forward_features(data_name: str, desc: dict) -> list[int]:
    canonical = data_name.upper().replace("-", "_")
    largest_prefixes = ("SD", "GBA", "GLA", "CA")
    if canonical.startswith(largest_prefixes) and int(desc.get("num_features", 1)) >= 3:
        return [0, 1, 2]
    return [0, 1]


def _select_scaler(data_name: str, desc: dict):
    scaler_name = os.environ.get("BASICTS_SCALER", "").strip().lower()
    if scaler_name in {"multi", "multichannel", "multi_channel", "all_features", "allfeat", "allfeatz"}:
        return MultiChannelZScoreScaler, "allfeatZ"
    if scaler_name in {"target", "target_only", "zscore", "target_zscore"}:
        return ZScoreScaler, "targetZ"
    canonical = data_name.upper().replace("-", "_")
    if canonical.startswith(("KNOWAIR", "CCAQ")) and int(desc.get("num_features", 1)) > 1:
        return MultiChannelZScoreScaler, "allfeatZ"
    return ZScoreScaler, "targetZ"


DATA_NAME = os.environ.get("BASICTS_DATA_NAME", "SD")
DESC_PATH = Path("datasets") / DATA_NAME / "desc.json"
DESC = json.loads(DESC_PATH.read_text(encoding="utf-8"))
REGULAR_SETTINGS = DESC["regular_settings"]
INPUT_LEN = REGULAR_SETTINGS["INPUT_LEN"]
OUTPUT_LEN = REGULAR_SETTINGS["OUTPUT_LEN"]
TRAIN_VAL_TEST_RATIO = REGULAR_SETTINGS["TRAIN_VAL_TEST_RATIO"]
NORM_EACH_CHANNEL = REGULAR_SETTINGS["NORM_EACH_CHANNEL"]
RESCALE = REGULAR_SETTINGS["RESCALE"]
NULL_VAL = REGULAR_SETTINGS["NULL_VAL"]
FORWARD_FEATURES = _env_int_list("BASICTS_FORWARD_FEATURES", _default_forward_features(DATA_NAME, DESC))
SCALER_TYPE, SCALER_TAG = _select_scaler(DATA_NAME, DESC)

ADJ_PATH = Path("datasets") / DATA_NAME / "adj_mx.pkl"
SUPPORTS = None
if ADJ_PATH.exists():
    adj_mx, raw_adj = load_adj(str(ADJ_PATH), "doubletransition")
    if np.asarray(raw_adj).sum() > 0:
        SUPPORTS = [torch.tensor(i, dtype=torch.float32) for i in adj_mx]

MODEL_ARCH = GraphWaveNetLastStep
MODEL_PARAM = {
    "num_nodes": int(DESC["num_nodes"]),
    "supports": SUPPORTS,
    "dropout": 0.3,
    "gcn_bool": True,
    "addaptadj": False,
    "aptinit": None,
    "in_dim": len(FORWARD_FEATURES),
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
GRAPH_TAG = os.environ.get("BASICTS_GRAPH_TAG", "dataset-graph" if SUPPORTS is not None else "no-graph")

CFG = EasyDict()
CFG.DESCRIPTION = f"GraphWaveNet no-adaptive baseline on {DATA_NAME}"
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
CFG.SCALER.TYPE = SCALER_TYPE
CFG.SCALER.PARAM = EasyDict(
    {
        "dataset_name": DATA_NAME,
        "train_ratio": TRAIN_VAL_TEST_RATIO[0],
        "norm_each_channel": NORM_EACH_CHANNEL,
        "rescale": RESCALE,
    }
)

CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "GraphWaveNetNoAdaptive"
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = FORWARD_FEATURES
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
ckpt_parts = [DATA_NAME, GRAPH_TAG, "noaddapt", str(NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)]
if SCALER_TAG != "targetZ":
    ckpt_parts.append(SCALER_TAG)
if RUN_TAG:
    ckpt_parts.append(RUN_TAG)
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints", CFG.MODEL.NAME, "_".join(ckpt_parts))
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {"lr": 0.002, "weight_decay": 0.0001}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {"milestones": [1, 50], "gamma": 0.5}
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
CFG.EVAL.HORIZONS = [h for h in [3, 6, 12] if h <= OUTPUT_LEN]
CFG.EVAL.USE_GPU = True

CFG.WANDB = EasyDict()
CFG.WANDB.PROJECT = os.environ.get("WANDB_PROJECT", "adaptive_threshold_dynamic_weight")
CFG.WANDB.MODE = os.environ.get("WANDB_MODE", "online")
CFG.WANDB.RUN_NAME = os.environ.get("WANDB_NAME", f"GWNet_{DATA_NAME}_{GRAPH_TAG}_noaddapt")
CFG.WANDB.GROUP = os.environ.get("WANDB_RUN_GROUP", "gwnet_module_ablation_noaddapt")
CFG.WANDB.TAGS = ["gwnet", DATA_NAME.lower(), "no-addaptadj", GRAPH_TAG, SCALER_TAG]
