import json
import os
import sys
from pathlib import Path

import torch
from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + "/../../.."))

from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import masked_mae, masked_mape, masked_rmse, masked_wape
from basicts.runners import WandBTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import load_adj

from .arch import D2STGNN

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

MODEL_ARCH = D2STGNN
adj_mx, _ = load_adj(f"datasets/{DATA_NAME}/adj_mx.pkl", "doubletransition")
MODEL_PARAM = {
    "num_feat": 1,
    "num_hidden": 32,
    "dropout": 0.1,
    "seq_length": OUTPUT_LEN,
    "k_t": 3,
    "k_s": 2,
    "gap": 3,
    "num_nodes": int(DESC["num_nodes"]),
    "adjs": [torch.tensor(adj) for adj in adj_mx],
    "num_layers": 5,
    "num_modalities": 2,
    "node_hidden": 12,
    "time_emb_dim": 12,
    "time_in_day_size": int(round(24 * 60 / DESC.get("frequency (minutes)", 15))),
    "day_in_week_size": 7,
}
NUM_EPOCHS = int(os.environ.get("BASICTS_NUM_EPOCHS", "100"))
RUN_TAG = os.environ.get("BASICTS_RUN_TAG", "").strip()

CFG = EasyDict()
CFG.DESCRIPTION = "D2STGNN on LargeST SD original graph"
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
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
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
ckpt_name_parts = [DATA_NAME, "original", str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)]
if RUN_TAG:
    ckpt_name_parts.append(RUN_TAG)
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints", MODEL_ARCH.__name__, "_".join(ckpt_name_parts))
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.002,
    "weight_decay": 1.0e-5,
    "eps": 1.0e-8,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [1, 30, 38, 46, 54, 62, 70, 80],
    "gamma": 0.5,
}
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 64
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}
CFG.TRAIN.CL = EasyDict()
CFG.TRAIN.CL.WARM_EPOCHS = 30
CFG.TRAIN.CL.CL_EPOCHS = 3
CFG.TRAIN.CL.PREDICTION_LENGTH = OUTPUT_LEN
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
CFG.WANDB.RUN_NAME = os.environ.get("WANDB_NAME", f"{MODEL_ARCH.__name__}_{DATA_NAME}_original_{RUN_TAG}".rstrip("_"))
CFG.WANDB.GROUP = os.environ.get("WANDB_RUN_GROUP", "sd_original_d2stgnn")
CFG.WANDB.TAGS = ["adaptive-threshold", "sd", "original", "d2stgnn", "baseline"]
