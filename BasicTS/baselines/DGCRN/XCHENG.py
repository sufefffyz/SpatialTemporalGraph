import json
import os
import random
import sys
from pathlib import Path

import torch
from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + "/../../.."))

from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import masked_mae, masked_mape, masked_rmse, masked_wape
from basicts.scaler import ZScoreScaler
from basicts.utils import load_adj

from .arch import DGCRN
from .runner import DGCRNRunner


def _null_val(value):
    if value is None:
        return float("nan")
    if isinstance(value, str) and value.lower() == "nan":
        return float("nan")
    return float(value)


DATA_NAME = os.environ.get("BASICTS_DATA_NAME", "XCHENG30D_5MIN_FLOW_TIME")
DESC_PATH = Path("datasets") / DATA_NAME / "desc.json"
DESC = json.loads(DESC_PATH.read_text(encoding="utf-8"))
REGULAR_SETTINGS = DESC["regular_settings"]
INPUT_LEN = int(REGULAR_SETTINGS["INPUT_LEN"])
OUTPUT_LEN = int(REGULAR_SETTINGS["OUTPUT_LEN"])
TRAIN_VAL_TEST_RATIO = REGULAR_SETTINGS["TRAIN_VAL_TEST_RATIO"]
NORM_EACH_CHANNEL = bool(REGULAR_SETTINGS["NORM_EACH_CHANNEL"])
RESCALE = bool(REGULAR_SETTINGS["RESCALE"])
NULL_VAL = _null_val(REGULAR_SETTINGS.get("NULL_VAL", "nan"))

MODEL_ARCH = DGCRN
adj_mx, _ = load_adj(f"datasets/{DATA_NAME}/adj_mx.pkl", "doubletransition")
MODEL_PARAM = {
    "gcn_depth": int(os.environ.get("BASICTS_DGCRN_GCN_DEPTH", "2")),
    "num_nodes": int(DESC["num_nodes"]),
    "predefined_A": [torch.Tensor(adj) for adj in adj_mx],
    "dropout": float(os.environ.get("BASICTS_DGCRN_DROPOUT", "0.3")),
    "subgraph_size": min(int(os.environ.get("BASICTS_DGCRN_SUBGRAPH_SIZE", "20")), int(DESC["num_nodes"])),
    "node_dim": int(os.environ.get("BASICTS_DGCRN_NODE_DIM", "40")),
    "middle_dim": int(os.environ.get("BASICTS_DGCRN_MIDDLE_DIM", "2")),
    "seq_length": OUTPUT_LEN,
    "in_dim": 2,
    "list_weight": [0.05, 0.95, 0.95],
    "tanhalpha": float(os.environ.get("BASICTS_DGCRN_TANHALPHA", "3")),
    "cl_decay_steps": int(os.environ.get("BASICTS_DGCRN_CL_DECAY_STEPS", "4000")),
    "rnn_size": int(os.environ.get("BASICTS_DGCRN_RNN_SIZE", "64")),
    "hyperGNN_dim": int(os.environ.get("BASICTS_DGCRN_HYPERGNN_DIM", "16")),
}
NUM_EPOCHS = int(os.environ.get("BASICTS_NUM_EPOCHS", "100"))
RUN_TAG = os.environ.get("BASICTS_RUN_TAG", "").strip()
BATCH_SIZE = int(os.environ.get("BASICTS_BATCH_SIZE", "8"))

CFG = EasyDict()
CFG.DESCRIPTION = f"DGCRN on {DATA_NAME}"
CFG.GPU_NUM = 1
CFG.RUNNER = DGCRNRunner
CFG._ = random.randint(-1000000, 1000000)

CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({"dataset_name": DATA_NAME, "train_val_test_ratio": TRAIN_VAL_TEST_RATIO, "input_len": INPUT_LEN, "output_len": OUTPUT_LEN})

CFG.SCALER = EasyDict()
CFG.SCALER.TYPE = ZScoreScaler
CFG.SCALER.PARAM = EasyDict({"dataset_name": DATA_NAME, "train_ratio": TRAIN_VAL_TEST_RATIO[0], "norm_each_channel": NORM_EACH_CHANNEL, "rescale": RESCALE})

CFG.MODEL = EasyDict()
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1]
CFG.MODEL.TARGET_FEATURES = [0]
CFG.MODEL.SETUP_GRAPH = True

CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict({"MAE": masked_mae, "MAPE": masked_mape, "RMSE": masked_rmse, "WAPE": masked_wape})
CFG.METRICS.TARGET = "MAE"
CFG.METRICS.NULL_VAL = NULL_VAL

CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
ckpt_name_parts = [DATA_NAME, "xuancheng", str(NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)]
if RUN_TAG:
    ckpt_name_parts.append(RUN_TAG)
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints", MODEL_ARCH.__name__, "_".join(ckpt_name_parts))
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {"lr": float(os.environ.get("BASICTS_LR", "0.001")), "weight_decay": float(os.environ.get("BASICTS_WEIGHT_DECAY", "0.0001"))}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {"milestones": [50, 100], "gamma": 0.5}
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = BATCH_SIZE
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}
CFG.TRAIN.CL = EasyDict()
CFG.TRAIN.CL.WARM_EPOCHS = 0
CFG.TRAIN.CL.CL_EPOCHS = 6
CFG.TRAIN.CL.PREDICTION_LENGTH = OUTPUT_LEN
CFG.TRAIN.EARLY_STOPPING_PATIENCE = int(os.environ.get("BASICTS_EARLY_STOPPING_PATIENCE", "30"))

CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = BATCH_SIZE

CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = NUM_EPOCHS
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = BATCH_SIZE

CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [OUTPUT_LEN] if OUTPUT_LEN <= 3 else [3, 6, OUTPUT_LEN]
CFG.EVAL.USE_GPU = True
