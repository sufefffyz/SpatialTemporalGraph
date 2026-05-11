import json
import os
import sys
from pathlib import Path

from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + "/../../.."))

from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import masked_mae, masked_mape, masked_rmse, masked_wape
from basicts.scaler import ZScoreScaler

from .arch import BasicTSFlowNet
from .runner import FlowNetOfficialWandBRunner


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

MODEL_ARCH = BasicTSFlowNet
DIST_MTX_PATH = os.environ.get(
    "FLOWNET_DIST_MTX",
    "/home/yuzhang_fei/stg_artifacts_archive/adaptive_threshold_dynamic_weight/"
    "distance_matrices/SD/SD_osrm_shortest_distance_m.npy",
)
MODEL_PARAM = {
    "num_nodes": int(DESC["num_nodes"]),
    "seq_len": INPUT_LEN,
    "pred_len": OUTPUT_LEN,
    "patch_len": 4,
    "stride": 2,
    "moving_avg": 3,
    "nhead": 4,
    "ffn_dim": 128,
    "d_model": 64,
    "n_expert": 16,
    "n_layer": 2,
    "dropout": 0.1,
    "rate": 4,
    "freq": "5min",
    "dist_mtx_path": DIST_MTX_PATH,
    "dist_norm": "max",
}
NUM_EPOCHS = 100
RUN_TAG = os.environ.get("BASICTS_RUN_TAG", "").strip()

CFG = EasyDict()
CFG.DESCRIPTION = "FlowNet on LargeST SD with OSRM shortest-path distance"
CFG.GPU_NUM = 1
CFG.RUNNER = FlowNetOfficialWandBRunner

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
CKPT_NAME_PARTS = [DATA_NAME, "osrm", str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)]
if RUN_TAG:
    CKPT_NAME_PARTS.append(RUN_TAG)
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    "FlowNet",
    "_".join(CKPT_NAME_PARTS),
)
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "AdamW"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.001,
    "weight_decay": 0.0001,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [10, 20, 30, 40],
    "gamma": 0.5,
}
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 8
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}
CFG.TRAIN.EARLY_STOPPING_WARMUP = 20
CFG.TRAIN.EARLY_STOPPING_PATIENCE = 10

CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 16

CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = NUM_EPOCHS
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 16

CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [3, 6, 12]
CFG.EVAL.USE_GPU = True
