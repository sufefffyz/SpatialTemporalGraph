import json
import os
import sys
from pathlib import Path

from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + "/../../.."))

from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import masked_mae, masked_mape, masked_rmse, masked_wape
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler

from .arch import STAEformer


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

MODEL_ARCH = STAEformer
MODEL_PARAM = {
    "num_nodes": int(DESC["num_nodes"]),
    "in_steps": INPUT_LEN,
    "out_steps": OUTPUT_LEN,
    "steps_per_day": int(round(24 * 60 / DESC.get("frequency (minutes)", 5))),
    "input_dim": 3,
    "output_dim": 1,
    "input_embedding_dim": int(os.environ.get("BASICTS_STAE_INPUT_EMB_DIM", "24")),
    "tod_embedding_dim": int(os.environ.get("BASICTS_STAE_TOD_EMB_DIM", "24")),
    "dow_embedding_dim": int(os.environ.get("BASICTS_STAE_DOW_EMB_DIM", "24")),
    "spatial_embedding_dim": int(os.environ.get("BASICTS_STAE_SPATIAL_EMB_DIM", "0")),
    "adaptive_embedding_dim": int(os.environ.get("BASICTS_STAE_ADAPTIVE_EMB_DIM", "80")),
    "feed_forward_dim": int(os.environ.get("BASICTS_STAE_FF_DIM", "256")),
    "num_heads": int(os.environ.get("BASICTS_STAE_NUM_HEADS", "4")),
    "num_layers": int(os.environ.get("BASICTS_STAE_NUM_LAYERS", "3")),
    "dropout": float(os.environ.get("BASICTS_STAE_DROPOUT", "0.1")),
    "use_mixed_proj": True,
}
NUM_EPOCHS = int(os.environ.get("BASICTS_NUM_EPOCHS", "100"))
RUN_TAG = os.environ.get("BASICTS_RUN_TAG", "").strip()
BATCH_SIZE = int(os.environ.get("BASICTS_BATCH_SIZE", "4"))

CFG = EasyDict()
CFG.DESCRIPTION = f"STAEformer on {DATA_NAME}"
CFG.GPU_NUM = 1
CFG.RUNNER = SimpleTimeSeriesForecastingRunner

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
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
CFG.MODEL.TARGET_FEATURES = [0]

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
CFG.TRAIN.OPTIM.PARAM = {"lr": float(os.environ.get("BASICTS_LR", "0.001")), "weight_decay": float(os.environ.get("BASICTS_WEIGHT_DECAY", "0.0003"))}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {"milestones": [20, 25], "gamma": 0.1}
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = BATCH_SIZE
CFG.TRAIN.DATA.SHUFFLE = True
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
