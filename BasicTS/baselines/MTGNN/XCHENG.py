import json
import os
import sys
from pathlib import Path

from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + "/../../.."))

from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import masked_mae, masked_mape, masked_rmse, masked_wape
from basicts.scaler import ZScoreScaler

from .arch import MTGNN
from .runner import MTGNNRunner


def _null_val(value):
    if value is None:
        return float("nan")
    if isinstance(value, str) and value.lower() == "nan":
        return float("nan")
    return float(value)


DATA_NAME = os.environ.get("BASICTS_DATA_NAME", "XCHENG30D_5MIN_FLOW")
DESC_PATH = Path("datasets") / DATA_NAME / "desc.json"
DESC = json.loads(DESC_PATH.read_text(encoding="utf-8"))
REGULAR_SETTINGS = DESC["regular_settings"]
INPUT_LEN = int(REGULAR_SETTINGS["INPUT_LEN"])
OUTPUT_LEN = int(REGULAR_SETTINGS["OUTPUT_LEN"])
TRAIN_VAL_TEST_RATIO = REGULAR_SETTINGS["TRAIN_VAL_TEST_RATIO"]
NORM_EACH_CHANNEL = bool(REGULAR_SETTINGS["NORM_EACH_CHANNEL"])
RESCALE = bool(REGULAR_SETTINGS["RESCALE"])
NULL_VAL = _null_val(REGULAR_SETTINGS.get("NULL_VAL", "nan"))

MODEL_ARCH = MTGNN
NUM_NODES = int(DESC["num_nodes"])
MODEL_PARAM = {
    "gcn_true": True,
    "buildA_true": True,
    "gcn_depth": int(os.environ.get("BASICTS_MTGNN_GCN_DEPTH", "2")),
    "num_nodes": NUM_NODES,
    "predefined_A": None,
    "dropout": float(os.environ.get("BASICTS_MTGNN_DROPOUT", "0.3")),
    "subgraph_size": min(int(os.environ.get("BASICTS_MTGNN_SUBGRAPH_SIZE", "20")), NUM_NODES),
    "node_dim": int(os.environ.get("BASICTS_MTGNN_NODE_DIM", "40")),
    "dilation_exponential": int(os.environ.get("BASICTS_MTGNN_DILATION_EXPONENTIAL", "1")),
    "conv_channels": int(os.environ.get("BASICTS_MTGNN_CONV_CHANNELS", "32")),
    "residual_channels": int(os.environ.get("BASICTS_MTGNN_RESIDUAL_CHANNELS", "32")),
    "skip_channels": int(os.environ.get("BASICTS_MTGNN_SKIP_CHANNELS", "64")),
    "end_channels": int(os.environ.get("BASICTS_MTGNN_END_CHANNELS", "128")),
    "seq_length": INPUT_LEN,
    "in_dim": 1,
    "out_dim": OUTPUT_LEN,
    "layers": int(os.environ.get("BASICTS_MTGNN_LAYERS", "3")),
    "propalpha": float(os.environ.get("BASICTS_MTGNN_PROPALPHA", "0.05")),
    "tanhalpha": float(os.environ.get("BASICTS_MTGNN_TANHALPHA", "3")),
    "layer_norm_affline": True,
}
NUM_EPOCHS = int(os.environ.get("BASICTS_NUM_EPOCHS", "100"))
RUN_TAG = os.environ.get("BASICTS_RUN_TAG", "").strip()
BATCH_SIZE = int(os.environ.get("BASICTS_BATCH_SIZE", "16"))

CFG = EasyDict()
CFG.DESCRIPTION = f"MTGNN on {DATA_NAME} with learned adjacency"
CFG.GPU_NUM = 1
CFG.RUNNER = MTGNNRunner

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
ckpt_name_parts = [DATA_NAME, "xuancheng", str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)]
if RUN_TAG:
    ckpt_name_parts.append(RUN_TAG)
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints", MODEL_ARCH.__name__, "_".join(ckpt_name_parts))
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": float(os.environ.get("BASICTS_LR", "0.001")),
    "weight_decay": float(os.environ.get("BASICTS_WEIGHT_DECAY", "0.0001")),
}
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = BATCH_SIZE
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}
CFG.TRAIN.CL = EasyDict()
CFG.TRAIN.CL.WARM_EPOCHS = 0
CFG.TRAIN.CL.CL_EPOCHS = 3
CFG.TRAIN.CL.PREDICTION_LENGTH = OUTPUT_LEN
CFG.TRAIN.CUSTOM = EasyDict()
CFG.TRAIN.CUSTOM.STEP_SIZE = int(os.environ.get("BASICTS_MTGNN_STEP_SIZE", "100"))
CFG.TRAIN.CUSTOM.NUM_NODES = NUM_NODES
CFG.TRAIN.CUSTOM.NUM_SPLIT = int(os.environ.get("BASICTS_MTGNN_NUM_SPLIT", "1"))
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

CFG.WANDB = EasyDict()
CFG.WANDB.PROJECT = os.environ.get("WANDB_PROJECT", "xuancheng_cityflow")
CFG.WANDB.MODE = os.environ.get("WANDB_MODE", "disabled")
CFG.WANDB.RUN_NAME = os.environ.get("WANDB_NAME", f"{MODEL_ARCH.__name__}_{DATA_NAME}_{RUN_TAG}".rstrip("_"))
CFG.WANDB.GROUP = os.environ.get("WANDB_RUN_GROUP", "xuancheng_basicts_mtgnn")
CFG.WANDB.TAGS = ["xuancheng", "cityflow", "mtgnn", DATA_NAME]
