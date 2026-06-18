import os
import sys

from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + "/../../.."))

from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import masked_mae, masked_mape, masked_rmse, masked_wape
from basicts.runners import WandBTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings, load_dataset_desc

from .arch.agcrn_arch import AGCRN
from .arch.adaptive_importance_agcrn_arch import AdaptiveImportanceAGCRN


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _num_nodes(data_name: str) -> int:
    desc = load_dataset_desc(data_name)
    if "num_nodes" in desc:
        return int(desc["num_nodes"])
    return int(desc["shape"][1])


DATA_NAME = os.environ.get("BASICTS_DATA_NAME", "METR-LA")
VARIANT = os.environ.get("BASICTS_AGCRN_VARIANT", "original")
RUN_STAGE = os.environ.get("BASICTS_RUN_STAGE", "smoke")
NUM_EPOCHS = int(os.environ.get("BASICTS_NUM_EPOCHS", "5"))
EARLY_STOPPING_PATIENCE = int(os.environ.get("BASICTS_EARLY_STOPPING_PATIENCE", "30"))
ENV_DETERMINISTIC = _env_bool("BASICTS_DETERMINISTIC", False)
CUDNN_BENCHMARK = _env_bool("BASICTS_CUDNN_BENCHMARK", True)
CUDNN_DETERMINISTIC = _env_bool("BASICTS_CUDNN_DETERMINISTIC", False)
ENV_TAG = os.environ.get(
    "BASICTS_ENV_TAG",
    f"det{int(ENV_DETERMINISTIC)}_cudnndet{int(CUDNN_DETERMINISTIC)}",
)

VALID_VARIANTS = {"original", "identity_support", "frozen_random_embedding", "graph_no_relu"}
if VARIANT not in VALID_VARIANTS:
    raise ValueError(f"BASICTS_AGCRN_VARIANT must be one of {sorted(VALID_VARIANTS)}, got {VARIANT}")

AGCRN_DATASET_DEFAULTS = {
    "METR-LA": {"input_dim": 2, "forward_features": [0, 1], "test_interval": 1},
    "PEMS04": {"input_dim": 1, "forward_features": [0], "test_interval": 1},
    "PEMS07": {"input_dim": 1, "forward_features": [0], "test_interval": 1},
    "SD": {"input_dim": 1, "forward_features": [0], "test_interval": NUM_EPOCHS},
}
if DATA_NAME not in AGCRN_DATASET_DEFAULTS:
    raise ValueError(f"Unsupported AGCRN adaptive-importance dataset: {DATA_NAME}")

regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings["INPUT_LEN"]
OUTPUT_LEN = regular_settings["OUTPUT_LEN"]
TRAIN_VAL_TEST_RATIO = regular_settings["TRAIN_VAL_TEST_RATIO"]
NORM_EACH_CHANNEL = regular_settings["NORM_EACH_CHANNEL"]
RESCALE = regular_settings["RESCALE"]
NULL_VAL = regular_settings["NULL_VAL"]
NUM_NODES = _num_nodes(DATA_NAME)
DATASET_DEFAULTS = AGCRN_DATASET_DEFAULTS[DATA_NAME]

MODEL_ARCH = AGCRN if VARIANT == "original" else AdaptiveImportanceAGCRN
MODEL_NAMES = {
    "original": "AGCRNAdaptiveImportanceOriginal",
    "identity_support": "AGCRNAdaptiveImportanceIdentitySupport",
    "frozen_random_embedding": "AGCRNAdaptiveImportanceFrozenRandomEmbedding",
    "graph_no_relu": "AGCRNAdaptiveImportanceGraphNoRelu",
}
MODEL_NAME = MODEL_NAMES[VARIANT]

MODEL_PARAM = {
    "num_nodes": NUM_NODES,
    "input_dim": DATASET_DEFAULTS["input_dim"],
    "rnn_units": 64,
    "output_dim": 1,
    "horizon": OUTPUT_LEN,
    "num_layers": 2,
    "default_graph": True,
    "embed_dim": 10,
    "cheb_k": 2,
}
if VARIANT != "original":
    MODEL_PARAM.update(
        {
            "support_mode": "identity" if VARIANT == "identity_support" else "adaptive",
            "graph_use_relu": VARIANT != "graph_no_relu",
            "freeze_node_embeddings": VARIANT == "frozen_random_embedding",
        }
    )

CFG = EasyDict()
CFG.DESCRIPTION = "AGCRN adaptive graph importance ablation"
CFG.GPU_NUM = 1
CFG.RUNNER = WandBTimeSeriesForecastingRunner

CFG.ENV = EasyDict()
CFG.ENV.SEED = 42
CFG.ENV.DETERMINISTIC = ENV_DETERMINISTIC
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True
CFG.ENV.CUDNN.BENCHMARK = CUDNN_BENCHMARK
CFG.ENV.CUDNN.DETERMINISTIC = CUDNN_DETERMINISTIC

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
CFG.MODEL.NAME = MODEL_NAME
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = DATASET_DEFAULTS["forward_features"]
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
ckpt_parts = [DATA_NAME, RUN_STAGE, VARIANT, ENV_TAG, str(NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)]
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints", MODEL_NAME, "_".join(ckpt_parts))
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {"lr": 0.003}
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 64
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.EARLY_STOPPING_PATIENCE = EARLY_STOPPING_PATIENCE

CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 64

CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = DATASET_DEFAULTS["test_interval"]
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 64

CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [h for h in [3, 6, 12] if h <= OUTPUT_LEN]
CFG.EVAL.USE_GPU = True

CFG.WANDB = EasyDict()
CFG.WANDB.PROJECT = os.environ.get("WANDB_PROJECT", "adaptive_graph_importance_ablation")
CFG.WANDB.MODE = os.environ.get("WANDB_MODE", "offline")
CFG.WANDB.RUN_NAME = os.environ.get("WANDB_NAME", f"{MODEL_NAME}_{DATA_NAME}_{RUN_STAGE}_{VARIANT}")
CFG.WANDB.GROUP = os.environ.get("WANDB_RUN_GROUP", f"{DATA_NAME}_{RUN_STAGE}")
CFG.WANDB.TAGS = ["adaptive-graph-importance", "agcrn", RUN_STAGE, DATA_NAME, VARIANT, ENV_TAG]
