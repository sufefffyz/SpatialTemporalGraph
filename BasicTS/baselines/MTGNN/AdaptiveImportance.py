import os
import sys

import torch
from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + "/../../.."))

from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import masked_mae, masked_mape, masked_rmse, masked_wape
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings, load_adj, load_dataset_desc

from .arch.adaptive_importance_mtgnn_arch import AdaptiveImportanceMTGNN
from .arch.mtgnn_arch import MTGNN
from .runner import MTGNNRunner


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


def _load_physical_adjacency(data_name: str, num_nodes: int):
    _, raw_adj = load_adj(os.path.join("datasets", data_name, "adj_mx.pkl"), "doubletransition")
    return torch.tensor(raw_adj, dtype=torch.float32) - torch.eye(num_nodes)


DATA_NAME = os.environ.get("BASICTS_DATA_NAME", "METR-LA")
VARIANT = os.environ.get("BASICTS_MTGNN_VARIANT", "original")
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

VALID_VARIANTS = {"original", "fixed_physical", "frozen_random_graph", "no_relu_score"}
if VARIANT not in VALID_VARIANTS:
    raise ValueError(f"BASICTS_MTGNN_VARIANT must be one of {sorted(VALID_VARIANTS)}, got {VARIANT}")

regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings["INPUT_LEN"]
OUTPUT_LEN = regular_settings["OUTPUT_LEN"]
TRAIN_VAL_TEST_RATIO = regular_settings["TRAIN_VAL_TEST_RATIO"]
NORM_EACH_CHANNEL = regular_settings["NORM_EACH_CHANNEL"]
RESCALE = regular_settings["RESCALE"]
NULL_VAL = regular_settings["NULL_VAL"]
NUM_NODES = _num_nodes(DATA_NAME)

BUILD_A_TRUE = VARIANT != "fixed_physical"
PREDEFINED_A = None if BUILD_A_TRUE else _load_physical_adjacency(DATA_NAME, NUM_NODES)
MODEL_ARCH = MTGNN if VARIANT in {"original", "fixed_physical"} else AdaptiveImportanceMTGNN
MODEL_NAMES = {
    "original": "MTGNNAdaptiveImportanceOriginal",
    "fixed_physical": "MTGNNAdaptiveImportanceFixedPhysical",
    "frozen_random_graph": "MTGNNAdaptiveImportanceFrozenRandomGraph",
    "no_relu_score": "MTGNNAdaptiveImportanceNoReluScore",
}
MODEL_NAME = MODEL_NAMES[VARIANT]

MODEL_PARAM = {
    "gcn_true": True,
    "buildA_true": BUILD_A_TRUE,
    "gcn_depth": 2,
    "num_nodes": NUM_NODES,
    "predefined_A": PREDEFINED_A,
    "dropout": 0.3,
    "subgraph_size": 20,
    "node_dim": 40,
    "dilation_exponential": 1,
    "conv_channels": 32,
    "residual_channels": 32,
    "skip_channels": 64,
    "end_channels": 128,
    "seq_length": INPUT_LEN,
    "in_dim": 2,
    "out_dim": OUTPUT_LEN,
    "layers": 3,
    "propalpha": 0.05,
    "tanhalpha": 3,
    "layer_norm_affline": True,
}
if MODEL_ARCH is AdaptiveImportanceMTGNN:
    MODEL_PARAM.update(
        {
            "adaptive_graph_use_relu": VARIANT != "no_relu_score",
            "freeze_graph_constructor": VARIANT == "frozen_random_graph",
        }
    )

CFG = EasyDict()
CFG.DESCRIPTION = "MTGNN adaptive graph importance ablation"
CFG.GPU_NUM = 1
CFG.RUNNER = MTGNNRunner

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
ckpt_parts = [DATA_NAME, RUN_STAGE, VARIANT, ENV_TAG, str(NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)]
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints", MODEL_NAME, "_".join(ckpt_parts))
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
CFG.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}
CFG.TRAIN.CL = EasyDict()
CFG.TRAIN.CL.WARM_EPOCHS = 0
CFG.TRAIN.CL.CL_EPOCHS = 3
CFG.TRAIN.CL.PREDICTION_LENGTH = OUTPUT_LEN
CFG.TRAIN.CUSTOM = EasyDict()
CFG.TRAIN.CUSTOM.STEP_SIZE = 100
CFG.TRAIN.CUSTOM.NUM_NODES = NUM_NODES
CFG.TRAIN.CUSTOM.NUM_SPLIT = 1
CFG.TRAIN.EARLY_STOPPING_PATIENCE = EARLY_STOPPING_PATIENCE

CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 64

CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = NUM_EPOCHS if DATA_NAME == "SD" else 10
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
CFG.WANDB.TAGS = ["adaptive-graph-importance", "mtgnn", RUN_STAGE, DATA_NAME, VARIANT, ENV_TAG]
