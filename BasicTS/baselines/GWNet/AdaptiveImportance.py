import os
import sys

import torch
from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + "/../../.."))

from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import masked_mae, masked_mape, masked_rmse, masked_wape
from basicts.runners import WandBTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings, load_adj, load_dataset_desc

from .arch import AdaptiveAblationGraphWaveNet, GraphWaveNet, SignalMLPGraphWaveNet


def _ratio_tag(value: float) -> str:
    return f"{float(value):.3f}".replace(".", "p")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


DATA_NAME = os.environ.get("BASICTS_DATA_NAME", "METR-LA")
VARIANT = os.environ.get("BASICTS_GWNET_VARIANT", "original")
RUN_STAGE = os.environ.get("BASICTS_RUN_STAGE", "smoke")
NUM_EPOCHS = int(os.environ.get("BASICTS_NUM_EPOCHS", "5"))
EARLY_STOPPING_PATIENCE = int(os.environ.get("BASICTS_EARLY_STOPPING_PATIENCE", "30"))
ADAPTIVE_EVAL_MODE = os.environ.get("BASICTS_ADAPTIVE_EVAL_MODE", "learned")
ADAPTIVE_KEEP_RATIO = float(os.environ.get("BASICTS_ADAPTIVE_KEEP_RATIO", "1.0"))
ADAPTIVE_KEEP_TAG = os.environ.get("BASICTS_ADAPTIVE_KEEP_TAG", _ratio_tag(ADAPTIVE_KEEP_RATIO))
ADAPTIVE_SHUFFLE_SEED = int(os.environ.get("BASICTS_ADAPTIVE_SHUFFLE_SEED", "2023"))
ENV_DETERMINISTIC = _env_bool("BASICTS_DETERMINISTIC", False)
ENV_TF32 = _env_bool("BASICTS_TF32", False)
CUDNN_BENCHMARK = _env_bool("BASICTS_CUDNN_BENCHMARK", True)
CUDNN_DETERMINISTIC = _env_bool("BASICTS_CUDNN_DETERMINISTIC", False)
ENV_TAG = os.environ.get(
    "BASICTS_ENV_TAG",
    f"det{int(ENV_DETERMINISTIC)}_cudnndet{int(CUDNN_DETERMINISTIC)}",
)

ADAPTIVE_VARIANTS = {"original", "frozen_random_adaptive_from_scratch", "learned_no_relu"}
SIGNAL_MLP_VARIANTS = {"signal_mlp_relu", "signal_mlp_no_relu"}
VALID_VARIANTS = ADAPTIVE_VARIANTS | SIGNAL_MLP_VARIANTS | {"no_adaptive_from_scratch"}

if VARIANT not in VALID_VARIANTS:
    raise ValueError(
        f"BASICTS_GWNET_VARIANT must be one of {sorted(VALID_VARIANTS)}, "
        f"got {VARIANT}"
    )

GRAPH_TAG = (
    ADAPTIVE_EVAL_MODE
    if VARIANT in ADAPTIVE_VARIANTS
    else VARIANT
    if VARIANT in SIGNAL_MLP_VARIANTS
    else "physical"
)
KEEP_TAG = ADAPTIVE_KEEP_TAG if VARIANT in ADAPTIVE_VARIANTS else "none"

regular_settings = get_regular_settings(DATA_NAME)
desc = load_dataset_desc(DATA_NAME)
INPUT_LEN = regular_settings["INPUT_LEN"]
OUTPUT_LEN = regular_settings["OUTPUT_LEN"]
TRAIN_VAL_TEST_RATIO = regular_settings["TRAIN_VAL_TEST_RATIO"]
NORM_EACH_CHANNEL = regular_settings["NORM_EACH_CHANNEL"]
RESCALE = regular_settings["RESCALE"]
NULL_VAL = regular_settings["NULL_VAL"]
NUM_NODES = int(desc["shape"][1])

adj_mx, _ = load_adj(os.path.join("datasets", DATA_NAME, "adj_mx.pkl"), "doubletransition")
SUPPORTS = [torch.tensor(i) for i in adj_mx]

if VARIANT in ADAPTIVE_VARIANTS:
    MODEL_ARCH = AdaptiveAblationGraphWaveNet
elif VARIANT in SIGNAL_MLP_VARIANTS:
    MODEL_ARCH = SignalMLPGraphWaveNet
else:
    MODEL_ARCH = GraphWaveNet
MODEL_NAMES = {
    "original": "GraphWaveNetAdaptiveImportance",
    "no_adaptive_from_scratch": "GraphWaveNetNoAdaptiveImportance",
    "frozen_random_adaptive_from_scratch": "GraphWaveNetFrozenRandomAdaptiveImportance",
    "learned_no_relu": "GraphWaveNetLearnedNoReluAdaptiveImportance",
    "signal_mlp_relu": "GraphWaveNetSignalMLPReluAdaptiveImportance",
    "signal_mlp_no_relu": "GraphWaveNetSignalMLPNoReluAdaptiveImportance",
}
MODEL_NAME = MODEL_NAMES[VARIANT]
ADDAPTADJ = VARIANT in ADAPTIVE_VARIANTS or VARIANT in SIGNAL_MLP_VARIANTS

MODEL_PARAM = {
    "num_nodes": NUM_NODES,
    "supports": SUPPORTS,
    "dropout": 0.3,
    "gcn_bool": True,
    "addaptadj": ADDAPTADJ,
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
if VARIANT in ADAPTIVE_VARIANTS:
    MODEL_PARAM.update(
        {
            "adaptive_eval_mode": ADAPTIVE_EVAL_MODE,
            "adaptive_keep_ratio": ADAPTIVE_KEEP_RATIO,
            "adaptive_shuffle_seed": ADAPTIVE_SHUFFLE_SEED,
            "freeze_adaptive_params": VARIANT == "frozen_random_adaptive_from_scratch",
            "adaptive_use_relu": VARIANT != "learned_no_relu",
        }
    )
elif VARIANT in SIGNAL_MLP_VARIANTS:
    MODEL_PARAM.update(
        {
            "signal_input_len": INPUT_LEN,
            "signal_feature_dim": 2,
            "signal_hidden_dim": 32,
            "signal_embedding_dim": 10,
            "signal_graph_relu": VARIANT == "signal_mlp_relu",
        }
    )

CFG = EasyDict()
CFG.DESCRIPTION = "GWNet adaptive graph importance ablation"
CFG.GPU_NUM = 1
CFG.RUNNER = WandBTimeSeriesForecastingRunner

CFG.ENV = EasyDict()
CFG.ENV.SEED = 42
CFG.ENV.TF32 = ENV_TF32
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
ckpt_parts = [
    DATA_NAME,
    RUN_STAGE,
    VARIANT,
    f"eval-{GRAPH_TAG}",
    f"keep-{KEEP_TAG}",
    ENV_TAG,
    str(NUM_EPOCHS),
    str(INPUT_LEN),
    str(OUTPUT_LEN),
]
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints", MODEL_NAME, "_".join(ckpt_parts))
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
CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 5.0,
}
CFG.TRAIN.EARLY_STOPPING_PATIENCE = EARLY_STOPPING_PATIENCE

CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 64

CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 10
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 64

CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [h for h in [3, 6, 12] if h <= OUTPUT_LEN]
CFG.EVAL.USE_GPU = True
CFG.EVAL.SAVE_RESULTS = False

CFG.WANDB = EasyDict()
CFG.WANDB.PROJECT = os.environ.get("WANDB_PROJECT", "adaptive_graph_importance_ablation")
CFG.WANDB.MODE = os.environ.get("WANDB_MODE", "offline")
CFG.WANDB.RUN_NAME = os.environ.get(
    "WANDB_NAME",
    f"{MODEL_NAME}_{DATA_NAME}_{RUN_STAGE}_{VARIANT}_{GRAPH_TAG}_keep{KEEP_TAG}",
)
CFG.WANDB.GROUP = os.environ.get("WANDB_RUN_GROUP", f"{DATA_NAME}_{RUN_STAGE}")
CFG.WANDB.TAGS = [
    "adaptive-graph-importance",
    RUN_STAGE,
    DATA_NAME,
    VARIANT,
    GRAPH_TAG,
    ENV_TAG,
]
