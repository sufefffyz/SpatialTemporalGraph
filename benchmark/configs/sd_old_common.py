import json
import os
import sys
from pathlib import Path

import torch
from easydict import EasyDict

REPO_ROOT = Path(__file__).resolve().parents[2]
BASICTS_ROOT = REPO_ROOT / "BasicTS"
if str(BASICTS_ROOT) not in sys.path:
    sys.path.append(str(BASICTS_ROOT))

from baselines.GWNet.arch import GraphWaveNet
from basicts.metrics import masked_mae, masked_mape, masked_rmse, masked_wape
from basicts.runners import WandBTimeSeriesForecastingRunner
from basicts.utils.serialization import load_adj, load_pkl
from stgraph_ext.dataset import ExplicitSplitTimeSeriesForecastingDataset
from stgraph_ext.scaler import ExplicitSplitZScoreScaler


DATASET_NAME = "SD"
DATASET_DIR = BASICTS_ROOT / "datasets" / DATASET_NAME
SD_PHYS_DIR = BASICTS_ROOT / "datasets" / "SD_phys"

WINDOW = os.environ.get("SD_OLD_WINDOW", "full")
GRAPH_VARIANT = os.environ.get("SD_OLD_GRAPH", "distthre")
SEED = int(os.environ.get("SD_OLD_SEED", "42"))


def _load_desc() -> dict:
    with (DATASET_DIR / "desc.json").open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _resolve_graph_path() -> Path | None:
    mapping = {
        "distthre": DATASET_DIR / "adj_mx.pkl",
        "phys": SD_PHYS_DIR / "adj_mx.pkl",
        "adaptive": None,
        "distthre+adaptive": DATASET_DIR / "adj_mx.pkl",
        "phys+adaptive": SD_PHYS_DIR / "adj_mx.pkl",
    }
    if GRAPH_VARIANT not in mapping:
        raise ValueError(f"Unsupported old SD graph variant: {GRAPH_VARIANT}")
    return mapping[GRAPH_VARIANT]


def build_gwnet_cfg() -> EasyDict:
    desc = _load_desc()
    regular_settings = desc["regular_settings"]
    input_len = int(regular_settings["INPUT_LEN"])
    output_len = int(regular_settings["OUTPUT_LEN"])
    train_ratio = float(regular_settings["TRAIN_VAL_TEST_RATIO"][0])
    norm_each_channel = bool(regular_settings["NORM_EACH_CHANNEL"])
    rescale = bool(regular_settings["RESCALE"])
    null_val = regular_settings["NULL_VAL"]
    split_filename = f"split_indices_{WINDOW}.npz"

    graph_path = _resolve_graph_path()
    supports = None
    addaptadj = True
    if GRAPH_VARIANT in {"distthre", "phys"}:
        adj_mx, _ = load_adj(str(graph_path), "doubletransition")
        supports = [torch.tensor(item, dtype=torch.float32) for item in adj_mx]
        addaptadj = False
    elif GRAPH_VARIANT in {"distthre+adaptive", "phys+adaptive"}:
        adj_mx, _ = load_adj(str(graph_path), "doubletransition")
        supports = [torch.tensor(item, dtype=torch.float32) for item in adj_mx]
        addaptadj = True
    elif GRAPH_VARIANT == "adaptive":
        supports = None
        addaptadj = True

    model_param = {
        "num_nodes": int(desc["num_nodes"]),
        "supports": supports,
        "dropout": 0.3,
        "gcn_bool": True,
        "addaptadj": addaptadj,
        "aptinit": None,
        "in_dim": 2,
        "out_dim": output_len,
        "residual_channels": 32,
        "dilation_channels": 32,
        "skip_channels": 256,
        "end_channels": 512,
        "kernel_size": 2,
        "blocks": 4,
        "layers": 2,
    }

    cfg = EasyDict()
    cfg.DESCRIPTION = f"Old SD GWNet benchmark: {GRAPH_VARIANT} / {WINDOW}"
    cfg.GPU_NUM = 1
    cfg.RUNNER = WandBTimeSeriesForecastingRunner

    cfg.ENV = EasyDict()
    cfg.ENV.SEED = SEED
    cfg.ENV.DETERMINISTIC = True
    cfg.ENV.CUDNN = EasyDict({"ENABLED": True, "BENCHMARK": True, "DETERMINISTIC": True})

    cfg.DATASET = EasyDict()
    cfg.DATASET.NAME = DATASET_NAME
    cfg.DATASET.TYPE = ExplicitSplitTimeSeriesForecastingDataset
    cfg.DATASET.PARAM = EasyDict(
        {
            "dataset_name": DATASET_NAME,
            "train_val_test_ratio": regular_settings["TRAIN_VAL_TEST_RATIO"],
            "input_len": input_len,
            "output_len": output_len,
            "split_filename": split_filename,
        }
    )

    cfg.SCALER = EasyDict()
    cfg.SCALER.TYPE = ExplicitSplitZScoreScaler
    cfg.SCALER.PARAM = EasyDict(
        {
            "dataset_name": DATASET_NAME,
            "train_ratio": train_ratio,
            "norm_each_channel": norm_each_channel,
            "rescale": rescale,
            "split_filename": split_filename,
        }
    )

    cfg.MODEL = EasyDict()
    cfg.MODEL.NAME = GraphWaveNet.__name__
    cfg.MODEL.ARCH = GraphWaveNet
    cfg.MODEL.PARAM = model_param
    cfg.MODEL.FORWARD_FEATURES = [0, 1]
    cfg.MODEL.TARGET_FEATURES = [0]

    cfg.METRICS = EasyDict()
    cfg.METRICS.FUNCS = EasyDict(
        {
            "MAE": masked_mae,
            "MAPE": masked_mape,
            "RMSE": masked_rmse,
            "WAPE": masked_wape,
        }
    )
    cfg.METRICS.TARGET = "MAE"
    cfg.METRICS.NULL_VAL = null_val

    cfg.TRAIN = EasyDict()
    cfg.TRAIN.NUM_EPOCHS = 30
    cfg.TRAIN.CKPT_SAVE_DIR = os.path.join(
        "checkpoints",
        GraphWaveNet.__name__,
        "_".join([DATASET_NAME, GRAPH_VARIANT, WINDOW, str(cfg.TRAIN.NUM_EPOCHS), str(input_len), str(output_len)]),
    )
    cfg.TRAIN.LOSS = masked_mae
    cfg.TRAIN.OPTIM = EasyDict({"TYPE": "Adam", "PARAM": {"lr": 0.002, "weight_decay": 0.0001}})
    cfg.TRAIN.LR_SCHEDULER = EasyDict({"TYPE": "MultiStepLR", "PARAM": {"milestones": [1, 50], "gamma": 0.5}})
    cfg.TRAIN.DATA = EasyDict({"BATCH_SIZE": 64, "SHUFFLE": True})
    cfg.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}
    cfg.TRAIN.EARLY_STOPPING_PATIENCE = 15

    cfg.VAL = EasyDict({"INTERVAL": 1, "DATA": EasyDict({"BATCH_SIZE": 64})})
    cfg.TEST = EasyDict({"INTERVAL": 10, "DATA": EasyDict({"BATCH_SIZE": 64})})
    cfg.EVAL = EasyDict({"HORIZONS": [3, 6, 12], "USE_GPU": True})
    return cfg
