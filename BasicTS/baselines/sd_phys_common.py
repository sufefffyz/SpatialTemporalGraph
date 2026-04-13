import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict


BASICTS_ROOT = Path(__file__).resolve().parents[1]
if str(BASICTS_ROOT) not in sys.path:
    sys.path.append(str(BASICTS_ROOT))

from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import masked_mae, masked_mape, masked_rmse, masked_wape
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils.adjacent_matrix_norm import calculate_transition_matrix
from basicts.utils.serialization import load_pkl
from baselines.DCRNN.arch import DCRNN
from baselines.GWNet.arch import GraphWaveNet


DATA_NAME = "SD_phys"
DESC = json.loads((BASICTS_ROOT / "datasets" / DATA_NAME / "desc.json").read_text(encoding="utf-8"))
REGULAR_SETTINGS = DESC["regular_settings"]
INPUT_LEN = REGULAR_SETTINGS["INPUT_LEN"]
OUTPUT_LEN = REGULAR_SETTINGS["OUTPUT_LEN"]
TRAIN_VAL_TEST_RATIO = REGULAR_SETTINGS["TRAIN_VAL_TEST_RATIO"]
NORM_EACH_CHANNEL = REGULAR_SETTINGS["NORM_EACH_CHANNEL"]
RESCALE = REGULAR_SETTINGS["RESCALE"]
NULL_VAL = REGULAR_SETTINGS["NULL_VAL"]
ADJ_PATH = BASICTS_ROOT / "datasets" / DATA_NAME / "adj_mx.pkl"
NUM_NODES = int(DESC["num_nodes"])
NUM_EPOCHS = 100
GRAPH_MODES = {"directed", "bidir", "adaptive"}


def _load_raw_adj() -> np.ndarray:
    raw = load_pkl(str(ADJ_PATH))
    if isinstance(raw, (list, tuple)):
        if len(raw) == 3:
            raw = raw[2]
        elif len(raw) == 1:
            raw = raw[0]
    array = np.asarray(raw, dtype=np.float32)
    if array.shape != (NUM_NODES, NUM_NODES):
        raise ValueError(f"Unexpected SD_phys adjacency shape: {array.shape}")
    return array


def _normalized_adj(mode: str) -> np.ndarray:
    raw = _load_raw_adj()
    if mode == "directed":
        return raw
    if mode == "bidir":
        return np.maximum(raw, raw.T).astype(np.float32)
    raise ValueError(f"Unsupported normalized adjacency mode: {mode}")


def _doubletransition_supports(adj: np.ndarray) -> list[torch.Tensor]:
    return [
        torch.tensor(calculate_transition_matrix(adj).T, dtype=torch.float32),
        torch.tensor(calculate_transition_matrix(adj.T).T, dtype=torch.float32),
    ]


def _base_cfg(description: str) -> EasyDict:
    cfg = EasyDict()
    cfg.DESCRIPTION = description
    cfg.GPU_NUM = 1
    cfg.RUNNER = SimpleTimeSeriesForecastingRunner

    cfg.ENV = EasyDict()
    cfg.ENV.SEED = 42
    cfg.ENV.DETERMINISTIC = True
    cfg.ENV.CUDNN = EasyDict()
    cfg.ENV.CUDNN.ENABLED = True
    cfg.ENV.CUDNN.BENCHMARK = True
    cfg.ENV.CUDNN.DETERMINISTIC = True

    cfg.DATASET = EasyDict()
    cfg.DATASET.NAME = DATA_NAME
    cfg.DATASET.TYPE = TimeSeriesForecastingDataset
    cfg.DATASET.PARAM = EasyDict(
        {
            "dataset_name": DATA_NAME,
            "train_val_test_ratio": TRAIN_VAL_TEST_RATIO,
            "input_len": INPUT_LEN,
            "output_len": OUTPUT_LEN,
        }
    )

    cfg.SCALER = EasyDict()
    cfg.SCALER.TYPE = ZScoreScaler
    cfg.SCALER.PARAM = EasyDict(
        {
            "dataset_name": DATA_NAME,
            "train_ratio": TRAIN_VAL_TEST_RATIO[0],
            "norm_each_channel": NORM_EACH_CHANNEL,
            "rescale": RESCALE,
        }
    )

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
    cfg.METRICS.NULL_VAL = NULL_VAL

    cfg.TRAIN = EasyDict()
    cfg.TRAIN.NUM_EPOCHS = NUM_EPOCHS
    cfg.TRAIN.DATA = EasyDict()
    cfg.TRAIN.DATA.BATCH_SIZE = 64
    cfg.TRAIN.DATA.SHUFFLE = True
    cfg.TRAIN.EARLY_STOPPING_PATIENCE = 15

    cfg.VAL = EasyDict()
    cfg.VAL.INTERVAL = 1
    cfg.VAL.DATA = EasyDict()
    cfg.VAL.DATA.BATCH_SIZE = 64

    cfg.TEST = EasyDict()
    cfg.TEST.DATA = EasyDict()
    cfg.TEST.DATA.BATCH_SIZE = 64

    cfg.EVAL = EasyDict()
    cfg.EVAL.HORIZONS = [3, 6, 12]
    cfg.EVAL.USE_GPU = True
    return cfg


def build_dcrnn_cfg(graph_mode: str) -> EasyDict:
    if graph_mode not in {"directed", "bidir"}:
        raise ValueError("DCRNN only supports 'directed' or 'bidir' in this experiment set")

    cfg = _base_cfg(f"DCRNN on SD_phys with {graph_mode} physical graph")
    cfg._ = random.randint(-1000000, 1000000)
    cfg.RUNNER = SimpleTimeSeriesForecastingRunner

    supports = _doubletransition_supports(_normalized_adj(graph_mode))
    model_param = {
        "cl_decay_steps": 2000,
        "horizon": OUTPUT_LEN,
        "input_dim": 3,
        "max_diffusion_step": 2,
        "num_nodes": NUM_NODES,
        "num_rnn_layers": 2,
        "output_dim": 1,
        "rnn_units": 64,
        "seq_len": INPUT_LEN,
        "adj_mx": supports,
        "use_curriculum_learning": True,
    }

    cfg.MODEL = EasyDict()
    cfg.MODEL.NAME = DCRNN.__name__
    cfg.MODEL.ARCH = DCRNN
    cfg.MODEL.PARAM = model_param
    cfg.MODEL.FORWARD_FEATURES = [0, 1, 2]
    cfg.MODEL.TARGET_FEATURES = [0]
    cfg.MODEL.SETUP_GRAPH = True

    cfg.TRAIN.CKPT_SAVE_DIR = os.path.join(
        "checkpoints",
        DCRNN.__name__,
        "_".join([DATA_NAME, graph_mode, str(cfg.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)]),
    )
    cfg.TRAIN.LOSS = masked_mae
    cfg.TRAIN.OPTIM = EasyDict()
    cfg.TRAIN.OPTIM.TYPE = "Adam"
    cfg.TRAIN.OPTIM.PARAM = {"lr": 0.003, "eps": 1e-3}
    cfg.TRAIN.LR_SCHEDULER = EasyDict()
    cfg.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
    cfg.TRAIN.LR_SCHEDULER.PARAM = {"milestones": [80], "gamma": 0.3}
    cfg.TEST.INTERVAL = 1
    return cfg


def build_gwnet_cfg(graph_mode: str) -> EasyDict:
    if graph_mode not in GRAPH_MODES:
        raise ValueError(f"Unsupported GWNet graph mode: {graph_mode}")

    description = {
        "directed": "GraphWaveNet on SD_phys with directed physical graph",
        "bidir": "GraphWaveNet on SD_phys with undirected physical graph",
        "adaptive": "GraphWaveNet on SD_phys with adaptive graph only",
    }[graph_mode]
    cfg = _base_cfg(description)
    cfg.RUNNER = SimpleTimeSeriesForecastingRunner

    if graph_mode == "adaptive":
        supports = None
        addaptadj = True
    else:
        supports = _doubletransition_supports(_normalized_adj(graph_mode))
        addaptadj = False

    model_param = {
        "num_nodes": NUM_NODES,
        "supports": supports,
        "dropout": 0.3,
        "gcn_bool": True,
        "addaptadj": addaptadj,
        "aptinit": None,
        "in_dim": 3,
        "out_dim": OUTPUT_LEN,
        "residual_channels": 32,
        "dilation_channels": 32,
        "skip_channels": 256,
        "end_channels": 512,
        "kernel_size": 2,
        "blocks": 4,
        "layers": 2,
    }

    cfg.MODEL = EasyDict()
    cfg.MODEL.NAME = GraphWaveNet.__name__
    cfg.MODEL.ARCH = GraphWaveNet
    cfg.MODEL.PARAM = model_param
    cfg.MODEL.FORWARD_FEATURES = [0, 1, 2]
    cfg.MODEL.TARGET_FEATURES = [0]

    cfg.TRAIN.CKPT_SAVE_DIR = os.path.join(
        "checkpoints",
        GraphWaveNet.__name__,
        "_".join([DATA_NAME, graph_mode, str(cfg.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)]),
    )
    cfg.TRAIN.LOSS = masked_mae
    cfg.TRAIN.OPTIM = EasyDict()
    cfg.TRAIN.OPTIM.TYPE = "Adam"
    cfg.TRAIN.OPTIM.PARAM = {"lr": 0.002, "weight_decay": 0.0001}
    cfg.TRAIN.LR_SCHEDULER = EasyDict()
    cfg.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
    cfg.TRAIN.LR_SCHEDULER.PARAM = {"milestones": [1, 50], "gamma": 0.5}
    cfg.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}
    cfg.TEST.INTERVAL = 10
    return cfg
