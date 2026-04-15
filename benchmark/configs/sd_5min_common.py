import json
import os
import random
import sys
from pathlib import Path

import torch
from easydict import EasyDict

REPO_ROOT = Path(__file__).resolve().parents[2]
BASICTS_ROOT = REPO_ROOT / "BasicTS"
if str(BASICTS_ROOT) not in sys.path:
    sys.path.append(str(BASICTS_ROOT))

from baselines.DCRNN.arch import DCRNN
from baselines.GWNet.arch import GraphWaveNet
from basicts.metrics import masked_mae, masked_mape, masked_rmse, masked_wape
from basicts.runners import SimpleTimeSeriesForecastingRunner, WandBTimeSeriesForecastingRunner
from basicts.utils.serialization import load_adj
from stgraph_ext.dataset import ExplicitSplitTimeSeriesForecastingDataset
from stgraph_ext.scaler import ExplicitSplitZScoreScaler


DATASET_NAME = "SD_5min_full"
DATASET_DIR = BASICTS_ROOT / "datasets" / DATASET_NAME
GRAPH_ROOT = REPO_ROOT / "graphs" / "SD"

WINDOW = os.environ.get("SD_BENCH_WINDOW", "full")
GRAPH_VARIANT = os.environ.get("SD_BENCH_GRAPH", "distthre")
SEED = int(os.environ.get("SD_BENCH_SEED", "42"))

GRAPH_PATHS = {
    "distthre": GRAPH_ROOT / "adj_mx_largeST_original.pkl",
    "directed": GRAPH_ROOT / "adj_mx_physical_forward.pkl",
    "phys_dir": GRAPH_ROOT / "adj_mx_physical_forward.pkl",
    "undirected": GRAPH_ROOT / "adj_mx_physical_bidir.pkl",
    "phys_bidir": GRAPH_ROOT / "adj_mx_physical_bidir.pkl",
    "adaptive": None,
    "adaptive_only": None,
    "adaptive_plus_phys": GRAPH_ROOT / "adj_mx_physical_forward.pkl",
    "phys+adaptive": GRAPH_ROOT / "adj_mx_physical_forward.pkl",
    "distthre+adaptive": GRAPH_ROOT / "adj_mx_largeST_original.pkl",
}


def _load_desc() -> dict:
    with (DATASET_DIR / "desc.json").open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _common_cfg(model_arch, runner_cls, model_param: dict, split_filename: str, loss_fn) -> EasyDict:
    desc = _load_desc()
    regular_settings = desc["regular_settings"]
    input_len = int(regular_settings["INPUT_LEN"])
    output_len = int(regular_settings["OUTPUT_LEN"])
    train_ratio = float(regular_settings["TRAIN_VAL_TEST_RATIO"][0])
    norm_each_channel = bool(regular_settings["NORM_EACH_CHANNEL"])
    rescale = bool(regular_settings["RESCALE"])
    null_val = regular_settings["NULL_VAL"]

    cfg = EasyDict()
    cfg.DESCRIPTION = f"SD 5min benchmark for {model_arch.__name__}: {GRAPH_VARIANT} / {WINDOW}"
    cfg.GPU_NUM = 1
    cfg.RUNNER = runner_cls

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
    cfg.MODEL.NAME = model_arch.__name__
    cfg.MODEL.ARCH = model_arch
    cfg.MODEL.PARAM = model_param
    cfg.MODEL.FORWARD_FEATURES = [0, 1, 2]
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
    cfg.TRAIN.NUM_EPOCHS = 100
    cfg.TRAIN.CKPT_SAVE_DIR = os.path.join(
        "checkpoints",
        model_arch.__name__,
        "_".join([DATASET_NAME, GRAPH_VARIANT, WINDOW, str(cfg.TRAIN.NUM_EPOCHS), str(input_len), str(output_len)]),
    )
    cfg.TRAIN.LOSS = loss_fn
    cfg.TRAIN.DATA = EasyDict({"BATCH_SIZE": 64, "SHUFFLE": True})
    cfg.TRAIN.EARLY_STOPPING_PATIENCE = 15

    cfg.VAL = EasyDict({"INTERVAL": 1, "DATA": EasyDict({"BATCH_SIZE": 64})})
    cfg.TEST = EasyDict({"INTERVAL": 10, "DATA": EasyDict({"BATCH_SIZE": 64})})
    cfg.EVAL = EasyDict({"HORIZONS": [3, 6, 12], "USE_GPU": True})

    return cfg


def build_gwnet_cfg() -> EasyDict:
    desc = _load_desc()
    input_len = int(desc["regular_settings"]["INPUT_LEN"])
    output_len = int(desc["regular_settings"]["OUTPUT_LEN"])
    split_filename = f"split_indices_{WINDOW}.npz"

    graph_path = GRAPH_PATHS[GRAPH_VARIANT]
    supports = None
    addaptadj = True
    if GRAPH_VARIANT in {"distthre", "directed", "phys_dir", "undirected", "phys_bidir"}:
        adj_mx, _ = load_adj(str(graph_path), "doubletransition")
        supports = [torch.tensor(item, dtype=torch.float32) for item in adj_mx]
        addaptadj = False
    elif GRAPH_VARIANT in {"adaptive_plus_phys", "phys+adaptive", "distthre+adaptive"}:
        adj_mx, _ = load_adj(str(graph_path), "doubletransition")
        supports = [torch.tensor(item, dtype=torch.float32) for item in adj_mx]
        addaptadj = True
    elif GRAPH_VARIANT in {"adaptive", "adaptive_only"}:
        supports = None
        addaptadj = True
    else:
        raise ValueError(f"Unsupported GWNet graph variant: {GRAPH_VARIANT}")

    model_param = {
        "num_nodes": int(desc["num_nodes"]),
        "supports": supports,
        "dropout": 0.3,
        "gcn_bool": True,
        "addaptadj": addaptadj,
        "aptinit": None,
        "in_dim": 3,
        "out_dim": output_len,
        "residual_channels": 32,
        "dilation_channels": 32,
        "skip_channels": 256,
        "end_channels": 512,
        "kernel_size": 2,
        "blocks": 4,
        "layers": 2,
    }
    cfg = _common_cfg(GraphWaveNet, WandBTimeSeriesForecastingRunner, model_param, split_filename, masked_mae)
    cfg.TRAIN.OPTIM = EasyDict({"TYPE": "Adam", "PARAM": {"lr": 0.002, "weight_decay": 0.0001}})
    cfg.TRAIN.LR_SCHEDULER = EasyDict({"TYPE": "MultiStepLR", "PARAM": {"milestones": [1, 50], "gamma": 0.5}})
    cfg.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}
    return cfg


def build_dcrnn_cfg() -> EasyDict:
    desc = _load_desc()
    input_len = int(desc["regular_settings"]["INPUT_LEN"])
    output_len = int(desc["regular_settings"]["OUTPUT_LEN"])
    split_filename = f"split_indices_{WINDOW}.npz"

    if GRAPH_VARIANT not in {"distthre", "directed", "undirected"}:
        raise ValueError(f"DCRNN does not support graph variant: {GRAPH_VARIANT}")

    graph_path = GRAPH_PATHS[GRAPH_VARIANT]
    adj_mx, _ = load_adj(str(graph_path), "doubletransition")
    model_param = {
        "cl_decay_steps": 2000,
        "horizon": output_len,
        "input_dim": 3,
        "max_diffusion_step": 2,
        "num_nodes": int(desc["num_nodes"]),
        "num_rnn_layers": 2,
        "output_dim": 1,
        "rnn_units": 64,
        "seq_len": input_len,
        "adj_mx": [torch.tensor(item, dtype=torch.float32) for item in adj_mx],
        "use_curriculum_learning": True,
    }
    cfg = _common_cfg(DCRNN, SimpleTimeSeriesForecastingRunner, model_param, split_filename, masked_mae)
    cfg._ = random.randint(-1000000, 1000000)
    cfg.MODEL.SETUP_GRAPH = True
    cfg.TRAIN.OPTIM = EasyDict({"TYPE": "Adam", "PARAM": {"lr": 0.003, "eps": 1e-3}})
    cfg.TRAIN.LR_SCHEDULER = EasyDict({"TYPE": "MultiStepLR", "PARAM": {"milestones": [80], "gamma": 0.3}})
    cfg.TEST.INTERVAL = 1
    return cfg
