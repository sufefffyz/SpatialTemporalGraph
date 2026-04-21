import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + "/../../.."))

from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import masked_mae, masked_mape, masked_rmse, masked_wape
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils.adjacent_matrix_norm import calculate_symmetric_normalized_laplacian
from basicts.utils.serialization import load_adj, load_pkl

from .arch import STGCN


def _unwrap_adj_payload(raw_adj):
    if isinstance(raw_adj, (list, tuple)):
        if len(raw_adj) == 3:
            raw_adj = raw_adj[2]
        elif len(raw_adj) == 1:
            raw_adj = raw_adj[0]
    return np.asarray(raw_adj, dtype=np.float32)


def _load_desc(data_name: str) -> dict:
    desc_path = Path("datasets") / data_name / "desc.json"
    return json.loads(desc_path.read_text(encoding="utf-8"))


def _build_gso(data_name: str, graph_variant: str) -> torch.Tensor:
    adj_path = Path("datasets") / data_name / "adj_mx.pkl"

    if graph_variant == "distthre":
        adj_mx, _ = load_adj(str(adj_path), "normlap")
        return torch.tensor(adj_mx[0], dtype=torch.float32)

    raw_adj = _unwrap_adj_payload(load_pkl(str(adj_path)))
    if graph_variant == "phys_bidir":
        raw_adj = np.maximum(raw_adj, raw_adj.T)
    elif graph_variant != "phys_dir":
        raise ValueError(f"Unsupported STGCN graph variant: {graph_variant}")

    gso = calculate_symmetric_normalized_laplacian(raw_adj).astype(np.float32).todense()
    return torch.tensor(gso, dtype=torch.float32)


def build_sd_cfg(graph_variant: str) -> EasyDict:
    if graph_variant == "distthre":
        data_name = "SD"
        description = "STGCN on SD with LargeST distance-threshold graph"
        ckpt_tag = "original"
    elif graph_variant == "phys_dir":
        data_name = "SD_phys"
        description = "STGCN on SD_phys with directed physical graph"
        ckpt_tag = "phys_directed"
    elif graph_variant == "phys_bidir":
        data_name = "SD_phys"
        description = "STGCN on SD_phys with symmetrized physical graph"
        ckpt_tag = "phys_bidir"
    else:
        raise ValueError(f"Unsupported graph variant: {graph_variant}")

    desc = _load_desc(data_name)
    regular_settings = desc["regular_settings"]
    input_len = regular_settings["INPUT_LEN"]
    output_len = regular_settings["OUTPUT_LEN"]
    train_val_test_ratio = regular_settings["TRAIN_VAL_TEST_RATIO"]
    norm_each_channel = regular_settings["NORM_EACH_CHANNEL"]
    rescale = regular_settings["RESCALE"]
    null_val = regular_settings["NULL_VAL"]

    gso = _build_gso(data_name, graph_variant)
    model_arch = STGCN
    model_param = {
        "Ks": 3,
        "Kt": 3,
        "blocks": [[1], [64, 16, 64], [64, 16, 64], [128, 128], [output_len]],
        "T": input_len,
        "n_vertex": int(desc["num_nodes"]),
        "act_func": "glu",
        "graph_conv_type": "cheb_graph_conv",
        "gso": gso,
        "bias": True,
        "droprate": 0.5,
    }
    num_epochs = 100

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
    cfg.DATASET.NAME = data_name
    cfg.DATASET.TYPE = TimeSeriesForecastingDataset
    cfg.DATASET.PARAM = EasyDict(
        {
            "dataset_name": data_name,
            "train_val_test_ratio": train_val_test_ratio,
            "input_len": input_len,
            "output_len": output_len,
        }
    )

    cfg.SCALER = EasyDict()
    cfg.SCALER.TYPE = ZScoreScaler
    cfg.SCALER.PARAM = EasyDict(
        {
            "dataset_name": data_name,
            "train_ratio": train_val_test_ratio[0],
            "norm_each_channel": norm_each_channel,
            "rescale": rescale,
        }
    )

    cfg.MODEL = EasyDict()
    cfg.MODEL.NAME = model_arch.__name__
    cfg.MODEL.ARCH = model_arch
    cfg.MODEL.PARAM = model_param
    cfg.MODEL.FORWARD_FEATURES = [0]
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
    cfg.TRAIN.NUM_EPOCHS = num_epochs
    cfg.TRAIN.CKPT_SAVE_DIR = os.path.join(
        "checkpoints",
        model_arch.__name__,
        "_".join([data_name, ckpt_tag, str(num_epochs), str(input_len), str(output_len)]),
    )
    cfg.TRAIN.LOSS = masked_mae
    cfg.TRAIN.OPTIM = EasyDict()
    cfg.TRAIN.OPTIM.TYPE = "Adam"
    cfg.TRAIN.OPTIM.PARAM = {
        "lr": 0.0004,
        "weight_decay": 0.0003,
    }
    cfg.TRAIN.LR_SCHEDULER = EasyDict()
    cfg.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
    cfg.TRAIN.LR_SCHEDULER.PARAM = {
        "milestones": [1, 50],
        "gamma": 0.5,
    }
    cfg.TRAIN.DATA = EasyDict()
    cfg.TRAIN.DATA.BATCH_SIZE = 64
    cfg.TRAIN.DATA.SHUFFLE = True
    cfg.TRAIN.EARLY_STOPPING_PATIENCE = 15

    cfg.VAL = EasyDict()
    cfg.VAL.INTERVAL = 1
    cfg.VAL.DATA = EasyDict()
    cfg.VAL.DATA.BATCH_SIZE = 64

    cfg.TEST = EasyDict()
    cfg.TEST.INTERVAL = 1
    cfg.TEST.DATA = EasyDict()
    cfg.TEST.DATA.BATCH_SIZE = 64

    cfg.EVAL = EasyDict()
    cfg.EVAL.HORIZONS = [3, 6, 12]
    cfg.EVAL.USE_GPU = True

    return cfg
