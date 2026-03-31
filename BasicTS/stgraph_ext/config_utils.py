import copy
import json
import os
import random

import numpy as np
import torch
from easydict import EasyDict

from basicts.metrics import masked_mae, masked_mape, masked_rmse, masked_wape
from basicts.utils import load_adj
from baselines.AGCRN.arch import AGCRN
from baselines.D2STGNN.arch import D2STGNN
from baselines.GTS.arch import GTS
from baselines.GTS.loss import gts_loss
from baselines.GWNet.arch import GraphWaveNet
from baselines.MTGNN.arch import MTGNN

from .dataset import ExplicitSplitTimeSeriesForecastingDataset
from .runner import GraphSnapshotTimeSeriesForecastingRunner
from .scaler import ExplicitSplitZScoreScaler


def _load_desc(dataset_name: str) -> dict:
    with open(os.path.join("datasets", dataset_name, "desc.json"), "r") as fp:
        return json.load(fp)


def _load_split_indices(dataset_name: str) -> dict[str, np.ndarray]:
    with np.load(os.path.join("datasets", dataset_name, "split_indices.npz")) as data:
        return {
            "train": np.asarray(data["train_idx"], dtype=np.int64),
            "val": np.asarray(data["val_idx"], dtype=np.int64),
            "test": np.asarray(data["test_idx"], dtype=np.int64),
        }


def _load_train_target_series(dataset_name: str) -> np.ndarray:
    desc = _load_desc(dataset_name)
    data = np.memmap(
        os.path.join("datasets", dataset_name, "data.dat"),
        dtype="float32",
        mode="r",
        shape=tuple(desc["shape"]),
    )
    train_idx = _load_split_indices(dataset_name)["train"]
    return np.asarray(data[train_idx, :, 0], dtype=np.float32).copy()


def _compute_gts_dim_fc(train_length: int) -> int:
    dim_fc = 16 * (train_length - 18)
    if dim_fc <= 0:
        raise ValueError(f"Training length {train_length} is too short for GTS conv stack.")
    return int(dim_fc)


def _time_in_day_size(dataset_name: str) -> int:
    desc = _load_desc(dataset_name)
    frequency_minutes = int(desc["frequency (minutes)"])
    if frequency_minutes <= 0:
        raise ValueError(f"Dataset {dataset_name} has invalid frequency: {frequency_minutes}.")
    if 1440 % frequency_minutes != 0:
        raise ValueError(
            f"Dataset {dataset_name} frequency {frequency_minutes} does not evenly divide one day."
        )
    return 1440 // frequency_minutes


def _base_cfg(dataset_name: str, model_arch, model_param: dict, loss_fn, num_epochs: int = 100) -> EasyDict:
    desc = _load_desc(dataset_name)
    regular_settings = desc["regular_settings"]
    input_len = regular_settings["INPUT_LEN"]
    output_len = regular_settings["OUTPUT_LEN"]
    train_ratio = regular_settings["TRAIN_VAL_TEST_RATIO"][0]
    norm_each_channel = regular_settings["NORM_EACH_CHANNEL"]
    rescale = regular_settings["RESCALE"]
    null_val = regular_settings["NULL_VAL"]

    cfg = EasyDict()
    cfg.DESCRIPTION = f"STGraph extension config for {model_arch.__name__} on {dataset_name}"
    cfg.GPU_NUM = 1
    cfg.RUNNER = GraphSnapshotTimeSeriesForecastingRunner

    cfg.ENV = EasyDict()
    cfg.ENV.SEED = 42
    cfg.ENV.DETERMINISTIC = True
    cfg.ENV.CUDNN = EasyDict({"ENABLED": True, "BENCHMARK": True, "DETERMINISTIC": True})

    cfg.DATASET = EasyDict()
    cfg.DATASET.NAME = dataset_name
    cfg.DATASET.TYPE = ExplicitSplitTimeSeriesForecastingDataset
    cfg.DATASET.PARAM = EasyDict(
        {
            "dataset_name": dataset_name,
            "train_val_test_ratio": regular_settings["TRAIN_VAL_TEST_RATIO"],
            "input_len": input_len,
            "output_len": output_len,
            "split_filename": "split_indices.npz",
        }
    )

    cfg.SCALER = EasyDict()
    cfg.SCALER.TYPE = ExplicitSplitZScoreScaler
    cfg.SCALER.PARAM = EasyDict(
        {
            "dataset_name": dataset_name,
            "train_ratio": train_ratio,
            "norm_each_channel": norm_each_channel,
            "rescale": rescale,
            "split_filename": "split_indices.npz",
        }
    )

    cfg.MODEL = EasyDict()
    cfg.MODEL.NAME = model_arch.__name__
    cfg.MODEL.ARCH = model_arch
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
    cfg.TRAIN.NUM_EPOCHS = num_epochs
    cfg.TRAIN.CKPT_SAVE_DIR = os.path.join(
        "checkpoints",
        model_arch.__name__,
        "_".join([dataset_name, str(num_epochs), str(input_len), str(output_len)]),
    )
    cfg.TRAIN.LOSS = loss_fn
    cfg.TRAIN.OPTIM = EasyDict()
    cfg.TRAIN.DATA = EasyDict({"BATCH_SIZE": 64, "SHUFFLE": True})
    cfg.TRAIN.EARLY_STOPPING_PATIENCE = 15

    cfg.VAL = EasyDict({"INTERVAL": 1, "DATA": EasyDict({"BATCH_SIZE": 64})})
    cfg.TEST = EasyDict({"INTERVAL": 10, "DATA": EasyDict({"BATCH_SIZE": 64})})
    cfg.EVAL = EasyDict({"HORIZONS": [3, 6, 12], "USE_GPU": True})
    cfg.GRAPH_SNAPSHOT = EasyDict({"ENABLED": True, "INTERVAL": 5, "SAMPLE_SPLIT": "valid"})

    return cfg


def build_agcrn_cfg(dataset_name: str, num_epochs: int = 100) -> EasyDict:
    desc = _load_desc(dataset_name)
    output_len = desc["regular_settings"]["OUTPUT_LEN"]
    model_param = {
        "num_nodes": desc["num_nodes"],
        "input_dim": 2,
        "rnn_units": 64,
        "output_dim": 1,
        "horizon": output_len,
        "num_layers": 2,
        "default_graph": True,
        "embed_dim": 10,
        "cheb_k": 2,
    }
    cfg = _base_cfg(dataset_name, AGCRN, model_param, masked_mae, num_epochs=num_epochs)
    cfg.TRAIN.OPTIM.TYPE = "Adam"
    cfg.TRAIN.OPTIM.PARAM = {"lr": 0.003}
    cfg.TEST.INTERVAL = 1
    return cfg


def build_gwnet_cfg(dataset_name: str, num_epochs: int = 100) -> EasyDict:
    desc = _load_desc(dataset_name)
    output_len = desc["regular_settings"]["OUTPUT_LEN"]
    adj_mx, _ = load_adj(os.path.join("datasets", dataset_name, "adj_mx.pkl"), "doubletransition")
    model_param = {
        "num_nodes": desc["num_nodes"],
        "supports": [torch.tensor(item) for item in adj_mx],
        "dropout": 0.3,
        "gcn_bool": True,
        "addaptadj": True,
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
    cfg = _base_cfg(dataset_name, GraphWaveNet, model_param, masked_mae, num_epochs=num_epochs)
    cfg.TRAIN.OPTIM.TYPE = "Adam"
    cfg.TRAIN.OPTIM.PARAM = {"lr": 0.002, "weight_decay": 0.0001}
    cfg.TRAIN.LR_SCHEDULER = EasyDict({"TYPE": "MultiStepLR", "PARAM": {"milestones": [1, 50], "gamma": 0.5}})
    cfg.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}
    return cfg


def build_mtgnn_cfg(dataset_name: str, num_epochs: int = 100) -> EasyDict:
    desc = _load_desc(dataset_name)
    input_len = desc["regular_settings"]["INPUT_LEN"]
    output_len = desc["regular_settings"]["OUTPUT_LEN"]
    model_param = {
        "gcn_true": True,
        "buildA_true": True,
        "gcn_depth": 2,
        "num_nodes": desc["num_nodes"],
        "predefined_A": None,
        "dropout": 0.3,
        "subgraph_size": min(20, desc["num_nodes"]),
        "node_dim": 40,
        "dilation_exponential": 1,
        "conv_channels": 32,
        "residual_channels": 32,
        "skip_channels": 64,
        "end_channels": 128,
        "seq_length": input_len,
        "in_dim": 2,
        "out_dim": output_len,
        "layers": 3,
        "propalpha": 0.05,
        "tanhalpha": 3,
        "layer_norm_affline": True,
    }
    cfg = _base_cfg(dataset_name, MTGNN, model_param, masked_mae, num_epochs=num_epochs)
    cfg.TRAIN.OPTIM.TYPE = "Adam"
    cfg.TRAIN.OPTIM.PARAM = {"lr": 0.001, "weight_decay": 0.0001}
    cfg.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}
    cfg.TRAIN.CL = EasyDict({"WARM_EPOCHS": 0, "CL_EPOCHS": 3, "PREDICTION_LENGTH": output_len})
    cfg.TRAIN.CUSTOM = EasyDict({"STEP_SIZE": 100, "NUM_NODES": desc["num_nodes"], "NUM_SPLIT": 1})
    return cfg


def build_gts_cfg(dataset_name: str, num_epochs: int = 100) -> EasyDict:
    desc = _load_desc(dataset_name)
    input_len = desc["regular_settings"]["INPUT_LEN"]
    output_len = desc["regular_settings"]["OUTPUT_LEN"]
    train_node_feats = _load_train_target_series(dataset_name)
    adj_mx, _ = load_adj(os.path.join("datasets", dataset_name, "adj_mx.pkl"), "original")
    model_param = {
        "cl_decay_steps": 2000,
        "filter_type": "dual_random_walk",
        "horizon": output_len,
        "input_dim": 2,
        "l1_decay": 0,
        "max_diffusion_step": 2,
        "num_nodes": desc["num_nodes"],
        "num_rnn_layers": 1,
        "output_dim": 1,
        "rnn_units": 128,
        "seq_len": input_len,
        "use_curriculum_learning": True,
        "dim_fc": _compute_gts_dim_fc(train_node_feats.shape[0]),
        "node_feats": train_node_feats,
        "temp": 0.5,
        "k": min(30, desc["num_nodes"]),
        "prior_adj": torch.tensor(adj_mx[0]),
        "lamda": 1,
    }
    cfg = _base_cfg(dataset_name, GTS, model_param, gts_loss, num_epochs=num_epochs)
    cfg.RUNNER = GraphSnapshotTimeSeriesForecastingRunner
    cfg.MODEL.SETUP_GRAPH = True
    cfg._ = random.randint(-10**6, 10**6)
    cfg.TRAIN.OPTIM.TYPE = "Adam"
    cfg.TRAIN.OPTIM.PARAM = {"lr": 0.001, "eps": 1e-3}
    cfg.TRAIN.LR_SCHEDULER = EasyDict({"TYPE": "MultiStepLR", "PARAM": {"milestones": [20, 30], "gamma": 0.1}})
    cfg.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}
    return cfg


def build_d2stgnn_cfg(dataset_name: str, num_epochs: int = 100) -> EasyDict:
    desc = _load_desc(dataset_name)
    input_len = desc["regular_settings"]["INPUT_LEN"]
    output_len = desc["regular_settings"]["OUTPUT_LEN"]
    adj_mx, _ = load_adj(os.path.join("datasets", dataset_name, "adj_mx.pkl"), "doubletransition")
    model_param = {
        "num_feat": 1,
        "num_hidden": 32,
        "dropout": 0.1,
        "seq_length": input_len,
        "k_t": 3,
        "k_s": 2,
        "gap": 3,
        "num_nodes": desc["num_nodes"],
        "adjs": [torch.tensor(adj) for adj in adj_mx],
        "num_layers": 5,
        "num_modalities": 2,
        "node_hidden": 12,
        "time_emb_dim": 12,
        "time_in_day_size": _time_in_day_size(dataset_name),
        "day_in_week_size": 7,
    }
    cfg = _base_cfg(dataset_name, D2STGNN, model_param, masked_mae, num_epochs=num_epochs)
    cfg.MODEL.FORWARD_FEATURES = [0, 1, 2]
    cfg.MODEL.TARGET_FEATURES = [0]
    cfg.TRAIN.OPTIM.TYPE = "Adam"
    cfg.TRAIN.OPTIM.PARAM = {"lr": 0.002, "weight_decay": 1.0e-5, "eps": 1.0e-8}
    cfg.TRAIN.LR_SCHEDULER = EasyDict(
        {"TYPE": "MultiStepLR", "PARAM": {"milestones": [1, 30, 38, 46, 54, 62, 70, 80], "gamma": 0.5}}
    )
    cfg.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}
    cfg.TRAIN.CL = EasyDict({"WARM_EPOCHS": 30, "CL_EPOCHS": 3, "PREDICTION_LENGTH": output_len})
    return cfg
