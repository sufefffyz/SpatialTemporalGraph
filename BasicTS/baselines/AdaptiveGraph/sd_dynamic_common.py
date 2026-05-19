import json
import os
import random
import sys
from pathlib import Path

from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + "/../../.."))

from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import masked_mae, masked_mape, masked_rmse, masked_wape
from basicts.runners import WandBTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler

from baselines.AdaptiveGraph import DynamicThresholdDCRNN, DynamicThresholdGraphWaveNet, DynamicThresholdSTGCN


DEFAULT_OSRM_DISTANCE = (
    "/home/yuzhang_fei/stg_artifacts_archive/adaptive_threshold_dynamic_weight/"
    "distance_matrices/SD/SD_osrm_shortest_distance_m.npy"
)
REFERENCE_SD_AVG_DEGREE = 17319 / 716


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or value == "" else float(value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


def _dynamic_graph_args(mode: str) -> dict:
    args = {
        "dist_mtx_path": os.environ.get("DYNAMIC_GRAPH_DIST_MTX", DEFAULT_OSRM_DISTANCE),
        "mode": mode,
        "d_model": _env_int("DYNAMIC_GRAPH_D_MODEL", 32),
        "target_avg_degree": _env_float("DYNAMIC_GRAPH_TARGET_AVG_DEGREE", REFERENCE_SD_AVG_DEGREE),
        "radius_scale": _env_float("DYNAMIC_GRAPH_RADIUS_SCALE", 1.0),
        "temperature": _env_float("DYNAMIC_GRAPH_TEMPERATURE", 0.05),
        "dist_norm": os.environ.get("DYNAMIC_GRAPH_DIST_NORM", "max"),
        "self_loops": os.environ.get("DYNAMIC_GRAPH_SELF_LOOPS", "1") != "0",
        "straight_through": os.environ.get("DYNAMIC_GRAPH_STRAIGHT_THROUGH", "1") != "0",
    }
    if os.environ.get("DYNAMIC_GRAPH_INIT_RADIUS"):
        args["init_radius"] = float(os.environ["DYNAMIC_GRAPH_INIT_RADIUS"])
    if os.environ.get("DYNAMIC_GRAPH_GAUSSIAN_SIGMA"):
        args["gaussian_sigma"] = float(os.environ["DYNAMIC_GRAPH_GAUSSIAN_SIGMA"])
    return args


def build_dynamic_sd_cfg(backbone: str) -> EasyDict:
    backbone = backbone.lower()
    mode = os.environ.get("DYNAMIC_GRAPH_MODE", "hard").lower()
    if mode not in {"soft", "hard"}:
        raise ValueError(f"DYNAMIC_GRAPH_MODE must be soft or hard, got {mode}.")

    data_name = os.environ.get("BASICTS_DATA_NAME", "SD")
    desc_path = Path("datasets") / data_name / "desc.json"
    desc = json.loads(desc_path.read_text(encoding="utf-8"))
    regular_settings = desc["regular_settings"]
    input_len = regular_settings["INPUT_LEN"]
    output_len = regular_settings["OUTPUT_LEN"]
    train_val_test_ratio = regular_settings["TRAIN_VAL_TEST_RATIO"]
    norm_each_channel = regular_settings["NORM_EACH_CHANNEL"]
    rescale = regular_settings["RESCALE"]
    null_val = regular_settings["NULL_VAL"]
    num_nodes = int(desc["num_nodes"])

    dynamic_graph = _dynamic_graph_args(mode)
    if backbone == "gwnet":
        model_arch = DynamicThresholdGraphWaveNet
        model_param = {
            "num_nodes": num_nodes,
            "seq_len": input_len,
            "dynamic_graph": dynamic_graph,
            "dropout": 0.3,
            "gcn_bool": True,
            "addaptadj": False,
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
        optim = ("Adam", {"lr": 0.002, "weight_decay": 0.0001})
        scheduler = {"milestones": [1, 50], "gamma": 0.5}
        batch_size = 64
        forward_features = [0, 1]
        setup_graph = False
    elif backbone == "dcrnn":
        model_arch = DynamicThresholdDCRNN
        model_param = {
            "dynamic_graph": dynamic_graph,
            "cl_decay_steps": 2000,
            "horizon": output_len,
            "input_dim": 2,
            "max_diffusion_step": 2,
            "num_nodes": num_nodes,
            "num_rnn_layers": 2,
            "output_dim": 1,
            "rnn_units": 64,
            "seq_len": input_len,
            "use_curriculum_learning": True,
        }
        optim = ("Adam", {"lr": 0.003, "eps": 1e-3})
        scheduler = {"milestones": [80], "gamma": 0.3}
        batch_size = 64
        forward_features = [0, 1]
        setup_graph = True
    elif backbone == "stgcn":
        model_arch = DynamicThresholdSTGCN
        model_param = {
            "Ks": 3,
            "Kt": 3,
            "blocks": [[1], [64, 16, 64], [64, 16, 64], [128, 128], [output_len]],
            "T": input_len,
            "n_vertex": num_nodes,
            "act_func": "glu",
            "graph_conv_type": "cheb_graph_conv",
            "dynamic_graph": dynamic_graph,
            "bias": True,
            "droprate": 0.5,
        }
        optim = ("Adam", {"lr": 0.0004, "weight_decay": 0.0003})
        scheduler = {"milestones": [1, 50], "gamma": 0.5}
        batch_size = 64
        forward_features = [0]
        setup_graph = False
    else:
        raise ValueError(f"Unsupported dynamic backbone: {backbone}")

    num_epochs = int(os.environ.get("BASICTS_NUM_EPOCHS", "100"))
    run_tag = os.environ.get("BASICTS_RUN_TAG", "").strip()

    cfg = EasyDict()
    cfg.DESCRIPTION = f"{model_arch.__name__} on {data_name} with {mode} dynamic OSRM threshold support"
    cfg.GPU_NUM = 1
    cfg.RUNNER = WandBTimeSeriesForecastingRunner
    cfg._ = random.randint(-1000000, 1000000)

    cfg.ENV = EasyDict()
    cfg.ENV.SEED = int(os.environ.get("BASICTS_SEED", "2023"))
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
    cfg.MODEL.FORWARD_FEATURES = forward_features
    cfg.MODEL.TARGET_FEATURES = [0]
    if setup_graph:
        cfg.MODEL.SETUP_GRAPH = True

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
    ckpt_name_parts = [data_name, f"dynamic_threshold_{mode}", backbone, str(num_epochs), str(input_len), str(output_len)]
    if run_tag:
        ckpt_name_parts.append(run_tag)
    cfg.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints", model_arch.__name__, "_".join(ckpt_name_parts))
    cfg.TRAIN.LOSS = masked_mae
    cfg.TRAIN.OPTIM = EasyDict()
    cfg.TRAIN.OPTIM.TYPE = optim[0]
    cfg.TRAIN.OPTIM.PARAM = optim[1]
    cfg.TRAIN.LR_SCHEDULER = EasyDict()
    cfg.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
    cfg.TRAIN.LR_SCHEDULER.PARAM = scheduler
    cfg.TRAIN.DATA = EasyDict()
    cfg.TRAIN.DATA.BATCH_SIZE = batch_size
    cfg.TRAIN.DATA.SHUFFLE = True
    cfg.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}
    cfg.TRAIN.EARLY_STOPPING_PATIENCE = 30

    cfg.VAL = EasyDict()
    cfg.VAL.INTERVAL = 1
    cfg.VAL.DATA = EasyDict()
    cfg.VAL.DATA.BATCH_SIZE = batch_size

    cfg.TEST = EasyDict()
    cfg.TEST.INTERVAL = num_epochs
    cfg.TEST.DATA = EasyDict()
    cfg.TEST.DATA.BATCH_SIZE = batch_size

    cfg.EVAL = EasyDict()
    cfg.EVAL.HORIZONS = [3, 6, 12]
    cfg.EVAL.USE_GPU = True

    cfg.WANDB = EasyDict()
    cfg.WANDB.PROJECT = os.environ.get("WANDB_PROJECT", "adaptive_threshold_dynamic_weight")
    cfg.WANDB.MODE = os.environ.get("WANDB_MODE", "online")
    cfg.WANDB.RUN_NAME = os.environ.get("WANDB_NAME", f"{model_arch.__name__}_{data_name}_{mode}_{run_tag}".rstrip("_"))
    cfg.WANDB.GROUP = os.environ.get("WANDB_RUN_GROUP", f"sd_dynamic_threshold_{mode}_{backbone}")
    cfg.WANDB.TAGS = ["adaptive-threshold", "sd", "dynamic-threshold", mode, backbone]
    return cfg
