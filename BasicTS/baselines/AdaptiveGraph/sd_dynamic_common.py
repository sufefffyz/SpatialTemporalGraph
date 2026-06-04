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
from basicts.scaler import MultiChannelZScoreScaler, ZScoreScaler

from baselines.AdaptiveGraph import (
    DynamicThresholdDCRNN,
    DynamicThresholdGraphWaveNet,
    DynamicThresholdMTGNN,
    DynamicThresholdSTGCN,
)


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


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def _env_int_list(name: str, default: list[int]) -> list[int]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _default_forward_features(data_name: str, desc: dict) -> list[int]:
    canonical = data_name.upper().replace("-", "_")
    largest_prefixes = ("SD", "GBA", "GLA", "CA")
    if canonical.startswith(largest_prefixes) and int(desc.get("num_features", 1)) >= 3:
        return [0, 1, 2]
    return [0, 1]


def _select_scaler(data_name: str, desc: dict):
    scaler_name = os.environ.get("BASICTS_SCALER", "").strip().lower()
    if scaler_name in {"multi", "multichannel", "multi_channel", "all_features", "allfeat", "allfeatz"}:
        return MultiChannelZScoreScaler, "allfeatZ"
    if scaler_name in {"target", "target_only", "zscore", "target_zscore"}:
        return ZScoreScaler, "targetZ"
    canonical = data_name.upper().replace("-", "_")
    if canonical.startswith(("KNOWAIR", "CCAQ")) and int(desc.get("num_features", 1)) > 1:
        return MultiChannelZScoreScaler, "allfeatZ"
    return ZScoreScaler, "targetZ"


def _default_distance_path(data_name: str) -> str:
    dataset_distance = Path("datasets") / data_name / "distance_m.npy"
    if dataset_distance.exists():
        return str(dataset_distance)
    return DEFAULT_OSRM_DISTANCE


def _candidate_tag(dynamic_graph: dict) -> str:
    candidate_path = dynamic_graph.get("candidate_adj_path")
    if not candidate_path:
        return "full-pair"
    upper_path = str(candidate_path).upper()
    for token in ("K032", "K064", "K128"):
        if token in upper_path:
            return f"candidate-{token.lower()}"
    return "candidate"


def _dynamic_graph_args(mode: str, data_name: str) -> dict:
    args = {
        "dist_mtx_path": os.environ.get("DYNAMIC_GRAPH_DIST_MTX", _default_distance_path(data_name)),
        "mode": mode,
        "d_model": _env_int("DYNAMIC_GRAPH_D_MODEL", 32),
        "target_avg_degree": _env_float("DYNAMIC_GRAPH_TARGET_AVG_DEGREE", REFERENCE_SD_AVG_DEGREE),
        "init_mode": os.environ.get("DYNAMIC_GRAPH_INIT_MODE", "scalar"),
        "init_degree_min": _env_int("DYNAMIC_GRAPH_INIT_DEGREE_MIN", 8),
        "init_degree_max": _env_int("DYNAMIC_GRAPH_INIT_DEGREE_MAX", 64),
        "init_seed": _env_int("DYNAMIC_GRAPH_INIT_SEED", _env_int("BASICTS_SEED", 2023)),
        "radius_scale": _env_float("DYNAMIC_GRAPH_RADIUS_SCALE", 1.0),
        "radius_param": os.environ.get("DYNAMIC_GRAPH_RADIUS_PARAM", "exp_tanh"),
        "quantile_min": _env_float("DYNAMIC_GRAPH_QUANTILE_MIN", 0.05),
        "quantile_max": _env_float("DYNAMIC_GRAPH_QUANTILE_MAX", 1.0),
        "degree_min": _env_float("DYNAMIC_GRAPH_DEGREE_MIN", 8.0),
        "degree_max": _env_float("DYNAMIC_GRAPH_DEGREE_MAX", 64.0),
        "degree_init": _env_float("DYNAMIC_GRAPH_DEGREE_INIT", 0.5 * (
            _env_float("DYNAMIC_GRAPH_DEGREE_MIN", 8.0) + _env_float("DYNAMIC_GRAPH_DEGREE_MAX", 64.0)
        )),
        "state_act": os.environ.get("DYNAMIC_GRAPH_STATE_ACT", "tanh"),
        "quantile_temperature": _env_float("DYNAMIC_GRAPH_QUANTILE_TEMPERATURE", 1.0),
        "temperature": _env_float("DYNAMIC_GRAPH_TEMPERATURE", 0.05),
        "dist_norm": os.environ.get("DYNAMIC_GRAPH_DIST_NORM", "max"),
        "weight_mode": os.environ.get("DYNAMIC_GRAPH_WEIGHT_MODE", "binary"),
        "self_loops": os.environ.get("DYNAMIC_GRAPH_SELF_LOOPS", "1") != "0",
        "straight_through": os.environ.get("DYNAMIC_GRAPH_STRAIGHT_THROUGH", "1") != "0",
        "edge_matmul_mode": os.environ.get("DYNAMIC_GRAPH_EDGE_MATMUL", "scatter"),
        "edge_output": _env_bool("DYNAMIC_GRAPH_EDGE_OUTPUT", False),
    }
    if os.environ.get("DYNAMIC_GRAPH_CANDIDATE_ADJ"):
        args["candidate_adj_path"] = os.environ["DYNAMIC_GRAPH_CANDIDATE_ADJ"]
    if os.environ.get("DYNAMIC_GRAPH_INIT_RADIUS"):
        args["init_radius"] = float(os.environ["DYNAMIC_GRAPH_INIT_RADIUS"])
    if os.environ.get("DYNAMIC_GRAPH_GAUSSIAN_SIGMA"):
        args["gaussian_sigma"] = float(os.environ["DYNAMIC_GRAPH_GAUSSIAN_SIGMA"])
    if os.environ.get("DYNAMIC_GRAPH_QUANTILE_INIT"):
        args["quantile_init"] = float(os.environ["DYNAMIC_GRAPH_QUANTILE_INIT"])
    return args


def build_dynamic_sd_cfg(backbone: str) -> EasyDict:
    backbone = backbone.lower()
    mode = os.environ.get("DYNAMIC_GRAPH_MODE", "hard").lower()
    if mode not in {"soft", "hard"}:
        raise ValueError(f"DYNAMIC_GRAPH_MODE must be soft or hard, got {mode}.")

    data_name = os.environ.get("BASICTS_DATA_NAME", "SD")
    graph_tag = os.environ.get("BASICTS_GRAPH_TAG", "").strip()
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
    data_name_canonical = data_name.upper().replace("_", "-")
    scaler_type, scaler_tag = _select_scaler(data_name, desc)

    dynamic_graph = _dynamic_graph_args(mode, data_name)
    weight_mode = dynamic_graph["weight_mode"]
    radius_param = dynamic_graph["radius_param"]
    if backbone == "gwnet":
        runner_class = WandBTimeSeriesForecastingRunner
        forward_features = _env_int_list("BASICTS_FORWARD_FEATURES", _default_forward_features(data_name, desc))
        addaptadj = _env_bool("DYNAMIC_GWNET_ADDAPTADJ", False)
        model_arch = DynamicThresholdGraphWaveNet
        model_param = {
            "num_nodes": num_nodes,
            "seq_len": input_len,
            "dynamic_graph": dynamic_graph,
            "dropout": 0.3,
            "gcn_bool": True,
            "addaptadj": addaptadj,
            "aptinit": None,
            "in_dim": len(forward_features),
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
        setup_graph = False
    elif backbone == "dcrnn":
        runner_class = WandBTimeSeriesForecastingRunner
        forward_features = _env_int_list("BASICTS_FORWARD_FEATURES", _default_forward_features(data_name, desc))
        model_arch = DynamicThresholdDCRNN
        model_param = {
            "dynamic_graph": dynamic_graph,
            "cl_decay_steps": 2000,
            "horizon": output_len,
            "input_dim": len(forward_features),
            "max_diffusion_step": 2,
            "num_nodes": num_nodes,
            "num_rnn_layers": 2,
            "output_dim": 1,
            "rnn_units": 64,
            "seq_len": input_len,
            "use_curriculum_learning": True,
        }
        if data_name_canonical == "METR-LA":
            optim = ("Adam", {"lr": 0.01, "eps": 1e-3})
            scheduler = {"milestones": [20, 30, 40, 50], "gamma": 0.1}
        else:
            optim = ("Adam", {"lr": 0.003, "eps": 1e-3})
            scheduler = {"milestones": [80], "gamma": 0.3}
        batch_size = 64
        setup_graph = True
    elif backbone == "stgcn":
        runner_class = WandBTimeSeriesForecastingRunner
        dynamic_graph = dict(dynamic_graph)
        if "DYNAMIC_GRAPH_SELF_LOOPS" not in os.environ:
            dynamic_graph["self_loops"] = False
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
    elif backbone == "mtgnn":
        from baselines.MTGNN.runner import MTGNNRunner

        runner_class = MTGNNRunner
        dynamic_graph = dict(dynamic_graph)
        if "DYNAMIC_GRAPH_SELF_LOOPS" not in os.environ:
            dynamic_graph["self_loops"] = False
        dynamic_graph["edge_output"] = False
        default_mtgnn_features = [0, 1] if int(desc.get("num_features", 1)) >= 2 else [0]
        forward_features = _env_int_list("BASICTS_FORWARD_FEATURES", default_mtgnn_features)
        model_arch = DynamicThresholdMTGNN
        model_param = {
            "dynamic_graph": dynamic_graph,
            "gcn_true": True,
            "buildA_true": False,
            "gcn_depth": 2,
            "num_nodes": num_nodes,
            "predefined_A": None,
            "dropout": 0.3,
            "subgraph_size": 20,
            "node_dim": 40,
            "dilation_exponential": 1,
            "conv_channels": 32,
            "residual_channels": 32,
            "skip_channels": 64,
            "end_channels": 128,
            "seq_length": input_len,
            "in_dim": len(forward_features),
            "out_dim": output_len,
            "layers": 3,
            "propalpha": 0.05,
            "tanhalpha": 3,
            "layer_norm_affline": True,
        }
        optim = ("Adam", {"lr": 0.001, "weight_decay": 0.0001})
        scheduler = None
        batch_size = 64
        setup_graph = False
    else:
        raise ValueError(f"Unsupported dynamic backbone: {backbone}")

    num_epochs = int(os.environ.get("BASICTS_NUM_EPOCHS", "100"))
    run_tag = os.environ.get("BASICTS_RUN_TAG", "").strip()

    cfg = EasyDict()
    cfg.DESCRIPTION = (
        f"{model_arch.__name__} on {data_name} with {mode} dynamic OSRM threshold support, "
        f"{weight_mode} weights, and {radius_param} radius parameterization"
    )
    cfg.GPU_NUM = 1
    cfg.RUNNER = runner_class
    cfg._ = random.randint(-1000000, 1000000)

    cfg.ENV = EasyDict()
    cfg.ENV.SEED = int(os.environ.get("BASICTS_SEED", "2023"))
    cfg.ENV.DETERMINISTIC = False
    cfg.ENV.CUDNN = EasyDict()
    cfg.ENV.CUDNN.ENABLED = True
    cfg.ENV.CUDNN.BENCHMARK = True
    cfg.ENV.CUDNN.DETERMINISTIC = False

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
    cfg.SCALER.TYPE = scaler_type
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
    ckpt_name_parts = [
        data_name,
        f"dynamic_threshold_{mode}",
        radius_param,
        backbone,
        str(num_epochs),
        str(input_len),
        str(output_len),
    ]
    if graph_tag:
        ckpt_name_parts.append(graph_tag)
    if backbone == "gwnet" and model_param.get("addaptadj", False):
        ckpt_name_parts.append("addaptadj")
    if scaler_tag != "targetZ":
        ckpt_name_parts.append(scaler_tag)
    if run_tag:
        ckpt_name_parts.append(run_tag)
    cfg.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints", model_arch.__name__, "_".join(ckpt_name_parts))
    cfg.TRAIN.LOSS = masked_mae
    cfg.TRAIN.OPTIM = EasyDict()
    cfg.TRAIN.OPTIM.TYPE = optim[0]
    cfg.TRAIN.OPTIM.PARAM = optim[1]
    if scheduler is not None:
        cfg.TRAIN.LR_SCHEDULER = EasyDict()
        cfg.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
        cfg.TRAIN.LR_SCHEDULER.PARAM = scheduler
    cfg.TRAIN.DATA = EasyDict()
    cfg.TRAIN.DATA.BATCH_SIZE = batch_size
    cfg.TRAIN.DATA.SHUFFLE = True
    cfg.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}
    if backbone == "mtgnn":
        cfg.TRAIN.CL = EasyDict()
        cfg.TRAIN.CL.WARM_EPOCHS = 0
        cfg.TRAIN.CL.CL_EPOCHS = 3
        cfg.TRAIN.CL.PREDICTION_LENGTH = output_len
        cfg.TRAIN.CUSTOM = EasyDict()
        cfg.TRAIN.CUSTOM.STEP_SIZE = 100
        cfg.TRAIN.CUSTOM.NUM_NODES = num_nodes
        cfg.TRAIN.CUSTOM.NUM_SPLIT = 1
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
    cfg.EVAL.HORIZONS = [h for h in [3, 6, 12] if h <= output_len]
    cfg.EVAL.USE_GPU = True

    cfg.WANDB = EasyDict()
    cfg.WANDB.PROJECT = os.environ.get("WANDB_PROJECT", "adaptive_threshold_dynamic_weight")
    cfg.WANDB.MODE = os.environ.get("WANDB_MODE", "online")
    cfg.WANDB.RUN_NAME = os.environ.get(
        "WANDB_NAME",
        f"{model_arch.__name__}_{data_name}_{graph_tag}_{mode}_{weight_mode}_{radius_param}_{run_tag}".rstrip("_"),
    )
    addaptadj_tag = "addaptadj" if backbone == "gwnet" and model_param.get("addaptadj", False) else "no-addaptadj"
    candidate_tag = _candidate_tag(dynamic_graph)
    cfg.WANDB.GROUP = os.environ.get(
        "WANDB_RUN_GROUP",
        f"sd_dynamic_threshold_{mode}_{weight_mode}_{radius_param}_{backbone}_{addaptadj_tag}_{candidate_tag}",
    )
    cfg.WANDB.TAGS = [
        "adaptive-threshold",
        data_name.lower(),
        "dynamic-threshold",
        mode,
        weight_mode,
        radius_param,
        backbone,
        addaptadj_tag,
        candidate_tag,
        scaler_tag,
    ]
    if graph_tag:
        cfg.WANDB.TAGS.append(graph_tag)
    return cfg
