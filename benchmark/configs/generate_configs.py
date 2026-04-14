#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASICTS_DATASETS_DIR = REPO_ROOT / "BasicTS" / "datasets"
DEFAULT_DCRNN_DIR = REPO_ROOT / "BasicTS" / "baselines" / "DCRNN"
DEFAULT_GWNET_DIR = REPO_ROOT / "BasicTS" / "baselines" / "GWNet"
DEFAULT_WINDOWS = ("full", "1m")
DEFAULT_GRAPHS = (
    "largeST_original",
    "physical_forward",
    "physical_bidir",
    "adaptive_only",
    "adaptive_plus_phys",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate BasicTS DCRNN/GWNet configs for SD LargeST benchmark datasets."
    )
    parser.add_argument("--basicts-datasets-dir", type=Path, default=DEFAULT_BASICTS_DATASETS_DIR)
    parser.add_argument("--dcrnn-dir", type=Path, default=DEFAULT_DCRNN_DIR)
    parser.add_argument("--gwnet-dir", type=Path, default=DEFAULT_GWNET_DIR)
    parser.add_argument("--windows", nargs="+", default=list(DEFAULT_WINDOWS))
    parser.add_argument("--graphs", nargs="+", default=list(DEFAULT_GRAPHS))
    return parser.parse_args()


def ensure_exists(path: Path, desc: str) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{desc} not found: {path}")
    return path


def dataset_name(window: str, graph: str) -> str:
    return f"SD_LargeST_{window}_{graph}"


def load_desc(dataset_dir: Path) -> dict:
    return json.loads((dataset_dir / "desc.json").read_text(encoding="utf-8"))


def render_dcrnn_config(name: str, desc: dict) -> str:
    num_nodes = int(desc["num_nodes"])
    input_len = int(desc["regular_settings"]["INPUT_LEN"])
    output_len = int(desc["regular_settings"]["OUTPUT_LEN"])
    return f"""import os
import sys
import torch
import random
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings, load_adj

from .arch import DCRNN

DATA_NAME = '{name}'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']
OUTPUT_LEN = regular_settings['OUTPUT_LEN']
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
RESCALE = regular_settings['RESCALE']
NULL_VAL = regular_settings['NULL_VAL']
MODEL_ARCH = DCRNN
adj_mx, _ = load_adj("datasets/" + DATA_NAME + "/adj_mx.pkl", "doubletransition")
MODEL_PARAM = {{
    "cl_decay_steps": 2000,
    "horizon": {output_len},
    "input_dim": 3,
    "max_diffusion_step": 2,
    "num_nodes": {num_nodes},
    "num_rnn_layers": 2,
    "output_dim": 1,
    "rnn_units": 64,
    "seq_len": {input_len},
    "adj_mx": [torch.tensor(i) for i in adj_mx],
    "use_curriculum_learning": True
}}
NUM_EPOCHS = 100

CFG = EasyDict()
CFG.DESCRIPTION = 'SD LargeST 5min benchmark config'
CFG.GPU_NUM = 1
CFG.RUNNER = SimpleTimeSeriesForecastingRunner
CFG._ = random.randint(-1000000, 1000000)

CFG.ENV = EasyDict()
CFG.ENV.SEED = 42
CFG.ENV.DETERMINISTIC = True
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True
CFG.ENV.CUDNN.BENCHMARK = True
CFG.ENV.CUDNN.DETERMINISTIC = True

CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({{
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
}})

CFG.SCALER = EasyDict()
CFG.SCALER.TYPE = ZScoreScaler
CFG.SCALER.PARAM = EasyDict({{
    'dataset_name': DATA_NAME,
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'norm_each_channel': NORM_EACH_CHANNEL,
    'rescale': RESCALE,
}})

CFG.MODEL = EasyDict()
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
CFG.MODEL.TARGET_FEATURES = [0]
CFG.MODEL.SETUP_GRAPH = True

CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict({{
    'MAE': masked_mae,
    'MAPE': masked_mape,
    'RMSE': masked_rmse,
}})
CFG.METRICS.TARGET = 'MAE'
CFG.METRICS.NULL_VAL = NULL_VAL

CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    MODEL_ARCH.__name__,
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {{
    "lr": 0.003,
    "eps": 1e-3
}}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {{
    "milestones": [80],
    "gamma": 0.3
}}
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 64
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.EARLY_STOPPING_PATIENCE = 15

CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 64

CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 64

CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [3, 6, 12]
CFG.EVAL.USE_GPU = True
"""


def render_gwnet_config(name: str, desc: dict) -> str:
    num_nodes = int(desc["num_nodes"])
    input_len = int(desc["regular_settings"]["INPUT_LEN"])
    output_len = int(desc["regular_settings"]["OUTPUT_LEN"])
    graph_name = name.split("_", 3)[-1] if name.startswith("SD_LargeST_") else name

    if graph_name == "largeST_original":
        supports_expr = '[torch.tensor(i) for i in adj_mx]'
        addaptadj = "True"
    elif graph_name == "physical_forward":
        supports_expr = '[torch.tensor(i) for i in adj_mx]'
        addaptadj = "False"
    elif graph_name == "physical_bidir":
        supports_expr = '[torch.tensor(i) for i in adj_mx]'
        addaptadj = "False"
    elif graph_name == "physical_ML_only":
        supports_expr = '[torch.tensor(i) for i in adj_mx]'
        addaptadj = "False"
    elif graph_name == "physical_reverse":
        supports_expr = '[torch.tensor(i) for i in adj_mx]'
        addaptadj = "False"
    elif graph_name == "adaptive_only":
        supports_expr = "None"
        addaptadj = "True"
    elif graph_name == "adaptive_plus_phys":
        supports_expr = '[torch.tensor(i) for i in adj_mx]'
        addaptadj = "True"
    else:
        raise ValueError(f"Unsupported GWNet graph variant for config generation: {graph_name}")

    return f"""import os
import sys
import torch
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse, masked_wape
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import WandBTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings, load_adj

from .arch import GraphWaveNet

DATA_NAME = '{name}'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']
OUTPUT_LEN = regular_settings['OUTPUT_LEN']
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
RESCALE = regular_settings['RESCALE']
NULL_VAL = regular_settings['NULL_VAL']
MODEL_ARCH = GraphWaveNet
adj_mx, _ = load_adj("datasets/" + DATA_NAME + "/adj_mx.pkl", "doubletransition")
MODEL_PARAM = {{
    "num_nodes": {num_nodes},
    "supports": {supports_expr},
    "dropout": 0.3,
    "gcn_bool": True,
    "addaptadj": {addaptadj},
    "aptinit": None,
    "in_dim": 3,
    "out_dim": {output_len},
    "residual_channels": 32,
    "dilation_channels": 32,
    "skip_channels": 256,
    "end_channels": 512,
    "kernel_size": 2,
    "blocks": 4,
    "layers": 2
}}
NUM_EPOCHS = 100

CFG = EasyDict()
CFG.DESCRIPTION = 'SD LargeST 5min benchmark config'
CFG.GPU_NUM = 1
CFG.RUNNER = WandBTimeSeriesForecastingRunner

CFG.ENV = EasyDict()
CFG.ENV.SEED = 42
CFG.ENV.DETERMINISTIC = True
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True
CFG.ENV.CUDNN.BENCHMARK = True
CFG.ENV.CUDNN.DETERMINISTIC = True

CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({{
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
}})

CFG.SCALER = EasyDict()
CFG.SCALER.TYPE = ZScoreScaler
CFG.SCALER.PARAM = EasyDict({{
    'dataset_name': DATA_NAME,
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'norm_each_channel': NORM_EACH_CHANNEL,
    'rescale': RESCALE,
}})

CFG.MODEL = EasyDict()
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
CFG.MODEL.TARGET_FEATURES = [0]

CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict({{
    'MAE': masked_mae,
    'MAPE': masked_mape,
    'RMSE': masked_rmse,
    'WAPE': masked_wape,
}})
CFG.METRICS.TARGET = 'MAE'
CFG.METRICS.NULL_VAL = NULL_VAL

CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    MODEL_ARCH.__name__,
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {{
    "lr": 0.002,
    "weight_decay": 0.0001,
}}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {{
    "milestones": [1, 50],
    "gamma": 0.5
}}
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 64
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.CLIP_GRAD_PARAM = {{
    "max_norm": 5.0
}}
CFG.TRAIN.EARLY_STOPPING_PATIENCE = 15

CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 64

CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 10
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 64

CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [3, 6, 12]
CFG.EVAL.USE_GPU = True
"""


def main() -> None:
    args = parse_args()
    basicts_datasets_dir = ensure_exists(args.basicts_datasets_dir, "BasicTS datasets dir")
    dcrnn_dir = ensure_exists(args.dcrnn_dir, "DCRNN baseline dir")
    gwnet_dir = ensure_exists(args.gwnet_dir, "GWNet baseline dir")

    summary = []
    for window in args.windows:
        for graph in args.graphs:
            name = dataset_name(window, graph)
            dataset_dir = ensure_exists(basicts_datasets_dir / name, f"Dataset dir for {name}")
            desc = load_desc(dataset_dir)

            dcrnn_path = dcrnn_dir / f"{name}.py"
            gwnet_path = gwnet_dir / f"{name}.py"
            dcrnn_path.write_text(render_dcrnn_config(name, desc), encoding="utf-8")
            gwnet_path.write_text(render_gwnet_config(name, desc), encoding="utf-8")

            summary.append(
                {
                    "dataset": name,
                    "num_nodes": int(desc["num_nodes"]),
                    "shape": desc["shape"],
                    "dcrnn_config": str(dcrnn_path),
                    "gwnet_config": str(gwnet_path),
                }
            )

    summary_path = REPO_ROOT / "benchmark" / "configs" / "generate_configs_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"num_configs": len(summary) * 2, "summary_path": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
