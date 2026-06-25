import glob
import os
import sys

import torch
from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + "/../../.."))

from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings, load_adj

from .arch import BigST, BigSTPreprocess
from .loss import bigst_loss
from .runner import BigSTPreprocessRunner

# BigST uses one week of long history. The official paper reports 2016
# steps for 5-minute California data; LargeST SD/GLA/GBA are 15-minute
# data, so one week is 96 * 7 = 672 steps.
BIGST_INPUT_LEN = int(os.environ.get("BIGST_INPUT_LEN", "672"))
BIGST_OUTPUT_LEN = int(os.environ.get("BIGST_OUTPUT_LEN", "12"))
BIGST_PREPROCESS_EPOCHS = int(os.environ.get("BIGST_PREPROCESS_EPOCHS", "100"))
BIGST_TINY_BATCH_SIZE = int(os.environ.get("BIGST_TINY_BATCH_SIZE", "64"))

_TRAIN_BATCH_SIZE = {
    "CA": 1,
    "GLA": 2,
    "GBA": 4,
    "SD": 8,
}


def _seed_from_env() -> int:
    return int(os.environ.get("BASICTS_SEED", "2023"))


def _build_common_cfg(dataset_name: str, num_epochs: int, input_len: int, output_len: int) -> tuple[EasyDict, dict]:
    regular_settings = get_regular_settings(dataset_name)
    train_val_test_ratio = regular_settings["TRAIN_VAL_TEST_RATIO"]
    norm_each_channel = regular_settings["NORM_EACH_CHANNEL"]
    rescale = regular_settings["RESCALE"]
    null_val = regular_settings["NULL_VAL"]

    cfg = EasyDict()
    cfg.DESCRIPTION = f"BigST config for {dataset_name}"
    cfg.GPU_NUM = 1

    cfg.ENV = EasyDict()
    cfg.ENV.SEED = _seed_from_env()

    cfg.DATASET = EasyDict()
    cfg.DATASET.NAME = dataset_name
    cfg.DATASET.TYPE = TimeSeriesForecastingDataset
    cfg.DATASET.PARAM = EasyDict(
        {
            "dataset_name": dataset_name,
            "train_val_test_ratio": train_val_test_ratio,
            "input_len": input_len,
            "output_len": output_len,
        }
    )

    cfg.SCALER = EasyDict()
    cfg.SCALER.TYPE = ZScoreScaler
    cfg.SCALER.PARAM = EasyDict(
        {
            "dataset_name": dataset_name,
            "train_ratio": train_val_test_ratio[0],
            "norm_each_channel": norm_each_channel,
            "rescale": rescale,
        }
    )

    cfg.METRICS = EasyDict()
    cfg.METRICS.FUNCS = EasyDict(
        {
            "MAE": masked_mae,
            "MAPE": masked_mape,
            "RMSE": masked_rmse,
        }
    )
    cfg.METRICS.TARGET = "MAE"
    cfg.METRICS.NULL_VAL = null_val

    cfg.TRAIN = EasyDict()
    cfg.TRAIN.NUM_EPOCHS = num_epochs
    cfg.TRAIN.EARLY_STOPPING_PATIENCE = 30
    cfg.TRAIN.OPTIM = EasyDict()
    cfg.TRAIN.OPTIM.TYPE = "AdamW"
    cfg.TRAIN.OPTIM.PARAM = {
        "lr": 0.002,
        "weight_decay": 0.0001,
    }
    cfg.TRAIN.LR_SCHEDULER = EasyDict()
    cfg.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
    cfg.TRAIN.LR_SCHEDULER.PARAM = {
        "milestones": [1, 50],
        "gamma": 0.5,
    }
    cfg.TRAIN.CLIP_GRAD_PARAM = {
        "max_norm": 5.0,
    }

    cfg.VAL = EasyDict()
    cfg.VAL.INTERVAL = 1
    cfg.VAL.DATA = EasyDict()

    cfg.TEST = EasyDict()
    cfg.TEST.INTERVAL = num_epochs
    cfg.TEST.DATA = EasyDict()

    cfg.EVAL = EasyDict()
    cfg.EVAL.HORIZONS = [3, 6, 12]
    cfg.EVAL.USE_GPU = True

    return cfg, regular_settings


def _resolve_preprocess_path(dataset_name: str) -> str:
    override = os.environ.get("BIGST_PREPROCESS_CKPT")
    if override:
        return override

    pattern = os.path.join(
        "checkpoints",
        BigSTPreprocess.__name__,
        "_".join(
            [
                dataset_name,
                str(BIGST_PREPROCESS_EPOCHS),
                str(BIGST_INPUT_LEN),
                str(BIGST_OUTPUT_LEN),
            ]
        ),
        "*",
        f"{BigSTPreprocess.__name__}_best_val_MAE.pt",
    )
    candidates = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        "BigST preprocess checkpoint not found for "
        f"{dataset_name}. Run `python experiments/train.py -c baselines/BigST/Preprocess{dataset_name}.py -g <gpu>` "
        "first, or set BIGST_PREPROCESS_CKPT to an existing checkpoint."
    )


def build_bigst_preprocess_cfg(dataset_name: str, num_nodes: int) -> EasyDict:
    cfg, regular_settings = _build_common_cfg(
        dataset_name=dataset_name,
        num_epochs=BIGST_PREPROCESS_EPOCHS,
        input_len=BIGST_INPUT_LEN,
        output_len=BIGST_OUTPUT_LEN,
    )

    cfg.RUNNER = BigSTPreprocessRunner

    cfg.MODEL = EasyDict()
    cfg.MODEL.NAME = BigSTPreprocess.__name__
    cfg.MODEL.ARCH = BigSTPreprocess
    cfg.MODEL.PARAM = {
        "num_nodes": num_nodes,
        "in_dim": 3,
        "dropout": 0.3,
        "input_length": BIGST_INPUT_LEN,
        "output_length": BIGST_OUTPUT_LEN,
        "nhid": 32,
        "tiny_batch_size": BIGST_TINY_BATCH_SIZE,
    }
    cfg.MODEL.FORWARD_FEATURES = [0, 1, 2]
    cfg.MODEL.TARGET_FEATURES = [0]

    cfg.TRAIN.CKPT_SAVE_DIR = os.path.join(
        "checkpoints",
        BigSTPreprocess.__name__,
        "_".join([dataset_name, str(cfg.TRAIN.NUM_EPOCHS), str(BIGST_INPUT_LEN), str(BIGST_OUTPUT_LEN)]),
    )
    cfg.TRAIN.LOSS = masked_mae
    cfg.TRAIN.DATA = EasyDict()
    cfg.TRAIN.DATA.BATCH_SIZE = 1
    cfg.TRAIN.DATA.SHUFFLE = True
    cfg.VAL.DATA.BATCH_SIZE = 1
    cfg.TEST.DATA.BATCH_SIZE = 1

    return cfg


def build_bigst_cfg(dataset_name: str, num_nodes: int) -> EasyDict:
    cfg, regular_settings = _build_common_cfg(
        dataset_name=dataset_name,
        num_epochs=100,
        input_len=BIGST_INPUT_LEN,
        output_len=BIGST_OUTPUT_LEN,
    )

    cfg.RUNNER = SimpleTimeSeriesForecastingRunner

    supports, _ = load_adj(os.path.join("datasets", dataset_name, "adj_mx.pkl"), "doubletransition")
    preprocess_path = _resolve_preprocess_path(dataset_name)

    cfg.MODEL = EasyDict()
    cfg.MODEL.NAME = BigST.__name__
    cfg.MODEL.ARCH = BigST
    cfg.MODEL.PARAM = {
        "bigst_args": {
            "num_nodes": num_nodes,
            "seq_num": 12,
            "in_dim": 3,
            "out_dim": BIGST_OUTPUT_LEN,
            "hid_dim": 32,
            "tau": 0.25,
            "random_feature_dim": 64,
            "node_emb_dim": 32,
            "time_emb_dim": 32,
            "use_residual": True,
            "use_bn": True,
            "use_long": True,
            "use_spatial": True,
            "dropout": 0.3,
            "supports": [torch.tensor(support, dtype=torch.float32) for support in supports],
            "time_of_day_size": 96,
            "day_of_week_size": 7,
        },
        "preprocess_path": preprocess_path,
        "preprocess_args": {
            "num_nodes": num_nodes,
            "in_dim": 3,
            "dropout": 0.3,
            "input_length": BIGST_INPUT_LEN,
            "output_length": BIGST_OUTPUT_LEN,
            "nhid": 32,
            "tiny_batch_size": BIGST_TINY_BATCH_SIZE,
        },
    }
    cfg.MODEL.FORWARD_FEATURES = [0, 1, 2]
    cfg.MODEL.TARGET_FEATURES = [0]

    batch_size = _TRAIN_BATCH_SIZE.get(dataset_name, 4)
    cfg.TRAIN.CKPT_SAVE_DIR = os.path.join(
        "checkpoints",
        BigST.__name__,
        "_".join([dataset_name, str(cfg.TRAIN.NUM_EPOCHS), str(BIGST_INPUT_LEN), str(BIGST_OUTPUT_LEN)]),
    )
    cfg.TRAIN.LOSS = bigst_loss if cfg.MODEL.PARAM["bigst_args"]["use_spatial"] else masked_mae
    cfg.TRAIN.DATA = EasyDict()
    cfg.TRAIN.DATA.BATCH_SIZE = batch_size
    cfg.TRAIN.DATA.SHUFFLE = True
    cfg.VAL.DATA.BATCH_SIZE = batch_size
    cfg.TEST.DATA.BATCH_SIZE = batch_size

    return cfg
