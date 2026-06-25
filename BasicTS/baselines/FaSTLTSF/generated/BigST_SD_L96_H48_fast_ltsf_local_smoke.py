import os
import sys
import torch
from easydict import EasyDict

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "mvp_experiments", "active", "largest2019_scalable_st", "ltsf_input_window", "scripts")))

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.utils import get_regular_settings, load_adj
from fast_ltsf_modules import FaSTIndexedTimeSeriesDataset, FaSTSampleFirstZScoreScaler
from baselines.BigST.arch import BigST as MODEL_ARCH
from baselines.BigST.loss import bigst_loss

DATA_NAME = "SD"
INPUT_LEN = 96
OUTPUT_LEN = 48
NUM_EPOCHS = int(os.environ.get("FAST_LTSF_NUM_EPOCHS", "1"))
BATCH_SIZE = int(os.environ.get("FAST_LTSF_BATCH_SIZE", "64"))
RUN_TAG = os.environ.get("FAST_LTSF_RUN_TAG", "fast_ltsf_local_smoke")

regular_settings = get_regular_settings(DATA_NAME)
TRAIN_VAL_TEST_RATIO = regular_settings["TRAIN_VAL_TEST_RATIO"]
NORM_EACH_CHANNEL = regular_settings["NORM_EACH_CHANNEL"]
RESCALE = regular_settings["RESCALE"]
NULL_VAL = regular_settings["NULL_VAL"]

adj_mx, _ = load_adj(os.path.join("datasets", DATA_NAME, "adj_mx.pkl"), "doubletransition")

MODEL_PARAM = {
    "bigst_args": {
        "num_nodes": 716,
        "seq_num": INPUT_LEN,
        "in_dim": 3,
        "out_dim": OUTPUT_LEN,
        "hid_dim": 32,
        "tau": 0.25,
        "random_feature_dim": 64,
        "node_emb_dim": 32,
        "time_emb_dim": 32,
        "use_residual": True,
        "use_bn": True,
        "use_long": False,
        "use_full_history": True,
        "use_spatial": True,
        "dropout": 0.3,
        "supports": [torch.tensor(i, dtype=torch.float32) for i in adj_mx],
        "time_of_day_size": 96,
        "day_of_week_size": 7,
    },
    "preprocess_path": "",
    "preprocess_args": {
        "num_nodes": 716,
        "in_dim": 3,
        "dropout": 0.3,
        "input_length": INPUT_LEN,
        "output_length": OUTPUT_LEN,
        "nhid": 32,
        "tiny_batch_size": 700,
    },
}

CFG = EasyDict()
CFG.DESCRIPTION = "FaST-style LargeST LTSF baseline: BigST SD 96->48"
CFG.GPU_NUM = 1
CFG.RUNNER = SimpleTimeSeriesForecastingRunner

CFG.ENV = EasyDict()
CFG.ENV.SEED = int(os.environ.get("BASICTS_SEED", "42"))
CFG.ENV.DETERMINISTIC = False
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.DETERMINISTIC = False
CFG.ENV.CUDNN.BENCHMARK = True

CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = FaSTIndexedTimeSeriesDataset
CFG.DATASET.PARAM = EasyDict({
    "dataset_name": DATA_NAME,
    "train_val_test_ratio": TRAIN_VAL_TEST_RATIO,
    "input_len": INPUT_LEN,
    "output_len": OUTPUT_LEN,
})

CFG.SCALER = EasyDict()
CFG.SCALER.TYPE = FaSTSampleFirstZScoreScaler
CFG.SCALER.PARAM = EasyDict({
    "dataset_name": DATA_NAME,
    "train_ratio": TRAIN_VAL_TEST_RATIO[0],
    "norm_each_channel": NORM_EACH_CHANNEL,
    "rescale": RESCALE,
    "input_len": INPUT_LEN,
    "output_len": OUTPUT_LEN,
})

CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "BigST_FaSTLTSF"
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
CFG.MODEL.TARGET_FEATURES = [0]

CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict({
    "MAE": masked_mae,
    "RMSE": masked_rmse,
    "MAPE": masked_mape,
})
CFG.METRICS.TARGET = "MAE"
CFG.METRICS.NULL_VAL = NULL_VAL

CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.EARLY_STOPPING_PATIENCE = int(os.environ.get("FAST_LTSF_PATIENCE", "30"))
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    "FaSTLTSF",
    "BigST",
    "_".join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), "L96", f"H{OUTPUT_LEN}", RUN_TAG]),
)
CFG.TRAIN.LOSS = bigst_loss
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "AdamW"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.002,
    "weight_decay": 0.0001,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [1, 35, 40] if "BigST" == "PatchSTG" else [1, 50],
    "gamma": 0.5,
}
CFG.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = BATCH_SIZE
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.PREFETCH = True
CFG.TRAIN.DATA.NUM_WORKERS = 4
CFG.TRAIN.DATA.PIN_MEMORY = True

CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = BATCH_SIZE
CFG.VAL.DATA.PREFETCH = True
CFG.VAL.DATA.NUM_WORKERS = 4
CFG.VAL.DATA.PIN_MEMORY = True

CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = BATCH_SIZE
CFG.TEST.DATA.PREFETCH = True
CFG.TEST.DATA.NUM_WORKERS = 4
CFG.TEST.DATA.PIN_MEMORY = True

CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = []
CFG.EVAL.USE_GPU = True
