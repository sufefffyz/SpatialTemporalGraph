import json
import os
from pathlib import Path

from easydict import EasyDict

from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import masked_mae, masked_mape, masked_rmse, masked_wape
from basicts.runners import WandBTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler


def build_sd_cfg(
    model_arch,
    model_param,
    *,
    tag,
    forward_features,
    lr,
    weight_decay=0.0005,
    batch_size=32,
    val_batch_size=64,
    num_epochs=100,
    milestones=(1, 25, 50),
    gamma=0.5,
    use_gpu=True,
    horizons=(3, 6, 12),
):
    data_name = "SD"
    desc = json.loads((Path("datasets") / data_name / "desc.json").read_text(encoding="utf-8"))
    regular_settings = desc["regular_settings"]
    input_len = regular_settings["INPUT_LEN"]
    output_len = regular_settings["OUTPUT_LEN"]
    train_val_test_ratio = regular_settings["TRAIN_VAL_TEST_RATIO"]

    cfg = EasyDict()
    cfg.DESCRIPTION = f"{model_arch.__name__} on LargeST SD for FlowNet-paper comparison"
    cfg.GPU_NUM = 1
    cfg.RUNNER = WandBTimeSeriesForecastingRunner

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
            "norm_each_channel": regular_settings["NORM_EACH_CHANNEL"],
            "rescale": regular_settings["RESCALE"],
        }
    )

    cfg.MODEL = EasyDict()
    cfg.MODEL.NAME = model_arch.__name__
    cfg.MODEL.ARCH = model_arch
    cfg.MODEL.PARAM = model_param
    cfg.MODEL.FORWARD_FEATURES = forward_features
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
    cfg.METRICS.NULL_VAL = regular_settings["NULL_VAL"]

    cfg.TRAIN = EasyDict()
    cfg.TRAIN.NUM_EPOCHS = int(os.environ.get("BASICTS_EPOCHS", str(num_epochs)))
    run_tag = os.environ.get("BASICTS_RUN_TAG", "").strip()
    ckpt_name_parts = [data_name, tag, str(cfg.TRAIN.NUM_EPOCHS), str(input_len), str(output_len)]
    if run_tag:
        ckpt_name_parts.append(run_tag)
    cfg.TRAIN.CKPT_SAVE_DIR = os.path.join(
        "checkpoints",
        model_arch.__name__,
        "_".join(ckpt_name_parts),
    )
    cfg.TRAIN.LOSS = masked_mae
    cfg.TRAIN.OPTIM = EasyDict()
    cfg.TRAIN.OPTIM.TYPE = "Adam"
    cfg.TRAIN.OPTIM.PARAM = {
        "lr": lr,
        "weight_decay": weight_decay,
    }
    cfg.TRAIN.LR_SCHEDULER = EasyDict()
    cfg.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
    cfg.TRAIN.LR_SCHEDULER.PARAM = {
        "milestones": list(milestones),
        "gamma": gamma,
    }
    cfg.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}
    cfg.TRAIN.EARLY_STOPPING_PATIENCE = 30
    cfg.TRAIN.DATA = EasyDict()
    cfg.TRAIN.DATA.BATCH_SIZE = batch_size
    cfg.TRAIN.DATA.SHUFFLE = True

    cfg.VAL = EasyDict()
    cfg.VAL.INTERVAL = 1
    cfg.VAL.DATA = EasyDict()
    cfg.VAL.DATA.BATCH_SIZE = val_batch_size

    cfg.TEST = EasyDict()
    cfg.TEST.INTERVAL = cfg.TRAIN.NUM_EPOCHS
    cfg.TEST.DATA = EasyDict()
    cfg.TEST.DATA.BATCH_SIZE = val_batch_size

    cfg.EVAL = EasyDict()
    cfg.EVAL.HORIZONS = list(horizons)
    cfg.EVAL.USE_GPU = use_gpu
    return cfg
