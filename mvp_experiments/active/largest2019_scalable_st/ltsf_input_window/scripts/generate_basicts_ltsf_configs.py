#!/usr/bin/env python3
"""Generate isolated BasicTS configs for LargeST-LTSF input-window probes."""

from __future__ import annotations

import argparse
from pathlib import Path


DATASETS = {
    "SD": {"nodes": 716, "features": 3, "time_of_day_size": 96, "null_val": 0.0},
    "GBA": {"nodes": 2352, "features": 3, "time_of_day_size": 96, "null_val": 0.0},
    "GLA": {"nodes": 3834, "features": 3, "time_of_day_size": 96, "null_val": 0.0},
}

DEFAULT_BATCH = {
    "STID": 64,
    "DLinear": 64,
    "CycleNet": 64,
    "TimeMixer": 16,
    "PatchTST": 16,
}


def csv_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def model_import(model: str) -> str:
    return {
        "STID": "from baselines.STID.arch import STID as MODEL_ARCH",
        "DLinear": "from baselines.DLinear.arch import DLinear as MODEL_ARCH",
        "CycleNet": "from baselines.CycleNet.arch import CycleNet as MODEL_ARCH",
        "TimeMixer": "from baselines.TimeMixer.arch import TimeMixer as MODEL_ARCH",
        "PatchTST": "from baselines.PatchTST.arch import PatchTST as MODEL_ARCH",
    }[model]


def model_param(model: str, dataset: str, input_len: int, horizon: int) -> tuple[str, list[int]]:
    meta = DATASETS[dataset]
    n = meta["nodes"]
    tod = meta["time_of_day_size"]
    if model == "STID":
        return (
            f"""{{
    "num_nodes": {n},
    "input_len": INPUT_LEN,
    "input_dim": 3,
    "embed_dim": 32,
    "output_len": OUTPUT_LEN,
    "num_layer": 4,
    "if_node": True,
    "node_dim": 64,
    "if_T_i_D": True,
    "if_D_i_W": True,
    "temp_dim_tid": 32,
    "temp_dim_diw": 32,
    "time_of_day_size": {tod},
    "day_of_week_size": 7,
}}""",
            [0, 1, 2],
        )
    if model == "DLinear":
        return (
            f"""{{
    "seq_len": INPUT_LEN,
    "pred_len": OUTPUT_LEN,
    "individual": False,
    "enc_in": {n},
}}""",
            [0],
        )
    if model == "CycleNet":
        return (
            f"""{{
    "seq_len": INPUT_LEN,
    "pred_len": OUTPUT_LEN,
    "enc_in": {n},
    "cycle_pattern": "daily",
    "cycle": {tod},
    "model_type": "mlp",
    "d_model": 512,
    "use_revin": True,
}}""",
            [0, 1, 2],
        )
    if model == "TimeMixer":
        return (
            f"""{{
    "enc_in": {n},
    "dec_in": {n},
    "c_out": {n},
    "seq_len": INPUT_LEN,
    "label_len": INPUT_LEN // 2,
    "pred_len": OUTPUT_LEN,
    "factor": 1,
    "down_sampling_window": 2,
    "down_sampling_layers": 3,
    "top_k": 5,
    "down_sampling_method": "avg",
    "channel_independence": True,
    "d_model": 16,
    "moving_avg": 25,
    "n_heads": 8,
    "e_layers": 3,
    "d_layers": 1,
    "d_ff": 32,
    "distil": True,
    "sigma": 0.2,
    "dropout": 0.1,
    "freq": "w",
    "use_norm": 0,
    "decomp_method": "moving_avg",
    "output_attention": False,
    "embed": "timeF",
    "activation": "gelu",
    "num_time_features": 2,
    "time_of_day_size": {tod},
    "day_of_week_size": 7,
    "day_of_month_size": 31,
    "day_of_year_size": 366,
}}""",
            [0, 1, 2],
        )
    if model == "PatchTST":
        return (
            f"""{{
    "enc_in": {n},
    "seq_len": INPUT_LEN,
    "pred_len": OUTPUT_LEN,
    "e_layers": 3,
    "n_heads": 16,
    "d_model": 128,
    "d_ff": 256,
    "dropout": 0.2,
    "fc_dropout": 0.2,
    "head_dropout": 0.0,
    "patch_len": 32,
    "stride": 16,
    "individual": 0,
    "padding_patch": "end",
    "revin": 1,
    "affine": 0,
    "subtract_last": 0,
    "decomposition": 0,
    "kernel_size": 25,
}}""",
            [0],
        )
    raise KeyError(model)


def config_text(
    model: str,
    dataset: str,
    input_len: int,
    horizon: int,
    num_epochs: int,
    batch_size: int,
    run_tag: str,
) -> str:
    meta = DATASETS[dataset]
    param_text, forward_features = model_param(model, dataset, input_len, horizon)
    lr = {
        "STID": 0.002,
        "DLinear": 0.002,
        "CycleNet": 0.01,
        "TimeMixer": 0.01,
        "PatchTST": 0.001,
    }[model]
    weight_decay = {"PatchTST": 0.0005}.get(model, 0.0001)
    milestones = {
        "STID": [1, 30, 60, 80],
        "DLinear": [1, 25],
        "CycleNet": [1, 25, 50],
        "TimeMixer": [1, 25, 50],
        "PatchTST": [1, 25, 50],
    }[model]
    horizons = [h for h in [48, 96, 192, 672] if h <= horizon]
    if not horizons:
        horizons = [horizon]
    return f'''import os
import sys
from easydict import EasyDict

sys.path.append(os.getcwd())

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
{model_import(model)}

DATA_NAME = "{dataset}"
INPUT_LEN = {input_len}
OUTPUT_LEN = {horizon}
TRAIN_VAL_TEST_RATIO = [0.6, 0.2, 0.2]
NORM_EACH_CHANNEL = False
RESCALE = True
NULL_VAL = {meta["null_val"]}
NUM_EPOCHS = int(os.environ.get("LARGEST_LTSF_NUM_EPOCHS", "{num_epochs}"))
RUN_TAG = os.environ.get("LARGEST_LTSF_RUN_TAG", "{run_tag}")

MODEL_PARAM = {param_text}

CFG = EasyDict()
CFG.DESCRIPTION = "LargeST-LTSF input-window probe: {model} {dataset} L{input_len} H{horizon}"
CFG.GPU_NUM = 1
CFG.RUNNER = SimpleTimeSeriesForecastingRunner

CFG.ENV = EasyDict()
CFG.ENV.SEED = int(os.environ.get("BASICTS_SEED", "2023"))
CFG.ENV.DETERMINISTIC = False
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.DETERMINISTIC = False
CFG.ENV.CUDNN.BENCHMARK = True

CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({{
    "dataset_name": DATA_NAME,
    "train_val_test_ratio": TRAIN_VAL_TEST_RATIO,
    "input_len": INPUT_LEN,
    "output_len": OUTPUT_LEN,
}})

CFG.SCALER = EasyDict()
CFG.SCALER.TYPE = ZScoreScaler
CFG.SCALER.PARAM = EasyDict({{
    "dataset_name": DATA_NAME,
    "train_ratio": TRAIN_VAL_TEST_RATIO[0],
    "norm_each_channel": NORM_EACH_CHANNEL,
    "rescale": RESCALE,
}})

CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "{model}_LargeSTLTSF"
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = {forward_features}
CFG.MODEL.TARGET_FEATURES = [0]

CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict({{
    "MAE": masked_mae,
    "MAPE": masked_mape,
    "RMSE": masked_rmse,
}})
CFG.METRICS.TARGET = "MAE"
CFG.METRICS.NULL_VAL = NULL_VAL

CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.EARLY_STOPPING_PATIENCE = int(os.environ.get("LARGEST_LTSF_PATIENCE", "10"))
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    "LargeSTLTSF",
    "{model}",
    "_".join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), f"L{{INPUT_LEN}}", f"H{{OUTPUT_LEN}}", RUN_TAG]),
)
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {{
    "lr": {lr},
    "weight_decay": {weight_decay},
}}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {{
    "milestones": {milestones},
    "gamma": 0.5,
}}
CFG.TRAIN.CLIP_GRAD_PARAM = {{
    "max_norm": 5.0,
}}
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = int(os.environ.get("LARGEST_LTSF_BATCH_SIZE", "{batch_size}"))
CFG.TRAIN.DATA.SHUFFLE = True

CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = int(os.environ.get("LARGEST_LTSF_EVAL_BATCH_SIZE", "{batch_size}"))

CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = int(os.environ.get("LARGEST_LTSF_EVAL_BATCH_SIZE", "{batch_size}"))

CFG.EVAL = EasyDict()
CFG.EVAL.USE_GPU = True
CFG.EVAL.HORIZONS = {horizons}
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="SD")
    parser.add_argument("--models", default="STID,DLinear,CycleNet,TimeMixer,PatchTST")
    parser.add_argument("--input-lengths", default="96,192,336,672")
    parser.add_argument("--horizon", type=int, default=672)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--run-tag", default="ltsf_smoke")
    parser.add_argument("--out-dir", default="mvp_experiments/active/largest2019_scalable_st/ltsf_input_window/outputs/generated_configs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for dataset in csv_list(args.datasets):
        if dataset not in DATASETS:
            raise KeyError(f"Unknown dataset: {dataset}")
        for model in csv_list(args.models):
            if model not in DEFAULT_BATCH:
                raise KeyError(f"Unknown model: {model}")
            for input_len in [int(x) for x in csv_list(args.input_lengths)]:
                filename = f"{model}_{dataset}_L{input_len}_H{args.horizon}_{args.run_tag}.py"
                path = out_dir / filename
                path.write_text(
                    config_text(
                        model=model,
                        dataset=dataset,
                        input_len=input_len,
                        horizon=args.horizon,
                        num_epochs=args.num_epochs,
                        batch_size=DEFAULT_BATCH[model],
                        run_tag=args.run_tag,
                    ),
                    encoding="utf-8",
                )
                written.append(path)

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
