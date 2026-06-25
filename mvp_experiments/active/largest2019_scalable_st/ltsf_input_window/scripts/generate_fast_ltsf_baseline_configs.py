#!/usr/bin/env python3
"""Generate FaST-style scalable baseline configs for LargeST LTSF."""

from __future__ import annotations

import argparse
from pathlib import Path


DATASETS = {
    "SD": {
        "nodes": 716,
        "spa_patchsize": 2,
        "spa_patchnum": 512,
        "factors": 32,
        "node_dims": 64,
        "recur": 9,
    },
    "GBA": {
        "nodes": 2352,
        "spa_patchsize": 2,
        "spa_patchnum": 2048,
        "factors": 128,
        "node_dims": 32,
        "recur": 11,
    },
    "GLA": {
        "nodes": 3834,
        "spa_patchsize": 2,
        "spa_patchnum": 2048,
        "factors": 32,
        "node_dims": 32,
        "recur": 11,
    },
}


def csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def model_import(model: str) -> str:
    if model == "PatchSTG":
        return (
            "from fast_ltsf_modules import FaSTPatchSTG as MODEL_ARCH\n"
            "from fast_ltsf_modules import reorder_data"
        )
    if model == "BigST":
        return "from baselines.BigST.arch import BigST as MODEL_ARCH\nfrom baselines.BigST.loss import bigst_loss"
    if model == "STID":
        return "from baselines.STID.arch import STID as MODEL_ARCH"
    if model == "CycleNet":
        return "from baselines.CycleNet.arch import CycleNet as MODEL_ARCH"
    raise KeyError(model)


def model_param(model: str, dataset: str, input_len: int, horizon: int) -> tuple[str, list[int], str]:
    meta = DATASETS[dataset]
    if model == "PatchSTG":
        return (
            f"""metapath = f"./datasets/{{DATA_NAME}}/meta.csv"
adjpath = f"./datasets/{{DATA_NAME}}/adj_mx.pkl"
ori_parts_idx, reo_parts_idx, reo_all_idx = reorder_data(metapath, adjpath, {meta["recur"]}, {meta["spa_patchsize"]})

MODEL_PARAM = {{
    "tem_patchsize": INPUT_LEN,
    "tem_patchnum": 1,
    "output_len": OUTPUT_LEN,
    "node_num": {meta["nodes"]},
    "spa_patchsize": {meta["spa_patchsize"]},
    "spa_patchnum": {meta["spa_patchnum"]},
    "tod": 96,
    "dow": 7,
    "layers": 5,
    "factors": {meta["factors"]},
    "input_dims": 64,
    "node_dims": {meta["node_dims"]},
    "tod_dims": 32,
    "dow_dims": 32,
    "ori_parts_idx": ori_parts_idx,
    "reo_parts_idx": reo_parts_idx,
    "reo_all_idx": reo_all_idx,
}}""",
            [0, 1, 2],
            "masked_mae",
        )
    if model == "BigST":
        return (
            f"""adj_mx, _ = load_adj(os.path.join("datasets", DATA_NAME, "adj_mx.pkl"), "doubletransition")

MODEL_PARAM = {{
    "bigst_args": {{
        "num_nodes": {meta["nodes"]},
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
    }},
    "preprocess_path": "",
    "preprocess_args": {{
        "num_nodes": {meta["nodes"]},
        "in_dim": 3,
        "dropout": 0.3,
        "input_length": INPUT_LEN,
        "output_length": OUTPUT_LEN,
        "nhid": 32,
        "tiny_batch_size": 700,
    }},
}}""",
            [0, 1, 2],
            "bigst_loss",
        )
    if model == "STID":
        return (
            f"""MODEL_PARAM = {{
    "num_nodes": {meta["nodes"]},
    "input_len": INPUT_LEN,
    "input_dim": 1,
    "embed_dim": 32,
    "output_len": OUTPUT_LEN,
    "num_layer": 4,
    "if_node": True,
    "node_dim": 64,
    "if_T_i_D": True,
    "if_D_i_W": True,
    "temp_dim_tid": 32,
    "temp_dim_diw": 32,
    "time_of_day_size": 96,
    "day_of_week_size": 7,
}}""",
            [0, 1, 2],
            "masked_mae",
        )
    if model == "CycleNet":
        return (
            f"""MODEL_PARAM = {{
    "seq_len": INPUT_LEN,
    "pred_len": OUTPUT_LEN,
    "enc_in": {meta["nodes"]},
    "cycle_pattern": "daily&weekly",
    "cycle": 96,
    "model_type": "mlp",
    "d_model": 512,
    "use_revin": True,
}}""",
            [0, 1, 2],
            "masked_mae",
        )
    raise KeyError(model)


def optim_text(model: str) -> str:
    if model == "CycleNet":
        return '''CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {"lr": 0.01}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
desc = load_dataset_desc(DATA_NAME)
train_steps = math.ceil(desc["num_time_steps"] * TRAIN_VAL_TEST_RATIO[0])
CFG.TRAIN.LR_SCHEDULER.TYPE = "OneCycleLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "pct_start": 0.3,
    "epochs": NUM_EPOCHS,
    "steps_per_epoch": train_steps,
    "max_lr": CFG.TRAIN.OPTIM.PARAM["lr"],
}'''
    if model == "STID":
        return '''CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.002,
    "weight_decay": 0.0001,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {"milestones": [1, 30, 60, 80], "gamma": 0.5}'''
    if model == "PatchSTG":
        return '''CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "AdamW"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.002,
    "weight_decay": 0.0001,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {"milestones": [1, 35, 40], "gamma": 0.5}'''
    if model == "BigST":
        return '''CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "AdamW"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.002,
    "weight_decay": 0.0001,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {"milestones": [1, 50], "gamma": 0.5}'''
    raise KeyError(model)


def config_text(model: str, dataset: str, horizon: int, num_epochs: int, run_tag: str) -> str:
    param_text, forward_features, loss_expr = model_param(model, dataset, 96, horizon)
    import_text = model_import(model)
    training_optim_text = optim_text(model)
    return f'''import os
import sys
import math
import torch
from easydict import EasyDict

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "mvp_experiments", "active", "largest2019_scalable_st", "ltsf_input_window", "scripts")))

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.utils import get_regular_settings, load_adj, load_dataset_desc
from fast_ltsf_modules import FaSTIndexedTimeSeriesDataset, FaSTSampleFirstZScoreScaler
{import_text}

DATA_NAME = "{dataset}"
INPUT_LEN = 96
OUTPUT_LEN = {horizon}
NUM_EPOCHS = int(os.environ.get("FAST_LTSF_NUM_EPOCHS", "{num_epochs}"))
BATCH_SIZE = int(os.environ.get("FAST_LTSF_BATCH_SIZE", "64"))
RUN_TAG = os.environ.get("FAST_LTSF_RUN_TAG", "{run_tag}")

regular_settings = get_regular_settings(DATA_NAME)
TRAIN_VAL_TEST_RATIO = regular_settings["TRAIN_VAL_TEST_RATIO"]
NORM_EACH_CHANNEL = regular_settings["NORM_EACH_CHANNEL"]
RESCALE = regular_settings["RESCALE"]
NULL_VAL = regular_settings["NULL_VAL"]

{param_text}

CFG = EasyDict()
CFG.DESCRIPTION = "FaST-style LargeST LTSF baseline: {model} {dataset} 96->{horizon}"
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
CFG.DATASET.PARAM = EasyDict({{
    "dataset_name": DATA_NAME,
    "train_val_test_ratio": TRAIN_VAL_TEST_RATIO,
    "input_len": INPUT_LEN,
    "output_len": OUTPUT_LEN,
}})

CFG.SCALER = EasyDict()
CFG.SCALER.TYPE = FaSTSampleFirstZScoreScaler
CFG.SCALER.PARAM = EasyDict({{
    "dataset_name": DATA_NAME,
    "train_ratio": TRAIN_VAL_TEST_RATIO[0],
    "norm_each_channel": NORM_EACH_CHANNEL,
    "rescale": RESCALE,
    "input_len": INPUT_LEN,
    "output_len": OUTPUT_LEN,
}})

CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "{model}_FaSTLTSF"
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = {forward_features}
CFG.MODEL.TARGET_FEATURES = [0]

CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict({{
    "MAE": masked_mae,
    "RMSE": masked_rmse,
    "MAPE": masked_mape,
}})
CFG.METRICS.TARGET = "MAE"
CFG.METRICS.NULL_VAL = NULL_VAL

CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.EARLY_STOPPING_PATIENCE = int(os.environ.get("FAST_LTSF_PATIENCE", "30"))
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    "FaSTLTSF",
    "{model}",
    "_".join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), "L96", f"H{{OUTPUT_LEN}}", RUN_TAG]),
)
CFG.TRAIN.LOSS = {loss_expr}
{training_optim_text}
CFG.TRAIN.CLIP_GRAD_PARAM = {{"max_norm": 5.0}}
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
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="SD")
    parser.add_argument("--models", default="PatchSTG,BigST")
    parser.add_argument("--horizons", default="48,96,192,672")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--run-tag", default="fast_ltsf_smoke")
    parser.add_argument("--out-dir", default="BasicTS/baselines/FaSTLTSF/generated")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for dataset in csv_list(args.datasets):
        if dataset not in DATASETS:
            raise KeyError(f"Unknown dataset: {dataset}")
        for model in csv_list(args.models):
            for horizon in [int(item) for item in csv_list(args.horizons)]:
                path = out_dir / f"{model}_{dataset}_L96_H{horizon}_{args.run_tag}.py"
                path.write_text(
                    config_text(
                        model=model,
                        dataset=dataset,
                        horizon=horizon,
                        num_epochs=args.num_epochs,
                        run_tag=args.run_tag,
                    ),
                    encoding="utf-8",
                )
                written.append(path)

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
