import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../.."))

from baselines.FEDformer.arch import FEDformer

from .common import build_sd_cfg


INPUT_LEN = 12
OUTPUT_LEN = 12
NUM_NODES = 716

MODEL_PARAM = {
    "enc_in": NUM_NODES,
    "dec_in": NUM_NODES,
    "c_out": NUM_NODES,
    "seq_len": INPUT_LEN,
    "label_len": INPUT_LEN // 2,
    "pred_len": OUTPUT_LEN,
    "d_model": 512,
    "version": "Fourier",
    "moving_avg": 3,
    "n_heads": 8,
    "e_layers": 2,
    "d_layers": 1,
    "d_ff": 2048,
    "dropout": 0.05,
    "output_attention": False,
    "embed": "timeF",
    "mode_select": "random",
    "modes": 8,
    "base": "legendre",
    "L": 3,
    "cross_activation": "tanh",
    "activation": "gelu",
    "num_time_features": 2,
    "time_of_day_size": 96,
    "day_of_week_size": 7,
}

CFG = build_sd_cfg(
    FEDformer,
    MODEL_PARAM,
    tag="flownetpaper",
    forward_features=[0, 1, 2],
    lr=0.0005,
    batch_size=64,
)
