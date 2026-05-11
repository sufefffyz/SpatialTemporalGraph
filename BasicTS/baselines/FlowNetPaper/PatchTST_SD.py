import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../.."))

from baselines.PatchTST.arch import PatchTST

from .common import build_sd_cfg


INPUT_LEN = 12
OUTPUT_LEN = 12
NUM_NODES = 716

MODEL_PARAM = {
    "enc_in": NUM_NODES,
    "seq_len": INPUT_LEN,
    "pred_len": OUTPUT_LEN,
    "e_layers": 3,
    "n_heads": 16,
    "d_model": 128,
    "d_ff": 256,
    "dropout": 0.2,
    "fc_dropout": 0.2,
    "head_dropout": 0.0,
    "patch_len": 4,
    "stride": 2,
    "individual": 0,
    "padding_patch": "end",
    "revin": 1,
    "affine": 0,
    "subtract_last": 0,
    "decomposition": 0,
    "kernel_size": 3,
}

CFG = build_sd_cfg(
    PatchTST,
    MODEL_PARAM,
    tag="flownetpaper",
    forward_features=[0],
    lr=0.001,
    batch_size=64,
)
