import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../.."))

from baselines.Crossformer.arch import Crossformer

from .common import build_sd_cfg


INPUT_LEN = 12
OUTPUT_LEN = 12
NUM_NODES = 716

MODEL_PARAM = {
    "data_dim": NUM_NODES,
    "in_len": INPUT_LEN,
    "out_len": OUTPUT_LEN,
    "seg_len": 3,
    "win_size": 2,
    "factor": 10,
    "d_model": 256,
    "d_ff": 512,
    "n_heads": 4,
    "e_layers": 3,
    "dropout": 0.2,
    "baseline": False,
}

CFG = build_sd_cfg(
    Crossformer,
    MODEL_PARAM,
    tag="flownetpaper",
    forward_features=[0],
    lr=0.0002,
    batch_size=16,
)
