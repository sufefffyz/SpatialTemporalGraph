import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../.."))

from baselines.iTransformer.arch import iTransformer

from .common import build_sd_cfg


INPUT_LEN = 12
OUTPUT_LEN = 12
NUM_NODES = 716

MODEL_PARAM = {
    "task_name": "forecast",
    "enc_in": NUM_NODES,
    "dec_in": NUM_NODES,
    "c_out": NUM_NODES,
    "seq_len": INPUT_LEN,
    "label_len": INPUT_LEN // 2,
    "pred_len": OUTPUT_LEN,
    "factor": 3,
    "p_hidden_dims": [128, 128],
    "p_hidden_layers": 2,
    "d_model": 512,
    "moving_avg": 3,
    "n_heads": 8,
    "e_layers": 4,
    "d_layers": 1,
    "d_ff": 512,
    "distil": True,
    "sigma": 0.2,
    "dropout": 0.1,
    "freq": "h",
    "use_norm": True,
    "output_attention": False,
    "embed": "timeF",
    "activation": "gelu",
    "num_time_features": 2,
    "time_of_day_size": 96,
    "day_of_week_size": 7,
}

CFG = build_sd_cfg(
    iTransformer,
    MODEL_PARAM,
    tag="flownetpaper",
    forward_features=[0, 1, 2],
    lr=0.001,
    weight_decay=0.0,
    batch_size=32,
)
