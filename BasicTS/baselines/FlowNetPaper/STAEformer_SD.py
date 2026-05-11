import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../.."))

from baselines.STAEformer.arch import STAEformer

from .common import build_sd_cfg


INPUT_LEN = 12
OUTPUT_LEN = 12
NUM_NODES = 716

MODEL_PARAM = {
    "num_nodes": NUM_NODES,
    "in_steps": INPUT_LEN,
    "out_steps": OUTPUT_LEN,
    "steps_per_day": 96,
    "input_dim": 3,
    "output_dim": 1,
    "input_embedding_dim": 24,
    "tod_embedding_dim": 24,
    "dow_embedding_dim": 24,
    "spatial_embedding_dim": 0,
    "adaptive_embedding_dim": 80,
    "feed_forward_dim": 256,
    "num_heads": 4,
    "num_layers": 3,
    "dropout": 0.1,
    "use_mixed_proj": True,
}

CFG = build_sd_cfg(
    STAEformer,
    MODEL_PARAM,
    tag="flownetpaper",
    forward_features=[0, 1, 2],
    lr=0.001,
    weight_decay=0.0003,
    batch_size=16,
)
