import os

from .SD_topk_prior import (
    CFG,
    DATA_NAME,
    GRAPH_TAG,
    INPUT_LEN,
    MODEL_PARAM,
    NUM_EPOCHS,
    OUTPUT_LEN,
    RUN_TAG,
)
from .arch import SparseGraphWaveNet


MODEL_ARCH = SparseGraphWaveNet
MODEL_PARAM["addaptadj"] = False
MODEL_PARAM["aptinit"] = None

CFG.DESCRIPTION = f"Sparse GraphWaveNet fixed top-k prior graph on {DATA_NAME} ({GRAPH_TAG})"
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM

CKPT_NAME_PARTS = [
    DATA_NAME,
    "topk_prior_sparse_fixed",
    str(NUM_EPOCHS),
    str(INPUT_LEN),
    str(OUTPUT_LEN),
]
if RUN_TAG:
    CKPT_NAME_PARTS.append(RUN_TAG)
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    MODEL_ARCH.__name__,
    "_".join(CKPT_NAME_PARTS),
)

CFG.WANDB.RUN_NAME = os.environ.get(
    "WANDB_NAME", f"{MODEL_ARCH.__name__}_{DATA_NAME}_{GRAPH_TAG}_topk_prior"
)
CFG.WANDB.GROUP = os.environ.get("WANDB_RUN_GROUP", "sd_topk_prior_sparsegwnet_fixed")
CFG.WANDB.TAGS = [
    "adaptive-threshold",
    "sd",
    "topk-prior",
    "sparse-operator",
    "sparsegwnet",
    GRAPH_TAG,
]
