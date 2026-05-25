import os

from .XCHENG import (
    BATCH_SIZE,
    CFG,
    DATA_NAME,
    INPUT_LEN,
    MODEL_PARAM,
    NUM_EPOCHS,
    OUTPUT_LEN,
    RUN_TAG,
)
from .arch import SparseGraphWaveNet


MODEL_ARCH = SparseGraphWaveNet
MODEL_PARAM["addaptadj"] = os.environ.get("BASICTS_SPARSE_ADDAPTADJ", "false").lower() in {
    "1",
    "true",
    "yes",
}
MODEL_PARAM["aptinit"] = None

CFG.DESCRIPTION = f"Sparse GraphWaveNet on {DATA_NAME}"
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM

CKPT_NAME_PARTS = [
    DATA_NAME,
    "xuancheng_sparse",
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
    "WANDB_NAME", f"SparseGraphWaveNet_{DATA_NAME}_{RUN_TAG or 'baseline'}"
)
CFG.WANDB.GROUP = os.environ.get("WANDB_RUN_GROUP", "xuancheng_basicts_sparsegwnet")
CFG.WANDB.TAGS = [
    "xuancheng",
    "cityflow",
    "sparsegwnet",
    DATA_NAME,
    f"batch{BATCH_SIZE}",
]
