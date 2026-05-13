import os

from .SD_osrm_gaussian_global import (
    CFG,
    DATA_NAME,
    GRAPH_TAG,
    INPUT_LEN,
    MODEL_ARCH,
    NUM_EPOCHS,
    OUTPUT_LEN,
    RUN_TAG,
)


CFG.DESCRIPTION = f"GraphWaveNet adaptive adjacency on {DATA_NAME} ({GRAPH_TAG})"
CFG.MODEL.PARAM["addaptadj"] = True
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    MODEL_ARCH.__name__,
    "_".join(
        [
            DATA_NAME,
            "osrm_gaussian_global_adaptive",
            str(NUM_EPOCHS),
            str(INPUT_LEN),
            str(OUTPUT_LEN),
            RUN_TAG,
        ]
    )
    if RUN_TAG
    else "_".join(
        [
            DATA_NAME,
            "osrm_gaussian_global_adaptive",
            str(NUM_EPOCHS),
            str(INPUT_LEN),
            str(OUTPUT_LEN),
        ]
    ),
)
CFG.WANDB.RUN_NAME = os.environ.get("WANDB_NAME", f"{MODEL_ARCH.__name__}_{DATA_NAME}_{GRAPH_TAG}_adaptive")
CFG.WANDB.GROUP = os.environ.get("WANDB_RUN_GROUP", "sd_osrm_gaussian_global_gwnet_adaptive")
CFG.WANDB.TAGS = ["adaptive-threshold", "sd", "osrm-gaussian-global", "gwnet-adaptive", GRAPH_TAG]
