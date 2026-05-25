import os

from .SD_osrm_gaussian_global import (
    CFG,
    DATA_NAME,
    GRAPH_TAG,
    INPUT_LEN,
    MODEL_ARCH,
    MODEL_PARAM,
    NUM_EPOCHS,
    OUTPUT_LEN,
    RUN_TAG,
)


CFG.DESCRIPTION = f"GraphWaveNet fixed top-k prior graph on {DATA_NAME} ({GRAPH_TAG})"
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    MODEL_ARCH.__name__,
    "_".join(
        [
            DATA_NAME,
            "topk_prior_fixed",
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
            "topk_prior_fixed",
            str(NUM_EPOCHS),
            str(INPUT_LEN),
            str(OUTPUT_LEN),
        ]
    ),
)
CFG.WANDB.RUN_NAME = os.environ.get("WANDB_NAME", f"{MODEL_ARCH.__name__}_{DATA_NAME}_{GRAPH_TAG}_topk_prior")
CFG.WANDB.GROUP = os.environ.get("WANDB_RUN_GROUP", "sd_topk_prior_gwnet_fixed")
CFG.WANDB.TAGS = ["adaptive-threshold", "sd", "topk-prior", "gwnet-fixed", GRAPH_TAG]
