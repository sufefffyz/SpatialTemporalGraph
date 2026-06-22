import os

from baselines.GWNet.NoAdaptive import (
    CFG,
    DATA_NAME,
    GRAPH_TAG,
    INPUT_LEN,
    MODEL_PARAM,
    NUM_EPOCHS,
    OUTPUT_LEN,
    RUN_TAG,
    SCALER_TAG,
)


MODEL_PARAM["addaptadj"] = True

CFG.DESCRIPTION = f"GraphWaveNet adaptive adjacency baseline on {DATA_NAME}"
CFG.MODEL.NAME = "GraphWaveNetAdaptive"
CFG.MODEL.PARAM = MODEL_PARAM

ckpt_parts = [DATA_NAME, GRAPH_TAG, "addapt", str(NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)]
if SCALER_TAG != "targetZ":
    ckpt_parts.append(SCALER_TAG)
if RUN_TAG:
    ckpt_parts.append(RUN_TAG)
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints", CFG.MODEL.NAME, "_".join(ckpt_parts))

CFG.WANDB.RUN_NAME = os.environ.get("WANDB_NAME", f"GWNet_{DATA_NAME}_{GRAPH_TAG}_addapt")
CFG.WANDB.GROUP = os.environ.get("WANDB_RUN_GROUP", "gwnet_module_ablation_addapt")
CFG.WANDB.TAGS = ["gwnet", DATA_NAME.lower(), "addaptadj", GRAPH_TAG, SCALER_TAG]
