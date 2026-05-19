import os

from easydict import EasyDict

from .SD import CFG, DATA_NAME, INPUT_LEN, MODEL_ARCH, MODEL_PARAM, NUM_EPOCHS, OUTPUT_LEN, RUN_TAG


DEFAULT_OSRM_DISTANCE = (
    "/home/yuzhang_fei/stg_artifacts_archive/adaptive_threshold_dynamic_weight/"
    "distance_matrices/SD/SD_osrm_shortest_distance_m.npy"
)

MODEL_PARAM["dist_mtx_path"] = os.environ.get("FLOWNET_DIST_MTX", DEFAULT_OSRM_DISTANCE)
MODEL_PARAM["dist_norm"] = os.environ.get("FLOWNET_DIST_NORM", "max")

CFG.DESCRIPTION = "FlowNet-style dense soft dynamic radius control on LargeST SD with OSRM distance"
CFG.MODEL.PARAM = MODEL_PARAM
ckpt_parts = [DATA_NAME, "osrm_soft_dense", str(NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)]
if RUN_TAG:
    ckpt_parts.append(RUN_TAG)
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints", MODEL_ARCH.__name__, "_".join(ckpt_parts))

CFG.WANDB = EasyDict()
CFG.WANDB.PROJECT = os.environ.get("WANDB_PROJECT", "adaptive_threshold_dynamic_weight")
CFG.WANDB.MODE = os.environ.get("WANDB_MODE", "online")
CFG.WANDB.RUN_NAME = os.environ.get("WANDB_NAME", f"{MODEL_ARCH.__name__}_{DATA_NAME}_osrm_soft_dense")
CFG.WANDB.GROUP = os.environ.get("WANDB_RUN_GROUP", "sd_flownet_soft_dense_osrm")
CFG.WANDB.TAGS = ["adaptive-threshold", "sd", "flownet-soft-dense", "osrm"]
