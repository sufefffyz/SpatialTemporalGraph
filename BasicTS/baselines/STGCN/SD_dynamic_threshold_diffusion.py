import os

from baselines.AdaptiveGraph.sd_dynamic_common import build_dynamic_sd_cfg


CFG = build_dynamic_sd_cfg("stgcn")
CFG.MODEL.NAME = "DynamicThresholdSTGCN_Diff"
CFG.MODEL.PARAM["graph_conv_type"] = "diffusion_graph_conv"
CFG.MODEL.PARAM["Ks"] = int(os.environ.get("STGCN_DIFFUSION_STEPS", "2"))
CFG.DESCRIPTION = CFG.DESCRIPTION.replace(
    "DynamicThresholdSTGCN",
    "DynamicThresholdSTGCN-Diff",
).replace(
    "dynamic OSRM threshold support",
    "dynamic OSRM threshold diffusion support",
)
CFG.TRAIN.CKPT_SAVE_DIR = CFG.TRAIN.CKPT_SAVE_DIR.replace(
    "checkpoints/DynamicThresholdSTGCN/",
    "checkpoints/DynamicThresholdSTGCN_Diff/",
)
if "diffusion" not in CFG.TRAIN.CKPT_SAVE_DIR:
    CFG.TRAIN.CKPT_SAVE_DIR = f"{CFG.TRAIN.CKPT_SAVE_DIR}_diffusion"

if "WANDB" in CFG:
    if "WANDB_NAME" not in os.environ:
        CFG.WANDB.RUN_NAME = f"{CFG.WANDB.RUN_NAME}_diffusion"
    if "WANDB_RUN_GROUP" not in os.environ:
        CFG.WANDB.GROUP = CFG.WANDB.GROUP.replace("stgcn_", "stgcn_diffusion_")
        if "stgcn_diffusion" not in CFG.WANDB.GROUP:
            CFG.WANDB.GROUP = f"{CFG.WANDB.GROUP}_stgcn_diffusion"
    tags = list(CFG.WANDB.TAGS)
    tags = ["stgcn-diffusion" if tag == "stgcn" else tag for tag in tags]
    if "diffusion-graph-conv" not in tags:
        tags.append("diffusion-graph-conv")
    CFG.WANDB.TAGS = tags
