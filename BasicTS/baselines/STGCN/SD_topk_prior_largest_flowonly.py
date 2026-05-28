from baselines.AdaptiveGraph.largest_flowonly import apply_largest_flowonly_cfg

from .sd_common import build_sd_cfg


CFG = apply_largest_flowonly_cfg(build_sd_cfg("osrm_gaussian_global"), "stgcn")
CFG.DESCRIPTION = CFG.DESCRIPTION.replace("OSRM Gaussian global-threshold graph", "fixed top-k prior graph")
CFG.TRAIN.CKPT_SAVE_DIR = CFG.TRAIN.CKPT_SAVE_DIR.replace(
    "osrm_gaussian_global_fixed",
    "topk_prior_fixed",
)
if "WANDB" in CFG:
    CFG.WANDB.RUN_NAME = CFG.WANDB.RUN_NAME.replace("largest_flowonly", "topk_prior_largest_flowonly")
    CFG.WANDB.GROUP = CFG.WANDB.GROUP.replace("sd_osrm_gaussian_global_stgcn", "sd_topk_prior_stgcn_fixed")
    CFG.WANDB.TAGS = [
        "topk-prior" if tag == "osrm-gaussian-global" else tag
        for tag in CFG.WANDB.TAGS
    ]
