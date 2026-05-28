from easydict import EasyDict

from baselines.AdaptiveGraph.largest_aligned import (
    _append_ckpt_tag,
    _append_wandb_tag,
    _env_bool,
    _set_common_large_st_controls,
    _set_scheduler,
)


def apply_largest_flowonly_cfg(cfg: EasyDict, backbone: str) -> EasyDict:
    """Use LargeST training controls while keeping only the traffic-flow input."""
    backbone = backbone.lower()
    _set_common_large_st_controls(cfg)

    cfg.MODEL.FORWARD_FEATURES = [0]
    cfg.MODEL.TARGET_FEATURES = [0]

    if backbone == "gwnet":
        cfg.MODEL.PARAM["in_dim"] = 1
        if "addaptadj" in cfg.MODEL.PARAM:
            cfg.MODEL.PARAM["addaptadj"] = _env_bool("GWNET_ADDAPTADJ", bool(cfg.MODEL.PARAM["addaptadj"]))
        cfg.TRAIN.OPTIM.TYPE = "Adam"
        cfg.TRAIN.OPTIM.PARAM = {"lr": 0.001, "weight_decay": 0.0001}
        _set_scheduler(cfg, None)
        cfg.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}
    elif backbone == "dcrnn":
        cfg.MODEL.PARAM["input_dim"] = 1
        cfg.TRAIN.OPTIM.TYPE = "Adam"
        cfg.TRAIN.OPTIM.PARAM = {"lr": 0.01, "weight_decay": 0.0}
        _set_scheduler(cfg, "MultiStepLR", {"milestones": [10, 50, 90], "gamma": 0.1})
        cfg.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}
    elif backbone == "stgcn":
        blocks = cfg.MODEL.PARAM.get("blocks")
        if blocks:
            blocks[0] = [1]
        cfg.TRAIN.OPTIM.TYPE = "Adam"
        cfg.TRAIN.OPTIM.PARAM = {"lr": 0.001, "weight_decay": 0.0005}
        _set_scheduler(cfg, "StepLR", {"step_size": 10, "gamma": 0.95})
        cfg.TRAIN.pop("CLIP_GRAD_PARAM", None)
    else:
        raise ValueError(f"Unsupported LargeST-flowonly backbone: {backbone}")

    if "LargeST-flowonly" not in cfg.DESCRIPTION:
        cfg.DESCRIPTION = f"{cfg.DESCRIPTION} [LargeST-flowonly]"
    _append_ckpt_tag(cfg, "largest_flowonly")
    _append_wandb_tag(cfg, "largest-flowonly")
    return cfg
