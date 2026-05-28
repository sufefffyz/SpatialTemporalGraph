import os

from easydict import EasyDict


def _set_scheduler(cfg: EasyDict, scheduler_type: str | None, params: dict | None = None) -> None:
    if scheduler_type is None:
        cfg.TRAIN.pop("LR_SCHEDULER", None)
        return
    cfg.TRAIN.LR_SCHEDULER = EasyDict()
    cfg.TRAIN.LR_SCHEDULER.TYPE = scheduler_type
    cfg.TRAIN.LR_SCHEDULER.PARAM = params or {}


def _append_ckpt_tag(cfg: EasyDict, tag: str) -> None:
    ckpt_dir = cfg.TRAIN.CKPT_SAVE_DIR
    root, name = os.path.split(ckpt_dir)
    if tag not in name:
        cfg.TRAIN.CKPT_SAVE_DIR = os.path.join(root, f"{name}_{tag}")


def _append_wandb_tag(cfg: EasyDict, tag: str) -> None:
    if "WANDB" not in cfg:
        return
    tags = list(getattr(cfg.WANDB, "TAGS", []))
    if tag not in tags:
        tags.append(tag)
    cfg.WANDB.TAGS = tags
    if "WANDB_NAME" not in os.environ and cfg.WANDB.get("RUN_NAME"):
        run_name = cfg.WANDB.RUN_NAME
        if tag.replace("-", "_") not in run_name:
            cfg.WANDB.RUN_NAME = f"{run_name}_{tag.replace('-', '_')}"
    if "WANDB_RUN_GROUP" not in os.environ and cfg.WANDB.get("GROUP"):
        group = cfg.WANDB.GROUP
        if tag.replace("-", "_") not in group:
            cfg.WANDB.GROUP = f"{group}_{tag.replace('-', '_')}"


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def _set_common_large_st_controls(cfg: EasyDict) -> None:
    cfg.ENV.DETERMINISTIC = False
    cfg.ENV.CUDNN.ENABLED = True
    cfg.ENV.CUDNN.BENCHMARK = False
    cfg.ENV.CUDNN.DETERMINISTIC = False
    cfg.TRAIN.NUM_EPOCHS = int(os.environ.get("BASICTS_NUM_EPOCHS", str(cfg.TRAIN.NUM_EPOCHS)))
    cfg.TRAIN.EARLY_STOPPING_PATIENCE = int(os.environ.get("BASICTS_PATIENCE", "30"))
    cfg.TRAIN.DATA.BATCH_SIZE = int(os.environ.get("BASICTS_BATCH_SIZE", "64"))
    cfg.VAL.DATA.BATCH_SIZE = cfg.TRAIN.DATA.BATCH_SIZE
    cfg.TEST.DATA.BATCH_SIZE = cfg.TRAIN.DATA.BATCH_SIZE
    cfg.TEST.INTERVAL = cfg.TRAIN.NUM_EPOCHS


def apply_largest_aligned_cfg(cfg: EasyDict, backbone: str) -> EasyDict:
    """Mutate a BasicTS SD config to match the official LargeST training protocol."""
    backbone = backbone.lower()
    _set_common_large_st_controls(cfg)

    cfg.MODEL.FORWARD_FEATURES = [0, 1, 2]
    cfg.MODEL.TARGET_FEATURES = [0]

    if backbone == "gwnet":
        cfg.MODEL.PARAM["in_dim"] = 3
        if "addaptadj" in cfg.MODEL.PARAM:
            cfg.MODEL.PARAM["addaptadj"] = _env_bool("GWNET_ADDAPTADJ", bool(cfg.MODEL.PARAM["addaptadj"]))
        cfg.TRAIN.OPTIM.TYPE = "Adam"
        cfg.TRAIN.OPTIM.PARAM = {"lr": 0.001, "weight_decay": 0.0001}
        _set_scheduler(cfg, None)
        cfg.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}
    elif backbone == "dcrnn":
        cfg.MODEL.PARAM["input_dim"] = 3
        cfg.TRAIN.OPTIM.TYPE = "Adam"
        cfg.TRAIN.OPTIM.PARAM = {"lr": 0.01, "weight_decay": 0.0}
        _set_scheduler(cfg, "MultiStepLR", {"milestones": [10, 50, 90], "gamma": 0.1})
        cfg.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}
    elif backbone == "stgcn":
        blocks = cfg.MODEL.PARAM.get("blocks")
        if blocks:
            blocks[0] = [3]
        cfg.TRAIN.OPTIM.TYPE = "Adam"
        cfg.TRAIN.OPTIM.PARAM = {"lr": 0.001, "weight_decay": 0.0005}
        _set_scheduler(cfg, "StepLR", {"step_size": 10, "gamma": 0.95})
        cfg.TRAIN.pop("CLIP_GRAD_PARAM", None)
    else:
        raise ValueError(f"Unsupported LargeST-aligned backbone: {backbone}")

    if "LargeST-aligned" not in cfg.DESCRIPTION:
        cfg.DESCRIPTION = f"{cfg.DESCRIPTION} [LargeST-aligned]"
    _append_ckpt_tag(cfg, "largest_aligned")
    _append_wandb_tag(cfg, "largest-aligned")
    return cfg
