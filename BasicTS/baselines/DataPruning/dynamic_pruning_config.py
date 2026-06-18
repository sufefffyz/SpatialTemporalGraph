import copy
import importlib.util
import os

from easydict import EasyDict

from basicts.runners import (DynamicPruningTimeSeriesForecastingRunner,
                             DynamicPruningWandBTimeSeriesForecastingRunner,
                             WandBTimeSeriesForecastingRunner)


def _ratio_tag(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _append_ckpt_tag(cfg: EasyDict, tag: str) -> None:
    root, name = os.path.split(cfg.TRAIN.CKPT_SAVE_DIR)
    if tag not in name:
        cfg.TRAIN.CKPT_SAVE_DIR = os.path.join(root, f"{name}_{tag}")


def _select_runner(runner_type):
    try:
        if issubclass(runner_type, WandBTimeSeriesForecastingRunner):
            return DynamicPruningWandBTimeSeriesForecastingRunner
    except TypeError:
        pass
    return DynamicPruningTimeSeriesForecastingRunner


def _env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no"}


def _rescale_default_for_strategy(strategy: str) -> str:
    """Strategies with explicit pruning weights default to rescaled losses."""

    return "1" if strategy in {"infobatch", "infobatch_norm", "proxy_gap", "node_hard_revisit"} else "0"


def _paper_optimizer_name() -> str:
    optimizer = os.environ.get("DYNAMIC_PRUNING_OPTIMIZER", "SGD").strip().lower()
    aliases = {
        "sgd": "sgd",
        "adam": "adam",
        "muon": "muon",
    }
    if optimizer not in aliases:
        raise ValueError(
            "DYNAMIC_PRUNING_OPTIMIZER must be one of {'SGD', 'Adam', 'Muon'}, "
            f"got {optimizer!r}."
        )
    return aliases[optimizer]


def _load_muon_optimizer_type():
    global MuonWithAdamFallback
    try:
        from .paper_optimizers import MuonWithAdamFallback
        return MuonWithAdamFallback
    except ImportError:
        module_path = os.path.join(os.path.dirname(__file__), "paper_optimizers.py")
        spec = importlib.util.spec_from_file_location("data_pruning_paper_optimizers", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        MuonWithAdamFallback = module.MuonWithAdamFallback
        return MuonWithAdamFallback


def _paper_optimizer_tag() -> str:
    return f"opt_{_paper_optimizer_name()}"


def _apply_paper_lr_scheduler(cfg: EasyDict) -> None:
    scheduler = os.environ.get("DYNAMIC_PRUNING_LR_SCHEDULER", "CosineAnnealingLR").strip().lower()

    cfg.TRAIN.LR_SCHEDULER = EasyDict()
    if scheduler in {"cosine", "cosineannealinglr"}:
        cfg.TRAIN.LR_SCHEDULER.TYPE = "CosineAnnealingLR"
        cfg.TRAIN.LR_SCHEDULER.PARAM = {
            "T_max": int(os.environ.get("DYNAMIC_PRUNING_COSINE_T_MAX", str(cfg.TRAIN.NUM_EPOCHS))),
            "eta_min": float(os.environ.get("DYNAMIC_PRUNING_MIN_LR", "0.0001")),
        }
    elif scheduler in {"step", "steplr"}:
        cfg.TRAIN.LR_SCHEDULER.TYPE = "StepLR"
        cfg.TRAIN.LR_SCHEDULER.PARAM = {
            "step_size": int(os.environ.get("DYNAMIC_PRUNING_STEP_SIZE", "50")),
            "gamma": float(os.environ.get("DYNAMIC_PRUNING_STEP_GAMMA", "0.5")),
        }
    else:
        raise ValueError(
            "DYNAMIC_PRUNING_LR_SCHEDULER must be one of "
            "{'CosineAnnealingLR', 'StepLR'}, "
            f"got {scheduler!r}."
        )


def _maybe_disable_grad_clip(cfg: EasyDict) -> None:
    if _env_flag("DYNAMIC_PRUNING_DISABLE_CLIP", "0"):
        cfg.TRAIN.CLIP_GRAD_PARAM = None


def _apply_paper_hparams(cfg: EasyDict) -> None:
    """Use the unified optimizer/scheduler/batch-size protocol from Chen et al. 2026."""

    optimizer = _paper_optimizer_name()
    lr = float(os.environ.get("DYNAMIC_PRUNING_LR", "0.001"))
    momentum = float(os.environ.get("DYNAMIC_PRUNING_MOMENTUM", "0.9"))
    weight_decay = float(os.environ.get("DYNAMIC_PRUNING_WEIGHT_DECAY", "0.0001"))

    cfg.TRAIN.OPTIM = EasyDict()
    if optimizer == "sgd":
        cfg.TRAIN.OPTIM.TYPE = "SGD"
        cfg.TRAIN.OPTIM.PARAM = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
        }
    elif optimizer == "adam":
        cfg.TRAIN.OPTIM.TYPE = "Adam"
        cfg.TRAIN.OPTIM.PARAM = {
            "lr": lr,
            "weight_decay": weight_decay,
        }
    else:
        cfg.TRAIN.OPTIM.TYPE = _load_muon_optimizer_type()
        cfg.TRAIN.OPTIM.PARAM = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
        }

    _apply_paper_lr_scheduler(cfg)
    _maybe_disable_grad_clip(cfg)

    batch_size = int(os.environ.get("DYNAMIC_PRUNING_BATCH_SIZE", "256"))
    cfg.TRAIN.DATA.BATCH_SIZE = batch_size
    if hasattr(cfg, "VAL") and hasattr(cfg.VAL, "DATA"):
        cfg.VAL.DATA.BATCH_SIZE = batch_size
    if hasattr(cfg, "TEST") and hasattr(cfg.TEST, "DATA"):
        cfg.TEST.DATA.BATCH_SIZE = batch_size


def apply_paper_hparam_full_cfg(cfg: EasyDict, backbone_tag: str) -> EasyDict:
    """Attach the paper hyperparameter protocol without enabling dynamic pruning."""

    cfg = copy.deepcopy(cfg)
    seed = int(os.environ.get("DYNAMIC_PRUNING_SEED", os.environ.get("BASICTS_SEED", "2023")))
    disable_early_stopping = _env_flag("DYNAMIC_PRUNING_DISABLE_EARLY_STOPPING", "1")
    cfg.ENV.SEED = seed
    _apply_paper_hparams(cfg)
    if disable_early_stopping:
        cfg.TRAIN.EARLY_STOPPING_PATIENCE = None
    optimizer_tag = _paper_optimizer_tag()
    _append_ckpt_tag(cfg, f"paperhp_full_{backbone_tag}_{optimizer_tag}_s{seed}")
    ckpt_suffix = os.environ.get("DYNAMIC_PRUNING_CKPT_SUFFIX", "").strip()
    if ckpt_suffix:
        _append_ckpt_tag(cfg, ckpt_suffix)
    cfg.DESCRIPTION = (
        f"{cfg.DESCRIPTION} [paperhp_full, optimizer={optimizer_tag}, "
        f"disable_early_stopping={disable_early_stopping}, seed={seed}]"
    )
    return cfg


def apply_dynamic_pruning_cfg(cfg: EasyDict, backbone_tag: str, strategy: str) -> EasyDict:
    """Attach opt-in dynamic pruning settings to an existing BasicTS config."""

    strategy = str(strategy).lower().replace("-", "_")
    strategy_tag = strategy
    reference_type_default = "seasonal"
    score_loss_type_default_override = None
    if strategy in {"eps_greedy", "epsilon"}:
        strategy = "epsilon_greedy"
    if strategy in {"infobatch_normalized", "normalized_infobatch"}:
        strategy = "infobatch_norm"
    if strategy in {"rho_prune", "proxy_gap_prune"}:
        strategy = "proxy_gap"
    if strategy in {"proxy_gap_lowpass", "rho_prune_lowpass", "rho_lowpass"}:
        strategy = "proxy_gap"
        score_loss_type_default_override = "lowpass_normalized_mae"
    if strategy in {"proxy_gap_lowpass_node_hard", "lowpass_node_hard", "node_hard_lowpass"}:
        strategy = "node_hard_revisit"
        score_loss_type_default_override = "lowpass_normalized_mae"
    if strategy in {"proxy_gap_dlinear", "rho_prune_dlinear", "rho_dlinear"}:
        strategy = "proxy_gap"
        reference_type_default = "dlinear"
    cfg = copy.deepcopy(cfg)

    ratio = float(os.environ.get("DYNAMIC_PRUNING_RATIO", "0.1"))
    reference_num_epochs = int(cfg.TRAIN.NUM_EPOCHS)
    infobatch_like = strategy in {"infobatch", "infobatch_norm", "proxy_gap"}
    prune_probability_default = "0.5" if infobatch_like else "0.0"
    prune_probability = float(os.environ.get("DYNAMIC_PRUNING_PRUNE_PROBABILITY", prune_probability_default))
    epsilon = float(os.environ.get("DYNAMIC_PRUNING_EPSILON", "0.1"))
    pruning_period = int(os.environ.get("DYNAMIC_PRUNING_PRUNING_PERIOD", "1"))
    seed = int(os.environ.get("DYNAMIC_PRUNING_SEED", os.environ.get("BASICTS_SEED", "2023")))
    warmup_default = "0"
    warmup_epochs = int(os.environ.get("DYNAMIC_PRUNING_WARMUP_EPOCHS", warmup_default))
    target_forward_ratio_default = ""
    target_forward_ratio_env = os.environ.get("DYNAMIC_PRUNING_TARGET_FORWARD_RATIO", target_forward_ratio_default)
    target_forward_ratio = float(target_forward_ratio_env) if target_forward_ratio_env else None
    if strategy == "node_hard_revisit":
        target_forward_ratio = None

    expected_retained_ratio = None
    match_epochs = os.environ.get("DYNAMIC_PRUNING_MATCH_EPOCHS", "0").strip().lower() not in {"0", "false", "no"}
    if infobatch_like and target_forward_ratio is not None and match_epochs:
        expected_retained_ratio = float(os.environ.get("DYNAMIC_PRUNING_EXPECTED_RETAINED_RATIO", "0.75"))
        if not 0.0 < expected_retained_ratio <= 1.0:
            raise ValueError(f"DYNAMIC_PRUNING_EXPECTED_RETAINED_RATIO must be in (0, 1], got {expected_retained_ratio}.")

    delta_default = "0.875" if infobatch_like else "1.0"
    delta = float(os.environ.get("DYNAMIC_PRUNING_DELTA", delta_default))
    final_full_ratio_default = "0.0" if strategy == "soft_random" else str(max(0.0, 1.0 - delta))
    final_full_ratio = float(os.environ.get("DYNAMIC_PRUNING_FINAL_FULL_RATIO", final_full_ratio_default))
    use_budget_final_full = infobatch_like and target_forward_ratio is not None and match_epochs
    final_full_epochs_default = 0 if use_budget_final_full else int(round(cfg.TRAIN.NUM_EPOCHS * final_full_ratio))
    if infobatch_like and not use_budget_final_full and final_full_ratio > 0.0 and cfg.TRAIN.NUM_EPOCHS > 1:
        final_full_epochs_default = max(1, final_full_epochs_default)
    final_full_epochs = int(os.environ.get("DYNAMIC_PRUNING_FINAL_FULL_EPOCHS", str(final_full_epochs_default)))
    final_full_forward_ratio = final_full_ratio if use_budget_final_full else 0.0
    score_momentum = float(os.environ.get("DYNAMIC_PRUNING_SCORE_MOMENTUM", "0.0"))
    score_alpha_default = "0.8" if strategy == "epsilon_greedy" else ""
    score_alpha_env = os.environ.get("DYNAMIC_PRUNING_SCORE_ALPHA", score_alpha_default)
    score_alpha = float(score_alpha_env) if score_alpha_env else None
    min_batch_size = int(os.environ.get("DYNAMIC_PRUNING_MIN_BATCH_SIZE", "1"))
    rescale_default = _rescale_default_for_strategy(strategy)
    rescale = os.environ.get("DYNAMIC_PRUNING_RESCALE", rescale_default).strip().lower() not in {"0", "false", "no"}
    score_loss_type_default = score_loss_type_default_override
    if score_loss_type_default is None:
        score_loss_type_default = "normalized_mae" if strategy in {"infobatch_norm", "proxy_gap"} else "raw_mae"
    score_loss_type = os.environ.get("DYNAMIC_PRUNING_SCORE_LOSS_TYPE", score_loss_type_default).strip().lower()
    lowpass_keep_ratio = float(os.environ.get("DYNAMIC_PRUNING_LOWPASS_KEEP_RATIO", "0.5"))
    node_loss_pruning = strategy == "node_hard_revisit"
    node_retention_ratio = float(os.environ.get("DYNAMIC_PRUNING_NODE_RATIO", str(ratio)))
    node_revisit_probability = float(os.environ.get("DYNAMIC_PRUNING_NODE_REVISIT_PROBABILITY", "0.5"))
    reference_type = os.environ.get("DYNAMIC_PRUNING_REFERENCE_TYPE", reference_type_default).strip().lower()
    reference_period = int(os.environ.get("DYNAMIC_PRUNING_REFERENCE_PERIOD", "288"))
    dlinear_reference_epochs = int(os.environ.get("DYNAMIC_PRUNING_DLINEAR_EPOCHS", "100"))
    dlinear_reference_patience = int(os.environ.get("DYNAMIC_PRUNING_DLINEAR_PATIENCE", "30"))
    dlinear_reference_lr = float(os.environ.get("DYNAMIC_PRUNING_DLINEAR_LR", "0.001"))
    dlinear_reference_weight_decay = float(os.environ.get("DYNAMIC_PRUNING_DLINEAR_WEIGHT_DECAY", "0.0"))
    dlinear_reference_individual = _env_flag("DYNAMIC_PRUNING_DLINEAR_INDIVIDUAL", "0")
    disable_early_stopping = _env_flag("DYNAMIC_PRUNING_DISABLE_EARLY_STOPPING", "1")
    align_paper_hparams = _env_flag("DYNAMIC_PRUNING_ALIGN_PAPER_HPARAMS", "1")

    cfg.RUNNER = _select_runner(cfg.RUNNER)
    cfg.ENV.SEED = seed
    if align_paper_hparams:
        _apply_paper_hparams(cfg)
    if disable_early_stopping:
        cfg.TRAIN.EARLY_STOPPING_PATIENCE = None

    cfg.TRAIN.DYNAMIC_PRUNING = EasyDict(
        {
            "ENABLED": True,
            "STRATEGY": strategy,
            "RETENTION_RATIO": ratio,
            "PRUNE_PROBABILITY": prune_probability,
            "EPSILON": epsilon,
            "PRUNING_PERIOD": pruning_period,
            "TARGET_FORWARD_RATIO": target_forward_ratio,
            "REFERENCE_NUM_EPOCHS": reference_num_epochs,
            "EXPECTED_RETAINED_RATIO": expected_retained_ratio,
            "SEED": seed,
            "WARMUP_EPOCHS": warmup_epochs,
            "FINAL_FULL_EPOCHS": final_full_epochs,
            "FINAL_FULL_FORWARD_RATIO": final_full_forward_ratio,
            "SCORE_MOMENTUM": score_momentum,
            "SCORE_ALPHA": score_alpha,
            "MIN_BATCH_SIZE": min_batch_size,
            "RESCALE": rescale,
            "SCORE_LOSS_TYPE": score_loss_type,
            "LOWPASS_KEEP_RATIO": lowpass_keep_ratio,
            "NODE_LOSS_PRUNING": node_loss_pruning,
            "NODE_RETENTION_RATIO": node_retention_ratio,
            "NODE_REVISIT_PROBABILITY": node_revisit_probability,
            "REFERENCE_TYPE": reference_type,
            "REFERENCE_PERIOD": reference_period,
            "DLINEAR_EPOCHS": dlinear_reference_epochs,
            "DLINEAR_PATIENCE": dlinear_reference_patience,
            "DLINEAR_LR": dlinear_reference_lr,
            "DLINEAR_WEIGHT_DECAY": dlinear_reference_weight_decay,
            "DLINEAR_INDIVIDUAL": dlinear_reference_individual,
            "DISABLE_EARLY_STOPPING": disable_early_stopping,
            "ALIGN_PAPER_HPARAMS": align_paper_hparams,
        }
    )

    hparam_tag = f"_paperhp_{_paper_optimizer_tag()}" if align_paper_hparams else ""
    if infobatch_like:
        budget_tag = f"_budget{_ratio_tag(target_forward_ratio)}" if target_forward_ratio is not None else ""
        _append_ckpt_tag(
            cfg,
            f"dynprune_{backbone_tag}_{strategy_tag}_p{_ratio_tag(prune_probability)}{budget_tag}{hparam_tag}_s{seed}",
        )
        cfg.DESCRIPTION = (
            f"{cfg.DESCRIPTION} [dynamic_pruning:{strategy_tag}, effective_strategy={strategy}, "
            f"prune_probability={prune_probability}, target_forward_ratio={target_forward_ratio}, "
            f"reference_epochs={reference_num_epochs}, num_epochs={cfg.TRAIN.NUM_EPOCHS}, "
            f"rescale={rescale}, score_loss_type={score_loss_type}, "
            f"lowpass_keep_ratio={lowpass_keep_ratio}, "
            f"reference_type={reference_type}, reference_period={reference_period}, "
            f"dlinear_epochs={dlinear_reference_epochs}, dlinear_patience={dlinear_reference_patience}, "
            f"disable_early_stopping={disable_early_stopping}, "
            f"align_paper_hparams={align_paper_hparams}, seed={seed}]"
        )
    elif strategy == "epsilon_greedy":
        _append_ckpt_tag(
            cfg,
            f"dynprune_{backbone_tag}_{strategy}_r{_ratio_tag(ratio)}_eps{_ratio_tag(epsilon)}_tp{pruning_period}{hparam_tag}_s{seed}",
        )
        cfg.DESCRIPTION = (
            f"{cfg.DESCRIPTION} [dynamic_pruning:{strategy}, ratio={ratio}, epsilon={epsilon}, "
            f"pruning_period={pruning_period}, score_alpha={score_alpha}, rescale={rescale}, "
            f"disable_early_stopping={disable_early_stopping}, "
            f"align_paper_hparams={align_paper_hparams}, seed={seed}]"
        )
    elif strategy == "node_hard_revisit":
        _append_ckpt_tag(
            cfg,
            f"dynprune_{backbone_tag}_{strategy_tag}_nr{_ratio_tag(node_retention_ratio)}{hparam_tag}_s{seed}",
        )
        cfg.DESCRIPTION = (
            f"{cfg.DESCRIPTION} [dynamic_pruning:{strategy_tag}, effective_strategy={strategy}, "
            f"node_retention_ratio={node_retention_ratio}, node_revisit_probability={node_revisit_probability}, "
            f"sample_forward=full, "
            f"rescale={rescale}, score_loss_type={score_loss_type}, "
            f"lowpass_keep_ratio={lowpass_keep_ratio}, "
            f"disable_early_stopping={disable_early_stopping}, "
            f"align_paper_hparams={align_paper_hparams}, seed={seed}]"
        )
    else:
        _append_ckpt_tag(cfg, f"dynprune_{backbone_tag}_{strategy_tag}_r{_ratio_tag(ratio)}{hparam_tag}_s{seed}")
        cfg.DESCRIPTION = (
            f"{cfg.DESCRIPTION} [dynamic_pruning:{strategy_tag}, effective_strategy={strategy}, ratio={ratio}, "
            f"rescale={rescale}, disable_early_stopping={disable_early_stopping}, "
            f"align_paper_hparams={align_paper_hparams}, seed={seed}]"
        )

    ckpt_suffix = os.environ.get("DYNAMIC_PRUNING_CKPT_SUFFIX", "").strip()
    if ckpt_suffix:
        _append_ckpt_tag(cfg, ckpt_suffix)
        cfg.DESCRIPTION = f"{cfg.DESCRIPTION} [dynamic_pruning_ckpt_suffix:{ckpt_suffix}]"
    return cfg


def _apply_dynamic_pruning_cfg_with_env_overrides(
    cfg: EasyDict,
    backbone_tag: str,
    strategy: str,
    overrides: dict,
) -> EasyDict:
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            os.environ[key] = value
        return apply_dynamic_pruning_cfg(cfg, backbone_tag, strategy)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def apply_cluster_subgraph_cfg(
    cfg: EasyDict,
    backbone_tag: str,
    strategy: str,
    cluster_type: str = None,
) -> EasyDict:
    """Attach node-cluster subgraph forward to a dynamic-pruning config.

    strategy="none" means spatial-only: all windows are retained while each
    training step forwards only the configured node clusters.
    """

    strategy = str(strategy).lower().replace("-", "_")
    spatial_only = strategy in {"none", "full", "spatial_only", "cluster_only", "no_temporal_pruning"}
    if spatial_only:
        cfg = _apply_dynamic_pruning_cfg_with_env_overrides(
            cfg,
            backbone_tag,
            "soft_random",
            {
                "DYNAMIC_PRUNING_RATIO": "1.0",
                "DYNAMIC_PRUNING_TARGET_FORWARD_RATIO": "",
                "DYNAMIC_PRUNING_MATCH_EPOCHS": "0",
                "DYNAMIC_PRUNING_FINAL_FULL_RATIO": "0.0",
            },
        )
        pruning_cfg = cfg.TRAIN.DYNAMIC_PRUNING
        pruning_cfg.STRATEGY = "soft_random"
        pruning_cfg.RETENTION_RATIO = 1.0
        pruning_cfg.PRUNE_PROBABILITY = 0.0
        pruning_cfg.RESCALE = False
        pruning_cfg.TARGET_FORWARD_RATIO = None
        pruning_cfg.EXPECTED_RETAINED_RATIO = None
        pruning_cfg.FINAL_FULL_FORWARD_RATIO = 0.0
        final_full_ratio = float(os.environ.get("CLUSTER_SUBGRAPH_FINAL_FULL_RATIO", "0.125"))
        pruning_cfg.FINAL_FULL_EPOCHS = int(
            os.environ.get(
                "CLUSTER_SUBGRAPH_FINAL_FULL_EPOCHS",
                str(max(1, int(round(cfg.TRAIN.NUM_EPOCHS * final_full_ratio)))),
            )
        )
        _append_ckpt_tag(cfg, "spatial_only")
    else:
        cfg = apply_dynamic_pruning_cfg(cfg, backbone_tag, strategy)
        pruning_cfg = cfg.TRAIN.DYNAMIC_PRUNING

    cluster_type = os.environ.get("CLUSTER_SUBGRAPH_TYPE", cluster_type or "spatial_kdtree").strip().lower()
    if cluster_type == "random_balanced":
        cluster_type = "random_balanced_clusters"
    if cluster_type not in {"spatial_kdtree", "signal_kmeans", "random_balanced_clusters"}:
        raise ValueError(
            "cluster_type must be one of {'spatial_kdtree', 'signal_kmeans', 'random_balanced_clusters'}, "
            f"got {cluster_type!r}."
        )

    num_clusters = int(os.environ.get("CLUSTER_SUBGRAPH_NUM_CLUSTERS", "8"))
    num_active_clusters = int(os.environ.get("CLUSTER_SUBGRAPH_ACTIVE_CLUSTERS", "2"))
    if num_clusters < 1:
        raise ValueError(f"CLUSTER_SUBGRAPH_NUM_CLUSTERS must be >= 1, got {num_clusters}.")
    if not 1 <= num_active_clusters <= num_clusters:
        raise ValueError(
            "CLUSTER_SUBGRAPH_ACTIVE_CLUSTERS must be in [1, num_clusters], "
            f"got {num_active_clusters} for num_clusters={num_clusters}."
        )

    seed = int(os.environ.get("CLUSTER_SUBGRAPH_SEED", str(pruning_cfg.SEED)))
    dataset_name = os.environ.get(
        "CLUSTER_SUBGRAPH_DATASET",
        getattr(getattr(cfg, "DATASET", EasyDict()), "NAME", backbone_tag),
    )
    pruning_cfg.CLUSTER_SUBGRAPH = EasyDict(
        {
            "ENABLED": True,
            "TYPE": cluster_type,
            "NUM_CLUSTERS": num_clusters,
            "NUM_ACTIVE_CLUSTERS": num_active_clusters,
            "ACTIVE_NODE_RATIO": float(num_active_clusters) / float(num_clusters),
            "SEED": seed,
            "DATASET": dataset_name,
            "STEPS_PER_DAY": int(os.environ.get("CLUSTER_SUBGRAPH_STEPS_PER_DAY", "96")),
            "KMEANS_MAX_ITER": int(os.environ.get("CLUSTER_SUBGRAPH_KMEANS_MAX_ITER", "50")),
            "META_CSV": os.environ.get("CLUSTER_SUBGRAPH_META_CSV", ""),
            "CACHE_DIR": os.environ.get(
                "CLUSTER_SUBGRAPH_CACHE_DIR",
                os.path.join("datasets", str(dataset_name), "cluster_cache"),
            ),
            "USE_CACHE": _env_flag("CLUSTER_SUBGRAPH_USE_CACHE", "1"),
        }
    )
    if not pruning_cfg.CLUSTER_SUBGRAPH.META_CSV:
        pruning_cfg.CLUSTER_SUBGRAPH.pop("META_CSV")

    target_effective_pair_ratio = float(os.environ.get("CLUSTER_SUBGRAPH_TARGET_EFFECTIVE_RATIO", "0.1"))
    if not 0.0 < target_effective_pair_ratio <= 1.0:
        raise ValueError(
            "CLUSTER_SUBGRAPH_TARGET_EFFECTIVE_RATIO must be in (0, 1], "
            f"got {target_effective_pair_ratio}."
        )
    active_node_ratio = float(pruning_cfg.CLUSTER_SUBGRAPH.ACTIVE_NODE_RATIO)
    target_window_ratio = target_effective_pair_ratio / active_node_ratio
    if target_window_ratio > 1.0:
        raise ValueError(
            "The requested total effective ratio is infeasible for the active node ratio: "
            f"target_effective={target_effective_pair_ratio}, active_node_ratio={active_node_ratio}. "
            "Increase active node ratio or lower CLUSTER_SUBGRAPH_TARGET_EFFECTIVE_RATIO."
        )
    pruning_cfg.CLUSTER_SUBGRAPH.TARGET_EFFECTIVE_PAIR_RATIO = target_effective_pair_ratio
    pruning_cfg.CLUSTER_SUBGRAPH.TARGET_WINDOW_RATIO = target_window_ratio

    if not spatial_only:
        pruning_cfg.RETENTION_RATIO = target_window_ratio
        pruning_cfg.TARGET_RETENTION_MODE = True
        if (
            "CLUSTER_SUBGRAPH_FINAL_FULL_EPOCHS" not in os.environ
            and "CLUSTER_SUBGRAPH_FINAL_FULL_RATIO" not in os.environ
        ):
            pruning_cfg.FINAL_FULL_EPOCHS = 0
            pruning_cfg.FINAL_FULL_FORWARD_RATIO = 0.0

    cluster_tag = f"cluster_{cluster_type}_k{num_clusters}_a{num_active_clusters}"
    _append_ckpt_tag(cfg, cluster_tag)
    cfg.DESCRIPTION = (
        f"{cfg.DESCRIPTION} [cluster_subgraph:type={cluster_type}, "
        f"num_clusters={num_clusters}, active_clusters={num_active_clusters}, "
        f"active_node_ratio={pruning_cfg.CLUSTER_SUBGRAPH.ACTIVE_NODE_RATIO:.4f}, "
        f"target_window_ratio={target_window_ratio:.4f}, "
        f"target_effective_pair_ratio={target_effective_pair_ratio:.4f}, "
        f"spatial_only={spatial_only}, seed={seed}]"
    )
    return cfg
