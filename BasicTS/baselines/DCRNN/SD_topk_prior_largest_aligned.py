from baselines.AdaptiveGraph.largest_aligned import apply_largest_aligned_cfg

from .SD_topk_prior import CFG


CFG = apply_largest_aligned_cfg(CFG, "dcrnn")
