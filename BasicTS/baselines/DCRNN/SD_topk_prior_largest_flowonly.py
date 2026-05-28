from baselines.AdaptiveGraph.largest_flowonly import apply_largest_flowonly_cfg

from .SD_topk_prior import CFG


CFG = apply_largest_flowonly_cfg(CFG, "dcrnn")
