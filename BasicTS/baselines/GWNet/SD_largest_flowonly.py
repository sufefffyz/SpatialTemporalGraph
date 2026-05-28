from baselines.AdaptiveGraph.largest_flowonly import apply_largest_flowonly_cfg

from .SD import CFG


CFG = apply_largest_flowonly_cfg(CFG, "gwnet")
