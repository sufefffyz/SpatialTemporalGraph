from baselines.AdaptiveGraph.largest_flowonly import apply_largest_flowonly_cfg

from .sd_common import build_sd_cfg


CFG = apply_largest_flowonly_cfg(build_sd_cfg("distthre"), "stgcn")
