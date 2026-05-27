from baselines.AdaptiveGraph.largest_aligned import apply_largest_aligned_cfg
from baselines.AdaptiveGraph.sd_dynamic_common import build_dynamic_sd_cfg


CFG = apply_largest_aligned_cfg(build_dynamic_sd_cfg("stgcn"), "stgcn")
