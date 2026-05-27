from baselines.AdaptiveGraph.largest_aligned import apply_largest_aligned_cfg

from .SD_osrm_gaussian_global import CFG


CFG = apply_largest_aligned_cfg(CFG, "dcrnn")
