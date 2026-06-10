from baselines.Coreset.coreset_config import apply_coreset_cfg

from .SD_largest_aligned import CFG


CFG = apply_coreset_cfg(CFG, "gwnet_sd_largest_aligned")
