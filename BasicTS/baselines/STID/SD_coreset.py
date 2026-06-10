from baselines.Coreset.coreset_config import apply_coreset_cfg

from .SD import CFG


CFG = apply_coreset_cfg(CFG, "stid_sd")
