from baselines.Coreset.coreset_config import apply_coreset_cfg

from .GLA import CFG


CFG = apply_coreset_cfg(CFG, "stid_gla")
