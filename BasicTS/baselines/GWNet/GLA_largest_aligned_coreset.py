from baselines.Coreset.coreset_config import apply_coreset_cfg

from .GLA_largest_aligned import CFG


CFG = apply_coreset_cfg(CFG, "gwnet_gla_largest_aligned")
