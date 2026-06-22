from baselines.Coreset.coreset_config import apply_coreset_cfg

from .GBA_largest_aligned import CFG


CFG = apply_coreset_cfg(CFG, "gwnet_gba_largest_aligned")
