from baselines.Coreset.coreset_config import apply_coreset_cfg

from .GBA import CFG


CFG = apply_coreset_cfg(CFG, "stid_gba")
