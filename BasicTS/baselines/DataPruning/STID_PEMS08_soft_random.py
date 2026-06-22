from baselines.DataPruning.dynamic_pruning_config import apply_dynamic_pruning_cfg
from baselines.STID.PEMS08 import CFG

CFG = apply_dynamic_pruning_cfg(CFG, "stid_pems08", "soft_random")
