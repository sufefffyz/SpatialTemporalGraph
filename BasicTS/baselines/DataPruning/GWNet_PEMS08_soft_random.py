from baselines.DataPruning.dynamic_pruning_config import apply_dynamic_pruning_cfg
from baselines.GWNet.PEMS08 import CFG

CFG = apply_dynamic_pruning_cfg(CFG, "gwnet_pems08", "soft_random")
