from baselines.DataPruning.dynamic_pruning_config import apply_dynamic_pruning_cfg
from baselines.GWNet.PEMS04 import CFG

CFG = apply_dynamic_pruning_cfg(CFG, "gwnet_pems04", "soft_random")
