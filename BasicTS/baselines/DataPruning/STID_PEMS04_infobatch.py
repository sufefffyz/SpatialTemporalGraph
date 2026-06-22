from baselines.DataPruning.dynamic_pruning_config import apply_dynamic_pruning_cfg
from baselines.STID.PEMS04 import CFG

CFG = apply_dynamic_pruning_cfg(CFG, "stid_pems04", "infobatch")
