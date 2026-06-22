from baselines.DataPruning.dynamic_pruning_config import apply_dynamic_pruning_cfg
from baselines.STID.SD import CFG

CFG = apply_dynamic_pruning_cfg(CFG, "stid_sd", "epsilon_greedy")
