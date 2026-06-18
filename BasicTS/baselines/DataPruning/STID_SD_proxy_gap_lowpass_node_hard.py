from baselines.STID.SD import CFG
from baselines.DataPruning.dynamic_pruning_config import apply_dynamic_pruning_cfg

CFG = apply_dynamic_pruning_cfg(CFG, "stid_sd", "proxy_gap_lowpass_node_hard")
