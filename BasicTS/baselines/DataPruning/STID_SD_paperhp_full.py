from baselines.DataPruning.dynamic_pruning_config import apply_paper_hparam_full_cfg
from baselines.STID.SD import CFG


CFG = apply_paper_hparam_full_cfg(CFG, "stid_sd")
