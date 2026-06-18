from baselines.DataPruning.dynamic_pruning_config import apply_cluster_subgraph_cfg
from baselines.STID.SD import CFG

CFG = apply_cluster_subgraph_cfg(CFG, "stid_sd", "none", cluster_type="spatial_kdtree")
