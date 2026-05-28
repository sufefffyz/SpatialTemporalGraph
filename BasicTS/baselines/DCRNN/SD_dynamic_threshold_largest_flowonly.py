from baselines.AdaptiveGraph.largest_flowonly import apply_largest_flowonly_cfg
from baselines.AdaptiveGraph.sd_dynamic_common import build_dynamic_sd_cfg


CFG = apply_largest_flowonly_cfg(build_dynamic_sd_cfg("dcrnn"), "dcrnn")
