import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../.."))

from baselines.sd_phys_common import build_dcrnn_cfg


CFG = build_dcrnn_cfg("directed")
