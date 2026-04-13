import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../.."))

from baselines.sd_phys_common import build_gwnet_cfg


CFG = build_gwnet_cfg("bidir")
