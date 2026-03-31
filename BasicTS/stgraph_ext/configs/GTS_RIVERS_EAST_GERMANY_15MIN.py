import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../.."))

from stgraph_ext.config_utils import build_gts_cfg


CFG = build_gts_cfg("RIVERS_EAST_GERMANY_15MIN", num_epochs=int(os.getenv("STGRAPH_NUM_EPOCHS", "100")))
