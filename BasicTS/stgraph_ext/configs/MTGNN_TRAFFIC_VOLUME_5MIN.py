import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../.."))

from stgraph_ext.config_utils import build_mtgnn_cfg


CFG = build_mtgnn_cfg("TRAFFIC_VOLUME_5MIN", num_epochs=int(os.getenv("STGRAPH_NUM_EPOCHS", "100")))
