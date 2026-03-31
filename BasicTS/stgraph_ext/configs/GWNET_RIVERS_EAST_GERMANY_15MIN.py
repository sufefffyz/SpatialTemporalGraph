import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../.."))

from stgraph_ext.config_utils import build_gwnet_cfg


CFG = build_gwnet_cfg(
    os.getenv("STGRAPH_DATASET_NAME", "RIVERS_EAST_GERMANY_15MIN"),
    num_epochs=int(os.getenv("STGRAPH_NUM_EPOCHS", "100")),
)
