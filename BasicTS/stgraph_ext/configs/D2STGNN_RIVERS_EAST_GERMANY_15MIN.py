import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../.."))

from stgraph_ext.config_utils import build_d2stgnn_cfg


CFG = build_d2stgnn_cfg(
    os.getenv("STGRAPH_DATASET_NAME", "RIVERS_EAST_GERMANY_15MIN"),
    num_epochs=int(os.getenv("STGRAPH_NUM_EPOCHS", "100")),
)
