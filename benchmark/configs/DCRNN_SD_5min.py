import os
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from sd_5min_common import build_dcrnn_cfg

CFG = build_dcrnn_cfg()
