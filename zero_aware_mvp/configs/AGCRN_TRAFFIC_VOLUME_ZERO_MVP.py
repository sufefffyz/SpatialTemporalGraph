from __future__ import annotations

import os
import sys
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent
if str(CONFIG_DIR) not in sys.path:
    sys.path.append(str(CONFIG_DIR))

from zero_aware_basicts_common import build_zero_aware_agcrn_cfg


CFG = build_zero_aware_agcrn_cfg()

