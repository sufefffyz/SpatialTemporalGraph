import os

from basicts.runners import SimpleTimeSeriesForecastingRunner, WandBTimeSeriesForecastingRunner

from .SD import CFG, DATA_NAME, INPUT_LEN, MODEL_PARAM, OUTPUT_LEN


VARIANT = os.environ.get("GWNET_TIMING_VARIANT", "baseline").strip().lower()
NUM_EPOCHS = int(os.environ.get("BASICTS_NUM_EPOCHS", "2"))
RUN_TAG = os.environ.get("BASICTS_RUN_TAG", VARIANT).strip()

CFG.DESCRIPTION = f"GraphWaveNet SD timing ablation ({VARIANT})"
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TEST.INTERVAL = NUM_EPOCHS

if VARIANT in {"no_wandb", "fast_env", "buffered_support"}:
    CFG.RUNNER = SimpleTimeSeriesForecastingRunner
elif VARIANT == "baseline":
    CFG.RUNNER = WandBTimeSeriesForecastingRunner
else:
    raise ValueError(f"Unknown GWNET_TIMING_VARIANT={VARIANT}")

if VARIANT in {"fast_env", "buffered_support"}:
    CFG.ENV.TF32 = True
    CFG.ENV.DETERMINISTIC = False
    CFG.ENV.CUDNN.DETERMINISTIC = False
    CFG.ENV.CUDNN.BENCHMARK = True

if VARIANT == "buffered_support":
    from .arch.gwnet_buffered_arch import GraphWaveNetBuffered

    CFG.MODEL.NAME = GraphWaveNetBuffered.__name__
    CFG.MODEL.ARCH = GraphWaveNetBuffered
    CFG.MODEL.PARAM = dict(MODEL_PARAM)

ckpt_parts = [DATA_NAME, "timing_ablation", VARIANT, str(NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)]
if RUN_TAG:
    ckpt_parts.append(RUN_TAG)
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints", CFG.MODEL.NAME, "_".join(ckpt_parts))
