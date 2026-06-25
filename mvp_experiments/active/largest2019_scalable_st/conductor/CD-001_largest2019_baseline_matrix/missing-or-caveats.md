# CD-001 Missing Items And Caveats

Generated: 2026-06-24 12:10 CST

## Missing Cell

| Cell | Current Evidence | Exact Blocker |
|---|---|---|
| MAGE GLA 2019 | Active screen `472029.cd001_mage_gla_full_g1_fix1` on `183.174.228.180`; PID `472035`; latest completed Epoch 61 at `2026-06-24 12:03:28 CST`; log path `/home/yuzhang_fei/code/SpatialTemporalGraph/references/mage_official/experiments/mage/gla/cd001_mage_gla_full_g1_fix1.outer.log`. | Final test metrics are not available because the run is still training. There is no `Average Test` row in the log yet. |

## Caveats

- Metric conventions are mixed: STID uses BasicTS `overall` JSON metrics, while PatchSTG/BiST/MAGE use each framework's average test logging.
- Peak GPU memory is only explicitly available for STID profiler runs. PatchSTG, BiST, and completed MAGE logs do not record peak memory. MAGE GLA has only a live current-memory observation, not a peak.
- MAGE official code required a minimal evaluation-stage compatibility patch in `references/mage_official/src/engines/MAGE_enine.py` before validation/test metrics could run. The patch does not change model architecture or hyperparameters, but it should be labeled in any reproduction note.
- PatchSTG used CD-001 run configs (`CD001_*`) to point at prepared data/device settings. The metric logs should be cited with those config names, not as unmodified default config filenames.
- No dynamic-threshold work was read, revived, or included in the baseline matrix.

## Proposed Global Decision

This is a proposal only, not binding: the master session should decide whether CD-001 may close with MAGE GLA recorded as a live-running blocker, or whether it should wait for the MAGE GLA final `Average Test` row before merging the baseline matrix.
