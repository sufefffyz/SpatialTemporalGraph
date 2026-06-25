# Branch Brief: CD-004 LargeST-LTSF Smoke And Pilot

Stable title: `[CD-004][W1][experiment] LargeST-LTSF smoke and pilot`

Purpose: run the LargeST-LTSF input-window smoke and pilot experiments using isolated configs and logs.

Not for: full paper claims, hyperparameter tuning, dynamic-threshold modules, or modifying original baseline configs.

Input: CD-003 protocol and scripts under `ltsf_input_window`.

Expected output:

- Smoke logs and metrics for SD.
- Pilot curves for SD and selected GBA models if smoke passes.
- Summary table with MAE/RMSE/MAPE, wall seconds, and sampled peak GPU memory.

Completion criteria:

- BasicTS configs are generated under `ltsf_input_window/outputs/generated_configs`.
- BiST uses explicit `2019_L{L}_H12` caches in the first probe and never reads the default 12-to-12 cache by accident.
- `collect_ltsf_results.py` writes a phase summary CSV.
- `plot_input_window_curve.py` generates MAE/RMSE curves from the summary.

Return condition: master can decide whether longer input windows are worth a full 50-epoch run.
