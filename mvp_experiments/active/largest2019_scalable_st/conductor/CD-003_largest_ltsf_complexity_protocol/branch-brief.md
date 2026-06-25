# Branch Brief: CD-003 LargeST-LTSF Complexity And Protocol

Stable title: `[CD-003][W1][research] LargeST-LTSF complexity and protocol`

Purpose: define the LTSF input-window experiment protocol and complexity triage for LargeST 2019.

Not for: dynamic-threshold experiments, new model design, or full leaderboard claims.

Input: master decision to first fix `H=12` and test `L in {96,192,336,672}`; local STID/BiST/BasicTS LTSF code; FaST paper setting as a later long-horizon reference.

Expected output:

- Resource estimates for SD/GBA/GLA.
- Model scalability classification.
- A concrete smoke/pilot matrix usable by CD-004.

Completion criteria:

- `estimate_ltsf_resources.py` writes `summary.csv` and `scalability_table.md`.
- The protocol distinguishes the first `H=12` long-input probe from later standard LTSF long-input/long-output settings.
- The result is usable without modifying baseline main configs.

Return condition: master can decide which models and datasets enter smoke/pilot.
