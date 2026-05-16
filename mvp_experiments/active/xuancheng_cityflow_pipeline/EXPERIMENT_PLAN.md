# Xuancheng CityFlow Road-Aggregation MVP

One-sentence thesis:

> Official Xuancheng CityFlow flows can produce a road-level simulated traffic tensor large enough for a first benchmark, but the released temporal coverage caps the first phase at one month.

## Scope

- Primary datasets: Xuancheng CityFlow `2023-04-01` to `2023-04-30`.
- Backbones: preprocessing first; downstream BasicTS baselines only after tensor sanity checks.
- Must not claim: PeMS/LargeST temporal duration, sensor-native observations, or exhaustive all-vehicle regional coverage.

## First Run Order

1. Diagnostics: inspect road count, lane count, daily flow sizes, time coverage.
2. Cheapest baseline: one-day 1,800-second CityFlow smoke aggregation.
3. Core MVP variant: one-day full 86,400-second road aggregation to 1-minute `csv.gz` and `.npz`.
4. Decisive ablation: fixed-time signals versus official RL-light setting with no agent, checking congestion artifacts.
5. Scale or robustness check: seven-day then full-month tensor generation; only then build BasicTS dataset wrappers.

## Success Gate

Proceed if:

- R001 finishes without CityFlow import/runtime errors on the server.
- Aggregated CSV has 1,744 roads per full bucket in dense mode.
- NPZ shape matches `[T, 1744, F]` with `T=1440` for one full day at 1-min resolution.
- Speed and active-vehicle distributions are not degenerate for the selected day.

Stop or pivot if:

- The server lacks both `cityflow` and Docker access.
- Fixed-time replay gridlocks immediately, and official agent-controlled replay is required before meaningful aggregation.
- One-day runtime is too slow to extrapolate to the released month.

## Folder Layout

```text
configs/      example configs and local defaults
scripts/      download, config, simulation aggregation, server launch scripts
src/          reserved for reusable package code
results/      compact metrics, summaries, figures
outputs/      raw logs and bulky generated artifacts
notes/        related work, decisions, scratch notes
```

## Tracker

| Run ID | Purpose | Dataset | Metrics | Status | Notes |
|---|---|---|---|---|---|
| R001 | Server smoke aggregation | Xuancheng 2023-04-03 first 1,800 s | runtime, rows, NPZ shape | TODO | Confirms CityFlow/Docker path. |
| R002 | Full-day aggregation | Xuancheng 2023-04-03 full day | runtime, file size, distributions | TODO | First PeMS-style daily tensor. |
| R003 | One-week pilot | Xuancheng 2023-04-01..2023-04-07 | per-day volume/speed stats | TODO | Checks temporal variability. |
| R004 | Full released month | Xuancheng 2023-04-01..2023-04-30 | monthly tensor shape, disk use | TODO | Maximum official temporal coverage. |
| R005 | Downstream BasicTS staging | R004 tensor | train/val/test split feasibility | TODO | Only after tensor sanity checks. |
