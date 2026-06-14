# Proxy-Gap Dynamic Sample Pruning Design

## Scope

Add an explicit opt-in dynamic pruning strategy named `proxy_gap` for BasicTS forecasting experiments. The strategy adapts the Selective Learning proxy-gap score to sample/window-level data pruning, while keeping the original forecasting objective unchanged.

This design does not implement selective loss masking, target-position masking, new backbone losses, or backbone hyperparameter changes. Selected samples are trained with the normal BasicTS forecasting loss path.

## Motivation

Selective Learning scores examples by comparing the current model residual to a simpler proxy residual. For CD-006, the same scoring idea will be used only to decide which training windows to forward. A high positive gap means the current model still struggles relative to the proxy, so the sample is prioritized. A low gap means the sample is a pruning candidate, but it must remain revisit-able so scores do not become stale.

## Strategy

For each train sample/window index `i`:

```text
proxy_gap_i = current_loss_ema_i - reference_loss_i
```

`current_loss_ema_i` is the EMA of per-sample forecasting loss observed when sample `i` is actually forwarded. `reference_loss_i` is a fixed per-sample proxy/reference forecasting loss precomputed before pruning decisions.

Selection uses global scores at pruning checkpoints:

- During warmup, use full data.
- If a sample is missing `current_loss_ema_i` or `reference_loss_i`, keep it.
- Otherwise keep the highest `RETENTION_RATIO` fraction by proxy gap.
- For low-score samples outside the retained set, sample them with `REVISIT_PROBABILITY` so their current loss can be refreshed later.
- In final epochs, disable pruning and return to full data using the existing `FINAL_FULL_EPOCHS` / `FINAL_FULL_RATIO` path.

The default loss rescaling for `proxy_gap` is off. This keeps the first candidate focused on selection behavior rather than mixing in InfoBatch-style gradient rescaling.

## Reference Losses

The first implemented reference is `seasonal`.

For a train sample starting at dataset index `i`:

- If `i - PROXY_PERIOD` is valid, use the target window from the previous same-time period as the proxy prediction.
- If it is not valid, fall back to persistence by repeating the last observed target feature from the input window across the forecast horizon.
- Compute reference loss with the same masked MAE per-sample reduction used by the dynamic pruning runner.

Default period is configurable through `DYNAMIC_PRUNING_PROXY_PERIOD`. Config wrappers may set dataset-specific defaults such as 96 for SD and 288 for PEMS-style 5-minute datasets.

## DLinear Reference

The `dlinear` reference is not part of the first implementation unless a clean checkpoint/precompute path already exists. The config should reserve the interface:

- `REFERENCE_TYPE=dlinear`
- optional reference cache or precompute path

If requested without a usable cache/checkpoint path, the code should fail clearly or document the blocker instead of silently training a proxy model or changing the experiment protocol.

## Integration Points

Changes stay inside the CD-006 dynamic pruning add-on:

- Extend `DynamicPruningBatchSelector` with `proxy_gap`.
- Extend the dynamic pruning runner to precompute seasonal reference losses after the indexed train loader is built.
- Update retained-sample current loss EMA after normal forward loss is computed.
- Extend `dynamic_pruning_config.py` to expose `proxy_gap` pruning controls.
- Add explicit opt-in configs such as `STID_SD_proxy_gap.py` and `STID_PEMS08_proxy_gap.py` only if low risk.

No existing Soft Random, InfoBatch-lite, or epsilon-greedy behavior should change.

## Config Defaults

Initial defaults:

- `STRATEGY=proxy_gap`
- `RETENTION_RATIO=0.1` via existing ratio env
- `WARMUP_EPOCHS=1`
- `PRUNING_PERIOD=1`
- `SCORE_MOMENTUM=0.0` unless overridden
- `REVISIT_PROBABILITY=0.5`
- `REFERENCE_TYPE=seasonal`
- `FINAL_FULL_RATIO` follows the existing non-Soft-Random default derived from `DELTA=0.875`
- `RESCALE=False`

Only pruning-control hyperparameters are introduced. Optimizer, scheduler, batch size, gradient clipping, dataset split, and model architecture remain whatever the selected base config already uses unless an existing explicit experiment wrapper changes them.

## Tests

Add focused unit tests before implementation:

- Proxy-gap score equals `current_loss_ema - reference_loss`.
- Warmup and missing scores keep all samples.
- High proxy-gap samples are retained, low proxy-gap samples are pruned unless revisited.
- Revisit probability can select low-score samples and assigns the correct inclusion probability.
- Final full-data epoch disables pruning.
- `proxy_gap` updates score memory after retained train batches, without changing normal loss computation.

## Non-Goals

- No selective target timestep mask.
- No replacement objective.
- No automatic DLinear training.
- No GPU experiment launch as part of implementation.
- No global changes to original BasicTS datasets or runners.
