# Proxy-Gap Pruning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `proxy_gap` dynamic sample pruning strategy that uses sample/window proxy-gap scores for selection while preserving the normal BasicTS forecasting objective.

**Architecture:** Extend the existing CD-006 dynamic pruning add-on rather than original BasicTS datasets/runners. `DynamicPruningBatchSelector` owns score memory, reference-loss memory, revisiting, active-set refresh, and final-full behavior. The dynamic pruning runner precomputes seasonal reference losses and updates current-loss EMA only for retained samples after normal forward loss computation.

**Tech Stack:** Python, PyTorch, NumPy, EasyTorch/BasicTS runner patterns, `unittest` tests under `BasicTS/tests`.

---

### Task 1: Add Proxy-Gap Selector Logic

**Files:**
- Modify: `BasicTS/basicts/runners/runner_zoo/dynamic_pruning_utils.py`
- Test: `BasicTS/tests/basicts_test/runner_test/test_dynamic_pruning.py`

- [ ] **Step 1: Write failing selector tests**

Add these imports near the top of `test_dynamic_pruning.py`:

```python
ProxyGapScoreMemory = MODULE.ProxyGapScoreMemory
```

Add these tests before `test_epsilon_greedy_builds_checkpoint_active_set_from_global_scores`:

```python
    def test_proxy_gap_score_is_current_loss_minus_reference_loss(self):
        memory = ProxyGapScoreMemory(score_momentum=0.0)

        memory.set_reference_losses(torch.tensor([10, 11]), torch.tensor([2.0, 5.0]))
        memory.update_current_losses(torch.tensor([10, 11]), torch.tensor([7.0, 8.0]))

        self.assertAlmostEqual(memory.proxy_gap(10), 5.0, places=6)
        self.assertAlmostEqual(memory.proxy_gap(11), 3.0, places=6)

    def test_proxy_gap_keeps_all_during_warmup_or_missing_scores(self):
        selector = DynamicPruningBatchSelector(
            strategy="proxy_gap",
            retention_ratio=0.5,
            seed=3,
            warmup_epochs=1,
            min_batch_size=1,
            revisit_probability=0.5,
        )
        batch_indices = torch.tensor([10, 11, 12, 13], dtype=torch.long)
        selector.set_dataset_indices(batch_indices)
        selector.set_num_epochs(10)

        warmup_selection = selector.select(batch_indices=batch_indices, epoch=1)
        self.assertTrue(torch.equal(warmup_selection.selected_positions, torch.arange(4)))
        self.assertTrue(torch.allclose(warmup_selection.inclusion_probabilities, torch.ones(4)))

        selector.begin_epoch(epoch=2)
        missing_selection = selector.select(batch_indices=batch_indices, epoch=2)
        self.assertTrue(torch.equal(missing_selection.selected_positions, torch.arange(4)))
        self.assertTrue(torch.allclose(missing_selection.inclusion_probabilities, torch.ones(4)))

    def test_proxy_gap_keeps_high_gap_and_revisits_low_gap_samples(self):
        selector = DynamicPruningBatchSelector(
            strategy="proxy_gap",
            retention_ratio=0.5,
            seed=3,
            warmup_epochs=0,
            min_batch_size=1,
            revisit_probability=0.5,
            pruning_period=1,
        )
        batch_indices = torch.tensor([10, 11, 12, 13], dtype=torch.long)
        selector.set_dataset_indices(batch_indices)
        selector.set_num_epochs(10)
        selector.set_reference_losses(batch_indices, torch.tensor([1.0, 1.0, 1.0, 1.0]))
        selector.update_scores(batch_indices, torch.tensor([10.0, 8.0, 4.0, 2.0]))
        selector.rng = FixedRng([0.25, 0.75])

        selector.begin_epoch(epoch=1)
        selection = selector.select(batch_indices=batch_indices, epoch=1)

        self.assertTrue(torch.equal(selection.selected_positions, torch.tensor([0, 1, 2])))
        self.assertTrue(torch.allclose(selection.inclusion_probabilities, torch.tensor([1.0, 1.0, 0.5])))

    def test_proxy_gap_final_full_epoch_disables_pruning(self):
        selector = DynamicPruningBatchSelector(
            strategy="proxy_gap",
            retention_ratio=0.5,
            seed=3,
            warmup_epochs=0,
            final_full_epochs=1,
            min_batch_size=1,
            revisit_probability=0.5,
        )
        batch_indices = torch.tensor([10, 11, 12, 13], dtype=torch.long)
        selector.set_dataset_indices(batch_indices)
        selector.set_num_epochs(10)
        selector.set_reference_losses(batch_indices, torch.ones(4))
        selector.update_scores(batch_indices, torch.tensor([10.0, 8.0, 4.0, 2.0]))

        selection = selector.select(batch_indices=batch_indices, epoch=10)

        self.assertTrue(torch.equal(selection.selected_positions, torch.arange(4)))
        self.assertTrue(torch.allclose(selection.inclusion_probabilities, torch.ones(4)))

    def test_proxy_gap_updates_scores_after_retained_train_batch(self):
        selector = DynamicPruningBatchSelector(
            strategy="proxy_gap",
            retention_ratio=0.5,
            seed=3,
        )

        self.assertTrue(selector.should_update_scores_after_train_batch())
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
cd /Users/richardo/Desktop/STproject/worktrees/cd-006-dynamic-pruning
python -m unittest BasicTS.tests.basicts_test.runner_test.test_dynamic_pruning.TestDynamicPruningUtils.test_proxy_gap_score_is_current_loss_minus_reference_loss
```

Expected: FAIL or ERROR because `ProxyGapScoreMemory` does not exist.

- [ ] **Step 3: Implement score memory and selector constructor fields**

In `dynamic_pruning_utils.py`, add this class above `DynamicPruningBatchSelector`:

```python
class ProxyGapScoreMemory:
    """Stores fixed reference losses and dynamic current-loss EMA for proxy-gap pruning."""

    def __init__(self, score_momentum: float = 0.0) -> None:
        self.score_momentum = float(score_momentum)
        self.current_loss_memory = {}
        self.reference_loss_memory = {}

    def set_reference_losses(self, batch_indices: torch.Tensor, reference_losses: torch.Tensor) -> None:
        batch_indices = batch_indices.detach().cpu().long().tolist()
        reference_losses = reference_losses.detach().cpu().float().tolist()
        for sample_index, loss in zip(batch_indices, reference_losses):
            self.reference_loss_memory[int(sample_index)] = float(loss)

    def update_current_losses(self, batch_indices: torch.Tensor, losses: torch.Tensor) -> None:
        batch_indices = batch_indices.detach().cpu().long().tolist()
        losses = losses.detach().cpu().float().tolist()
        for sample_index, loss in zip(batch_indices, losses):
            sample_index = int(sample_index)
            previous = self.current_loss_memory.get(sample_index)
            if previous is None:
                self.current_loss_memory[sample_index] = float(loss)
            else:
                self.current_loss_memory[sample_index] = (
                    self.score_momentum * float(previous) + (1.0 - self.score_momentum) * float(loss)
                )

    def has_score(self, sample_index: int) -> bool:
        sample_index = int(sample_index)
        return sample_index in self.current_loss_memory and sample_index in self.reference_loss_memory

    def proxy_gap(self, sample_index: int) -> float:
        sample_index = int(sample_index)
        return float(self.current_loss_memory[sample_index]) - float(self.reference_loss_memory[sample_index])
```

Extend `DynamicPruningBatchSelector.__init__` signature:

```python
        revisit_probability: float = 0.5,
```

Add validation after `epsilon` validation:

```python
        if not 0.0 <= revisit_probability <= 1.0:
            raise ValueError(f"revisit_probability must be in [0, 1], got {revisit_probability}.")
```

Add fields after `self.low_score_keep_probability`:

```python
        self.revisit_probability = float(revisit_probability)
```

Add after `self.score_memory = {}`:

```python
        self.proxy_gap_memory = ProxyGapScoreMemory(score_momentum=self.score_momentum)
```

- [ ] **Step 4: Implement proxy-gap selection methods**

Update `begin_epoch` supported strategies:

```python
        if self.strategy not in {"soft_random", "epsilon_greedy", "infobatch", "proxy_gap"}:
            return
```

Add after the infobatch branch:

```python
        if self.strategy == "proxy_gap":
            if self.active_indices is not None and not self._is_pruning_checkpoint(epoch):
                self.active_epoch = epoch
                return
            self._refresh_proxy_gap_active_set(epoch)
            return
```

Update `select`:

```python
        if self.strategy == "proxy_gap":
            return self._proxy_gap(batch_indices, epoch)
```

Update `update_scores`:

```python
        if self.strategy == "proxy_gap":
            self.proxy_gap_memory.update_current_losses(batch_indices, losses)
            return
```

Add public reference-loss setter:

```python
    def set_reference_losses(self, batch_indices: torch.Tensor, reference_losses: torch.Tensor) -> None:
        self.proxy_gap_memory.set_reference_losses(batch_indices, reference_losses)
```

Update `should_update_scores_after_train_batch`:

```python
        return self.strategy in {"infobatch", "proxy_gap"}
```

Add methods before `_refresh_soft_random_active_set`:

```python
    def _proxy_gap(self, batch_indices: torch.Tensor, epoch: int) -> PruningSelection:
        batch_size = int(batch_indices.shape[0])
        if self._is_full_data_epoch(epoch):
            return self._full_selection(batch_size)

        if self.active_indices is None:
            self._refresh_proxy_gap_active_set(epoch)
        if self.active_indices is None:
            return self._full_selection(batch_size)

        return self._selection_from_active_set(batch_indices)

    def _refresh_proxy_gap_active_set(self, epoch: int) -> None:
        if self.dataset_indices is None:
            raise ValueError("proxy_gap requires dataset indices before checkpoint selection.")

        sample_indices = [int(sample_index) for sample_index in self.dataset_indices.tolist()]
        if any(not self.proxy_gap_memory.has_score(sample_index) for sample_index in sample_indices):
            self.active_epoch = epoch
            self.active_indices = None
            self.active_inclusion_probabilities = {}
            return

        scored_items = [
            (sample_index, self.proxy_gap_memory.proxy_gap(sample_index))
            for sample_index in sample_indices
        ]
        scored_items.sort(key=lambda item: (-float(item[1]), int(item[0])))
        keep_count = self._target_keep_count(len(scored_items))
        if keep_count >= len(scored_items):
            self.active_epoch = epoch
            self.active_indices = None
            self.active_inclusion_probabilities = {}
            return

        retained = scored_items[:keep_count]
        low = scored_items[keep_count:]
        selected_probabilities = {int(sample_index): 1.0 for sample_index, _ in retained}
        if self.revisit_probability > 0.0 and low:
            keep_low = self.rng.random(len(low)) < self.revisit_probability
            for (sample_index, _), keep in zip(low, keep_low.tolist()):
                if keep:
                    selected_probabilities[int(sample_index)] = self.revisit_probability

        self.active_epoch = epoch
        self.active_indices = set(selected_probabilities.keys())
        self.active_inclusion_probabilities = selected_probabilities
```

- [ ] **Step 5: Run selector tests to verify GREEN**

Run:

```bash
cd /Users/richardo/Desktop/STproject/worktrees/cd-006-dynamic-pruning
python -m unittest BasicTS.tests.basicts_test.runner_test.test_dynamic_pruning
```

Expected: PASS. If local Python lacks dependencies, run the same command on the remote STGraph environment and capture the result.

- [ ] **Step 6: Commit selector logic**

```bash
git add BasicTS/basicts/runners/runner_zoo/dynamic_pruning_utils.py BasicTS/tests/basicts_test/runner_test/test_dynamic_pruning.py
git commit -m "feat: add proxy-gap pruning selector"
```

### Task 2: Add Seasonal Reference Precompute In Runner

**Files:**
- Modify: `BasicTS/basicts/runners/runner_zoo/dynamic_pruning_tsf_runner.py`
- Test: `BasicTS/tests/basicts_test/runner_test/test_proxy_gap_reference.py`

- [ ] **Step 1: Write failing seasonal reference tests**

Create `test_proxy_gap_reference.py`:

```python
import importlib.util
from pathlib import Path
import sys
import unittest

import torch

MODULE_PATH = Path(__file__).resolve().parents[3] / "basicts" / "runners" / "runner_zoo" / "dynamic_pruning_tsf_runner.py"
MODULE_SPEC = importlib.util.spec_from_file_location("dynamic_pruning_tsf_runner_under_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
sys.modules[MODULE_SPEC.name] = MODULE
MODULE_SPEC.loader.exec_module(MODULE)

build_seasonal_reference_prediction = MODULE.build_seasonal_reference_prediction


class TestProxyGapReference(unittest.TestCase):
    def test_seasonal_reference_uses_previous_period_target_when_available(self):
        targets_by_index = {
            4: torch.tensor([[[1.0]], [[2.0]]]),
            10: torch.tensor([[[9.0]], [[9.0]]]),
        }
        inputs = torch.tensor([[[[5.0]]], [[[6.0]]]])
        batch_indices = torch.tensor([10], dtype=torch.long)

        prediction = build_seasonal_reference_prediction(
            batch_indices=batch_indices,
            inputs=inputs,
            targets_by_index=targets_by_index,
            proxy_period=6,
            target_feature_count=1,
        )

        self.assertTrue(torch.equal(prediction, targets_by_index[4].unsqueeze(0)))

    def test_seasonal_reference_falls_back_to_persistence_for_early_indices(self):
        targets_by_index = {}
        inputs = torch.tensor([[[[5.0]], [[6.0]]]])
        batch_indices = torch.tensor([2], dtype=torch.long)

        prediction = build_seasonal_reference_prediction(
            batch_indices=batch_indices,
            inputs=inputs,
            targets_by_index=targets_by_index,
            proxy_period=6,
            target_feature_count=1,
        )

        expected = torch.tensor([[[[6.0]], [[6.0]]]])
        self.assertTrue(torch.equal(prediction, expected))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
cd /Users/richardo/Desktop/STproject/worktrees/cd-006-dynamic-pruning
python -m unittest BasicTS.tests.basicts_test.runner_test.test_proxy_gap_reference
```

Expected: FAIL or ERROR because `build_seasonal_reference_prediction` is not defined.

- [ ] **Step 3: Implement pure seasonal helper**

In `dynamic_pruning_tsf_runner.py`, add above `_DynamicPruningRunnerMixin`:

```python
def build_seasonal_reference_prediction(
    batch_indices: torch.Tensor,
    inputs: torch.Tensor,
    targets_by_index: Dict[int, torch.Tensor],
    proxy_period: int,
    target_feature_count: int,
) -> torch.Tensor:
    """Build seasonal proxy predictions with persistence fallback."""

    proxy_period = int(proxy_period)
    predictions = []
    for row, sample_index in enumerate(batch_indices.detach().cpu().long().tolist()):
        previous_index = int(sample_index) - proxy_period
        previous_target = targets_by_index.get(previous_index)
        if previous_target is not None:
            predictions.append(previous_target.to(inputs.device, dtype=inputs.dtype))
            continue

        last_observed = inputs[row, -1:, :, :target_feature_count]
        output_len = next(iter(targets_by_index.values())).shape[0] if targets_by_index else inputs.shape[1]
        predictions.append(last_observed.repeat(output_len, 1, 1))

    return torch.stack(predictions, dim=0)
```

- [ ] **Step 4: Wire reference precompute into runner**

In `_DynamicPruningRunnerMixin.__init__`, add fields after `self.dynamic_pruning_pending_backward_samples = 0`:

```python
        self.dynamic_pruning_reference_type = str(pruning_cfg.get("REFERENCE_TYPE", "seasonal")).lower()
        self.dynamic_pruning_proxy_period = int(pruning_cfg.get("PROXY_PERIOD", 96))
        self.dynamic_pruning_reference_ready = False
```

In `init_training`, after `self._init_forward_budget()` add:

```python
            self._init_proxy_gap_reference_losses()
```

Add this method before `_score_full_training_dataset_for_pruning`:

```python
    @torch.no_grad()
    def _init_proxy_gap_reference_losses(self) -> None:
        if self.dynamic_pruning_selector.strategy != "proxy_gap":
            return
        if self.dynamic_pruning_reference_type != "seasonal":
            raise ValueError(
                "proxy_gap currently supports REFERENCE_TYPE='seasonal' only; "
                f"got {self.dynamic_pruning_reference_type!r}."
            )

        targets_by_index = {}
        for data in self.dynamic_pruning_full_train_data_loader:
            batch_indices = self._to_long_tensor(data["index"])
            target = data["target"].detach().cpu()
            for row, sample_index in enumerate(batch_indices.tolist()):
                targets_by_index[int(sample_index)] = target[row]

        reference_count = 0
        target_feature_count = len(self.cfg["MODEL"].get("TARGET_FEATURES", [0]))
        for data in self.dynamic_pruning_full_train_data_loader:
            batch_indices = self._to_long_tensor(data["index"])
            inputs = data["inputs"]
            target = data["target"]
            prediction = build_seasonal_reference_prediction(
                batch_indices=batch_indices,
                inputs=inputs,
                targets_by_index=targets_by_index,
                proxy_period=self.dynamic_pruning_proxy_period,
                target_feature_count=target_feature_count,
            )
            reference_loss = masked_mae_per_sample(prediction.to(target.device), target, null_val=self.null_val)
            self.dynamic_pruning_selector.set_reference_losses(batch_indices, reference_loss)
            reference_count += int(batch_indices.shape[0])

        self.dynamic_pruning_reference_ready = True
        self.logger.info(
            "Proxy-gap seasonal reference initialized for %d samples with period=%d.",
            reference_count,
            self.dynamic_pruning_proxy_period,
        )
```

- [ ] **Step 5: Run reference tests**

Run:

```bash
cd /Users/richardo/Desktop/STproject/worktrees/cd-006-dynamic-pruning
python -m unittest BasicTS.tests.basicts_test.runner_test.test_proxy_gap_reference
```

Expected: PASS.

- [ ] **Step 6: Commit runner reference precompute**

```bash
git add BasicTS/basicts/runners/runner_zoo/dynamic_pruning_tsf_runner.py BasicTS/tests/basicts_test/runner_test/test_proxy_gap_reference.py
git commit -m "feat: add seasonal proxy-gap reference losses"
```

### Task 3: Add Config And Opt-In Files

**Files:**
- Modify: `BasicTS/baselines/DataPruning/dynamic_pruning_config.py`
- Create: `BasicTS/baselines/DataPruning/STID_SD_proxy_gap.py`
- Create: `BasicTS/baselines/DataPruning/STID_PEMS08_proxy_gap.py`
- Test: `BasicTS/tests/basicts_test/runner_test/test_dynamic_pruning_config.py`

- [ ] **Step 1: Write failing config test**

Add to `test_dynamic_pruning_config.py`:

```python
    def test_proxy_gap_defaults_use_epoch_refresh_and_infobatch_like_revisit(self):
        module = load_config_module()
        env = {
            "DYNAMIC_PRUNING_ALIGN_PAPER_HPARAMS": "0",
            "DYNAMIC_PRUNING_RATIO": "0.1",
        }

        with mock.patch.dict(os.environ, env, clear=True):
            cfg = module.apply_dynamic_pruning_cfg(build_cfg(), "stid_sd", "proxy_gap")

        self.assertEqual(cfg.TRAIN.DYNAMIC_PRUNING.STRATEGY, "proxy_gap")
        self.assertEqual(cfg.TRAIN.DYNAMIC_PRUNING.PRUNING_PERIOD, 1)
        self.assertAlmostEqual(cfg.TRAIN.DYNAMIC_PRUNING.REVISIT_PROBABILITY, 0.5)
        self.assertEqual(cfg.TRAIN.DYNAMIC_PRUNING.REFERENCE_TYPE, "seasonal")
        self.assertEqual(cfg.TRAIN.DYNAMIC_PRUNING.PROXY_PERIOD, 96)
        self.assertFalse(cfg.TRAIN.DYNAMIC_PRUNING.RESCALE)
```

- [ ] **Step 2: Run config test to verify RED**

Run:

```bash
cd /Users/richardo/Desktop/STproject/worktrees/cd-006-dynamic-pruning
python -m unittest BasicTS.tests.basicts_test.runner_test.test_dynamic_pruning_config.TestDynamicPruningConfig.test_proxy_gap_defaults_use_epoch_refresh_and_infobatch_like_revisit
```

Expected: FAIL because `REVISIT_PROBABILITY`, `REFERENCE_TYPE`, or `PROXY_PERIOD` are missing/default wrong.

- [ ] **Step 3: Implement config defaults**

In `dynamic_pruning_config.py`, update strategy normalization:

```python
    if strategy in {"rho_prune", "rho", "proxygap"}:
        strategy = "proxy_gap"
```

Change pruning-period default:

```python
    pruning_period_default = "1" if strategy == "proxy_gap" else "10"
    pruning_period = int(os.environ.get("DYNAMIC_PRUNING_PRUNING_PERIOD", pruning_period_default))
```

Change score/revisit/reference defaults after score-alpha setup:

```python
    revisit_probability_default = "0.5" if strategy == "proxy_gap" else "0.0"
    revisit_probability = float(os.environ.get("DYNAMIC_PRUNING_REVISIT_PROBABILITY", revisit_probability_default))
    reference_type = os.environ.get("DYNAMIC_PRUNING_REFERENCE_TYPE", "seasonal").strip().lower()
    proxy_period_default = "96" if "sd" in backbone_tag.lower() else "288"
    proxy_period = int(os.environ.get("DYNAMIC_PRUNING_PROXY_PERIOD", proxy_period_default))
```

Change rescale default helper:

```python
    return "1" if strategy == "infobatch" else "0"
```

Add fields to `cfg.TRAIN.DYNAMIC_PRUNING`:

```python
            "REVISIT_PROBABILITY": revisit_probability,
            "REFERENCE_TYPE": reference_type,
            "PROXY_PERIOD": proxy_period,
```

Add a proxy-gap checkpoint tag branch before epsilon-greedy:

```python
    elif strategy == "proxy_gap":
        _append_ckpt_tag(
            cfg,
            f"dynprune_{backbone_tag}_{strategy}_r{_ratio_tag(ratio)}_rev{_ratio_tag(revisit_probability)}{hparam_tag}_s{seed}",
        )
        cfg.DESCRIPTION = (
            f"{cfg.DESCRIPTION} [dynamic_pruning:{strategy}, ratio={ratio}, "
            f"revisit_probability={revisit_probability}, reference_type={reference_type}, "
            f"proxy_period={proxy_period}, pruning_period={pruning_period}, rescale={rescale}, "
            f"disable_early_stopping={disable_early_stopping}, "
            f"align_paper_hparams={align_paper_hparams}, seed={seed}]"
        )
```

- [ ] **Step 4: Create opt-in STID configs**

Create `STID_SD_proxy_gap.py`:

```python
from baselines.DataPruning.dynamic_pruning_config import apply_dynamic_pruning_cfg
from baselines.STID.SD import CFG


CFG = apply_dynamic_pruning_cfg(CFG, "stid_sd", "proxy_gap")
```

Create `STID_PEMS08_proxy_gap.py`:

```python
from baselines.DataPruning.dynamic_pruning_config import apply_dynamic_pruning_cfg
from baselines.STID.PEMS08 import CFG


CFG = apply_dynamic_pruning_cfg(CFG, "stid_pems08", "proxy_gap")
```

- [ ] **Step 5: Run config tests**

Run:

```bash
cd /Users/richardo/Desktop/STproject/worktrees/cd-006-dynamic-pruning
python -m unittest BasicTS.tests.basicts_test.runner_test.test_dynamic_pruning_config
```

Expected: PASS.

- [ ] **Step 6: Commit config integration**

```bash
git add BasicTS/baselines/DataPruning/dynamic_pruning_config.py BasicTS/baselines/DataPruning/STID_SD_proxy_gap.py BasicTS/baselines/DataPruning/STID_PEMS08_proxy_gap.py BasicTS/tests/basicts_test/runner_test/test_dynamic_pruning_config.py
git commit -m "feat: add proxy-gap pruning configs"
```

### Task 4: Verification And No-Launch Smoke

**Files:**
- No production edits unless failures require fixes.

- [ ] **Step 1: Run focused local tests**

Run:

```bash
cd /Users/richardo/Desktop/STproject/worktrees/cd-006-dynamic-pruning
python -m unittest BasicTS.tests.basicts_test.runner_test.test_dynamic_pruning BasicTS.tests.basicts_test.runner_test.test_proxy_gap_reference BasicTS.tests.basicts_test.runner_test.test_dynamic_pruning_config
```

Expected: PASS. If local Python lacks project dependencies, run the same focused tests in the remote `STGraph` environment after syncing only the touched files.

- [ ] **Step 2: Run config import smoke without starting training**

Run locally or remotely:

```bash
cd /home/yuzhang_fei/code/SpatialTemporalGraph/BasicTS
DYNAMIC_PRUNING_ALIGN_PAPER_HPARAMS=0 DYNAMIC_PRUNING_RATIO=0.1 python -c "from baselines.DataPruning.STID_SD_proxy_gap import CFG; print(CFG.TRAIN.DYNAMIC_PRUNING)"
```

Expected: prints `STRATEGY: proxy_gap`, `PRUNING_PERIOD: 1`, `REVISIT_PROBABILITY: 0.5`, `REFERENCE_TYPE: seasonal`, and no training starts.

- [ ] **Step 3: Confirm no GPU jobs were launched**

Run:

```bash
ssh -p 5102 yuzhang_fei@183.174.228.180 "screen -ls && nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader"
```

Expected: no new CD-006 proxy-gap screen exists. Existing unrelated jobs, if any, should be reported but not modified.

- [ ] **Step 4: Final status**

Report:

- Files changed.
- Tests run and pass/fail output.
- Confirmation that no proxy-gap GPU experiment was launched.
- DLinear status: interface reserved, automatic DLinear reference training not implemented.

