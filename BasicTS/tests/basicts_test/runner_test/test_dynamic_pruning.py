import importlib.util
from pathlib import Path
import sys
import unittest

import torch

MODULE_PATH = Path(__file__).resolve().parents[3] / "basicts" / "runners" / "runner_zoo" / "dynamic_pruning_utils.py"
MODULE_SPEC = importlib.util.spec_from_file_location("dynamic_pruning_utils", MODULE_PATH)
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
sys.modules[MODULE_SPEC.name] = MODULE
MODULE_SPEC.loader.exec_module(MODULE)

DynamicPruningBatchSelector = MODULE.DynamicPruningBatchSelector
DynamicPruningBudgetCounter = MODULE.DynamicPruningBudgetCounter
inverse_probability_weighted_loss = MODULE.inverse_probability_weighted_loss
limit_selection_to_forward_budget = MODULE.limit_selection_to_forward_budget
masked_mae_per_sample = MODULE.masked_mae_per_sample
normalized_masked_mae_per_sample = MODULE.normalized_masked_mae_per_sample
lowpass_normalized_masked_mae_per_sample = MODULE.lowpass_normalized_masked_mae_per_sample
masked_mae_per_node = MODULE.masked_mae_per_node
normalized_masked_mae_per_node = MODULE.normalized_masked_mae_per_node
node_hard_revisit_rescaled_loss = MODULE.node_hard_revisit_rescaled_loss
PruningSelection = MODULE.PruningSelection


class FixedRng:
    def __init__(self, values):
        self.values = torch.tensor(values)

    def random(self, size):
        return self.values[:size]


class FixedChoiceRng:
    def choice(self, candidates, size, replace=False):
        del replace
        return torch.as_tensor(candidates)[-size:].numpy()


class FirstChoiceRng:
    def choice(self, candidates, size, replace=False):
        del replace
        return torch.as_tensor(candidates)[:size].numpy()


class TestDynamicPruningUtils(unittest.TestCase):
    def test_budget_counter_tracks_epoch_and_total_forward_backward_steps(self):
        counter = DynamicPruningBudgetCounter()

        counter.begin_epoch()
        counter.record_forward(5)
        counter.record_backward(5)
        counter.record_forward(3)

        self.assertEqual(counter.epoch_forward_samples, 8)
        self.assertEqual(counter.epoch_backward_samples, 5)
        self.assertEqual(counter.epoch_optimizer_steps, 1)
        self.assertEqual(counter.total_forward_samples, 8)
        self.assertEqual(counter.total_backward_samples, 5)
        self.assertEqual(counter.total_optimizer_steps, 1)
        self.assertEqual(counter.epoch_snapshot(), (8, 5, 1))
        self.assertEqual(counter.total_snapshot(), (8, 5, 1))

        counter.begin_epoch()

        self.assertEqual(counter.epoch_snapshot(), (0, 0, 0))
        self.assertEqual(counter.total_snapshot(), (8, 5, 1))

    def test_masked_mae_per_sample_reduces_non_batch_axes(self):
        prediction = torch.tensor(
            [
                [[[2.0], [4.0]]],
                [[[2.0], [5.0]]],
            ]
        )
        target = torch.tensor(
            [
                [[[1.0], [1.0]]],
                [[[0.0], [float("nan")]]],
            ]
        )

        losses = masked_mae_per_sample(prediction, target)

        self.assertTrue(torch.allclose(losses, torch.tensor([2.0, 2.0])))

    def test_normalized_masked_mae_per_sample_uses_node_scale(self):
        prediction = torch.tensor([[[[12.0], [120.0]]]])
        target = torch.tensor([[[[10.0], [100.0]]]])
        mean = torch.tensor([[10.0, 100.0]])
        std = torch.tensor([[1.0, 10.0]])

        raw_losses = masked_mae_per_sample(prediction, target)
        normalized_losses = normalized_masked_mae_per_sample(prediction, target, mean=mean, std=std)

        self.assertTrue(torch.allclose(raw_losses, torch.tensor([11.0])))
        self.assertTrue(torch.allclose(normalized_losses, torch.tensor([2.0])))

    def test_masked_mae_per_node_reduces_time_and_channel_axes(self):
        prediction = torch.tensor(
            [
                [
                    [[2.0], [10.0]],
                    [[6.0], [float("nan")]],
                ]
            ]
        )
        target = torch.tensor(
            [
                [
                    [[1.0], [8.0]],
                    [[2.0], [float("nan")]],
                ]
            ]
        )

        losses = masked_mae_per_node(prediction, target)

        self.assertEqual(tuple(losses.shape), (1, 2))
        self.assertTrue(torch.allclose(losses, torch.tensor([[2.5, 2.0]])))

    def test_normalized_masked_mae_per_node_uses_node_scale(self):
        prediction = torch.tensor([[[[12.0], [120.0]]]])
        target = torch.tensor([[[[10.0], [100.0]]]])
        mean = torch.tensor([[10.0, 100.0]])
        std = torch.tensor([[1.0, 10.0]])

        losses = normalized_masked_mae_per_node(prediction, target, mean=mean, std=std)

        self.assertTrue(torch.allclose(losses, torch.tensor([[2.0, 2.0]])))

    def test_node_hard_revisit_rescaled_loss_keeps_hard_and_importance_weights_easy_nodes(self):
        per_node_loss = torch.tensor([[1.0, 2.0, 10.0, 20.0]])
        per_node_score = torch.tensor([[0.0, 1.0, 2.0, 3.0]])

        loss, retained_ratio = node_hard_revisit_rescaled_loss(
            per_node_loss,
            per_node_score,
            retention_ratio=0.5,
            revisit_probability=0.5,
            rescale=True,
            random_values=torch.tensor([[0.25, 0.75, 0.0, 0.0]]),
        )

        self.assertAlmostEqual(retained_ratio, 0.75)
        self.assertAlmostEqual(loss.item(), 8.0, places=6)

    def test_node_hard_revisit_rescaled_loss_uses_unweighted_selected_mean_when_disabled(self):
        per_node_loss = torch.tensor([[1.0, 2.0, 10.0, 20.0]])
        per_node_score = torch.tensor([[0.0, 1.0, 2.0, 3.0]])

        loss, retained_ratio = node_hard_revisit_rescaled_loss(
            per_node_loss,
            per_node_score,
            retention_ratio=0.5,
            revisit_probability=0.5,
            rescale=False,
            random_values=torch.tensor([[0.25, 0.75, 0.0, 0.0]]),
        )

        self.assertAlmostEqual(retained_ratio, 0.75)
        self.assertAlmostEqual(loss.item(), (1.0 + 10.0 + 20.0) / 3.0, places=6)

    def test_lowpass_normalized_masked_mae_filters_high_frequency_error(self):
        prediction = torch.tensor([[[[1.0]], [[-1.0]], [[1.0]], [[-1.0]], [[1.0]], [[-1.0]], [[1.0]], [[-1.0]]]])
        target = torch.zeros_like(prediction)
        mean = torch.tensor([[0.0]])
        std = torch.tensor([[1.0]])

        raw_losses = normalized_masked_mae_per_sample(prediction, target, mean=mean, std=std)
        lowpass_losses = lowpass_normalized_masked_mae_per_sample(
            prediction,
            target,
            mean=mean,
            std=std,
            lowpass_keep_ratio=0.25,
        )

        self.assertTrue(torch.allclose(raw_losses, torch.tensor([1.0])))
        self.assertLess(lowpass_losses.item(), 1e-5)

    def test_inverse_probability_weighted_loss_matches_manual_formula(self):
        losses = torch.tensor([2.0, 6.0])
        probabilities = torch.tensor([0.5, 0.25])

        weighted_loss = inverse_probability_weighted_loss(losses, probabilities, original_batch_size=4)

        self.assertAlmostEqual(weighted_loss.item(), 7.0, places=6)

    def test_soft_random_selection_is_seeded_and_budgeted(self):
        selector_a = DynamicPruningBatchSelector(
            strategy="soft_random",
            retention_ratio=0.5,
            seed=7,
            warmup_epochs=0,
            min_batch_size=1,
        )
        selector_b = DynamicPruningBatchSelector(
            strategy="soft_random",
            retention_ratio=0.5,
            seed=7,
            warmup_epochs=0,
            min_batch_size=1,
        )
        batch_indices = torch.arange(6, dtype=torch.long)

        selector_a.set_dataset_indices(batch_indices)
        selector_b.set_dataset_indices(batch_indices)
        selector_a.begin_epoch(epoch=2)
        selector_b.begin_epoch(epoch=2)
        selection_a = selector_a.select(batch_indices=batch_indices, epoch=2)
        selection_b = selector_b.select(batch_indices=batch_indices, epoch=2)

        self.assertEqual(len(selection_a.selected_positions), 3)
        self.assertTrue(torch.equal(selection_a.selected_positions, selection_b.selected_positions))
        self.assertTrue(torch.allclose(selection_a.inclusion_probabilities, torch.full((3,), 0.5)))

    def test_soft_random_reuses_global_active_set_until_next_pruning_checkpoint(self):
        selector = DynamicPruningBatchSelector(
            strategy="soft_random",
            retention_ratio=0.5,
            seed=3,
            warmup_epochs=0,
            min_batch_size=1,
            pruning_period=10,
        )
        batch_indices = torch.tensor([10, 11, 12, 13], dtype=torch.long)
        selector.set_dataset_indices(batch_indices)
        selector.rng = FixedChoiceRng()

        selector.begin_epoch(epoch=1)
        first_selection = selector.select(batch_indices=batch_indices, epoch=1)

        selector.rng = FirstChoiceRng()
        selector.begin_epoch(epoch=2)
        reused_selection = selector.select(batch_indices=batch_indices, epoch=2)

        selector.begin_epoch(epoch=11)
        refreshed_selection = selector.select(batch_indices=batch_indices, epoch=11)

        self.assertTrue(torch.equal(first_selection.selected_positions, torch.tensor([2, 3])))
        self.assertTrue(torch.equal(reused_selection.selected_positions, torch.tensor([2, 3])))
        self.assertTrue(torch.equal(refreshed_selection.selected_positions, torch.tensor([0, 1])))
        self.assertTrue(torch.allclose(first_selection.inclusion_probabilities, torch.full((2,), 0.5)))

    def test_soft_random_exposes_epoch_subset_positions_for_step_pruning(self):
        selector = DynamicPruningBatchSelector(
            strategy="soft_random",
            retention_ratio=0.1,
            seed=3,
            warmup_epochs=0,
            min_batch_size=1,
        )
        selector.set_dataset_indices(torch.arange(100, dtype=torch.long))

        selector.begin_epoch(epoch=1)
        subset = selector.active_dataset_positions()

        self.assertEqual(len(subset.dataset_positions), 10)
        self.assertEqual(len(subset.sample_indices), 10)
        self.assertTrue(torch.allclose(torch.as_tensor(subset.inclusion_probabilities), torch.full((10,), 0.1)))
        self.assertAlmostEqual(subset.retained_ratio, 0.1)

    def test_infobatch_uses_global_mean_threshold_and_low_score_probability(self):
        selector = DynamicPruningBatchSelector(
            strategy="infobatch",
            retention_ratio=0.5,
            prune_probability=0.5,
            seed=3,
            warmup_epochs=1,
            final_full_epochs=1,
            min_batch_size=1,
        )
        selector.set_num_epochs(10)
        batch_indices = torch.tensor([10, 11, 12, 13], dtype=torch.long)

        warmup_selection = selector.select(batch_indices=batch_indices, epoch=1)
        self.assertTrue(torch.equal(warmup_selection.selected_positions, torch.arange(4)))
        self.assertTrue(torch.allclose(warmup_selection.inclusion_probabilities, torch.ones(4)))

        selector.update_scores(batch_indices, torch.tensor([4.0, 3.0, 1.0, 0.0]))
        selector.rng = FixedRng([0.25, 0.75])
        selection = selector.select(batch_indices=batch_indices, epoch=2)

        self.assertTrue(torch.equal(selection.selected_positions, torch.tensor([0, 1, 2])))
        self.assertTrue(torch.allclose(selection.inclusion_probabilities, torch.tensor([1.0, 1.0, 0.5])))

        final_selection = selector.select(batch_indices=batch_indices, epoch=10)
        self.assertTrue(torch.equal(final_selection.selected_positions, torch.arange(4)))
        self.assertTrue(torch.allclose(final_selection.inclusion_probabilities, torch.ones(4)))

    def test_infobatch_target_retention_mode_uses_retention_ratio_as_window_budget(self):
        selector = DynamicPruningBatchSelector(
            strategy="infobatch",
            retention_ratio=0.4,
            prune_probability=0.5,
            seed=3,
            warmup_epochs=0,
            min_batch_size=1,
            target_retention_mode=True,
        )
        batch_indices = torch.arange(10, dtype=torch.long)
        selector.update_scores(batch_indices, torch.arange(10, dtype=torch.float32))

        selection = selector.select(batch_indices=batch_indices, epoch=2)

        self.assertTrue(torch.equal(selection.selected_positions, torch.tensor([6, 7, 8, 9])))
        self.assertAlmostEqual(selection.retained_ratio, 0.4)
        self.assertTrue(torch.allclose(selection.inclusion_probabilities, torch.ones(4)))

    def test_infobatch_can_be_forced_to_full_data_for_budget_tail(self):
        selector = DynamicPruningBatchSelector(
            strategy="infobatch",
            retention_ratio=0.5,
            prune_probability=0.5,
            seed=3,
            warmup_epochs=0,
            min_batch_size=1,
        )
        batch_indices = torch.tensor([10, 11, 12, 13], dtype=torch.long)
        selector.update_scores(batch_indices, torch.tensor([4.0, 3.0, 1.0, 0.0]))
        selector.rng = FixedRng([0.75, 0.75])

        selector.set_force_full_data(True)
        selection = selector.select(batch_indices=batch_indices, epoch=2)

        self.assertTrue(torch.equal(selection.selected_positions, torch.arange(4)))
        self.assertTrue(torch.allclose(selection.inclusion_probabilities, torch.ones(4)))

    def test_infobatch_prune_probability_is_not_low_score_retention(self):
        selector = DynamicPruningBatchSelector(
            strategy="infobatch",
            retention_ratio=0.1,
            prune_probability=0.1,
            seed=3,
            warmup_epochs=0,
            min_batch_size=1,
        )
        batch_indices = torch.tensor([10, 11, 12, 13], dtype=torch.long)
        selector.update_scores(batch_indices, torch.tensor([4.0, 3.0, 1.0, 0.0]))
        selector.rng = FixedRng([0.89, 0.95])

        selection = selector.select(batch_indices=batch_indices, epoch=2)

        self.assertTrue(torch.equal(selection.selected_positions, torch.tensor([0, 1, 2])))
        self.assertTrue(torch.allclose(selection.inclusion_probabilities, torch.tensor([1.0, 1.0, 0.9])))

    def test_forward_budget_clips_only_the_final_retained_samples(self):
        selection = PruningSelection(
            selected_positions=torch.tensor([0, 2, 4], dtype=torch.long),
            inclusion_probabilities=torch.tensor([1.0, 0.9, 0.9]),
            retained_ratio=0.6,
        )

        clipped = limit_selection_to_forward_budget(selection, remaining_forward_samples=2, original_batch_size=5)
        empty = limit_selection_to_forward_budget(selection, remaining_forward_samples=0, original_batch_size=5)

        self.assertTrue(torch.equal(clipped.selected_positions, torch.tensor([0, 2])))
        self.assertTrue(torch.allclose(clipped.inclusion_probabilities, torch.tensor([1.0, 0.9])))
        self.assertAlmostEqual(clipped.retained_ratio, 0.4)
        self.assertEqual(int(empty.selected_positions.shape[0]), 0)

    def test_score_alpha_uses_raju_ema_update_order(self):
        selector = DynamicPruningBatchSelector(
            strategy="epsilon_greedy",
            retention_ratio=0.5,
            seed=3,
            score_alpha=0.8,
        )
        batch_indices = torch.tensor([10], dtype=torch.long)

        selector.update_scores(batch_indices, torch.tensor([10.0]))
        selector.update_scores(batch_indices, torch.tensor([20.0]))

        self.assertAlmostEqual(selector.score_memory[10], 18.0, places=6)

    def test_only_score_memory_strategies_update_scores_after_retained_train_batch(self):
        epsilon_selector = DynamicPruningBatchSelector(
            strategy="epsilon_greedy",
            retention_ratio=0.5,
            seed=3,
        )
        soft_random_selector = DynamicPruningBatchSelector(
            strategy="soft_random",
            retention_ratio=0.5,
            seed=3,
        )
        infobatch_selector = DynamicPruningBatchSelector(
            strategy="infobatch",
            retention_ratio=0.5,
            seed=3,
        )
        infobatch_norm_selector = DynamicPruningBatchSelector(
            strategy="infobatch_norm",
            retention_ratio=0.5,
            seed=3,
        )
        proxy_gap_selector = DynamicPruningBatchSelector(
            strategy="proxy_gap",
            retention_ratio=0.5,
            seed=3,
        )

        self.assertFalse(epsilon_selector.should_update_scores_after_train_batch())
        self.assertFalse(soft_random_selector.should_update_scores_after_train_batch())
        self.assertTrue(infobatch_selector.should_update_scores_after_train_batch())
        self.assertTrue(infobatch_norm_selector.should_update_scores_after_train_batch())
        self.assertTrue(proxy_gap_selector.should_update_scores_after_train_batch())

    def test_infobatch_norm_uses_infobatch_threshold_selection(self):
        selector = DynamicPruningBatchSelector(
            strategy="infobatch_norm",
            retention_ratio=0.5,
            prune_probability=0.5,
            seed=3,
            warmup_epochs=0,
            min_batch_size=1,
        )
        batch_indices = torch.tensor([10, 11, 12, 13], dtype=torch.long)

        selector.update_scores(batch_indices, torch.tensor([4.0, 3.0, 1.0, 0.0]))
        selector.rng = FixedRng([0.25, 0.75])
        selection = selector.select(batch_indices=batch_indices, epoch=2)

        self.assertTrue(torch.equal(selection.selected_positions, torch.tensor([0, 1, 2])))
        self.assertTrue(torch.allclose(selection.inclusion_probabilities, torch.tensor([1.0, 1.0, 0.5])))

    def test_proxy_gap_selects_high_current_minus_reference_scores_and_revisits_easy_samples(self):
        selector = DynamicPruningBatchSelector(
            strategy="proxy_gap",
            retention_ratio=0.5,
            prune_probability=0.5,
            seed=3,
            warmup_epochs=0,
            min_batch_size=1,
        )
        batch_indices = torch.tensor([10, 11, 12, 13], dtype=torch.long)

        selector.update_reference_scores(batch_indices, torch.tensor([1.0, 1.0, 1.0, 1.0]))
        selector.update_scores(batch_indices, torch.tensor([10.0, 8.0, 4.0, 2.0]))
        selector.rng = FixedRng([0.25, 0.75])
        selection = selector.select(batch_indices=batch_indices, epoch=2)

        self.assertTrue(torch.equal(selection.selected_positions, torch.tensor([0, 1, 2])))
        self.assertTrue(torch.allclose(selection.inclusion_probabilities, torch.tensor([1.0, 1.0, 0.5])))

    def test_epsilon_greedy_builds_checkpoint_active_set_from_global_scores(self):
        selector = DynamicPruningBatchSelector(
            strategy="epsilon_greedy",
            retention_ratio=4 / 6,
            seed=3,
            warmup_epochs=1,
            min_batch_size=1,
            epsilon=0.25,
            pruning_period=10,
        )
        batch_indices = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.long)

        warmup_selection = selector.select(batch_indices=batch_indices, epoch=1)
        self.assertTrue(torch.equal(warmup_selection.selected_positions, torch.arange(6)))

        selector.update_scores(batch_indices, torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]))
        selector.rng = FixedChoiceRng()
        selector.begin_epoch(epoch=2)
        selection = selector.select(batch_indices=batch_indices, epoch=2)

        self.assertTrue(torch.equal(selection.selected_positions, torch.tensor([0, 1, 2, 5])))
        self.assertTrue(torch.allclose(selection.inclusion_probabilities, torch.tensor([1.0, 1.0, 1.0, 1 / 3])))

    def test_epsilon_greedy_reuses_active_set_until_next_pruning_checkpoint(self):
        selector = DynamicPruningBatchSelector(
            strategy="epsilon_greedy",
            retention_ratio=0.5,
            seed=3,
            warmup_epochs=0,
            min_batch_size=1,
            epsilon=0.0,
            pruning_period=10,
        )
        batch_indices = torch.tensor([10, 11, 12, 13], dtype=torch.long)

        selector.update_scores(batch_indices, torch.tensor([4.0, 3.0, 2.0, 1.0]))
        selector.begin_epoch(epoch=1)
        first_selection = selector.select(batch_indices=batch_indices, epoch=1)

        selector.update_scores(batch_indices, torch.tensor([1.0, 2.0, 3.0, 4.0]))
        selector.begin_epoch(epoch=2)
        reused_selection = selector.select(batch_indices=batch_indices, epoch=2)

        selector.begin_epoch(epoch=11)
        refreshed_selection = selector.select(batch_indices=batch_indices, epoch=11)

        self.assertTrue(torch.equal(first_selection.selected_positions, torch.tensor([0, 1])))
        self.assertTrue(torch.equal(reused_selection.selected_positions, torch.tensor([0, 1])))
        self.assertTrue(torch.equal(refreshed_selection.selected_positions, torch.tensor([2, 3])))


if __name__ == "__main__":
    unittest.main()
