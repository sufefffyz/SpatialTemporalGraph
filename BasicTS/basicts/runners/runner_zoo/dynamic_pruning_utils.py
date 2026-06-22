from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class PruningSelection:
    selected_positions: torch.Tensor
    inclusion_probabilities: torch.Tensor
    retained_ratio: float


@dataclass
class PruningSubset:
    dataset_positions: np.ndarray
    sample_indices: np.ndarray
    inclusion_probabilities: np.ndarray
    retained_ratio: float


@dataclass
class DynamicPruningBudgetCounter:
    epoch_forward_samples: int = 0
    epoch_backward_samples: int = 0
    epoch_optimizer_steps: int = 0
    total_forward_samples: int = 0
    total_backward_samples: int = 0
    total_optimizer_steps: int = 0

    def begin_epoch(self) -> None:
        self.epoch_forward_samples = 0
        self.epoch_backward_samples = 0
        self.epoch_optimizer_steps = 0

    def record_forward(self, sample_count: int) -> None:
        sample_count = int(sample_count)
        self.epoch_forward_samples += sample_count
        self.total_forward_samples += sample_count

    def record_backward(self, sample_count: int) -> None:
        sample_count = int(sample_count)
        self.epoch_backward_samples += sample_count
        self.total_backward_samples += sample_count
        self.epoch_optimizer_steps += 1
        self.total_optimizer_steps += 1

    def epoch_snapshot(self):
        return self.epoch_forward_samples, self.epoch_backward_samples, self.epoch_optimizer_steps

    def total_snapshot(self):
        return self.total_forward_samples, self.total_backward_samples, self.total_optimizer_steps


class RandomNodeSubsetSelector:
    """Epoch-level random node subset selector with coverage accounting."""

    def __init__(self, num_nodes: int, node_retention_ratio: float, seed: int, min_nodes: int = 1) -> None:
        if num_nodes < 1:
            raise ValueError(f"num_nodes must be >= 1, got {num_nodes}.")
        if not 0.0 < node_retention_ratio <= 1.0:
            raise ValueError(f"node_retention_ratio must be in (0, 1], got {node_retention_ratio}.")
        if min_nodes < 1:
            raise ValueError(f"min_nodes must be >= 1, got {min_nodes}.")

        self.num_nodes = int(num_nodes)
        self.node_retention_ratio = float(node_retention_ratio)
        self.min_nodes = int(min_nodes)
        self.rng = np.random.default_rng(seed)
        self.active_epoch = None
        self.active_node_indices = None
        self.coverage_counts = np.zeros(self.num_nodes, dtype=np.int64)
        self.epochs_seen = 0

    def begin_epoch(self, epoch: int, force_full: bool = False) -> torch.Tensor:
        if force_full:
            selected = np.arange(self.num_nodes, dtype=np.int64)
        else:
            keep_count = int(round(self.num_nodes * self.node_retention_ratio))
            keep_count = max(self.min_nodes, min(self.num_nodes, keep_count))
            if keep_count >= self.num_nodes:
                selected = np.arange(self.num_nodes, dtype=np.int64)
            else:
                selected = np.asarray(
                    self.rng.choice(np.arange(self.num_nodes, dtype=np.int64), size=keep_count, replace=False),
                    dtype=np.int64,
                )
                selected = np.sort(selected)

        self.coverage_counts[selected] += 1
        self.epochs_seen += 1
        self.active_epoch = int(epoch)
        self.active_node_indices = torch.as_tensor(selected, dtype=torch.long)
        return self.active_node_indices

    @property
    def active_retained_ratio(self) -> float:
        if self.active_node_indices is None:
            return 1.0
        return float(int(self.active_node_indices.shape[0])) / float(self.num_nodes)

    def coverage_summary(self) -> dict:
        counts = self.coverage_counts
        return {
            "epochs": int(self.epochs_seen),
            "selected_nodes": int(self.active_node_indices.shape[0]) if self.active_node_indices is not None else self.num_nodes,
            "retained_ratio": self.active_retained_ratio,
            "coverage_min": int(counts.min()) if counts.size else 0,
            "coverage_max": int(counts.max()) if counts.size else 0,
            "coverage_mean": float(counts.mean()) if counts.size else 0.0,
            "coverage_zero_count": int(np.sum(counts == 0)),
        }


def masked_mae_per_sample(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """Masked MAE reduced over non-batch axes while preserving batch dimension."""

    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        null_tensor = torch.tensor(null_val, device=target.device, dtype=target.dtype)
        mask = ~torch.isclose(target, null_tensor.expand_as(target), atol=eps, rtol=0.0)

    mask = mask.float()
    valid_counts = mask.reshape(mask.shape[0], -1).sum(dim=1).clamp_min(1.0)
    loss = torch.abs(prediction - target) * mask
    loss = torch.nan_to_num(loss)
    return loss.reshape(loss.shape[0], -1).sum(dim=1) / valid_counts


def masked_mae_per_node(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """Masked MAE reduced over horizon/channel axes while preserving batch and node dimensions."""

    if prediction.ndim < 4:
        raise ValueError(
            "masked_mae_per_node expects prediction with shape [B, T, N, C], "
            f"got {tuple(prediction.shape)}."
        )
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        null_tensor = torch.tensor(null_val, device=target.device, dtype=target.dtype)
        mask = ~torch.isclose(target, null_tensor.expand_as(target), atol=eps, rtol=0.0)

    mask = mask.float()
    valid_counts = mask.sum(dim=(1, 3)).clamp_min(1.0)
    loss = torch.abs(prediction - target) * mask
    loss = torch.nan_to_num(loss)
    return loss.sum(dim=(1, 3)) / valid_counts


def _scale_to_prediction_shape(scale, reference: torch.Tensor) -> torch.Tensor:
    scale = torch.as_tensor(scale, device=reference.device, dtype=reference.dtype)
    if scale.ndim == 0:
        return scale
    if scale.ndim == 1:
        return scale.reshape(1, 1, -1, 1)
    if scale.ndim == 2:
        return scale.unsqueeze(0).unsqueeze(-1)
    while scale.ndim < reference.ndim:
        scale = scale.unsqueeze(0)
    return scale


def normalized_masked_mae_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mean,
    std,
    null_val: float = np.nan,
) -> torch.Tensor:
    """Per-sample MAE after node/channel z-score normalization for pruning scores."""

    mean = _scale_to_prediction_shape(mean, prediction)
    std = _scale_to_prediction_shape(std, prediction).clamp_min(1e-6)
    normalized_prediction = (prediction - mean) / std
    normalized_target = (target - mean) / std
    return masked_mae_per_sample(normalized_prediction, normalized_target, null_val=null_val)


def normalized_masked_mae_per_node(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mean,
    std,
    null_val: float = np.nan,
) -> torch.Tensor:
    """Per-node MAE after node/channel z-score normalization for pruning scores."""

    mean = _scale_to_prediction_shape(mean, prediction)
    std = _scale_to_prediction_shape(std, prediction).clamp_min(1e-6)
    normalized_prediction = (prediction - mean) / std
    normalized_target = (target - mean) / std
    return masked_mae_per_node(normalized_prediction, normalized_target, null_val=null_val)


def fft_lowpass_time(data: torch.Tensor, keep_ratio: float = 0.5) -> torch.Tensor:
    """Low-pass filter along the forecasting horizon using real FFT."""

    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}.")
    horizon = int(data.shape[1])
    if horizon <= 1 or keep_ratio >= 1.0:
        return data

    spectrum = torch.fft.rfft(data, dim=1)
    keep_count = max(1, int(round(spectrum.shape[1] * keep_ratio)))
    filtered = spectrum.clone()
    filtered[:, keep_count:, ...] = 0
    return torch.fft.irfft(filtered, n=horizon, dim=1)


def lowpass_normalized_masked_mae_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mean,
    std,
    null_val: float = np.nan,
    lowpass_keep_ratio: float = 0.5,
) -> torch.Tensor:
    """Per-sample normalized MAE after filtering high-frequency horizon noise."""

    prediction = fft_lowpass_time(prediction, keep_ratio=lowpass_keep_ratio)
    target = fft_lowpass_time(target, keep_ratio=lowpass_keep_ratio)
    return normalized_masked_mae_per_sample(prediction, target, mean=mean, std=std, null_val=null_val)


def lowpass_normalized_masked_mae_per_node(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mean,
    std,
    null_val: float = np.nan,
    lowpass_keep_ratio: float = 0.5,
) -> torch.Tensor:
    """Per-node normalized MAE after filtering high-frequency horizon noise."""

    prediction = fft_lowpass_time(prediction, keep_ratio=lowpass_keep_ratio)
    target = fft_lowpass_time(target, keep_ratio=lowpass_keep_ratio)
    return normalized_masked_mae_per_node(prediction, target, mean=mean, std=std, null_val=null_val)


def inverse_probability_weighted_loss(
    losses: torch.Tensor, inclusion_probabilities: torch.Tensor, original_batch_size: int
) -> torch.Tensor:
    """Horvitz-Thompson style batch-mean estimator over retained samples."""

    probs = inclusion_probabilities.to(losses.device).clamp_min(1e-8)
    return torch.sum(losses / probs) / float(original_batch_size)


def node_hard_revisit_rescaled_loss(
    per_node_loss: torch.Tensor,
    per_node_score: torch.Tensor,
    retention_ratio: float,
    revisit_probability: float = 0.5,
    rescale: bool = True,
    random_values: torch.Tensor = None,
) -> tuple[torch.Tensor, float]:
    """Keep hard nodes, revisit easy nodes, and importance-weight the selected node loss."""

    if per_node_loss.shape != per_node_score.shape:
        raise ValueError(
            "per_node_loss and per_node_score must have the same shape, "
            f"got {tuple(per_node_loss.shape)} and {tuple(per_node_score.shape)}."
        )
    if not 0.0 < retention_ratio <= 1.0:
        raise ValueError(f"retention_ratio must be in (0, 1], got {retention_ratio}.")
    if not 0.0 <= revisit_probability <= 1.0:
        raise ValueError(f"revisit_probability must be in [0, 1], got {revisit_probability}.")

    flat_loss = per_node_loss.reshape(-1)
    flat_score = torch.nan_to_num(per_node_score.reshape(-1), nan=-float("inf"))
    total_nodes = int(flat_loss.numel())
    if total_nodes == 0:
        raise ValueError("Cannot apply node-hard pruning to an empty node loss tensor.")

    keep_count = max(1, min(total_nodes, int(round(total_nodes * float(retention_ratio)))))
    hard_positions = torch.topk(flat_score, k=keep_count, largest=True, sorted=False).indices
    hard_mask = torch.zeros(total_nodes, dtype=torch.bool, device=flat_loss.device)
    hard_mask[hard_positions] = True

    probabilities = torch.full(
        (total_nodes,),
        float(revisit_probability),
        dtype=flat_loss.dtype,
        device=flat_loss.device,
    )
    probabilities[hard_mask] = 1.0
    if revisit_probability <= 0.0:
        selected_mask = hard_mask
    else:
        if random_values is None:
            random_values = torch.rand(per_node_loss.shape, device=flat_loss.device, dtype=flat_loss.dtype)
        random_values = random_values.to(device=flat_loss.device, dtype=flat_loss.dtype).reshape(-1)
        selected_mask = hard_mask | (random_values < revisit_probability)

    selected_loss = flat_loss[selected_mask]
    selected_probabilities = probabilities[selected_mask].clamp_min(1e-8)
    retained_ratio = float(selected_mask.float().mean().item())
    if rescale:
        loss = torch.sum(selected_loss / selected_probabilities) / float(total_nodes)
    else:
        loss = selected_loss.mean()
    return loss, retained_ratio


def limit_selection_to_forward_budget(
    selection: PruningSelection, remaining_forward_samples: int, original_batch_size: int
) -> PruningSelection:
    """Clip the final retained batch to an exact forward-sample budget."""

    remaining_forward_samples = int(remaining_forward_samples)
    if remaining_forward_samples <= 0:
        return PruningSelection(
            selected_positions=torch.empty(0, dtype=torch.long),
            inclusion_probabilities=torch.empty(0, dtype=torch.float32),
            retained_ratio=0.0,
        )

    selected_count = int(selection.selected_positions.shape[0])
    if selected_count <= remaining_forward_samples:
        return selection

    return PruningSelection(
        selected_positions=selection.selected_positions[:remaining_forward_samples],
        inclusion_probabilities=selection.inclusion_probabilities[:remaining_forward_samples],
        retained_ratio=float(remaining_forward_samples) / float(original_batch_size),
    )


class DynamicPruningBatchSelector:
    """Stateful batch selector for dynamic data-pruning baselines."""

    def __init__(
        self,
        strategy: str,
        retention_ratio: float,
        seed: int,
        warmup_epochs: int = 0,
        final_full_epochs: int = 0,
        min_batch_size: int = 1,
        score_momentum: float = 0.0,
        prune_probability: float = 0.5,
        epsilon: float = 0.1,
        pruning_period: int = 10,
        score_alpha: float = None,
        target_retention_mode: bool = False,
    ) -> None:
        if not 0.0 < retention_ratio <= 1.0:
            raise ValueError(f"retention_ratio must be in (0, 1], got {retention_ratio}.")
        if min_batch_size < 1:
            raise ValueError(f"min_batch_size must be >= 1, got {min_batch_size}.")
        if not 0.0 <= prune_probability < 1.0:
            raise ValueError(f"prune_probability must be in [0, 1), got {prune_probability}.")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}.")
        if pruning_period < 1:
            raise ValueError(f"pruning_period must be >= 1, got {pruning_period}.")
        if score_alpha is not None and not 0.0 <= score_alpha <= 1.0:
            raise ValueError(f"score_alpha must be in [0, 1], got {score_alpha}.")

        self.strategy = self._normalize_strategy(strategy)
        self.retention_ratio = float(retention_ratio)
        self.prune_probability = float(prune_probability)
        self.low_score_keep_probability = 1.0 - self.prune_probability
        self.epsilon = float(epsilon)
        self.pruning_period = int(pruning_period)
        self.target_retention_mode = bool(target_retention_mode)
        self.warmup_epochs = int(warmup_epochs)
        self.final_full_epochs = int(final_full_epochs)
        self.min_batch_size = int(min_batch_size)
        self.score_momentum = float(score_momentum)
        self.score_alpha = score_alpha
        self.rng = np.random.default_rng(seed)
        self.score_memory = {}
        self.reference_score_memory = {}
        self.dataset_indices = None
        self.num_epochs = None
        self.active_epoch = None
        self.active_indices = None
        self.active_inclusion_probabilities = {}
        self.force_full_data = False

    def set_num_epochs(self, num_epochs: int) -> None:
        self.num_epochs = int(num_epochs)

    def set_dataset_indices(self, dataset_indices) -> None:
        self.dataset_indices = np.asarray(dataset_indices, dtype=np.int64)

    def set_force_full_data(self, force_full_data: bool) -> None:
        self.force_full_data = bool(force_full_data)

    def should_score_epoch(self, epoch: int) -> bool:
        if self.strategy != "epsilon_greedy":
            return False
        if self._is_full_data_epoch(epoch):
            return False
        return self._is_pruning_checkpoint(epoch)

    def begin_epoch(self, epoch: int) -> None:
        if self.strategy not in {
            "soft_random",
            "block_random_node_window",
            "epsilon_greedy",
            "infobatch",
            "infobatch_norm",
            "proxy_gap",
            "node_hard_revisit",
        }:
            return
        if self._is_full_data_epoch(epoch):
            self.active_epoch = epoch
            self.active_indices = None
            self.active_inclusion_probabilities = {}
            return
        if self.strategy == "block_random_node_window":
            self._refresh_soft_random_active_set(epoch)
            return
        if self.strategy == "soft_random":
            if self.active_indices is not None and not self._is_pruning_checkpoint(epoch):
                self.active_epoch = epoch
                return
            self._refresh_soft_random_active_set(epoch)
            return
        if self.strategy in {"infobatch", "infobatch_norm", "proxy_gap"}:
            self._refresh_infobatch_active_set(epoch)
            return
        if self.active_indices is not None and not self._is_pruning_checkpoint(epoch):
            self.active_epoch = epoch
            return
        self._refresh_epsilon_greedy_active_set(epoch)

    def select(self, batch_indices: torch.Tensor, epoch: int) -> PruningSelection:
        batch_indices = batch_indices.detach().cpu().long()
        batch_size = int(batch_indices.shape[0])
        if batch_size == 0:
            raise ValueError("Cannot prune an empty batch.")

        if self.strategy == "soft_random":
            return self._soft_random(batch_indices, epoch)
        if self.strategy == "block_random_node_window":
            return self._soft_random(batch_indices, epoch)
        if self.strategy == "epsilon_greedy":
            return self._epsilon_greedy(batch_indices, epoch)
        if self.strategy in {"infobatch", "infobatch_norm"}:
            return self._infobatch(batch_indices, epoch)
        if self.strategy == "proxy_gap":
            return self._proxy_gap(batch_indices, epoch)
        if self.strategy == "node_hard_revisit":
            return self._full_selection(batch_size)
        raise ValueError(f"Unsupported dynamic pruning strategy: {self.strategy}")

    def update_scores(self, batch_indices: torch.Tensor, losses: torch.Tensor) -> None:
        self._update_score_dict(self.score_memory, batch_indices, losses)

    def update_reference_scores(self, batch_indices: torch.Tensor, losses: torch.Tensor) -> None:
        self._update_score_dict(self.reference_score_memory, batch_indices, losses)

    def _update_score_dict(self, score_dict, batch_indices: torch.Tensor, losses: torch.Tensor) -> None:
        batch_indices = batch_indices.detach().cpu().long().tolist()
        losses = losses.detach().cpu().float().tolist()
        for sample_index, loss in zip(batch_indices, losses):
            previous = score_dict.get(sample_index)
            if previous is None:
                score_dict[sample_index] = float(loss)
            elif self.score_alpha is not None:
                score_dict[sample_index] = (
                    self.score_alpha * float(loss) + (1.0 - self.score_alpha) * float(previous)
                )
            else:
                score_dict[sample_index] = (
                    self.score_momentum * float(previous) + (1.0 - self.score_momentum) * float(loss)
                )

    def should_update_scores_after_train_batch(self) -> bool:
        """Whether retained training batches should update score memory."""

        return self.strategy in {"infobatch", "infobatch_norm", "proxy_gap"}

    def active_dataset_positions(self):
        """Return active dataset positions for the current epoch, or None for full data."""

        if self.active_indices is None:
            return None
        if self.dataset_indices is None:
            raise ValueError(f"{self.strategy} requires dataset indices before subset selection.")

        positions = []
        sample_indices = []
        probabilities = []
        for position, sample_index in enumerate(self.dataset_indices.tolist()):
            sample_index = int(sample_index)
            if sample_index in self.active_indices:
                positions.append(position)
                sample_indices.append(sample_index)
                probabilities.append(self.active_inclusion_probabilities.get(sample_index, 1.0))

        return PruningSubset(
            dataset_positions=np.asarray(positions, dtype=np.int64),
            sample_indices=np.asarray(sample_indices, dtype=np.int64),
            inclusion_probabilities=np.asarray(probabilities, dtype=np.float32),
            retained_ratio=float(len(positions)) / float(len(self.dataset_indices)),
        )

    def _soft_random(self, batch_indices: torch.Tensor, epoch: int) -> PruningSelection:
        batch_size = int(batch_indices.shape[0])
        if self._is_full_data_epoch(epoch):
            return self._full_selection(batch_size)

        if self.active_indices is None:
            self._refresh_soft_random_active_set(epoch)
        if self.active_indices is None:
            return self._full_selection(batch_size)
        return self._selection_from_active_set(batch_indices)

    def _infobatch(self, batch_indices: torch.Tensor, epoch: int) -> PruningSelection:
        batch_size = int(batch_indices.shape[0])
        if self._is_full_data_epoch(epoch) or self.prune_probability <= 0.0:
            return self._full_selection(batch_size)
        if self.active_indices is not None:
            return self._selection_from_active_set(batch_indices)

        score_memory = self._effective_score_memory()
        if score_memory is None:
            return self._full_selection(batch_size)
        scores = self._scores_for_batch(batch_indices, score_memory)
        if scores is None:
            return self._full_selection(batch_size)

        return self._threshold_revisit_selection(scores, batch_size)

    def _proxy_gap(self, batch_indices: torch.Tensor, epoch: int) -> PruningSelection:
        return self._infobatch(batch_indices, epoch)

    def _threshold_revisit_selection(self, scores, batch_size: int) -> PruningSelection:
        if self.target_retention_mode:
            return self._target_retention_selection(scores, batch_size)

        threshold = float(np.mean(list(self._effective_score_memory().values())))
        high_positions = [pos for pos, score in enumerate(scores) if score >= threshold]
        low_positions = [pos for pos, score in enumerate(scores) if score < threshold]

        selected_with_prob = [(int(pos), 1.0) for pos in high_positions]
        if low_positions:
            keep_low = self.rng.random(len(low_positions)) < self.low_score_keep_probability
            selected_with_prob.extend(
                (int(pos), self.low_score_keep_probability)
                for pos, keep in zip(low_positions, keep_low.tolist())
                if keep
            )

        if len(selected_with_prob) < self.min_batch_size:
            selected_positions = {pos for pos, _ in selected_with_prob}
            fallback_order = np.argsort(np.asarray(scores, dtype=np.float32))[::-1]
            for pos in fallback_order.tolist():
                if pos in selected_positions:
                    continue
                probability = 1.0 if scores[pos] >= threshold else self.low_score_keep_probability
                selected_with_prob.append((int(pos), float(probability)))
                selected_positions.add(pos)
                if len(selected_with_prob) >= self.min_batch_size:
                    break

        selected_with_prob.sort(key=lambda item: item[0])
        selected_positions = torch.tensor([item[0] for item in selected_with_prob], dtype=torch.long)
        probabilities = torch.tensor([item[1] for item in selected_with_prob], dtype=torch.float32)
        return PruningSelection(
            selected_positions=selected_positions,
            inclusion_probabilities=probabilities,
            retained_ratio=float(len(selected_with_prob)) / float(batch_size),
        )

    def _target_retention_selection(self, scores, batch_size: int) -> PruningSelection:
        scores = np.asarray(scores, dtype=np.float32)
        keep_count = self._target_keep_count(batch_size)
        keep_count = min(int(keep_count), int(scores.shape[0]))
        selected_positions = np.argsort(scores)[::-1][:keep_count]
        selected_positions = torch.tensor(sorted(int(pos) for pos in selected_positions.tolist()), dtype=torch.long)
        return PruningSelection(
            selected_positions=selected_positions,
            inclusion_probabilities=torch.ones(int(selected_positions.shape[0]), dtype=torch.float32),
            retained_ratio=float(selected_positions.shape[0]) / float(batch_size),
        )

    def _epsilon_greedy(self, batch_indices: torch.Tensor, epoch: int) -> PruningSelection:
        batch_size = int(batch_indices.shape[0])
        if self._is_full_data_epoch(epoch):
            return self._full_selection(batch_size)

        if self.active_indices is None:
            self._refresh_epsilon_greedy_active_set(epoch)
        if self.active_indices is None:
            return self._full_selection(batch_size)

        return self._selection_from_active_set(batch_indices)

    def _refresh_soft_random_active_set(self, epoch: int) -> None:
        if self.dataset_indices is None:
            raise ValueError("soft_random requires dataset indices before checkpoint selection.")

        dataset_size = int(self.dataset_indices.shape[0])
        keep_count = self._target_keep_count(dataset_size)
        if keep_count >= dataset_size:
            self.active_epoch = epoch
            self.active_indices = None
            self.active_inclusion_probabilities = {}
            return

        selected_indices = self.rng.choice(self.dataset_indices, size=keep_count, replace=False).tolist()
        probability = float(keep_count) / float(dataset_size)
        self.active_epoch = epoch
        self.active_indices = {int(sample_index) for sample_index in selected_indices}
        self.active_inclusion_probabilities = {
            int(sample_index): probability for sample_index in selected_indices
        }

    def _refresh_infobatch_active_set(self, epoch: int) -> None:
        if self.dataset_indices is None:
            raise ValueError(f"{self.strategy} requires dataset indices before checkpoint selection.")
        score_memory = self._effective_score_memory()
        if self.prune_probability <= 0.0 or not score_memory:
            self.active_epoch = epoch
            self.active_indices = None
            self.active_inclusion_probabilities = {}
            return

        scores = []
        for sample_index in self.dataset_indices.tolist():
            score = score_memory.get(int(sample_index))
            if score is None:
                self.active_epoch = epoch
                self.active_indices = None
                self.active_inclusion_probabilities = {}
                return
            scores.append(float(score))

        if self.target_retention_mode:
            keep_count = self._target_keep_count(len(scores))
            order = np.argsort(np.asarray(scores, dtype=np.float32))[::-1][:keep_count]
            selected_probabilities = {
                int(self.dataset_indices[position]): 1.0
                for position in order.tolist()
            }
            self.active_epoch = epoch
            self.active_indices = set(selected_probabilities.keys())
            self.active_inclusion_probabilities = selected_probabilities
            return

        threshold = float(np.mean(scores))
        selected_probabilities = {}
        for sample_index, score in zip(self.dataset_indices.tolist(), scores):
            sample_index = int(sample_index)
            if score >= threshold:
                selected_probabilities[sample_index] = 1.0
            elif self.rng.random() < self.low_score_keep_probability:
                selected_probabilities[sample_index] = self.low_score_keep_probability

        if not selected_probabilities:
            best_position = int(np.argmax(np.asarray(scores, dtype=np.float32)))
            best_index = int(self.dataset_indices[best_position])
            selected_probabilities[best_index] = 1.0

        self.active_epoch = epoch
        self.active_indices = set(selected_probabilities.keys())
        self.active_inclusion_probabilities = selected_probabilities

    def _scores_for_batch(self, batch_indices: torch.Tensor, score_memory):
        scores = []
        for sample_index in batch_indices.tolist():
            score = score_memory.get(int(sample_index))
            if score is None:
                return None
            scores.append(float(score))
        return scores

    def _effective_score_memory(self):
        if self.strategy != "proxy_gap":
            return self.score_memory
        if not self.score_memory or not self.reference_score_memory:
            return None
        score_memory = {}
        for sample_index, current_score in self.score_memory.items():
            reference_score = self.reference_score_memory.get(sample_index)
            if reference_score is not None:
                score_memory[int(sample_index)] = float(current_score) - float(reference_score)
        return score_memory

    def _refresh_epsilon_greedy_active_set(self, epoch: int) -> None:
        if not self.score_memory:
            self.active_epoch = epoch
            self.active_indices = None
            self.active_inclusion_probabilities = {}
            return

        scored_items = sorted(self.score_memory.items(), key=lambda item: (-float(item[1]), int(item[0])))
        dataset_size = len(scored_items)
        keep_count = self._target_keep_count(dataset_size)
        if keep_count >= dataset_size:
            self.active_epoch = epoch
            self.active_indices = None
            self.active_inclusion_probabilities = {}
            return

        explore_count = int(round(keep_count * self.epsilon))
        if self.epsilon > 0.0 and explore_count == 0:
            explore_count = 1
        explore_count = min(explore_count, keep_count)
        greedy_count = keep_count - explore_count

        greedy_indices = [int(sample_index) for sample_index, _ in scored_items[:greedy_count]]
        greedy_set = set(greedy_indices)
        remaining_indices = [int(sample_index) for sample_index, _ in scored_items if int(sample_index) not in greedy_set]

        exploration_indices = []
        exploration_probability = 0.0
        if explore_count > 0 and remaining_indices:
            explore_count = min(explore_count, len(remaining_indices))
            exploration_indices = self.rng.choice(remaining_indices, size=explore_count, replace=False).tolist()
            exploration_probability = float(explore_count) / float(len(remaining_indices))

        active_probabilities = {sample_index: 1.0 for sample_index in greedy_indices}
        active_probabilities.update({int(sample_index): exploration_probability for sample_index in exploration_indices})
        self.active_epoch = epoch
        self.active_indices = set(active_probabilities.keys())
        self.active_inclusion_probabilities = active_probabilities

    def _selection_from_active_set(self, batch_indices: torch.Tensor) -> PruningSelection:
        batch_size = int(batch_indices.shape[0])
        selected_with_prob = []
        for position, sample_index in enumerate(batch_indices.tolist()):
            sample_index = int(sample_index)
            if sample_index in self.active_indices:
                probability = self.active_inclusion_probabilities.get(sample_index, 1.0)
                selected_with_prob.append((int(position), float(probability)))
        selected_with_prob.sort(key=lambda item: item[0])

        selected_positions = torch.tensor([item[0] for item in selected_with_prob], dtype=torch.long)
        probabilities = torch.tensor([item[1] for item in selected_with_prob], dtype=torch.float32)
        return PruningSelection(
            selected_positions=selected_positions,
            inclusion_probabilities=probabilities,
            retained_ratio=float(len(selected_with_prob)) / float(batch_size),
        )

    def _full_selection(self, batch_size: int) -> PruningSelection:
        return PruningSelection(
            selected_positions=torch.arange(batch_size, dtype=torch.long),
            inclusion_probabilities=torch.ones(batch_size, dtype=torch.float32),
            retained_ratio=1.0,
        )

    def _target_keep_count(self, batch_size: int) -> int:
        keep_count = int(round(batch_size * self.retention_ratio))
        return max(self.min_batch_size, min(batch_size, keep_count))

    def _is_full_data_epoch(self, epoch: int) -> bool:
        if self.force_full_data:
            return True
        if epoch <= self.warmup_epochs:
            return True
        if self.num_epochs is None or self.final_full_epochs <= 0:
            return False
        return epoch > self.num_epochs - self.final_full_epochs

    def _is_pruning_checkpoint(self, epoch: int) -> bool:
        first_pruned_epoch = self.warmup_epochs + 1
        if epoch < first_pruned_epoch:
            return False
        return (epoch - first_pruned_epoch) % self.pruning_period == 0

    @staticmethod
    def _normalize_strategy(strategy: str) -> str:
        strategy = str(strategy).lower().replace("-", "_")
        if strategy in {"eps_greedy", "epsilon"}:
            return "epsilon_greedy"
        if strategy in {"infobatch_normalized", "normalized_infobatch"}:
            return "infobatch_norm"
        if strategy in {"rho_prune", "proxy_gap_prune"}:
            return "proxy_gap"
        if strategy in {"node_hard", "lowpass_node_hard", "node_hard_lowpass"}:
            return "node_hard_revisit"
        if strategy in {"block_random", "random_node_window", "block_random_node_window"}:
            return "block_random_node_window"
        return strategy
