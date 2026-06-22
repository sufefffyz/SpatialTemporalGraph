import copy
import os
import warnings
from typing import Dict, Union

import numpy as np
import torch
from easytorch.core.data_loader import build_data_loader, build_data_loader_ddp
from torch.utils.data import Subset

from basicts.data.coreset_tsf_dataset import CoresetTimeSeriesForecastingDataset
from basicts.data.indexed_tsf_dataset import (IndexedCoresetTimeSeriesForecastingDataset,
                                              IndexedTimeSeriesForecastingDataset)
from basicts.data.simple_tsf_dataset import TimeSeriesForecastingDataset
from basicts.metrics import masked_mae

from .dynamic_pruning_utils import (DynamicPruningBatchSelector,
                                    DynamicPruningBudgetCounter,
                                    RandomNodeSubsetSelector,
                                    inverse_probability_weighted_loss,
                                    limit_selection_to_forward_budget,
                                    lowpass_normalized_masked_mae_per_node,
                                    lowpass_normalized_masked_mae_per_sample,
                                    masked_mae_per_node,
                                    masked_mae_per_sample,
                                    node_hard_revisit_rescaled_loss,
                                    normalized_masked_mae_per_node,
                                    normalized_masked_mae_per_sample)
from .cluster_subgraph_utils import (ClusterSubgraphScheduler,
                                     build_balanced_signal_kmeans_clusters,
                                     build_balanced_spatial_kdtree_clusters,
                                     build_random_balanced_clusters,
                                     cluster_assignment_cache_path,
                                     load_cluster_assignment,
                                     load_spatial_coordinates,
                                     reduce_node_scores_to_samples,
                                     save_cluster_assignment)
from .simple_tsf_runner import SimpleTimeSeriesForecastingRunner
from .wandb_tsf_runner import WandBTimeSeriesForecastingRunner


class _DynamicPruningRunnerMixin:
    """Mixin that adds opt-in dynamic batch pruning to forecasting runners."""

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        pruning_cfg = cfg.get("TRAIN", {}).get("DYNAMIC_PRUNING", {})
        self.dynamic_pruning_cfg = pruning_cfg
        self.dynamic_pruning_enabled = bool(pruning_cfg.get("ENABLED", False))
        self.dynamic_pruning_selector = None
        self.dynamic_pruning_rescale = bool(pruning_cfg.get("RESCALE", True))
        self.dynamic_pruning_target_forward_ratio = pruning_cfg.get("TARGET_FORWARD_RATIO", None)
        self.dynamic_pruning_reference_num_epochs = pruning_cfg.get("REFERENCE_NUM_EPOCHS", None)
        self.dynamic_pruning_final_full_forward_ratio = float(pruning_cfg.get("FINAL_FULL_FORWARD_RATIO", 0.0))
        self.dynamic_pruning_target_forward_samples = None
        self.dynamic_pruning_final_full_forward_samples = None
        self.dynamic_pruning_forwarded_samples = 0
        self.dynamic_pruning_budget_exhausted = False
        self.dynamic_pruning_budget_logged = False
        self.dynamic_pruning_full_train_data_loader = None
        self.dynamic_pruning_train_data_cfg = None
        self.dynamic_pruning_epoch_retained_ratio = 1.0
        self.dynamic_pruning_budget_counter = DynamicPruningBudgetCounter()
        self.dynamic_pruning_pending_backward_samples = 0
        self.dynamic_pruning_score_loss_type = str(pruning_cfg.get("SCORE_LOSS_TYPE", "raw_mae")).lower()
        self.dynamic_pruning_lowpass_keep_ratio = float(pruning_cfg.get("LOWPASS_KEEP_RATIO", 0.5))
        self.dynamic_pruning_node_loss_pruning = bool(pruning_cfg.get("NODE_LOSS_PRUNING", False))
        self.dynamic_pruning_node_retention_ratio = float(
            pruning_cfg.get("NODE_RETENTION_RATIO", pruning_cfg.get("RETENTION_RATIO", 1.0))
        )
        self.block_random_node_window_enabled = (
            bool(self.dynamic_pruning_enabled)
            and str(pruning_cfg.get("STRATEGY", "")).lower().replace("-", "_") == "block_random_node_window"
        )
        self.block_random_node_selector = None
        self.block_random_active_node_indices = None
        self.dynamic_pruning_node_revisit_probability = float(pruning_cfg.get("NODE_REVISIT_PROBABILITY", 0.5))
        self.dynamic_pruning_reference_type = str(pruning_cfg.get("REFERENCE_TYPE", "seasonal")).lower()
        self.dynamic_pruning_reference_period = int(pruning_cfg.get("REFERENCE_PERIOD", 288))
        self.dynamic_pruning_dlinear_epochs = int(pruning_cfg.get("DLINEAR_EPOCHS", 100))
        self.dynamic_pruning_dlinear_patience = int(pruning_cfg.get("DLINEAR_PATIENCE", 30))
        self.dynamic_pruning_dlinear_lr = float(pruning_cfg.get("DLINEAR_LR", 0.001))
        self.dynamic_pruning_dlinear_weight_decay = float(pruning_cfg.get("DLINEAR_WEIGHT_DECAY", 0.0))
        self.dynamic_pruning_dlinear_individual = bool(pruning_cfg.get("DLINEAR_INDIVIDUAL", False))
        self.cluster_subgraph_cfg = pruning_cfg.get("CLUSTER_SUBGRAPH", {})
        self.cluster_subgraph_enabled = bool(self.cluster_subgraph_cfg.get("ENABLED", False))
        self.cluster_subgraph_scheduler = None
        self.cluster_subgraph_epoch_force_full = False
        self.cluster_subgraph_reference_node_scores = {}
        self.cluster_subgraph_reference_score_cache = {}

        if not self.dynamic_pruning_enabled:
            return

        self.dynamic_pruning_selector = DynamicPruningBatchSelector(
            strategy=pruning_cfg.get("STRATEGY", "soft_random"),
            retention_ratio=pruning_cfg.get("RETENTION_RATIO", 1.0),
            seed=pruning_cfg.get("SEED", 2023),
            warmup_epochs=pruning_cfg.get("WARMUP_EPOCHS", 0),
            final_full_epochs=pruning_cfg.get("FINAL_FULL_EPOCHS", 0),
            min_batch_size=pruning_cfg.get("MIN_BATCH_SIZE", 1),
            score_momentum=pruning_cfg.get("SCORE_MOMENTUM", 0.0),
            prune_probability=pruning_cfg.get("PRUNE_PROBABILITY", 0.5),
            epsilon=pruning_cfg.get("EPSILON", 0.1),
            pruning_period=pruning_cfg.get("PRUNING_PERIOD", 10),
            score_alpha=pruning_cfg.get("SCORE_ALPHA", None),
            target_retention_mode=pruning_cfg.get("TARGET_RETENTION_MODE", False),
        )
        self.dynamic_pruning_selector.set_num_epochs(cfg["TRAIN"].get("NUM_EPOCHS", 0))

        if self.loss is not masked_mae:
            warnings.warn(
                "Dynamic pruning uses masked_mae_per_sample for weighted training loss; "
                "validation/test still use the configured loss.",
                stacklevel=2,
            )

    def build_train_dataset(self, cfg: Dict):
        if not self.dynamic_pruning_enabled:
            return super().build_train_dataset(cfg)

        dataset_cfg = copy.deepcopy(cfg)
        if "DATASET" in dataset_cfg:
            dataset_cfg["DATASET"]["TYPE"] = self._indexed_dataset_type(dataset_cfg["DATASET"]["TYPE"])
        else:
            dataset_type = dataset_cfg["TRAIN"]["DATA"]["DATASET"]["TYPE"]
            dataset_cfg["TRAIN"]["DATA"]["DATASET"]["TYPE"] = self._indexed_dataset_type(dataset_type)
        return super().build_train_dataset(dataset_cfg)

    def init_training(self, cfg: Dict):
        super().init_training(cfg)
        if self.dynamic_pruning_enabled:
            self.dynamic_pruning_full_train_data_loader = self.train_data_loader
            self.dynamic_pruning_train_data_cfg = copy.deepcopy(cfg["TRAIN"]["DATA"])
            self.dynamic_pruning_selector.set_dataset_indices(self._train_dataset_indices())
            self._init_forward_budget()
            if self.cluster_subgraph_enabled:
                self._init_cluster_subgraph()
            if self.block_random_node_window_enabled:
                self._init_block_random_node_window()
            self._init_reference_scores_for_pruning()
            self.register_epoch_meter("train/retained_ratio", "train", "{:.4f}")
            self.register_epoch_meter("train/window_retained_ratio", "train", "{:.4f}")
            if self.dynamic_pruning_node_loss_pruning or self.block_random_node_window_enabled:
                self.register_epoch_meter("train/node_retained_ratio", "train", "{:.4f}")
            if self.block_random_node_window_enabled:
                self.register_epoch_meter("train/effective_node_window_ratio", "train", "{:.4f}")
                self.register_epoch_meter("train/node_coverage_min", "train", "{:.0f}")
                self.register_epoch_meter("train/node_coverage_mean", "train", "{:.4f}")
                self.register_epoch_meter("train/node_coverage_max", "train", "{:.0f}")
                self.register_epoch_meter("train/node_coverage_zero_count", "train", "{:.0f}")
            if self.cluster_subgraph_enabled:
                self.register_epoch_meter("train/active_node_ratio", "train", "{:.4f}")
                self.register_epoch_meter("train/effective_pair_ratio", "train", "{:.4f}")
                self.register_epoch_meter("train/num_active_clusters", "train", "{:.0f}")
            self.register_epoch_meter("train/forward_budget_ratio", "train", "{:.4f}")
            self.register_epoch_meter("train/forward_samples", "train", "{:.0f}")
            self.register_epoch_meter("train/backward_samples", "train", "{:.0f}")
            self.register_epoch_meter("train/optimizer_steps", "train", "{:.0f}")
            self.register_epoch_meter("train/total_forward_samples", "train", "{:.0f}")
            self.register_epoch_meter("train/total_backward_samples", "train", "{:.0f}")
            self.register_epoch_meter("train/total_optimizer_steps", "train", "{:.0f}")

    def on_epoch_start(self, epoch: int):
        if not self.dynamic_pruning_enabled:
            super().on_epoch_start(epoch)
            return
        self.dynamic_pruning_selector.set_force_full_data(self._use_forward_budget_final_full_data())
        if self.dynamic_pruning_selector.should_score_epoch(epoch):
            self._score_full_training_dataset_for_pruning(epoch)
        self.dynamic_pruning_selector.begin_epoch(epoch)
        if self.cluster_subgraph_enabled:
            self._begin_cluster_subgraph_epoch(epoch)
            if (
                self.dynamic_pruning_selector.strategy == "proxy_gap"
                and not self.cluster_subgraph_epoch_force_full
            ):
                # ProxyGap+cluster needs per-step active-node references, so avoid a
                # stale epoch-level subset built from full-node reference scores.
                self.dynamic_pruning_selector.active_indices = None
                self.dynamic_pruning_selector.active_inclusion_probabilities = {}
        if self.block_random_node_window_enabled:
            force_full_nodes = self.dynamic_pruning_selector._is_full_data_epoch(epoch)
            self.block_random_active_node_indices = self.block_random_node_selector.begin_epoch(
                epoch=epoch,
                force_full=force_full_nodes,
            )
            summary = self.block_random_node_selector.coverage_summary()
            self.logger.info(
                "Block random node subset: selected=%d/%d ratio=%.4f "
                "coverage_min=%d coverage_mean=%.4f coverage_max=%d coverage_zero=%d.",
                summary["selected_nodes"],
                self.block_random_node_selector.num_nodes,
                summary["retained_ratio"],
                summary["coverage_min"],
                summary["coverage_mean"],
                summary["coverage_max"],
                summary["coverage_zero_count"],
            )
        self.dynamic_pruning_budget_counter.begin_epoch()
        self.dynamic_pruning_pending_backward_samples = 0
        self._refresh_epoch_train_data_loader()
        super().on_epoch_start(epoch)

    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Dict]) -> torch.Tensor:
        if not self.dynamic_pruning_enabled:
            return super().train_iters(epoch, iter_index, data)
        if self._forward_budget_exhausted():
            self.dynamic_pruning_budget_exhausted = True
            return None
        if not isinstance(data, dict) or "index" not in data:
            raise ValueError("Dynamic pruning runner requires a dataset that returns batch indices.")

        batch_indices = self._to_long_tensor(data["index"])
        original_batch_size = int(batch_indices.shape[0])
        cluster_selection = self._select_cluster_subgraph(epoch, iter_index)
        if cluster_selection is not None:
            self._update_proxy_gap_reference_scores_for_active_nodes(batch_indices, cluster_selection)
        selection = self.dynamic_pruning_selector.select(batch_indices=batch_indices, epoch=epoch)
        selection = self._limit_selection_to_forward_budget(selection, original_batch_size)
        if int(selection.selected_positions.shape[0]) == 0:
            if self._forward_budget_exhausted():
                self.dynamic_pruning_budget_exhausted = True
            return None

        pruned_data = self._slice_batch(data, selection.selected_positions)
        if cluster_selection is not None:
            pruned_data = self._slice_nodes(pruned_data, cluster_selection.node_indices)
        elif self.block_random_node_window_enabled:
            pruned_data = self._slice_nodes(pruned_data, self.block_random_active_node_indices)

        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index
        forward_return = self.forward(data=pruned_data, epoch=epoch, iter_num=iter_num, train=True)

        if self.cl_param:
            cl_length = self.curriculum_learning(epoch=epoch)
            forward_return["prediction"] = forward_return["prediction"][:, :cl_length, :, :]
            forward_return["target"] = forward_return["target"][:, :cl_length, :, :]

        score_per_sample_loss = None
        node_retained_ratio = None
        if self.dynamic_pruning_node_loss_pruning:
            per_node_loss = masked_mae_per_node(
                forward_return["prediction"], forward_return["target"], null_val=self.null_val
            )
            score_per_node_loss = self._score_per_node_loss(forward_return)
            loss, node_retained_ratio = node_hard_revisit_rescaled_loss(
                per_node_loss,
                score_per_node_loss,
                retention_ratio=self.dynamic_pruning_node_retention_ratio,
                revisit_probability=self.dynamic_pruning_node_revisit_probability,
                rescale=self.dynamic_pruning_rescale,
            )
        else:
            per_sample_loss = masked_mae_per_sample(
                forward_return["prediction"], forward_return["target"], null_val=self.null_val
            )
            score_per_sample_loss = self._score_per_sample_loss(forward_return)
            if self.dynamic_pruning_rescale:
                loss = inverse_probability_weighted_loss(
                    per_sample_loss, selection.inclusion_probabilities, original_batch_size
                )
            else:
                loss = torch.mean(per_sample_loss)

        retained_sample_indices = batch_indices[selection.selected_positions]
        if score_per_sample_loss is not None and self.dynamic_pruning_selector.should_update_scores_after_train_batch():
            self.dynamic_pruning_selector.update_scores(retained_sample_indices, score_per_sample_loss)
        selected_sample_count = int(selection.selected_positions.shape[0])
        self.dynamic_pruning_forwarded_samples += selected_sample_count
        self.dynamic_pruning_budget_counter.record_forward(selected_sample_count)
        self.dynamic_pruning_pending_backward_samples = selected_sample_count
        if self._forward_budget_exhausted():
            self.dynamic_pruning_budget_exhausted = True

        window_retained_ratio = (
            self.dynamic_pruning_epoch_retained_ratio
            if self.dynamic_pruning_selector.active_indices is not None
            else selection.retained_ratio
        )
        active_node_ratio = cluster_selection.active_node_ratio if cluster_selection is not None else 1.0
        self.update_epoch_meter("train/loss", loss.item(), original_batch_size)
        self.update_epoch_meter("train/retained_ratio", window_retained_ratio, original_batch_size)
        self.update_epoch_meter("train/window_retained_ratio", window_retained_ratio, original_batch_size)
        if self.block_random_node_window_enabled:
            node_retained_ratio = self.block_random_node_selector.active_retained_ratio
        if node_retained_ratio is not None:
            self.update_epoch_meter("train/node_retained_ratio", node_retained_ratio, original_batch_size)
        if self.block_random_node_window_enabled:
            self.update_epoch_meter(
                "train/effective_node_window_ratio",
                window_retained_ratio * node_retained_ratio,
                original_batch_size,
            )
        if cluster_selection is not None:
            self.update_epoch_meter("train/active_node_ratio", active_node_ratio, original_batch_size)
            self.update_epoch_meter(
                "train/effective_pair_ratio",
                window_retained_ratio * active_node_ratio,
                original_batch_size,
            )
            self.update_epoch_meter("train/num_active_clusters", len(cluster_selection.cluster_ids), original_batch_size)
        self.update_epoch_meter("train/forward_budget_ratio", self._forward_budget_ratio(), original_batch_size)

        weight = self._get_metric_weight(forward_return["target"])
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f"train/{metric_name}", metric_item.item(), weight)
        return loss

    def backward(self, loss: torch.Tensor):
        super().backward(loss)
        if self.dynamic_pruning_enabled and self.dynamic_pruning_pending_backward_samples > 0:
            self.dynamic_pruning_budget_counter.record_backward(self.dynamic_pruning_pending_backward_samples)
            self.dynamic_pruning_pending_backward_samples = 0

    def on_epoch_end(self, epoch: int) -> None:
        if self.dynamic_pruning_enabled:
            epoch_forward, epoch_backward, epoch_steps = self.dynamic_pruning_budget_counter.epoch_snapshot()
            total_forward, total_backward, total_steps = self.dynamic_pruning_budget_counter.total_snapshot()
            self.update_epoch_meter("train/forward_samples", epoch_forward)
            self.update_epoch_meter("train/backward_samples", epoch_backward)
            self.update_epoch_meter("train/optimizer_steps", epoch_steps)
            self.update_epoch_meter("train/total_forward_samples", total_forward)
            self.update_epoch_meter("train/total_backward_samples", total_backward)
            self.update_epoch_meter("train/total_optimizer_steps", total_steps)
            self.logger.info(
                "Dynamic pruning budget counters: epoch_forward_samples=%d, "
                "epoch_backward_samples=%d, epoch_optimizer_steps=%d, "
                "total_forward_samples=%d, total_backward_samples=%d, total_optimizer_steps=%d.",
                epoch_forward,
                epoch_backward,
                epoch_steps,
                total_forward,
                total_backward,
                total_steps,
            )
            if self.block_random_node_window_enabled:
                summary = self.block_random_node_selector.coverage_summary()
                self.update_epoch_meter("train/node_coverage_min", summary["coverage_min"])
                self.update_epoch_meter("train/node_coverage_mean", summary["coverage_mean"])
                self.update_epoch_meter("train/node_coverage_max", summary["coverage_max"])
                self.update_epoch_meter("train/node_coverage_zero_count", summary["coverage_zero_count"])
                self.logger.info(
                    "Block random node coverage: epochs=%d min=%d mean=%.4f max=%d zero_count=%d.",
                    summary["epochs"],
                    summary["coverage_min"],
                    summary["coverage_mean"],
                    summary["coverage_max"],
                    summary["coverage_zero_count"],
                )
        super().on_epoch_end(epoch)

    def check_early_stopping(self) -> bool:
        if self.dynamic_pruning_enabled and self.dynamic_pruning_budget_exhausted:
            if not self.dynamic_pruning_budget_logged:
                self.logger.info(
                    "Dynamic pruning forward budget exhausted: %s/%s samples.",
                    self.dynamic_pruning_forwarded_samples,
                    self.dynamic_pruning_target_forward_samples,
                )
                self.dynamic_pruning_budget_logged = True
            return True
        return super().check_early_stopping()

    @staticmethod
    def _indexed_dataset_type(dataset_type):
        try:
            if issubclass(dataset_type, CoresetTimeSeriesForecastingDataset):
                return IndexedCoresetTimeSeriesForecastingDataset
            if issubclass(dataset_type, TimeSeriesForecastingDataset):
                return IndexedTimeSeriesForecastingDataset
        except TypeError:
            pass
        return dataset_type

    @staticmethod
    def _to_long_tensor(indices) -> torch.Tensor:
        if isinstance(indices, torch.Tensor):
            return indices.detach().cpu().long()
        return torch.as_tensor(indices, dtype=torch.long)

    @staticmethod
    def _slice_batch(data: Dict, positions: torch.Tensor) -> Dict:
        positions = positions.detach().cpu().long()
        batch_size = int(data["index"].shape[0]) if hasattr(data["index"], "shape") else len(data["index"])
        sliced = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor) and value.shape and value.shape[0] == batch_size:
                sliced[key] = value.index_select(0, positions.to(value.device))
            elif isinstance(value, np.ndarray) and value.shape and value.shape[0] == batch_size:
                sliced[key] = value[positions.numpy()]
            else:
                sliced[key] = value
        return sliced

    @staticmethod
    def _slice_nodes(data: Dict, node_indices) -> Dict:
        if "inputs" not in data:
            raise ValueError("Cluster subgraph slicing requires an 'inputs' tensor.")
        node_indices = torch.as_tensor(node_indices, dtype=torch.long)
        num_nodes = int(data["inputs"].shape[2])
        sliced = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor) and value.ndim >= 3 and int(value.shape[2]) == num_nodes:
                sliced[key] = value.index_select(2, node_indices.to(value.device))
            elif isinstance(value, np.ndarray) and value.ndim >= 3 and int(value.shape[2]) == num_nodes:
                sliced[key] = value[:, :, node_indices.numpy(), ...]
            else:
                sliced[key] = value
        sliced["idx"] = node_indices
        return sliced

    def _init_cluster_subgraph(self) -> None:
        assignment = self._build_cluster_assignment()
        num_active_clusters = int(self.cluster_subgraph_cfg.get("NUM_ACTIVE_CLUSTERS", 2))
        seed = int(self.cluster_subgraph_cfg.get("SEED", self.dynamic_pruning_cfg.get("SEED", 2023)))
        self.cluster_subgraph_scheduler = ClusterSubgraphScheduler(
            assignment=assignment,
            num_active_clusters=num_active_clusters,
            seed=seed,
        )
        if hasattr(self, "logger") and self.logger is not None:
            sizes = [len(cluster) for cluster in assignment.clusters]
            self.logger.info(
                "Cluster subgraph initialized: type=%s clusters=%d active_clusters=%d "
                "nodes=%d size_min=%d size_max=%d.",
                assignment.cluster_type,
                assignment.num_clusters,
                num_active_clusters,
                len(assignment.labels),
                min(sizes),
                max(sizes),
            )

    def _build_cluster_assignment(self):
        dataset = self._unwrap_dataset(self.dynamic_pruning_full_train_data_loader.dataset)
        dataset_name = str(getattr(dataset, "dataset_name", self.cluster_subgraph_cfg.get("DATASET", "")))
        cluster_type = str(self.cluster_subgraph_cfg.get("TYPE", "spatial_kdtree")).lower()
        num_clusters = int(self.cluster_subgraph_cfg.get("NUM_CLUSTERS", 8))
        seed = int(self.cluster_subgraph_cfg.get("SEED", self.dynamic_pruning_cfg.get("SEED", 2023)))
        cache_dir = self.cluster_subgraph_cfg.get(
            "CACHE_DIR",
            os.path.join("datasets", dataset_name, "cluster_cache"),
        )
        cache_path = cluster_assignment_cache_path(
            cache_dir=cache_dir,
            dataset_name=dataset_name,
            cluster_type=cluster_type,
            num_clusters=num_clusters,
            seed=seed,
        )
        if bool(self.cluster_subgraph_cfg.get("USE_CACHE", True)) and os.path.exists(cache_path):
            return load_cluster_assignment(cache_path)

        if cluster_type == "spatial_kdtree":
            meta_csv = self._resolve_spatial_meta_csv(dataset_name)
            coords = load_spatial_coordinates(meta_csv)
            assignment = build_balanced_spatial_kdtree_clusters(
                coords,
                num_clusters=num_clusters,
                dataset_name=dataset_name,
                seed=seed,
            )
        elif cluster_type == "signal_kmeans":
            steps_per_day = int(self.cluster_subgraph_cfg.get("STEPS_PER_DAY", 96))
            max_iter = int(self.cluster_subgraph_cfg.get("KMEANS_MAX_ITER", 50))
            assignment = build_balanced_signal_kmeans_clusters(
                dataset.data,
                num_clusters=num_clusters,
                dataset_name=dataset_name,
                seed=seed,
                steps_per_day=steps_per_day,
                max_iter=max_iter,
            )
        elif cluster_type in {"random_balanced", "random_balanced_clusters"}:
            num_nodes = int(dataset.data.shape[1])
            assignment = build_random_balanced_clusters(
                num_nodes=num_nodes,
                num_clusters=num_clusters,
                dataset_name=dataset_name,
                seed=seed,
            )
        else:
            raise ValueError(
                "CLUSTER_SUBGRAPH.TYPE must be one of "
                "{'spatial_kdtree', 'signal_kmeans', 'random_balanced_clusters'}, "
                f"got {cluster_type!r}."
            )

        if bool(self.cluster_subgraph_cfg.get("USE_CACHE", True)):
            save_cluster_assignment(assignment, cache_path)
        return assignment

    def _init_block_random_node_window(self) -> None:
        if self.cluster_subgraph_enabled:
            raise ValueError("block_random_node_window cannot be combined with CLUSTER_SUBGRAPH in this runner.")

        dataset = self._unwrap_dataset(self.dynamic_pruning_full_train_data_loader.dataset)
        if not hasattr(dataset, "data"):
            raise ValueError("block_random_node_window requires a train dataset with a `data` array.")
        num_nodes = int(dataset.data.shape[1])
        seed = int(self.dynamic_pruning_cfg.get("SEED", 2023))
        self.block_random_node_selector = RandomNodeSubsetSelector(
            num_nodes=num_nodes,
            node_retention_ratio=self.dynamic_pruning_node_retention_ratio,
            seed=seed,
        )
        if hasattr(self, "logger") and self.logger is not None:
            self.logger.info(
                "Block random node-window initialized: num_nodes=%d node_retention_ratio=%.4f.",
                num_nodes,
                self.dynamic_pruning_node_retention_ratio,
            )

    def _resolve_spatial_meta_csv(self, dataset_name: str) -> str:
        explicit_meta_csv = self.cluster_subgraph_cfg.get("META_CSV", None)
        if explicit_meta_csv:
            if os.path.exists(explicit_meta_csv):
                return explicit_meta_csv
            raise FileNotFoundError(f"Configured CLUSTER_SUBGRAPH.META_CSV does not exist: {explicit_meta_csv}")

        lower_name = str(dataset_name).lower()
        candidates = [
            os.path.join("datasets", dataset_name, "meta.csv"),
            os.path.join("BasicTS", "datasets", dataset_name, "meta.csv"),
            os.path.join("PatchSTG", "data", dataset_name, f"{lower_name}_meta.csv"),
            os.path.join("..", "PatchSTG", "data", dataset_name, f"{lower_name}_meta.csv"),
            os.path.join("LargeST", "experiments", "patchstg", "data", dataset_name, f"{lower_name}_meta.csv"),
            os.path.join("..", "LargeST", "experiments", "patchstg", "data", dataset_name, f"{lower_name}_meta.csv"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(
            "Could not resolve spatial meta csv for cluster_subgraph. Tried: "
            + ", ".join(candidates)
            + ". Set CLUSTER_SUBGRAPH_META_CSV to an explicit Lat/Lng csv path."
        )

    @staticmethod
    def _unwrap_dataset(dataset):
        while isinstance(dataset, Subset):
            dataset = dataset.dataset
        return dataset

    def _begin_cluster_subgraph_epoch(self, epoch: int) -> None:
        self.cluster_subgraph_epoch_force_full = (
            self.dynamic_pruning_selector.force_full_data
            or self.dynamic_pruning_selector._is_full_data_epoch(epoch)
        )
        self.cluster_subgraph_scheduler.begin_epoch(
            epoch=epoch,
            force_full=self.cluster_subgraph_epoch_force_full,
        )

    def _select_cluster_subgraph(self, epoch: int, iter_index: int):
        del epoch
        if not self.cluster_subgraph_enabled:
            return None
        return self.cluster_subgraph_scheduler.select(iter_index)

    def _update_proxy_gap_reference_scores_for_active_nodes(self, batch_indices: torch.Tensor, cluster_selection) -> None:
        del batch_indices
        if self.dynamic_pruning_selector.strategy != "proxy_gap":
            return
        if cluster_selection.active_node_ratio >= 1.0:
            return
        if not self.cluster_subgraph_reference_node_scores:
            return

        combo_key = tuple(sorted(int(cluster_id) for cluster_id in cluster_selection.cluster_ids))
        reference_scores = self.cluster_subgraph_reference_score_cache.get(combo_key)
        if reference_scores is None:
            reference_scores = {}
            for sample_index, node_scores in self.cluster_subgraph_reference_node_scores.items():
                reduced_score = reduce_node_scores_to_samples(
                    node_scores.unsqueeze(0),
                    cluster_selection.node_indices,
                ).squeeze(0)
                reference_scores[int(sample_index)] = float(reduced_score.item())
            self.cluster_subgraph_reference_score_cache[combo_key] = reference_scores
        self.dynamic_pruning_selector.reference_score_memory.update(reference_scores)

    def _update_cluster_subgraph_reference_node_scores(
        self,
        batch_indices: torch.Tensor,
        node_losses: torch.Tensor,
    ) -> None:
        if not self.cluster_subgraph_enabled:
            return
        for position, sample_index in enumerate(batch_indices.detach().cpu().long().tolist()):
            self.cluster_subgraph_reference_node_scores[int(sample_index)] = node_losses[position].detach().cpu().float()

    def _train_dataset_indices(self):
        dataset = self.dynamic_pruning_full_train_data_loader.dataset
        if hasattr(dataset, "resolve_index"):
            return [dataset.resolve_index(index) for index in range(len(dataset))]
        return list(range(len(dataset)))

    def _refresh_epoch_train_data_loader(self) -> None:
        subset = self.dynamic_pruning_selector.active_dataset_positions()
        if subset is None:
            self.train_data_loader = self.dynamic_pruning_full_train_data_loader
            self.dynamic_pruning_epoch_retained_ratio = 1.0
            return

        dataset = Subset(
            self.dynamic_pruning_full_train_data_loader.dataset,
            subset.dataset_positions.tolist(),
        )
        self.train_data_loader = self._build_dynamic_pruning_train_data_loader(dataset)
        self.dynamic_pruning_epoch_retained_ratio = subset.retained_ratio
        if hasattr(self, "logger") and self.logger is not None:
            self.logger.info(
                "Dynamic pruning epoch subset: retained=%d/%d ratio=%.4f batches=%d.",
                len(dataset),
                len(self.dynamic_pruning_full_train_data_loader.dataset),
                subset.retained_ratio,
                len(self.train_data_loader),
            )

    def _build_dynamic_pruning_train_data_loader(self, dataset):
        if torch.distributed.is_initialized():
            return build_data_loader_ddp(dataset, self.dynamic_pruning_train_data_cfg)
        return build_data_loader(dataset, self.dynamic_pruning_train_data_cfg)

    def _init_forward_budget(self) -> None:
        self.dynamic_pruning_forwarded_samples = 0
        self.dynamic_pruning_budget_exhausted = False
        self.dynamic_pruning_budget_logged = False
        self.dynamic_pruning_target_forward_samples = None

        if self.dynamic_pruning_target_forward_ratio is None:
            return

        target_ratio = float(self.dynamic_pruning_target_forward_ratio)
        if not 0.0 < target_ratio <= 1.0:
            raise ValueError(f"TARGET_FORWARD_RATIO must be in (0, 1], got {target_ratio}.")

        dataset_size = len(self.dynamic_pruning_full_train_data_loader.dataset)
        reference_num_epochs = (
            int(self.dynamic_pruning_reference_num_epochs)
            if self.dynamic_pruning_reference_num_epochs is not None
            else int(self.num_epochs)
        )
        full_forward_samples = int(dataset_size) * reference_num_epochs
        self.dynamic_pruning_target_forward_samples = max(1, int(round(full_forward_samples * target_ratio)))
        if self.dynamic_pruning_final_full_forward_ratio < 0.0 or self.dynamic_pruning_final_full_forward_ratio > 1.0:
            raise ValueError(
                "FINAL_FULL_FORWARD_RATIO must be in [0, 1], "
                f"got {self.dynamic_pruning_final_full_forward_ratio}."
            )
        if self.dynamic_pruning_final_full_forward_ratio > 0.0:
            self.dynamic_pruning_final_full_forward_samples = max(
                1,
                int(round(self.dynamic_pruning_target_forward_samples * self.dynamic_pruning_final_full_forward_ratio)),
            )

    def _remaining_forward_budget(self):
        if self.dynamic_pruning_target_forward_samples is None:
            return None
        return self.dynamic_pruning_target_forward_samples - self.dynamic_pruning_forwarded_samples

    def _forward_budget_exhausted(self) -> bool:
        remaining = self._remaining_forward_budget()
        return remaining is not None and remaining <= 0

    def _forward_budget_ratio(self) -> float:
        if self.dynamic_pruning_target_forward_samples is None:
            return 0.0
        return min(1.0, float(self.dynamic_pruning_forwarded_samples) / float(self.dynamic_pruning_target_forward_samples))

    def _use_forward_budget_final_full_data(self) -> bool:
        if self.dynamic_pruning_target_forward_samples is None:
            return False
        if self.dynamic_pruning_final_full_forward_samples is None:
            return False
        return self._remaining_forward_budget() <= self.dynamic_pruning_final_full_forward_samples

    def _limit_selection_to_forward_budget(self, selection, original_batch_size: int):
        remaining = self._remaining_forward_budget()
        if remaining is None:
            return selection
        return limit_selection_to_forward_budget(selection, remaining, original_batch_size)

    @torch.no_grad()
    def _score_full_training_dataset_for_pruning(self, epoch: int) -> None:
        was_training = self.model.training
        self.model.eval()
        scored_samples = 0

        for iter_index, data in enumerate(self.dynamic_pruning_full_train_data_loader):
            if not isinstance(data, dict) or "index" not in data:
                raise ValueError("Dynamic pruning scoring requires a dataset that returns batch indices.")

            batch_indices = self._to_long_tensor(data["index"])
            forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_index, train=False)

            if self.cl_param:
                cl_length = self.curriculum_learning(epoch=epoch)
                forward_return["prediction"] = forward_return["prediction"][:, :cl_length, :, :]
                forward_return["target"] = forward_return["target"][:, :cl_length, :, :]

            per_sample_loss = masked_mae_per_sample(
                forward_return["prediction"], forward_return["target"], null_val=self.null_val
            )
            score_per_sample_loss = self._score_per_sample_loss(forward_return)
            self.dynamic_pruning_selector.update_scores(batch_indices, score_per_sample_loss)
            scored_samples += int(batch_indices.shape[0])

        if was_training:
            self.model.train()
        self.logger.info("Dynamic pruning scored %d training samples at epoch %d.", scored_samples, epoch)

    def _score_per_sample_loss(self, forward_return: Dict) -> torch.Tensor:
        if self.dynamic_pruning_score_loss_type in {"raw_mae", "mae", ""}:
            return masked_mae_per_sample(forward_return["prediction"], forward_return["target"], null_val=self.null_val)
        if self.dynamic_pruning_score_loss_type in {"normalized_mae", "zscore_mae", "norm_mae"}:
            return self._normalized_score_loss(forward_return["prediction"], forward_return["target"])
        if self.dynamic_pruning_score_loss_type in {"lowpass_normalized_mae", "fft_lowpass_normalized_mae"}:
            return self._lowpass_normalized_score_loss(forward_return["prediction"], forward_return["target"])
        raise ValueError(f"Unsupported DYNAMIC_PRUNING.SCORE_LOSS_TYPE: {self.dynamic_pruning_score_loss_type}.")

    def _score_per_node_loss(self, forward_return: Dict) -> torch.Tensor:
        if self.dynamic_pruning_score_loss_type in {"raw_mae", "mae", ""}:
            return masked_mae_per_node(forward_return["prediction"], forward_return["target"], null_val=self.null_val)
        if self.dynamic_pruning_score_loss_type in {"normalized_mae", "zscore_mae", "norm_mae"}:
            return self._normalized_score_loss_per_node(forward_return["prediction"], forward_return["target"])
        if self.dynamic_pruning_score_loss_type in {"lowpass_normalized_mae", "fft_lowpass_normalized_mae"}:
            return self._lowpass_normalized_score_loss_per_node(forward_return["prediction"], forward_return["target"])
        raise ValueError(f"Unsupported DYNAMIC_PRUNING.SCORE_LOSS_TYPE: {self.dynamic_pruning_score_loss_type}.")

    def _normalized_score_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.scaler is None or not hasattr(self.scaler, "mean") or not hasattr(self.scaler, "std"):
            warnings.warn(
                "normalized pruning score requested but scaler mean/std are unavailable; falling back to raw MAE.",
                stacklevel=2,
            )
            return masked_mae_per_sample(prediction, target, null_val=self.null_val)
        return normalized_masked_mae_per_sample(
            prediction,
            target,
            mean=self.scaler.mean,
            std=self.scaler.std,
            null_val=self.null_val,
        )

    def _normalized_score_loss_per_node(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.scaler is None or not hasattr(self.scaler, "mean") or not hasattr(self.scaler, "std"):
            warnings.warn(
                "normalized pruning score requested but scaler mean/std are unavailable; falling back to raw MAE.",
                stacklevel=2,
            )
            return masked_mae_per_node(prediction, target, null_val=self.null_val)
        return normalized_masked_mae_per_node(
            prediction,
            target,
            mean=self.scaler.mean,
            std=self.scaler.std,
            null_val=self.null_val,
        )

    def _lowpass_normalized_score_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.scaler is None or not hasattr(self.scaler, "mean") or not hasattr(self.scaler, "std"):
            warnings.warn(
                "lowpass normalized pruning score requested but scaler mean/std are unavailable; falling back to raw MAE.",
                stacklevel=2,
            )
            return masked_mae_per_sample(prediction, target, null_val=self.null_val)
        return lowpass_normalized_masked_mae_per_sample(
            prediction,
            target,
            mean=self.scaler.mean,
            std=self.scaler.std,
            null_val=self.null_val,
            lowpass_keep_ratio=self.dynamic_pruning_lowpass_keep_ratio,
        )

    def _lowpass_normalized_score_loss_per_node(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.scaler is None or not hasattr(self.scaler, "mean") or not hasattr(self.scaler, "std"):
            warnings.warn(
                "lowpass normalized pruning score requested but scaler mean/std are unavailable; falling back to raw MAE.",
                stacklevel=2,
            )
            return masked_mae_per_node(prediction, target, null_val=self.null_val)
        return lowpass_normalized_masked_mae_per_node(
            prediction,
            target,
            mean=self.scaler.mean,
            std=self.scaler.std,
            null_val=self.null_val,
            lowpass_keep_ratio=self.dynamic_pruning_lowpass_keep_ratio,
        )

    def _init_reference_scores_for_pruning(self) -> None:
        if self.dynamic_pruning_selector.strategy != "proxy_gap":
            return
        if self.dynamic_pruning_reference_type == "seasonal":
            with torch.no_grad():
                self._init_seasonal_reference_scores_for_pruning()
            return
        if self.dynamic_pruning_reference_type == "dlinear":
            self._init_dlinear_reference_scores_for_pruning()
            return
        raise ValueError(
            "proxy_gap reference must be one of {'seasonal', 'dlinear'}, "
            f"got {self.dynamic_pruning_reference_type}."
        )

    def _init_seasonal_reference_scores_for_pruning(self) -> None:
        target_by_index = {}
        raw_batches = []
        for data in self.dynamic_pruning_full_train_data_loader:
            if not isinstance(data, dict) or "index" not in data:
                raise ValueError("proxy_gap reference scoring requires a dataset that returns batch indices.")
            batch_indices = self._to_long_tensor(data["index"])
            target = self._reference_target_tensor(data["target"])
            target_by_index.update(
                {
                    int(sample_index): target[position].detach().cpu()
                    for position, sample_index in enumerate(batch_indices.tolist())
                }
            )
            raw_batches.append((batch_indices, data))

        scored_samples = 0
        for batch_indices, data in raw_batches:
            target = self._reference_target_tensor(data["target"])
            prediction = []
            fallback = self._persistence_reference_prediction(data["inputs"], target.shape[1])
            for position, sample_index in enumerate(batch_indices.tolist()):
                seasonal_target = target_by_index.get(int(sample_index) - self.dynamic_pruning_reference_period)
                if seasonal_target is None or list(seasonal_target.shape) != list(target[position].shape):
                    prediction.append(fallback[position])
                else:
                    prediction.append(seasonal_target.to(target.device, dtype=target.dtype))
            prediction = torch.stack(prediction, dim=0)
            reference_losses = self._normalized_score_loss(prediction, target)
            reference_node_losses = self._score_per_node_loss(
                {"prediction": prediction, "target": target}
            )
            self.dynamic_pruning_selector.update_reference_scores(batch_indices, reference_losses)
            self._update_cluster_subgraph_reference_node_scores(batch_indices, reference_node_losses)
            scored_samples += int(batch_indices.shape[0])

        if hasattr(self, "logger") and self.logger is not None:
            self.logger.info(
                "Dynamic pruning initialized %d proxy_gap reference scores using %s(period=%d).",
                scored_samples,
                self.dynamic_pruning_reference_type,
                self.dynamic_pruning_reference_period,
            )

    def _init_dlinear_reference_scores_for_pruning(self) -> None:
        try:
            from baselines.DLinear.arch import DLinear
        except ImportError as exc:
            raise ImportError("DLinear proxy_gap reference requires baselines.DLinear.arch.DLinear.") from exc

        first_batch = next(iter(self.dynamic_pruning_full_train_data_loader))
        history, target = self._dlinear_reference_training_batch(first_batch)
        _, history_len, num_nodes, _ = history.shape
        horizon = int(target.shape[1])
        reference_model = DLinear(
            seq_len=int(history_len),
            pred_len=horizon,
            enc_in=int(num_nodes),
            individual=self.dynamic_pruning_dlinear_individual,
        ).to(history.device)
        optimizer = torch.optim.Adam(
            reference_model.parameters(),
            lr=self.dynamic_pruning_dlinear_lr,
            weight_decay=self.dynamic_pruning_dlinear_weight_decay,
        )

        reference_model.train()
        best_val_loss = float("inf")
        best_state = None
        best_epoch = 0
        stale_epochs = 0
        final_epoch = 0
        for reference_epoch in range(1, self.dynamic_pruning_dlinear_epochs + 1):
            final_epoch = reference_epoch
            total_loss = 0.0
            total_samples = 0
            for data in self.dynamic_pruning_full_train_data_loader:
                history, target = self._dlinear_reference_training_batch(data)
                optimizer.zero_grad(set_to_none=True)
                prediction = reference_model(
                    history_data=history,
                    future_data=target,
                    batch_seen=None,
                    epoch=reference_epoch,
                    train=True,
                )
                loss = masked_mae_per_sample(prediction, target, null_val=self.null_val).mean()
                loss.backward()
                optimizer.step()
                batch_size = int(target.shape[0])
                total_loss += float(loss.item()) * batch_size
                total_samples += batch_size
            train_loss = total_loss / max(1, total_samples)
            val_loss = self._evaluate_dlinear_reference_model(reference_model)
            if hasattr(self, "logger") and self.logger is not None:
                self.logger.info(
                    "Dynamic pruning DLinear reference epoch %d/%d: normalized_train_mae=%.6f, "
                    "normalized_val_mae=%.6f.",
                    reference_epoch,
                    self.dynamic_pruning_dlinear_epochs,
                    train_loss,
                    val_loss,
                )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = reference_epoch
                best_state = {key: value.detach().cpu().clone() for key, value in reference_model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if self.dynamic_pruning_dlinear_patience > 0 and stale_epochs >= self.dynamic_pruning_dlinear_patience:
                    if hasattr(self, "logger") and self.logger is not None:
                        self.logger.info(
                            "Dynamic pruning DLinear reference early stop at epoch %d "
                            "(best_epoch=%d, best_val_mae=%.6f, patience=%d).",
                            reference_epoch,
                            best_epoch,
                            best_val_loss,
                            self.dynamic_pruning_dlinear_patience,
                        )
                    break

        if best_state is not None:
            reference_model.load_state_dict(best_state)
        if hasattr(self, "logger") and self.logger is not None:
            self.logger.info(
                "Dynamic pruning DLinear reference selected epoch %d/%d after %d trained epochs.",
                best_epoch,
                self.dynamic_pruning_dlinear_epochs,
                final_epoch,
            )

        reference_model.eval()
        scored_samples = 0
        with torch.no_grad():
            for data in self.dynamic_pruning_full_train_data_loader:
                if not isinstance(data, dict) or "index" not in data:
                    raise ValueError("proxy_gap DLinear reference scoring requires a dataset that returns batch indices.")
                batch_indices = self._to_long_tensor(data["index"])
                history, normalized_target = self._dlinear_reference_training_batch(data)
                normalized_prediction = reference_model(
                    history_data=history,
                    future_data=normalized_target,
                    batch_seen=None,
                    epoch=None,
                    train=False,
                )
                prediction = self._inverse_reference_prediction(normalized_prediction)
                target = self._reference_target_tensor(data["target"])
                reference_losses = self._normalized_score_loss(prediction, target)
                reference_node_losses = self._score_per_node_loss(
                    {"prediction": prediction, "target": target}
                )
                self.dynamic_pruning_selector.update_reference_scores(batch_indices, reference_losses)
                self._update_cluster_subgraph_reference_node_scores(batch_indices, reference_node_losses)
                scored_samples += int(batch_indices.shape[0])

        if hasattr(self, "logger") and self.logger is not None:
            self.logger.info(
                "Dynamic pruning initialized %d proxy_gap reference scores using DLinear(epochs=%d, lr=%.2e).",
                scored_samples,
                final_epoch,
                self.dynamic_pruning_dlinear_lr,
            )

    @torch.no_grad()
    def _evaluate_dlinear_reference_model(self, reference_model) -> float:
        data_loader = getattr(self, "val_data_loader", None) or self.dynamic_pruning_full_train_data_loader
        was_training = reference_model.training
        reference_model.eval()
        total_loss = 0.0
        total_samples = 0
        for data in data_loader:
            history, target = self._dlinear_reference_training_batch(data)
            prediction = reference_model(
                history_data=history,
                future_data=target,
                batch_seen=None,
                epoch=None,
                train=False,
            )
            losses = masked_mae_per_sample(prediction, target, null_val=self.null_val)
            batch_size = int(target.shape[0])
            total_loss += float(losses.mean().item()) * batch_size
            total_samples += batch_size
        if was_training:
            reference_model.train()
        return total_loss / max(1, total_samples)

    def _reference_target_tensor(self, target: torch.Tensor) -> torch.Tensor:
        target = self.to_running_device(target)
        target = self.select_target_features(target)
        if self.target_time_series is not None:
            target = self.select_target_time_series(target)
        return target

    def _dlinear_reference_training_batch(self, data: Dict):
        history = self.to_running_device(data["inputs"])
        target = self.to_running_device(data["target"])
        if self.scaler is not None:
            history = self.scaler.transform(history)
            target = self.scaler.transform(target)
        history = self.select_target_features(history)
        target = self.select_target_features(target)
        if self.target_time_series is not None:
            history = self.select_target_time_series(history)
            target = self.select_target_time_series(target)
        return history, target

    def _inverse_reference_prediction(self, prediction: torch.Tensor) -> torch.Tensor:
        if self.scaler is None:
            return prediction
        return self.scaler.inverse_transform(prediction)

    def _persistence_reference_prediction(self, inputs: torch.Tensor, horizon: int) -> torch.Tensor:
        inputs = self.to_running_device(inputs)
        inputs = self.select_target_features(inputs)
        if self.target_time_series is not None:
            inputs = self.select_target_time_series(inputs)
        return inputs[:, -1:, :, :].repeat(1, horizon, 1, 1)


class DynamicPruningTimeSeriesForecastingRunner(_DynamicPruningRunnerMixin, SimpleTimeSeriesForecastingRunner):
    """Simple forecasting runner with opt-in dynamic batch pruning."""


class DynamicPruningWandBTimeSeriesForecastingRunner(_DynamicPruningRunnerMixin, WandBTimeSeriesForecastingRunner):
    """WandB forecasting runner with opt-in dynamic batch pruning."""
