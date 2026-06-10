import json
import os
from typing import Dict, Iterable, List, Optional

import numpy as np

from .simple_tsf_dataset import TimeSeriesForecastingDataset


class CoresetTimeSeriesForecastingDataset(TimeSeriesForecastingDataset):
    """Training-window subset wrapper for BasicTS forecasting datasets.

    Validation and test modes keep the original contiguous split. Coreset
    selection is applied only to training sample start indices, so the task,
    scaler, validation set, and test set stay aligned with the base config.
    """

    def __init__(
        self,
        dataset_name: str,
        train_val_test_ratio: List[float],
        mode: str,
        input_len: int,
        output_len: int,
        memmap: bool = False,
        overlap: bool = False,
        logger=None,
        coreset_ratio: float = 1.0,
        coreset_strategy: str = "full",
        coreset_seed: int = 2023,
        coreset_temporal_period: int = 96,
        coreset_target_channel: int = 0,
        coreset_spatial_bins: int = 32,
        coreset_cache_dir: Optional[str] = None,
    ) -> None:
        self.coreset_ratio = float(coreset_ratio)
        self.coreset_strategy = str(coreset_strategy).lower()
        self.coreset_seed = int(coreset_seed)
        self.coreset_temporal_period = int(coreset_temporal_period)
        self.coreset_target_channel = int(coreset_target_channel)
        self.coreset_spatial_bins = int(coreset_spatial_bins)
        self.coreset_cache_dir = coreset_cache_dir
        super().__init__(
            dataset_name=dataset_name,
            train_val_test_ratio=train_val_test_ratio,
            mode=mode,
            input_len=input_len,
            output_len=output_len,
            memmap=memmap,
            overlap=overlap,
            logger=logger,
        )

        if self.mode == "train" and self.coreset_strategy != "full" and self.coreset_ratio < 1.0:
            self.selected_indices = self._load_or_select_indices()
            if logger is not None:
                logger.info(
                    "Coreset train subset: strategy=%s ratio=%.4f selected=%d/%d",
                    self.coreset_strategy,
                    self.coreset_ratio,
                    len(self.selected_indices),
                    super().__len__(),
                )

    def __len__(self) -> int:
        if hasattr(self, "selected_indices"):
            return len(self.selected_indices)
        return super().__len__()

    def __getitem__(self, index: int) -> dict:
        if hasattr(self, "selected_indices"):
            index = int(self.selected_indices[index])
        return super().__getitem__(index)

    def _load_or_select_indices(self) -> np.ndarray:
        total = super().__len__()
        budget = self._budget(total)
        if budget >= total:
            return np.arange(total, dtype=np.int64)

        cache_path = self._cache_path(total, budget)
        if cache_path and os.path.exists(cache_path):
            payload = np.load(cache_path)
            return np.asarray(payload["indices"], dtype=np.int64)

        rng = np.random.default_rng(self.coreset_seed)
        candidates = np.arange(total, dtype=np.int64)
        indices = self._select(candidates, budget, rng)
        indices = np.asarray(sorted(set(indices.tolist())), dtype=np.int64)
        if len(indices) != budget:
            indices = self._repair_budget(indices, candidates, budget, rng)

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez_compressed(
                cache_path,
                indices=indices,
                metadata=json.dumps(
                    {
                        "dataset_name": self.dataset_name,
                        "strategy": self.coreset_strategy,
                        "ratio": self.coreset_ratio,
                        "budget": budget,
                        "seed": self.coreset_seed,
                        "input_len": self.input_len,
                        "output_len": self.output_len,
                        "temporal_period": self.coreset_temporal_period,
                    },
                    sort_keys=True,
                ),
            )
        return indices

    def _budget(self, total: int) -> int:
        if not 0.0 < self.coreset_ratio <= 1.0:
            raise ValueError(f"coreset_ratio must be in (0, 1], got {self.coreset_ratio}.")
        return max(1, min(total, int(round(total * self.coreset_ratio))))

    def _cache_path(self, total: int, budget: int) -> Optional[str]:
        if self.coreset_cache_dir is None:
            return None
        cache_dir = self.coreset_cache_dir or os.path.join("datasets", self.dataset_name, "coreset_indices")
        name = (
            f"{self.dataset_name}_{self.coreset_strategy}"
            f"_r{self.coreset_ratio:.6f}_n{total}_k{budget}"
            f"_seed{self.coreset_seed}_in{self.input_len}_out{self.output_len}"
            f"_p{self.coreset_temporal_period}.npz"
        )
        return os.path.join(cache_dir, name)

    def _select(self, candidates: np.ndarray, budget: int, rng: np.random.Generator) -> np.ndarray:
        if self.coreset_strategy == "random":
            return self._select_random(candidates, budget, rng)
        if self.coreset_strategy == "temporal":
            return self._select_by_strata(self._temporal_strata(candidates), budget, rng, "random")
        if self.coreset_strategy == "kcenter":
            return self._select_kcenter(candidates, budget)
        if self.coreset_strategy == "temporal_kcenter":
            return self._select_by_strata(self._temporal_strata(candidates), budget, rng, "kcenter")
        if self.coreset_strategy == "temporal_kcenter_difficulty":
            return self._select_by_strata(self._temporal_difficulty_strata(candidates), budget, rng, "kcenter")
        raise ValueError(f"Unsupported coreset_strategy: {self.coreset_strategy}")

    def _select_random(self, candidates: np.ndarray, budget: int, rng: np.random.Generator) -> np.ndarray:
        if budget >= len(candidates):
            return candidates
        return np.sort(rng.choice(candidates, size=budget, replace=False))

    def _select_by_strata(
        self,
        strata: Dict[object, np.ndarray],
        budget: int,
        rng: np.random.Generator,
        inner_strategy: str,
    ) -> np.ndarray:
        allocations = self._allocate_budget(strata, budget)
        selected = []
        for key in sorted(strata, key=str):
            members = strata[key]
            count = allocations.get(key, 0)
            if count <= 0:
                continue
            if inner_strategy == "kcenter":
                selected.append(self._select_kcenter(members, count))
            else:
                selected.append(self._select_random(members, count, rng))
        if not selected:
            return np.array([], dtype=np.int64)
        return np.concatenate(selected).astype(np.int64)

    def _allocate_budget(self, strata: Dict[object, np.ndarray], budget: int) -> Dict[object, int]:
        non_empty = {key: members for key, members in strata.items() if len(members) > 0}
        total = sum(len(members) for members in non_empty.values())
        if total == 0:
            return {}

        raw = {key: budget * len(members) / total for key, members in non_empty.items()}
        allocation = {key: min(len(non_empty[key]), int(np.floor(value))) for key, value in raw.items()}
        remainder = budget - sum(allocation.values())

        order = sorted(non_empty, key=lambda key: (raw[key] - np.floor(raw[key]), len(non_empty[key])), reverse=True)
        while remainder > 0:
            progressed = False
            for key in order:
                if allocation[key] < len(non_empty[key]):
                    allocation[key] += 1
                    remainder -= 1
                    progressed = True
                    if remainder == 0:
                        break
            if not progressed:
                break
        return allocation

    def _temporal_strata(self, candidates: np.ndarray) -> Dict[int, np.ndarray]:
        period = max(1, self.coreset_temporal_period)
        buckets = (candidates + self.input_len) % period
        return self._group_by_labels(candidates, buckets)

    def _temporal_difficulty_strata(self, candidates: np.ndarray) -> Dict[tuple, np.ndarray]:
        temporal = (candidates + self.input_len) % max(1, self.coreset_temporal_period)
        difficulty = self._difficulty_scores(candidates)
        if len(np.unique(difficulty)) < 3:
            difficulty_bucket = np.ones(len(candidates), dtype=np.int64)
        else:
            thresholds = np.quantile(difficulty, [1.0 / 3.0, 2.0 / 3.0])
            difficulty_bucket = np.searchsorted(thresholds, difficulty, side="right")
        labels = list(zip(temporal.tolist(), difficulty_bucket.tolist()))
        return self._group_by_labels(candidates, labels)

    @staticmethod
    def _group_by_labels(candidates: np.ndarray, labels: Iterable) -> Dict[object, np.ndarray]:
        groups: Dict[object, list] = {}
        for index, label in zip(candidates.tolist(), labels):
            groups.setdefault(label, []).append(index)
        return {key: np.asarray(value, dtype=np.int64) for key, value in groups.items()}

    def _select_kcenter(self, candidates: np.ndarray, budget: int) -> np.ndarray:
        if budget >= len(candidates):
            return np.sort(candidates)
        embeddings = self._window_embeddings(candidates)
        center = np.mean(embeddings, axis=0, keepdims=True)
        first = int(np.argmin(np.sum((embeddings - center) ** 2, axis=1)))
        selected_positions = [first]
        min_dist = np.sum((embeddings - embeddings[first:first + 1]) ** 2, axis=1)
        min_dist[first] = -1.0

        while len(selected_positions) < budget:
            next_pos = int(np.argmax(min_dist))
            selected_positions.append(next_pos)
            dist = np.sum((embeddings - embeddings[next_pos:next_pos + 1]) ** 2, axis=1)
            min_dist = np.minimum(min_dist, dist)
            min_dist[selected_positions] = -1.0
        return np.sort(candidates[np.asarray(selected_positions, dtype=np.int64)])

    def _window_embeddings(self, candidates: np.ndarray) -> np.ndarray:
        flow = np.asarray(self.data[..., self.coreset_target_channel], dtype=np.float32)
        num_nodes = flow.shape[1]
        bins = np.array_split(np.arange(num_nodes), max(1, min(self.coreset_spatial_bins, num_nodes)))
        features = np.empty((len(candidates), self.input_len * 2 + len(bins) * 2), dtype=np.float32)
        for row, start in enumerate(candidates.tolist()):
            window = flow[start:start + self.input_len]
            offset = 0
            features[row, offset:offset + self.input_len] = window.mean(axis=1)
            offset += self.input_len
            features[row, offset:offset + self.input_len] = window.std(axis=1)
            offset += self.input_len
            for node_bin in bins:
                values = window[:, node_bin]
                features[row, offset] = values.mean()
                features[row, offset + 1] = values.std()
                offset += 2
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        return (features - mean) / std

    def _difficulty_scores(self, candidates: np.ndarray) -> np.ndarray:
        flow = np.asarray(self.data[..., self.coreset_target_channel], dtype=np.float32)
        scores = np.empty(len(candidates), dtype=np.float32)
        for row, start in enumerate(candidates.tolist()):
            history = flow[start:start + self.input_len]
            future = flow[start + self.input_len:start + self.input_len + self.output_len]
            naive = np.repeat(history[-1:, :], self.output_len, axis=0)
            scores[row] = np.mean(np.abs(future - naive))
        return scores

    def _repair_budget(
        self,
        selected: np.ndarray,
        candidates: np.ndarray,
        budget: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        selected_set = set(selected.tolist())
        if len(selected_set) < budget:
            remaining = np.asarray([idx for idx in candidates.tolist() if idx not in selected_set], dtype=np.int64)
            extra = self._select_random(remaining, budget - len(selected_set), rng)
            selected = np.concatenate([selected, extra])
        elif len(selected_set) > budget:
            selected = self._select_random(np.asarray(sorted(selected_set), dtype=np.int64), budget, rng)
        return np.asarray(sorted(set(selected.tolist())), dtype=np.int64)
