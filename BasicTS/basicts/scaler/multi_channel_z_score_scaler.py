import json

import numpy as np
import torch

from .base_scaler import BaseScaler


class MultiChannelZScoreScaler(BaseScaler):
    """Z-score scaler for multi-feature inputs with target-only inverse support.

    Statistics are fitted on the training split. When ``norm_each_channel`` is
    false, each feature channel gets one global mean/std across time and nodes.
    When true, each node-feature pair gets its own mean/std.
    """

    def __init__(
        self,
        dataset_name: str,
        train_ratio: float,
        norm_each_channel: bool,
        rescale: bool,
        target_channel: int = 0,
    ):
        super().__init__(dataset_name, train_ratio, norm_each_channel, rescale)
        self.target_channel = int(target_channel)

        description_file_path = f"datasets/{dataset_name}/desc.json"
        with open(description_file_path, "r") as f:
            description = json.load(f)
        data_file_path = f"datasets/{dataset_name}/data.dat"
        data = np.memmap(data_file_path, dtype="float32", mode="r", shape=tuple(description["shape"]))

        train_size = int(len(data) * train_ratio)
        train_data = data[:train_size].copy()
        if train_data.shape[-1] <= self.target_channel:
            raise ValueError(
                f"target_channel={self.target_channel} is outside feature dimension {train_data.shape[-1]}."
            )

        if norm_each_channel:
            mean = np.mean(train_data, axis=0, keepdims=True)
            std = np.std(train_data, axis=0, keepdims=True)
        else:
            mean = np.mean(train_data, axis=(0, 1), keepdims=True)
            std = np.std(train_data, axis=(0, 1), keepdims=True)
        std[std == 0] = 1.0

        self.num_features = int(train_data.shape[-1])
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def _select_stats(self, input_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_features = int(input_data.shape[-1])
        mean = self.mean.to(input_data.device)
        std = self.std.to(input_data.device)
        if num_features == self.num_features:
            return mean, std
        if num_features == 1:
            return mean[..., self.target_channel:self.target_channel + 1], std[..., self.target_channel:self.target_channel + 1]
        if num_features < self.num_features:
            return mean[..., :num_features], std[..., :num_features]
        raise ValueError(
            f"Input feature dimension {num_features} exceeds fitted feature dimension {self.num_features}."
        )

    def transform(self, input_data: torch.Tensor) -> torch.Tensor:
        mean, std = self._select_stats(input_data)
        return (input_data - mean) / std

    def inverse_transform(self, input_data: torch.Tensor) -> torch.Tensor:
        mean, std = self._select_stats(input_data)
        return input_data * std + mean
