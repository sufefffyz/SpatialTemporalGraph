import json
import os

import numpy as np
import torch

from basicts.scaler.base_scaler import BaseScaler


class ExplicitSplitZScoreScaler(BaseScaler):
    """Z-score scaler fitted on explicit training timestamps."""

    def __init__(
        self,
        dataset_name: str,
        train_ratio: float,
        norm_each_channel: bool,
        rescale: bool,
        split_filename: str = "split_indices.npz",
        target_channel: int = 0,
    ):
        super().__init__(dataset_name, train_ratio, norm_each_channel, rescale)
        self.target_channel = target_channel

        base_dir = os.path.join("datasets", dataset_name)
        with open(os.path.join(base_dir, "desc.json"), "r") as fp:
            description = json.load(fp)
        data = np.memmap(
            os.path.join(base_dir, "data.dat"),
            dtype="float32",
            mode="r",
            shape=tuple(description["shape"]),
        )
        with np.load(os.path.join(base_dir, split_filename)) as split_data:
            train_idx = np.asarray(split_data["train_idx"], dtype=np.int64)

        train_data = np.asarray(data[train_idx, :, self.target_channel]).copy()
        if norm_each_channel:
            mean = np.mean(train_data, axis=0, keepdims=True)
            std = np.std(train_data, axis=0, keepdims=True)
            std[std == 0] = 1.0
        else:
            mean = np.mean(train_data)
            std = np.std(train_data)
            if std == 0:
                std = 1.0

        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def transform(self, input_data: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(input_data.device)
        std = self.std.to(input_data.device)
        input_data = input_data.clone()
        input_data[..., self.target_channel] = (input_data[..., self.target_channel] - mean) / std
        return input_data

    def inverse_transform(self, input_data: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(input_data.device)
        std = self.std.to(input_data.device)
        input_data = input_data.clone()
        input_data[..., self.target_channel] = input_data[..., self.target_channel] * std + mean
        return input_data
