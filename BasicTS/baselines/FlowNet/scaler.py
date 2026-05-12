from pathlib import Path

import numpy as np
import torch

from basicts.scaler.base_scaler import BaseScaler


class FlowNetOfficialScaler(BaseScaler):
    """Use FlowNet's released full-data per-node mean/std files."""

    def __init__(
        self,
        dataset_name: str,
        train_ratio: float,
        norm_each_channel: bool,
        rescale: bool,
        mean_file: str = "x_mean.npy",
        std_file: str = "x_std.npy",
        target_channel: int = 0,
    ) -> None:
        super().__init__(dataset_name, train_ratio, norm_each_channel, rescale)
        self.target_channel = target_channel
        dataset_path = Path("datasets") / dataset_name
        mean = np.load(dataset_path / mean_file).astype("float32")
        std = np.load(dataset_path / std_file).astype("float32")
        std[std == 0] = 1.0
        self.mean = torch.from_numpy(mean)
        self.std = torch.from_numpy(std)

    def transform(self, input_data: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(input_data.device)
        std = self.std.to(input_data.device)
        input_data[..., self.target_channel] = (input_data[..., self.target_channel] - mean) / (std + 1e-6)
        return input_data

    def inverse_transform(self, input_data: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(input_data.device)
        std = self.std.to(input_data.device)
        input_data = input_data.clone()
        input_data[..., self.target_channel] = input_data[..., self.target_channel] * (std + 1e-6) + mean
        return input_data
