import json
import logging
from pathlib import Path
from typing import List

import numpy as np
from torch.utils.data import Dataset


class FlowNetOfficialDataset(Dataset):
    """BasicTS-style dataset that mirrors FlowNet's official PEMS04F split indices."""

    def __init__(
        self,
        dataset_name: str,
        train_val_test_ratio: List[float],
        mode: str,
        input_len: int,
        output_len: int,
        memmap: bool = False,
        overlap: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        del overlap, logger
        if mode not in {"train", "valid", "test"}:
            raise ValueError(f"Invalid mode: {mode}.")

        self.dataset_name = dataset_name
        self.mode = mode
        self.input_len = input_len
        self.output_len = output_len
        self.memmap = memmap

        dataset_path = Path("datasets") / dataset_name
        with (dataset_path / "desc.json").open("r", encoding="utf-8") as f:
            self.description = json.load(f)
        self.shape = tuple(self.description["shape"])
        self.data = np.memmap(dataset_path / "data.dat", dtype="float32", mode="r", shape=self.shape)
        self.indices = self._build_indices(train_val_test_ratio)

    def _build_indices(self, train_val_test_ratio: List[float]) -> np.ndarray:
        data_len = self.shape[0]
        train_val_point = int(np.round(data_len * train_val_test_ratio[0]))
        val_test_point = int(np.round(data_len * (train_val_test_ratio[0] + train_val_test_ratio[1])))
        seq_len = self.input_len + self.output_len

        if self.mode == "train":
            return np.arange(0, train_val_point - seq_len, step=1, dtype=np.int64)
        if self.mode == "valid":
            return np.arange(
                train_val_point - self.input_len,
                val_test_point - seq_len,
                step=self.output_len,
                dtype=np.int64,
            )
        return np.arange(val_test_point - self.input_len, data_len - seq_len + 1, step=self.output_len, dtype=np.int64)

    def __getitem__(self, index: int) -> dict:
        start = int(self.indices[index])
        history_data = self.data[start : start + self.input_len]
        future_data = self.data[start + self.input_len : start + self.input_len + self.output_len]
        if not self.memmap:
            history_data = history_data.copy()
            future_data = future_data.copy()
        return {"inputs": history_data, "target": future_data}

    def __len__(self) -> int:
        return len(self.indices)
