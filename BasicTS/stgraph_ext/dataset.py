import inspect
import json
import logging
import os
from typing import List

import numpy as np

from basicts.data.base_dataset import BaseDataset


class ExplicitSplitTimeSeriesForecastingDataset(BaseDataset):
    """Forecasting dataset with explicit timestamp splits.

    The dataset stores a full [L, N, C] array in BasicTS format and uses
    split_indices.npz to define train/valid/test timestamps. Training samples
    require both history and targets to stay inside the train split. Validation
    and test samples require the target window to stay inside their split while
    allowing the history window to look back into earlier timestamps.
    """

    def __init__(
        self,
        dataset_name: str,
        train_val_test_ratio: List[float],
        mode: str,
        input_len: int,
        output_len: int,
        memmap: bool = False,
        logger: logging.Logger | None = None,
        split_filename: str = "split_indices.npz",
        allow_val_test_context: bool = True,
    ) -> None:
        assert mode in ["train", "valid", "test"], f"Invalid mode: {mode}."
        super().__init__(dataset_name, train_val_test_ratio, mode, memmap)
        self.input_len = input_len
        self.output_len = output_len
        self.logger = logger
        self.split_filename = split_filename
        self.allow_val_test_context = allow_val_test_context

        self.data_file_path = os.path.join("datasets", dataset_name, "data.dat")
        self.description_file_path = os.path.join("datasets", dataset_name, "desc.json")
        self.split_file_path = os.path.join("datasets", dataset_name, split_filename)

        self.description = self._load_description()
        self.data = self._load_data()
        self.prediction_starts = self._build_prediction_starts()

    def _load_description(self) -> dict:
        with open(self.description_file_path, "r") as fp:
            return json.load(fp)

    def _load_data(self) -> np.ndarray:
        data = np.memmap(self.data_file_path, dtype="float32", mode="r", shape=tuple(self.description["shape"]))
        if not self.memmap:
            data = data.copy()
        return data

    def _load_split_indices(self) -> dict[str, np.ndarray]:
        with np.load(self.split_file_path) as data:
            return {
                "train": np.asarray(data["train_idx"], dtype=np.int64),
                "valid": np.asarray(data["val_idx"], dtype=np.int64),
                "test": np.asarray(data["test_idx"], dtype=np.int64),
            }

    def _build_prediction_starts(self) -> np.ndarray:
        split_indices = self._load_split_indices()
        split_idx = split_indices[self.mode]
        total_len = len(self.data)

        split_mask = np.zeros(total_len, dtype=np.int64)
        split_mask[split_idx] = 1
        prefix = np.concatenate([[0], np.cumsum(split_mask)])

        def window_inside_split(start: int, length: int) -> bool:
            end = start + length
            return (prefix[end] - prefix[start]) == length

        prediction_starts = []
        for target_start in split_idx:
            if target_start < self.input_len:
                continue
            if target_start + self.output_len > total_len:
                continue
            if not window_inside_split(int(target_start), self.output_len):
                continue
            history_start = int(target_start) - self.input_len
            if self.mode == "train" or not self.allow_val_test_context:
                if not window_inside_split(history_start, self.input_len):
                    continue
            prediction_starts.append(int(target_start))

        if len(prediction_starts) == 0:
            current_frame = inspect.currentframe()
            file_name = inspect.getfile(current_frame) if current_frame else __file__
            message = (
                f"No valid windows found for dataset={self.dataset_name}, mode={self.mode}, "
                f"input_len={self.input_len}, output_len={self.output_len}. "
                f"See {file_name}."
            )
            if self.logger is not None:
                self.logger.warning(message)
            else:
                print(message)

        return np.asarray(prediction_starts, dtype=np.int64)

    def __getitem__(self, index: int) -> dict:
        target_start = int(self.prediction_starts[index])
        history_start = target_start - self.input_len
        history_data = self.data[history_start:target_start]
        future_data = self.data[target_start:target_start + self.output_len]
        if self.memmap:
            history_data = history_data.copy()
            future_data = future_data.copy()
        return {"inputs": history_data, "target": future_data}

    def __len__(self) -> int:
        return int(len(self.prediction_starts))
