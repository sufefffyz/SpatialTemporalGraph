import inspect
import json
import logging
from typing import List, Optional

import numpy as np

from .base_dataset import BaseDataset


class RecentWindowTimeSeriesForecastingDataset(BaseDataset):
    """
    A copy of TimeSeriesForecastingDataset with an extra mechanism for restricting
    the training split to its most recent N days / steps while keeping validation
    and test splits unchanged.
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
        logger: logging.Logger = None,
        train_recent_days: Optional[int] = None,
        train_recent_steps: Optional[int] = None,
    ) -> None:
        assert mode in ['train', 'valid', 'test'], f"Invalid mode: {mode}. Must be one of ['train', 'valid', 'test']."
        super().__init__(dataset_name, train_val_test_ratio, mode, memmap)
        self.input_len = input_len
        self.output_len = output_len
        self.overlap = overlap
        self.logger = logger
        self.train_recent_days = train_recent_days
        self.train_recent_steps = train_recent_steps

        self.data_file_path = f'datasets/{dataset_name}/data.dat'
        self.description_file_path = f'datasets/{dataset_name}/desc.json'
        self.description = self._load_description()
        self.data = self._load_data()

    def _load_description(self) -> dict:
        try:
            with open(self.description_file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Description file not found: {self.description_file_path}') from e
        except json.JSONDecodeError as e:
            raise ValueError(f'Error decoding JSON file: {self.description_file_path}') from e

    def _resolve_recent_train_steps(self) -> Optional[int]:
        if self.mode != 'train':
            return None
        if self.train_recent_days is not None and self.train_recent_steps is not None:
            raise ValueError('Specify only one of train_recent_days or train_recent_steps.')
        if self.train_recent_steps is not None:
            steps = int(self.train_recent_steps)
            if steps <= 0:
                raise ValueError('train_recent_steps must be a positive integer.')
            return steps
        if self.train_recent_days is None:
            return None

        days = int(self.train_recent_days)
        if days <= 0:
            raise ValueError('train_recent_days must be a positive integer.')
        freq_minutes = self.description.get('frequency (minutes)')
        if freq_minutes is None:
            raise ValueError('train_recent_days requires `frequency (minutes)` in desc.json.')
        freq_minutes = int(freq_minutes)
        if freq_minutes <= 0 or 1440 % freq_minutes != 0:
            raise ValueError(f'Unsupported frequency (minutes): {freq_minutes}')
        return days * (1440 // freq_minutes)

    def _load_data(self) -> np.ndarray:
        try:
            data = np.memmap(self.data_file_path, dtype='float32', mode='r', shape=tuple(self.description['shape']))
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f'Error loading data file: {self.data_file_path}') from e

        total_len = len(data)
        valid_len = int(total_len * self.train_val_test_ratio[1])
        test_len = int(total_len * self.train_val_test_ratio[2])
        train_len = total_len - valid_len - test_len
        recent_train_steps = self._resolve_recent_train_steps()
        effective_train_len = train_len

        minimal_len = self.input_len + self.output_len
        if recent_train_steps is not None:
            if recent_train_steps > train_len:
                raise ValueError(
                    f'train_recent window ({recent_train_steps} steps) exceeds available training split ({train_len} steps).'
                )
            effective_train_len = recent_train_steps
        if minimal_len > {'train': effective_train_len, 'valid': valid_len, 'test': test_len}[self.mode]:
            self.overlap = True
            current_frame = inspect.currentframe()
            file_name = inspect.getfile(current_frame)
            line_number = current_frame.f_lineno - 7
            dataset = {'train': 'Training', 'valid': 'Validation', 'test': 'Test'}[self.mode]
            if self.logger is not None:
                self.logger.info(f'{dataset} dataset is too short, enabling overlap. See details in {file_name} at line {line_number}.')
            else:
                print(f'{dataset} dataset is too short, enabling overlap. See details in {file_name} at line {line_number}.')

        if self.mode == 'train':
            train_start = train_len - effective_train_len
            offset = self.output_len if self.overlap else 0
            seg = data[train_start:train_len + offset]
        elif self.mode == 'valid':
            offset_left = self.input_len - 1 if self.overlap else 0
            offset_right = self.output_len if self.overlap else 0
            seg = data[train_len - offset_left : train_len + valid_len + offset_right]
        else:
            offset = self.input_len - 1 if self.overlap else 0
            seg = data[train_len + valid_len - offset:]

        if not self.memmap:
            seg = seg.copy()
        return seg

    def __getitem__(self, index: int) -> dict:
        history_data = self.data[index:index + self.input_len]
        future_data = self.data[index + self.input_len:index + self.input_len + self.output_len]
        if self.memmap:
            history_data = history_data.copy()
            future_data = future_data.copy()
        return {'inputs': history_data, 'target': future_data}

    def __len__(self) -> int:
        return len(self.data) - self.input_len - self.output_len + 1
