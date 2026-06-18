import json
import unittest
from unittest.mock import mock_open, patch

import numpy as np

from basicts.data.indexed_tsf_dataset import (
    IndexedCoresetTimeSeriesForecastingDataset,
    IndexedTimeSeriesForecastingDataset,
)
from basicts.data.simple_tsf_dataset import TimeSeriesForecastingDataset


class TestDynamicPruningDatasetIndexing(unittest.TestCase):
    def setUp(self):
        self.dataset_name = "test_dynamic_pruning_dataset"
        self.train_val_test_ratio = [0.6, 0.2, 0.2]
        self.input_len = 3
        self.output_len = 2
        self.data = np.arange(100, dtype="float32")

    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps({"shape": [100]}))
    @patch("numpy.memmap")
    def test_default_dataset_keys_stay_unchanged(self, mock_memmap, mocked_open):
        mock_memmap.return_value = self.data

        dataset = TimeSeriesForecastingDataset(
            dataset_name=self.dataset_name,
            train_val_test_ratio=self.train_val_test_ratio,
            mode="train",
            input_len=self.input_len,
            output_len=self.output_len,
        )

        sample = dataset[4]
        self.assertEqual(set(sample.keys()), {"inputs", "target"})

    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps({"shape": [100]}))
    @patch("numpy.memmap")
    def test_dataset_can_return_sample_index(self, mock_memmap, mocked_open):
        mock_memmap.return_value = self.data

        dataset = IndexedTimeSeriesForecastingDataset(
            dataset_name=self.dataset_name,
            train_val_test_ratio=self.train_val_test_ratio,
            mode="train",
            input_len=self.input_len,
            output_len=self.output_len,
        )

        sample = dataset[7]
        self.assertEqual(sample["index"], 7)
        self.assertEqual(set(sample.keys()), {"inputs", "target", "index"})

    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps({"shape": [100]}))
    @patch("numpy.memmap")
    def test_coreset_dataset_returns_original_selected_index(self, mock_memmap, mocked_open):
        mock_memmap.return_value = self.data

        dataset = IndexedCoresetTimeSeriesForecastingDataset(
            dataset_name=self.dataset_name,
            train_val_test_ratio=self.train_val_test_ratio,
            mode="train",
            input_len=self.input_len,
            output_len=self.output_len,
            coreset_ratio=0.5,
            coreset_strategy="random",
            coreset_seed=11,
        )

        sample = dataset[0]
        self.assertEqual(sample["index"], int(dataset.selected_indices[0]))


if __name__ == "__main__":
    unittest.main()
