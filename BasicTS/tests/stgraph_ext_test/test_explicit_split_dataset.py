import json
import os
import shutil
import unittest

import numpy as np

from stgraph_ext.dataset import ExplicitSplitTimeSeriesForecastingDataset


class TestExplicitSplitTimeSeriesForecastingDataset(unittest.TestCase):
    def setUp(self):
        self.dataset_name = "test_explicit_split_dataset"
        self.dataset_dir = os.path.join("datasets", self.dataset_name)
        os.makedirs(self.dataset_dir, exist_ok=True)

        self.shape = (20, 2, 3)
        self.data = np.arange(np.prod(self.shape), dtype=np.float32).reshape(self.shape)
        with open(os.path.join(self.dataset_dir, "desc.json"), "w") as fp:
            json.dump({"shape": list(self.shape)}, fp)
        fp = np.memmap(os.path.join(self.dataset_dir, "data.dat"), dtype="float32", mode="w+", shape=self.shape)
        fp[:] = self.data[:]
        fp.flush()
        del fp
        np.savez_compressed(
            os.path.join(self.dataset_dir, "split_indices.npz"),
            train_idx=np.arange(0, 10, dtype=np.int64),
            val_idx=np.arange(10, 14, dtype=np.int64),
            test_idx=np.arange(14, 20, dtype=np.int64),
        )

    def tearDown(self):
        shutil.rmtree(self.dataset_dir)

    def test_train_windows_respect_explicit_split(self):
        dataset = ExplicitSplitTimeSeriesForecastingDataset(
            dataset_name=self.dataset_name,
            train_val_test_ratio=[0.7, 0.1, 0.2],
            mode="train",
            input_len=3,
            output_len=2,
        )
        self.assertEqual(dataset.prediction_starts.tolist(), [3, 4, 5, 6, 7, 8])
        sample = dataset[0]
        self.assertEqual(sample["inputs"].shape, (3, 2, 3))
        self.assertEqual(sample["target"].shape, (2, 2, 3))

    def test_valid_windows_allow_history_context(self):
        dataset = ExplicitSplitTimeSeriesForecastingDataset(
            dataset_name=self.dataset_name,
            train_val_test_ratio=[0.7, 0.1, 0.2],
            mode="valid",
            input_len=3,
            output_len=2,
        )
        self.assertEqual(dataset.prediction_starts.tolist(), [10, 11, 12])

    def test_valid_windows_can_disable_context(self):
        dataset = ExplicitSplitTimeSeriesForecastingDataset(
            dataset_name=self.dataset_name,
            train_val_test_ratio=[0.7, 0.1, 0.2],
            mode="valid",
            input_len=3,
            output_len=2,
            allow_val_test_context=False,
        )
        self.assertEqual(dataset.prediction_starts.tolist(), [])


if __name__ == "__main__":
    unittest.main()
