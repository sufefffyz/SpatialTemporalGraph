import json
import os
import shutil
import unittest

import numpy as np

from basicts.data.coreset_tsf_dataset import CoresetTimeSeriesForecastingDataset


class TestCoresetTimeSeriesForecastingDataset(unittest.TestCase):
    def setUp(self):
        self.dataset_name = "test_coreset_tsf_dataset"
        self.dataset_dir = os.path.join("datasets", self.dataset_name)
        os.makedirs(self.dataset_dir, exist_ok=True)

        self.shape = (40, 3, 3)
        data = np.zeros(self.shape, dtype=np.float32)
        for t in range(self.shape[0]):
            data[t, :, 0] = t + np.arange(self.shape[1], dtype=np.float32)
            data[t, :, 1] = (t % 4) / 4.0
            data[t, :, 2] = (t // 4) % 7
        self.data = data

        with open(os.path.join(self.dataset_dir, "desc.json"), "w") as fp:
            json.dump({"shape": list(self.shape)}, fp)
        fp = np.memmap(os.path.join(self.dataset_dir, "data.dat"), dtype="float32", mode="w+", shape=self.shape)
        fp[:] = self.data[:]
        fp.flush()
        del fp

    def tearDown(self):
        shutil.rmtree(self.dataset_dir)

    def build_dataset(self, mode="train", strategy="random", ratio=0.5, seed=7):
        return CoresetTimeSeriesForecastingDataset(
            dataset_name=self.dataset_name,
            train_val_test_ratio=[0.6, 0.2, 0.2],
            mode=mode,
            input_len=3,
            output_len=2,
            coreset_ratio=ratio,
            coreset_strategy=strategy,
            coreset_seed=seed,
            coreset_temporal_period=4,
            coreset_cache_dir=None,
        )

    def test_random_train_subset_is_reproducible(self):
        dataset_a = self.build_dataset(strategy="random", ratio=0.5, seed=11)
        dataset_b = self.build_dataset(strategy="random", ratio=0.5, seed=11)

        self.assertEqual(len(dataset_a), 10)
        self.assertEqual(dataset_a.selected_indices.tolist(), dataset_b.selected_indices.tolist())
        first_original_index = int(dataset_a.selected_indices[0])
        sample = dataset_a[0]
        np.testing.assert_array_equal(sample["inputs"], dataset_a.data[first_original_index:first_original_index + 3])

    def test_valid_split_ignores_coreset_selection(self):
        valid = self.build_dataset(mode="valid", strategy="random", ratio=0.25)
        full_valid = self.build_dataset(mode="valid", strategy="full", ratio=1.0)

        self.assertEqual(len(valid), len(full_valid))
        self.assertFalse(hasattr(valid, "selected_indices"))

    def test_temporal_strategy_covers_all_time_buckets(self):
        dataset = self.build_dataset(strategy="temporal", ratio=0.5, seed=13)
        buckets = sorted(set((dataset.selected_indices + dataset.input_len) % 4))

        self.assertEqual(buckets, [0, 1, 2, 3])
        self.assertEqual(len(dataset), 10)

    def test_temporal_kcenter_difficulty_returns_budgeted_subset(self):
        dataset = self.build_dataset(strategy="temporal_kcenter_difficulty", ratio=0.4, seed=17)

        self.assertEqual(len(dataset), 8)
        self.assertEqual(len(set(dataset.selected_indices.tolist())), 8)
        self.assertTrue(np.all(dataset.selected_indices[:-1] <= dataset.selected_indices[1:]))


if __name__ == "__main__":
    unittest.main()
