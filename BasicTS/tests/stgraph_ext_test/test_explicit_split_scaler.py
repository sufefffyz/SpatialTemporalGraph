import json
import os
import shutil
import unittest

import numpy as np
import torch

from stgraph_ext.scaler import ExplicitSplitZScoreScaler


class TestExplicitSplitZScoreScaler(unittest.TestCase):
    def setUp(self):
        self.dataset_name = "test_explicit_split_scaler"
        self.dataset_dir = os.path.join("datasets", self.dataset_name)
        os.makedirs(self.dataset_dir, exist_ok=True)

        data = np.zeros((10, 2, 3), dtype=np.float32)
        values = np.arange(20, dtype=np.float32).reshape(10, 2)
        data[:, :, 0] = values
        data[:, :, 1] = 0.5
        data[:, :, 2] = 0.25
        self.data = data

        with open(os.path.join(self.dataset_dir, "desc.json"), "w") as fp:
            json.dump({"shape": list(data.shape)}, fp)
        fp = np.memmap(os.path.join(self.dataset_dir, "data.dat"), dtype="float32", mode="w+", shape=data.shape)
        fp[:] = data[:]
        fp.flush()
        del fp
        np.savez_compressed(
            os.path.join(self.dataset_dir, "split_indices.npz"),
            train_idx=np.arange(0, 6, dtype=np.int64),
            val_idx=np.arange(6, 8, dtype=np.int64),
            test_idx=np.arange(8, 10, dtype=np.int64),
        )

    def tearDown(self):
        shutil.rmtree(self.dataset_dir)

    def test_transform_and_inverse_transform(self):
        scaler = ExplicitSplitZScoreScaler(
            dataset_name=self.dataset_name,
            train_ratio=0.6,
            norm_each_channel=True,
            rescale=True,
        )
        input_data = torch.tensor(self.data[:6], dtype=torch.float32)
        transformed = scaler.transform(input_data)
        restored = scaler.inverse_transform(transformed)

        target_values = transformed[..., 0]
        self.assertTrue(torch.allclose(target_values.mean(dim=0, keepdim=True), torch.zeros_like(target_values.mean(dim=0, keepdim=True)), atol=1e-6))
        self.assertTrue(torch.allclose(target_values.std(dim=0, keepdim=True, unbiased=False), torch.ones_like(target_values.std(dim=0, keepdim=True, unbiased=False)), atol=1e-6))
        self.assertTrue(torch.allclose(restored, input_data, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
