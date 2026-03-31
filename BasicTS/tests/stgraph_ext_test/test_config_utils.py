import json
import os
import pickle
import shutil
import unittest

import numpy as np

from stgraph_ext.config_utils import build_d2stgnn_cfg


class TestConfigUtils(unittest.TestCase):
    def setUp(self):
        self.dataset_names = []

    def tearDown(self):
        for dataset_name in self.dataset_names:
            shutil.rmtree(os.path.join("datasets", dataset_name))

    def _create_dataset(self, dataset_name: str, frequency_minutes: int) -> None:
        dataset_dir = os.path.join("datasets", dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        self.dataset_names.append(dataset_name)

        desc = {
            "name": dataset_name,
            "shape": [20, 2, 3],
            "num_nodes": 2,
            "frequency (minutes)": frequency_minutes,
            "regular_settings": {
                "INPUT_LEN": 12,
                "OUTPUT_LEN": 12,
                "TRAIN_VAL_TEST_RATIO": [0.7, 0.1, 0.2],
                "NORM_EACH_CHANNEL": False,
                "RESCALE": True,
                "NULL_VAL": 0.0,
            },
        }
        with open(os.path.join(dataset_dir, "desc.json"), "w") as fp:
            json.dump(desc, fp)

        adjacency = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        payload = (["0", "1"], {"0": 0, "1": 1}, adjacency)
        with open(os.path.join(dataset_dir, "adj_mx.pkl"), "wb") as fp:
            pickle.dump(payload, fp)

    def test_build_d2stgnn_cfg_uses_all_input_features_and_dynamic_time_size(self):
        self._create_dataset("test_d2stgnn_5min", frequency_minutes=5)
        self._create_dataset("test_d2stgnn_6h", frequency_minutes=360)

        cfg_5min = build_d2stgnn_cfg("test_d2stgnn_5min", num_epochs=1)
        cfg_6h = build_d2stgnn_cfg("test_d2stgnn_6h", num_epochs=1)

        self.assertEqual(cfg_5min.MODEL.FORWARD_FEATURES, [0, 1, 2])
        self.assertEqual(cfg_6h.MODEL.FORWARD_FEATURES, [0, 1, 2])
        self.assertEqual(cfg_5min.MODEL.TARGET_FEATURES, [0])
        self.assertEqual(cfg_5min.MODEL.PARAM["time_in_day_size"], 288)
        self.assertEqual(cfg_6h.MODEL.PARAM["time_in_day_size"], 4)
        self.assertEqual(cfg_5min.MODEL.PARAM["num_modalities"], 2)


if __name__ == "__main__":
    unittest.main()
