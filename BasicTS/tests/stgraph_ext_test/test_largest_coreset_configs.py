import importlib
import json
import os
import pickle
import shutil
import sys
import types
import unittest

import numpy as np


class TestLargestCoresetConfigs(unittest.TestCase):
    def setUp(self):
        self.dataset_names = ["GLA", "GBA"]
        self._module_backup = {}
        self._install_dependency_stubs()
        for dataset_name, num_nodes in [("GLA", 4), ("GBA", 3)]:
            self._create_dataset(dataset_name, num_nodes)

    def tearDown(self):
        for dataset_name in self.dataset_names:
            shutil.rmtree(os.path.join("datasets", dataset_name), ignore_errors=True)

        self._restore_dependency_stubs()

        for module_name in [
            "baselines.STID.GLA_coreset",
            "baselines.STID.GBA_coreset",
            "baselines.GWNet.GLA_largest_aligned",
            "baselines.GWNet.GBA_largest_aligned",
            "baselines.GWNet.GLA_largest_aligned_coreset",
            "baselines.GWNet.GBA_largest_aligned_coreset",
        ]:
            sys.modules.pop(module_name, None)

    def _create_dataset(self, dataset_name: str, num_nodes: int) -> None:
        dataset_dir = os.path.join("datasets", dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        desc = {
            "name": dataset_name,
            "shape": [40, num_nodes, 3],
            "num_nodes": num_nodes,
            "num_features": 3,
            "frequency (minutes)": 15,
            "regular_settings": {
                "INPUT_LEN": 12,
                "OUTPUT_LEN": 12,
                "TRAIN_VAL_TEST_RATIO": [0.6, 0.2, 0.2],
                "NORM_EACH_CHANNEL": False,
                "RESCALE": True,
                "NULL_VAL": 0.0,
            },
        }
        with open(os.path.join(dataset_dir, "desc.json"), "w", encoding="utf-8") as fp:
            json.dump(desc, fp)

        adjacency = np.eye(num_nodes, dtype=np.float32)
        payload = ([str(i) for i in range(num_nodes)], {str(i): i for i in range(num_nodes)}, adjacency)
        with open(os.path.join(dataset_dir, "adj_mx.pkl"), "wb") as fp:
            pickle.dump(payload, fp)

    def _import_module(self, module_name: str):
        importlib.invalidate_caches()
        sys.modules.pop(module_name, None)
        return importlib.import_module(module_name)

    def _install_dependency_stubs(self) -> None:
        class FakeEasyDict(dict):
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError as exc:
                    raise AttributeError(name) from exc

            def __setattr__(self, name, value):
                self[name] = value

            def pop(self, key, default=None):
                return super().pop(key, default)

        def _save(name: str, module: types.ModuleType) -> None:
            self._module_backup[name] = sys.modules.get(name)
            sys.modules[name] = module

        torch_module = types.ModuleType("torch")
        torch_module.tensor = lambda value, **kwargs: np.asarray(value)
        torch_module.float32 = np.float32
        _save("torch", torch_module)

        easydict_module = types.ModuleType("easydict")
        easydict_module.EasyDict = FakeEasyDict
        _save("easydict", easydict_module)

        metrics_module = types.ModuleType("basicts.metrics")
        for name in ("masked_mae", "masked_mape", "masked_rmse", "masked_wape"):
            setattr(metrics_module, name, lambda *args, **kwargs: 0.0)
        _save("basicts.metrics", metrics_module)

        data_module = types.ModuleType("basicts.data")
        data_module.TimeSeriesForecastingDataset = type("TimeSeriesForecastingDataset", (), {})
        _save("basicts.data", data_module)

        coreset_data_module = types.ModuleType("basicts.data.coreset_tsf_dataset")
        coreset_data_module.CoresetTimeSeriesForecastingDataset = type(
            "CoresetTimeSeriesForecastingDataset",
            (),
            {},
        )
        _save("basicts.data.coreset_tsf_dataset", coreset_data_module)

        runners_module = types.ModuleType("basicts.runners")
        runners_module.SimpleTimeSeriesForecastingRunner = type("SimpleTimeSeriesForecastingRunner", (), {})
        runners_module.WandBTimeSeriesForecastingRunner = type("WandBTimeSeriesForecastingRunner", (), {})
        _save("basicts.runners", runners_module)

        scaler_module = types.ModuleType("basicts.scaler")
        scaler_module.ZScoreScaler = type("ZScoreScaler", (), {})
        scaler_module.MultiChannelZScoreScaler = type("MultiChannelZScoreScaler", (), {})
        _save("basicts.scaler", scaler_module)

        utils_module = types.ModuleType("basicts.utils")

        def get_regular_settings(dataset_name: str):
            with open(os.path.join("datasets", dataset_name, "desc.json"), "r", encoding="utf-8") as fp:
                return json.load(fp)["regular_settings"]

        def load_adj(*args, **kwargs):
            adjacency = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            return [adjacency, adjacency], adjacency

        utils_module.get_regular_settings = get_regular_settings
        utils_module.load_adj = load_adj
        _save("basicts.utils", utils_module)

        adaptive_graph_package = types.ModuleType("baselines.AdaptiveGraph")
        adaptive_graph_package.__path__ = []
        _save("baselines.AdaptiveGraph", adaptive_graph_package)

        largest_aligned_module = types.ModuleType("baselines.AdaptiveGraph.largest_aligned")

        def apply_largest_aligned_cfg(cfg, backbone: str):
            cfg.MODEL.FORWARD_FEATURES = [0, 1, 2]
            cfg.MODEL.TARGET_FEATURES = [0]
            cfg.MODEL.PARAM["in_dim"] = 3
            cfg.TRAIN.OPTIM.PARAM = {"lr": 0.001, "weight_decay": 0.0001}
            cfg.TRAIN.pop("LR_SCHEDULER", None)
            cfg.TRAIN.CKPT_SAVE_DIR = f"{cfg.TRAIN.CKPT_SAVE_DIR}_largest_aligned"
            return cfg

        largest_aligned_module.apply_largest_aligned_cfg = apply_largest_aligned_cfg
        _save("baselines.AdaptiveGraph.largest_aligned", largest_aligned_module)

        stid_arch_module = types.ModuleType("baselines.STID.arch")
        stid_arch_module.STID = type("STID", (), {})
        _save("baselines.STID.arch", stid_arch_module)

        gwnet_arch_module = types.ModuleType("baselines.GWNet.arch")
        gwnet_arch_module.GraphWaveNet = type("GraphWaveNet", (), {})
        _save("baselines.GWNet.arch", gwnet_arch_module)

    def _restore_dependency_stubs(self) -> None:
        for name, original in self._module_backup.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    def test_stid_gla_gba_coreset_wrappers_attach_train_subset_dataset(self):
        original_strategy = os.environ.get("CORESET_STRATEGY")
        original_ratio = os.environ.get("CORESET_RATIO")
        try:
            os.environ["CORESET_STRATEGY"] = "temporal"
            os.environ["CORESET_RATIO"] = "0.1"

            gla = self._import_module("baselines.STID.GLA_coreset")
            gba = self._import_module("baselines.STID.GBA_coreset")

            self.assertEqual(gla.CFG.DATASET.NAME, "GLA")
            self.assertEqual(gba.CFG.DATASET.NAME, "GBA")
            self.assertEqual(gla.CFG.DATASET.TYPE.__name__, "CoresetTimeSeriesForecastingDataset")
            self.assertEqual(gba.CFG.DATASET.TYPE.__name__, "CoresetTimeSeriesForecastingDataset")
            self.assertEqual(gla.CFG.DATASET.PARAM["coreset_strategy"], "temporal")
            self.assertEqual(gba.CFG.DATASET.PARAM["coreset_ratio"], 0.1)
        finally:
            if original_strategy is None:
                os.environ.pop("CORESET_STRATEGY", None)
            else:
                os.environ["CORESET_STRATEGY"] = original_strategy

            if original_ratio is None:
                os.environ.pop("CORESET_RATIO", None)
            else:
                os.environ["CORESET_RATIO"] = original_ratio

    def test_gwnet_gla_gba_largest_aligned_configs_use_largest_protocol(self):
        gla = self._import_module("baselines.GWNet.GLA_largest_aligned")
        gba = self._import_module("baselines.GWNet.GBA_largest_aligned")

        self.assertEqual(gla.CFG.DATASET.NAME, "GLA")
        self.assertEqual(gba.CFG.DATASET.NAME, "GBA")
        self.assertEqual(gla.CFG.MODEL.FORWARD_FEATURES, [0, 1, 2])
        self.assertEqual(gba.CFG.MODEL.PARAM["in_dim"], 3)
        self.assertEqual(gla.CFG.TRAIN.OPTIM.PARAM["lr"], 0.001)
        self.assertFalse(hasattr(gla.CFG.TRAIN, "LR_SCHEDULER"))
        self.assertIn("largest_aligned", gla.CFG.TRAIN.CKPT_SAVE_DIR)

    def test_gwnet_gla_gba_coreset_wrappers_attach_coreset_selection(self):
        original_strategy = os.environ.get("CORESET_STRATEGY")
        try:
            os.environ["CORESET_STRATEGY"] = "temporal_kcenter"

            gla = self._import_module("baselines.GWNet.GLA_largest_aligned_coreset")
            gba = self._import_module("baselines.GWNet.GBA_largest_aligned_coreset")

            self.assertEqual(gla.CFG.DATASET.TYPE.__name__, "CoresetTimeSeriesForecastingDataset")
            self.assertEqual(gba.CFG.DATASET.PARAM["coreset_strategy"], "temporal_kcenter")
            self.assertEqual(gla.CFG.MODEL.PARAM["in_dim"], 3)
            self.assertIn("coreset_gwnet_gla_largest_aligned", gla.CFG.TRAIN.CKPT_SAVE_DIR)
            self.assertIn("coreset_gwnet_gba_largest_aligned", gba.CFG.TRAIN.CKPT_SAVE_DIR)
        finally:
            if original_strategy is None:
                os.environ.pop("CORESET_STRATEGY", None)
            else:
                os.environ["CORESET_STRATEGY"] = original_strategy


if __name__ == "__main__":
    unittest.main()
