import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

from easydict import EasyDict


CONFIG_PATH = Path(__file__).resolve().parents[3] / "baselines" / "DataPruning" / "dynamic_pruning_config.py"


class DummyRunner:
    pass


class DummyWandBRunner:
    pass


def load_config_module():
    runners_module = types.ModuleType("basicts.runners")
    runners_module.DynamicPruningTimeSeriesForecastingRunner = DummyRunner
    runners_module.DynamicPruningWandBTimeSeriesForecastingRunner = DummyRunner
    runners_module.WandBTimeSeriesForecastingRunner = DummyWandBRunner

    basicts_module = types.ModuleType("basicts")
    basicts_module.runners = runners_module

    with mock.patch.dict(sys.modules, {"basicts": basicts_module, "basicts.runners": runners_module}):
        spec = importlib.util.spec_from_file_location("dynamic_pruning_config_under_test", CONFIG_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


def build_cfg():
    return EasyDict(
        {
            "DESCRIPTION": "unit",
            "RUNNER": DummyRunner,
            "ENV": EasyDict({"SEED": 0}),
            "TRAIN": EasyDict(
                {
                    "NUM_EPOCHS": 100,
                    "CKPT_SAVE_DIR": "/tmp/checkpoints/unit",
                    "DATA": EasyDict({"BATCH_SIZE": 64}),
                }
            ),
            "VAL": EasyDict({"DATA": EasyDict({"BATCH_SIZE": 64})}),
            "TEST": EasyDict({"DATA": EasyDict({"BATCH_SIZE": 64})}),
        }
    )


class TestDynamicPruningConfig(unittest.TestCase):
    def test_strategy_specific_rescale_defaults_match_baseline_semantics(self):
        module = load_config_module()

        for strategy in ("soft_random", "epsilon_greedy"):
            with self.subTest(strategy=strategy):
                env = {"DYNAMIC_PRUNING_ALIGN_PAPER_HPARAMS": "0"}
                with mock.patch.dict(os.environ, env, clear=True):
                    cfg = module.apply_dynamic_pruning_cfg(build_cfg(), "stid_sd", strategy)
                self.assertFalse(cfg.TRAIN.DYNAMIC_PRUNING.RESCALE)

        with mock.patch.dict(os.environ, {"DYNAMIC_PRUNING_ALIGN_PAPER_HPARAMS": "0"}, clear=True):
            cfg = module.apply_dynamic_pruning_cfg(build_cfg(), "stid_sd", "infobatch")
        self.assertTrue(cfg.TRAIN.DYNAMIC_PRUNING.RESCALE)

    def test_rescale_environment_override_still_applies_to_epsilon_greedy(self):
        module = load_config_module()
        env = {
            "DYNAMIC_PRUNING_ALIGN_PAPER_HPARAMS": "0",
            "DYNAMIC_PRUNING_RESCALE": "1",
        }

        with mock.patch.dict(os.environ, env, clear=True):
            cfg = module.apply_dynamic_pruning_cfg(build_cfg(), "stid_sd", "epsilon_greedy")

        self.assertTrue(cfg.TRAIN.DYNAMIC_PRUNING.RESCALE)

    def test_infobatch_budget_matching_keeps_reference_epoch_cap_for_exact_budget_stop(self):
        module = load_config_module()
        env = {
            "DYNAMIC_PRUNING_RATIO": "0.1",
            "DYNAMIC_PRUNING_TARGET_FORWARD_RATIO": "0.1",
            "DYNAMIC_PRUNING_MATCH_EPOCHS": "1",
            "DYNAMIC_PRUNING_EXPECTED_RETAINED_RATIO": "0.78",
            "DYNAMIC_PRUNING_ALIGN_PAPER_HPARAMS": "0",
        }

        with mock.patch.dict(os.environ, env, clear=False):
            cfg = module.apply_dynamic_pruning_cfg(build_cfg(), "stid_sd", "infobatch")

        self.assertEqual(cfg.TRAIN.NUM_EPOCHS, 100)
        self.assertEqual(cfg.TRAIN.DYNAMIC_PRUNING.REFERENCE_NUM_EPOCHS, 100)
        self.assertAlmostEqual(cfg.TRAIN.DYNAMIC_PRUNING.TARGET_FORWARD_RATIO, 0.1)
        self.assertAlmostEqual(cfg.TRAIN.DYNAMIC_PRUNING.FINAL_FULL_FORWARD_RATIO, 0.125)
        self.assertEqual(cfg.TRAIN.DYNAMIC_PRUNING.FINAL_FULL_EPOCHS, 0)

    def test_paper_hparams_default_to_sgd_optimizer(self):
        module = load_config_module()

        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = module.apply_paper_hparam_full_cfg(build_cfg(), "stid_sd")

        self.assertEqual(cfg.TRAIN.OPTIM.TYPE, "SGD")
        self.assertAlmostEqual(cfg.TRAIN.OPTIM.PARAM["lr"], 0.001)
        self.assertEqual(cfg.TRAIN.LR_SCHEDULER.TYPE, "CosineAnnealingLR")
        self.assertEqual(cfg.TRAIN.DATA.BATCH_SIZE, 256)
        self.assertIn("opt_sgd", cfg.TRAIN.CKPT_SAVE_DIR)

    def test_paper_hparams_can_switch_to_adam_without_changing_other_protocol_knobs(self):
        module = load_config_module()
        env = {"DYNAMIC_PRUNING_OPTIMIZER": "Adam"}

        with mock.patch.dict(os.environ, env, clear=True):
            cfg = module.apply_paper_hparam_full_cfg(build_cfg(), "stid_sd")

        self.assertEqual(cfg.TRAIN.OPTIM.TYPE, "Adam")
        self.assertAlmostEqual(cfg.TRAIN.OPTIM.PARAM["lr"], 0.001)
        self.assertAlmostEqual(cfg.TRAIN.OPTIM.PARAM["weight_decay"], 0.0001)
        self.assertEqual(cfg.TRAIN.LR_SCHEDULER.TYPE, "CosineAnnealingLR")
        self.assertEqual(cfg.TRAIN.DATA.BATCH_SIZE, 256)
        self.assertIn("opt_adam", cfg.TRAIN.CKPT_SAVE_DIR)

    def test_paper_hparams_can_switch_to_muon_with_incremental_data_pruning_optimizer(self):
        module = load_config_module()
        env = {"DYNAMIC_PRUNING_OPTIMIZER": "Muon"}

        with mock.patch.dict(os.environ, env, clear=True):
            cfg = module.apply_paper_hparam_full_cfg(build_cfg(), "stid_sd")

        self.assertIs(cfg.TRAIN.OPTIM.TYPE, module.MuonWithAdamFallback)
        self.assertAlmostEqual(cfg.TRAIN.OPTIM.PARAM["lr"], 0.001)
        self.assertAlmostEqual(cfg.TRAIN.OPTIM.PARAM["weight_decay"], 0.0001)
        self.assertEqual(cfg.TRAIN.LR_SCHEDULER.TYPE, "CosineAnnealingLR")
        self.assertEqual(cfg.TRAIN.DATA.BATCH_SIZE, 256)
        self.assertIn("opt_muon", cfg.TRAIN.CKPT_SAVE_DIR)

    def test_paper_hparams_can_disable_grad_clip_and_use_step_lr(self):
        module = load_config_module()
        cfg = build_cfg()
        cfg.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}
        env = {
            "DYNAMIC_PRUNING_DISABLE_CLIP": "1",
            "DYNAMIC_PRUNING_LR_SCHEDULER": "StepLR",
            "DYNAMIC_PRUNING_STEP_SIZE": "50",
            "DYNAMIC_PRUNING_STEP_GAMMA": "0.5",
        }

        with mock.patch.dict(os.environ, env, clear=True):
            cfg = module.apply_paper_hparam_full_cfg(cfg, "stid_sd")

        self.assertIsNone(cfg.TRAIN.CLIP_GRAD_PARAM)
        self.assertEqual(cfg.TRAIN.LR_SCHEDULER.TYPE, "StepLR")
        self.assertEqual(cfg.TRAIN.LR_SCHEDULER.PARAM["step_size"], 50)
        self.assertAlmostEqual(cfg.TRAIN.LR_SCHEDULER.PARAM["gamma"], 0.5)

    def test_proxy_gap_dlinear_alias_keeps_proxy_gap_strategy_with_dlinear_reference(self):
        module = load_config_module()

        with mock.patch.dict(os.environ, {"DYNAMIC_PRUNING_ALIGN_PAPER_HPARAMS": "0"}, clear=True):
            cfg = module.apply_dynamic_pruning_cfg(build_cfg(), "stid_sd", "proxy_gap_dlinear")

        self.assertEqual(cfg.TRAIN.DYNAMIC_PRUNING.STRATEGY, "proxy_gap")
        self.assertEqual(cfg.TRAIN.DYNAMIC_PRUNING.REFERENCE_TYPE, "dlinear")
        self.assertEqual(cfg.TRAIN.DYNAMIC_PRUNING.SCORE_LOSS_TYPE, "normalized_mae")
        self.assertEqual(cfg.TRAIN.DYNAMIC_PRUNING.DLINEAR_EPOCHS, 100)
        self.assertEqual(cfg.TRAIN.DYNAMIC_PRUNING.DLINEAR_PATIENCE, 30)
        self.assertFalse(cfg.TRAIN.DYNAMIC_PRUNING.DLINEAR_INDIVIDUAL)
        self.assertTrue(cfg.TRAIN.DYNAMIC_PRUNING.RESCALE)
        self.assertIn("proxy_gap_dlinear", cfg.TRAIN.CKPT_SAVE_DIR)

    def test_proxy_gap_lowpass_alias_uses_proxy_gap_with_lowpass_score(self):
        module = load_config_module()

        with mock.patch.dict(os.environ, {"DYNAMIC_PRUNING_ALIGN_PAPER_HPARAMS": "0"}, clear=True):
            cfg = module.apply_dynamic_pruning_cfg(build_cfg(), "stid_sd", "proxy_gap_lowpass")

        self.assertEqual(cfg.TRAIN.DYNAMIC_PRUNING.STRATEGY, "proxy_gap")
        self.assertEqual(cfg.TRAIN.DYNAMIC_PRUNING.REFERENCE_TYPE, "seasonal")
        self.assertEqual(cfg.TRAIN.DYNAMIC_PRUNING.SCORE_LOSS_TYPE, "lowpass_normalized_mae")
        self.assertAlmostEqual(cfg.TRAIN.DYNAMIC_PRUNING.LOWPASS_KEEP_RATIO, 0.5)
        self.assertTrue(cfg.TRAIN.DYNAMIC_PRUNING.RESCALE)
        self.assertIn("proxy_gap_lowpass", cfg.TRAIN.CKPT_SAVE_DIR)

    def test_proxy_gap_lowpass_node_hard_alias_revisits_easy_nodes_without_forward_budget(self):
        module = load_config_module()
        env = {
            "DYNAMIC_PRUNING_ALIGN_PAPER_HPARAMS": "0",
            "DYNAMIC_PRUNING_RATIO": "0.1",
            "DYNAMIC_PRUNING_TARGET_FORWARD_RATIO": "0.1",
            "DYNAMIC_PRUNING_MATCH_EPOCHS": "1",
        }

        with mock.patch.dict(os.environ, env, clear=True):
            cfg = module.apply_dynamic_pruning_cfg(build_cfg(), "stid_sd", "proxy_gap_lowpass_node_hard")

        self.assertEqual(cfg.TRAIN.DYNAMIC_PRUNING.STRATEGY, "node_hard_revisit")
        self.assertTrue(cfg.TRAIN.DYNAMIC_PRUNING.NODE_LOSS_PRUNING)
        self.assertEqual(cfg.TRAIN.DYNAMIC_PRUNING.SCORE_LOSS_TYPE, "lowpass_normalized_mae")
        self.assertAlmostEqual(cfg.TRAIN.DYNAMIC_PRUNING.NODE_RETENTION_RATIO, 0.1)
        self.assertAlmostEqual(cfg.TRAIN.DYNAMIC_PRUNING.NODE_REVISIT_PROBABILITY, 0.5)
        self.assertIsNone(cfg.TRAIN.DYNAMIC_PRUNING.TARGET_FORWARD_RATIO)
        self.assertEqual(cfg.TRAIN.DYNAMIC_PRUNING.FINAL_FULL_EPOCHS, 0)
        self.assertTrue(cfg.TRAIN.DYNAMIC_PRUNING.RESCALE)
        self.assertIn("proxy_gap_lowpass_node_hard", cfg.TRAIN.CKPT_SAVE_DIR)

    def test_cluster_subgraph_cfg_enables_joint_infobatch_with_default_8_by_2_clusters(self):
        module = load_config_module()
        env = {
            "DYNAMIC_PRUNING_ALIGN_PAPER_HPARAMS": "0",
            "DYNAMIC_PRUNING_RATIO": "0.1",
        }

        with mock.patch.dict(os.environ, env, clear=True):
            cfg = module.apply_cluster_subgraph_cfg(
                build_cfg(),
                backbone_tag="stid_sd",
                strategy="infobatch",
                cluster_type="signal_kmeans",
            )

        self.assertTrue(cfg.TRAIN.DYNAMIC_PRUNING.CLUSTER_SUBGRAPH.ENABLED)
        self.assertEqual(cfg.TRAIN.DYNAMIC_PRUNING.CLUSTER_SUBGRAPH.TYPE, "signal_kmeans")
        self.assertEqual(cfg.TRAIN.DYNAMIC_PRUNING.CLUSTER_SUBGRAPH.NUM_CLUSTERS, 8)
        self.assertEqual(cfg.TRAIN.DYNAMIC_PRUNING.CLUSTER_SUBGRAPH.NUM_ACTIVE_CLUSTERS, 2)
        self.assertAlmostEqual(cfg.TRAIN.DYNAMIC_PRUNING.CLUSTER_SUBGRAPH.ACTIVE_NODE_RATIO, 0.25)
        self.assertAlmostEqual(cfg.TRAIN.DYNAMIC_PRUNING.CLUSTER_SUBGRAPH.TARGET_EFFECTIVE_PAIR_RATIO, 0.1)
        self.assertAlmostEqual(cfg.TRAIN.DYNAMIC_PRUNING.CLUSTER_SUBGRAPH.TARGET_WINDOW_RATIO, 0.4)
        self.assertEqual(cfg.TRAIN.DYNAMIC_PRUNING.STRATEGY, "infobatch")
        self.assertAlmostEqual(cfg.TRAIN.DYNAMIC_PRUNING.RETENTION_RATIO, 0.4)
        self.assertTrue(cfg.TRAIN.DYNAMIC_PRUNING.TARGET_RETENTION_MODE)
        self.assertEqual(cfg.TRAIN.DYNAMIC_PRUNING.FINAL_FULL_EPOCHS, 0)
        self.assertIn("cluster_signal_kmeans_k8_a2", cfg.TRAIN.CKPT_SAVE_DIR)

    def test_cluster_subgraph_cfg_spatial_only_keeps_all_windows(self):
        module = load_config_module()
        env = {
            "DYNAMIC_PRUNING_ALIGN_PAPER_HPARAMS": "0",
            "DYNAMIC_PRUNING_RATIO": "0.1",
        }

        with mock.patch.dict(os.environ, env, clear=True):
            cfg = module.apply_cluster_subgraph_cfg(
                build_cfg(),
                backbone_tag="stid_sd",
                strategy="none",
                cluster_type="spatial_kdtree",
            )

        self.assertEqual(cfg.TRAIN.DYNAMIC_PRUNING.STRATEGY, "soft_random")
        self.assertAlmostEqual(cfg.TRAIN.DYNAMIC_PRUNING.RETENTION_RATIO, 1.0)
        self.assertAlmostEqual(cfg.TRAIN.DYNAMIC_PRUNING.PRUNE_PROBABILITY, 0.0)
        self.assertFalse(cfg.TRAIN.DYNAMIC_PRUNING.RESCALE)
        self.assertTrue(cfg.TRAIN.DYNAMIC_PRUNING.CLUSTER_SUBGRAPH.ENABLED)
        self.assertEqual(cfg.TRAIN.DYNAMIC_PRUNING.CLUSTER_SUBGRAPH.TYPE, "spatial_kdtree")
        self.assertIn("spatial_only", cfg.TRAIN.CKPT_SAVE_DIR)


if __name__ == "__main__":
    unittest.main()
