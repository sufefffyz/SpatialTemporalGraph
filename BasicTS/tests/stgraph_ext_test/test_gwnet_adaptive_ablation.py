import unittest

import torch

from baselines.GWNet.arch.adaptive_ablation_gwnet_arch import AdaptiveAblationGraphWaveNet
from baselines.GWNet.arch.gwnet_arch import nconv


class TestAdaptiveAblationGraphWaveNet(unittest.TestCase):
    def _model(self, mode="learned", keep_ratio=1.0, freeze_adaptive_params=False):
        return AdaptiveAblationGraphWaveNet(
            num_nodes=4,
            supports=[torch.eye(4)],
            dropout=0.0,
            gcn_bool=True,
            addaptadj=True,
            in_dim=2,
            out_dim=12,
            residual_channels=4,
            dilation_channels=4,
            skip_channels=8,
            end_channels=8,
            blocks=1,
            layers=1,
            adaptive_eval_mode=mode,
            adaptive_keep_ratio=keep_ratio,
            adaptive_shuffle_seed=7,
            freeze_adaptive_params=freeze_adaptive_params,
        )

    def test_batched_nconv_matches_per_sample_static_nconv(self):
        conv = nconv()
        x = torch.randn(2, 3, 4, 5)
        support = torch.randn(2, 4, 4)

        actual = conv(x, support)
        expected = torch.stack([conv(x[i:i + 1], support[i]).squeeze(0) for i in range(2)], dim=0)

        self.assertTrue(torch.allclose(actual, expected))

    def test_learned_mode_keeps_adaptive_matrix_unchanged(self):
        model = self._model(mode="learned")
        model.eval()
        adp = torch.softmax(torch.randn(4, 4), dim=-1)

        transformed = model.transform_adaptive_adj(adp)

        self.assertTrue(torch.allclose(transformed, adp))

    def test_topk_mode_keeps_per_row_edges_without_renormalizing(self):
        model = self._model(mode="topk", keep_ratio=0.5)
        model.eval()
        adp = torch.tensor(
            [
                [0.1, 0.4, 0.3, 0.2],
                [0.7, 0.15, 0.1, 0.05],
                [0.26, 0.24, 0.2, 0.3],
                [0.05, 0.15, 0.6, 0.2],
            ]
        )

        transformed = model.transform_adaptive_adj(adp)

        self.assertEqual(torch.count_nonzero(transformed, dim=-1).tolist(), [2, 2, 2, 2])
        expected = torch.tensor(
            [
                [0.0, 0.4, 0.3, 0.0],
                [0.7, 0.15, 0.0, 0.0],
                [0.26, 0.0, 0.0, 0.3],
                [0.0, 0.0, 0.6, 0.2],
            ]
        )
        self.assertTrue(torch.allclose(transformed, expected))
        self.assertTrue(torch.all(transformed.sum(dim=-1) < 1.0))
        self.assertEqual(torch.count_nonzero(transformed[0]).item(), 2)
        self.assertGreater(transformed[0, 1].item(), 0.0)
        self.assertGreater(transformed[0, 2].item(), 0.0)

    def test_shuffle_mode_uses_fixed_node_permutation(self):
        model = self._model(mode="shuffle")
        model.eval()
        adp = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        adp = adp / adp.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        first = model.transform_adaptive_adj(adp)
        second = model.transform_adaptive_adj(adp)

        self.assertTrue(torch.allclose(first, second))
        self.assertFalse(torch.allclose(first, adp))
        self.assertTrue(torch.allclose(first.sum(dim=-1), torch.ones(4)))

    def test_off_mode_removes_adaptive_support_in_eval_only(self):
        model = self._model(mode="off")
        adp = torch.softmax(torch.randn(4, 4), dim=-1)

        model.eval()
        self.assertTrue(torch.allclose(model.transform_adaptive_adj(adp), torch.zeros_like(adp)))

        model.train()
        self.assertTrue(torch.allclose(model.transform_adaptive_adj(adp), adp))

    def test_freeze_adaptive_params_only_freezes_node_embeddings(self):
        model = self._model(freeze_adaptive_params=True)

        self.assertFalse(model.nodevec1.requires_grad)
        self.assertFalse(model.nodevec2.requires_grad)
        self.assertTrue(model.start_conv.weight.requires_grad)

    def test_learned_no_relu_uses_raw_embedding_product(self):
        model = AdaptiveAblationGraphWaveNet(
            num_nodes=2,
            supports=[torch.eye(2)],
            dropout=0.0,
            gcn_bool=True,
            addaptadj=True,
            in_dim=2,
            out_dim=12,
            residual_channels=4,
            dilation_channels=4,
            skip_channels=8,
            end_channels=8,
            blocks=1,
            layers=1,
            adaptive_use_relu=False,
        )
        with torch.no_grad():
            model.nodevec1.zero_()
            model.nodevec2.zero_()
            model.nodevec1[:, :2] = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
            model.nodevec2[:2, :] = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])

        expected = torch.softmax(torch.mm(model.nodevec1, model.nodevec2), dim=1)

        self.assertTrue(torch.allclose(model.build_adaptive_adj(), expected))

    def test_default_learned_adaptive_graph_keeps_relu_formula(self):
        model = self._model(mode="learned")
        with torch.no_grad():
            model.nodevec1[:, :] = 0.0
            model.nodevec2[:, :] = 0.0
            model.nodevec1[0, 0] = -1.0
            model.nodevec2[0, 1] = 1.0

        expected = torch.softmax(torch.relu(torch.mm(model.nodevec1, model.nodevec2)), dim=1)

        self.assertTrue(torch.allclose(model.build_adaptive_adj(), expected))

    def test_signal_mlp_graph_has_no_nodevec_params_and_rows_sum_to_one(self):
        from baselines.GWNet.arch.adaptive_ablation_gwnet_arch import SignalMLPGraphWaveNet

        model = SignalMLPGraphWaveNet(
            num_nodes=4,
            supports=[torch.eye(4)],
            dropout=0.0,
            gcn_bool=True,
            addaptadj=True,
            in_dim=2,
            out_dim=12,
            residual_channels=4,
            dilation_channels=4,
            skip_channels=8,
            end_channels=8,
            blocks=1,
            layers=1,
            signal_input_len=12,
            signal_feature_dim=2,
            signal_graph_relu=False,
        )
        history = torch.randn(2, 12, 4, 2)
        adp = model.build_signal_adaptive_adj(history)

        self.assertFalse(hasattr(model, "nodevec1"))
        self.assertFalse(hasattr(model, "nodevec2"))
        self.assertEqual(adp.shape, (2, 4, 4))
        self.assertTrue(torch.allclose(adp.sum(dim=-1), torch.ones(2, 4), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
