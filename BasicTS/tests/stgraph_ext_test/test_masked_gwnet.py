import unittest

import torch

from baselines.GWNet.arch import MaskedGraphWaveNet


class TestMaskedGraphWaveNet(unittest.TestCase):
    def test_attention_mask_and_rowsum(self):
        prior = torch.tensor(
            [
                [0.0, 0.8, 0.0],
                [0.1, 0.0, 0.3],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        model = MaskedGraphWaveNet(
            num_nodes=3,
            prior_adj=prior,
            in_dim=3,
            out_dim=12,
            residual_channels=4,
            dilation_channels=4,
            skip_channels=8,
            end_channels=8,
            blocks=1,
            layers=1,
        )

        attn = model.compute_attention_adj().detach()
        mask = model.candidate_mask.detach()

        self.assertEqual(tuple(attn.shape), (3, 3))
        self.assertTrue(torch.allclose(attn[~mask], torch.zeros_like(attn[~mask]), atol=1e-6))
        self.assertTrue(torch.allclose(attn.sum(dim=-1), torch.ones(3), atol=1e-6))
        self.assertGreater(float(attn[2, 2]), 0.999)

    def test_forward_shape(self):
        prior = torch.tensor(
            [
                [0.0, 0.5],
                [0.2, 0.0],
            ],
            dtype=torch.float32,
        )
        model = MaskedGraphWaveNet(
            num_nodes=2,
            prior_adj=prior,
            in_dim=3,
            out_dim=12,
            residual_channels=4,
            dilation_channels=4,
            skip_channels=8,
            end_channels=8,
            blocks=1,
            layers=1,
        )
        history = torch.randn(2, 12, 2, 3)
        future = torch.randn(2, 12, 2, 3)
        pred = model(history, future, batch_seen=0, epoch=1, train=True)
        self.assertEqual(tuple(pred.shape), (2, 12, 2, 1))


if __name__ == "__main__":
    unittest.main()
