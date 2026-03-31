import os
import tempfile
import unittest

import numpy as np
import torch

from stgraph_ext.adjacency import extract_learned_graph, save_adjacency_snapshot


class _FakeAGCRNModel:
    def __init__(self):
        self.node_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)


class _FakeGWNetModel:
    def __init__(self):
        self.nodevec1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        self.nodevec2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)


class _FakeGC:
    def __init__(self):
        self.emb1 = type("Emb", (), {"weight": torch.zeros(2, 2)})()

    def fullA(self, idx):
        _ = idx
        return torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)


class _FakeMTGNNModel:
    def __init__(self):
        self.gc = _FakeGC()
        self.idx = torch.tensor([0, 1], dtype=torch.long)


class _FakeD2STGNNModel:
    def __init__(self):
        self.node_emb_d = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        self.node_emb_u = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float32)


class _FakeGTSModel:
    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False

    def train(self, mode=True):
        self.training = mode


class _FakeLogger:
    def info(self, *args, **kwargs):
        _ = args, kwargs


class _FakeRunner:
    def __init__(self, model_name, model, ckpt_dir):
        self.model_name = model_name
        self.model = model
        self.ckpt_save_dir = ckpt_dir
        self.dataset_name = "TEST_DATASET"
        self.logger = _FakeLogger()

    def forward(self, sample_batch, epoch=None, iter_num=None, train=False):
        _ = sample_batch, epoch, iter_num, train
        return {
            "prediction": torch.zeros((1, 1, 2, 1), dtype=torch.float32),
            "pred_adj": torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32),
            "prior_adj": torch.tensor([[0.0, 0.5], [0.5, 0.0]], dtype=torch.float32),
        }


class TestAdjacencyExtraction(unittest.TestCase):
    def test_extract_learned_graph(self):
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2, tempfile.TemporaryDirectory() as tmp3, tempfile.TemporaryDirectory() as tmp4, tempfile.TemporaryDirectory() as tmp5:
            agcrn_runner = _FakeRunner("AGCRN", _FakeAGCRNModel(), ckpt_dir=tmp1)
            gwnet_runner = _FakeRunner("GraphWaveNet", _FakeGWNetModel(), ckpt_dir=tmp2)
            mtgnn_runner = _FakeRunner("MTGNN", _FakeMTGNNModel(), ckpt_dir=tmp3)
            gts_runner = _FakeRunner("GTS", _FakeGTSModel(), ckpt_dir=tmp4)
            d2stgnn_runner = _FakeRunner("D2STGNN", _FakeD2STGNNModel(), ckpt_dir=tmp5)

            self.assertEqual(extract_learned_graph(agcrn_runner)["adj"].shape, (2, 2))
            self.assertEqual(extract_learned_graph(gwnet_runner)["adj"].shape, (2, 2))
            self.assertEqual(extract_learned_graph(mtgnn_runner)["adj"].shape, (2, 2))
            d2stgnn_payload = extract_learned_graph(d2stgnn_runner)
            self.assertEqual(d2stgnn_payload["adj"].shape, (2, 2))
            self.assertEqual(d2stgnn_payload["graph_semantics"].item(), "static_directed")
            gts_payload = extract_learned_graph(gts_runner, sample_batch={"inputs": None, "target": None})
            self.assertIn("adj", gts_payload)
            self.assertIn("prior_adj", gts_payload)

    def test_save_adjacency_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _FakeRunner("AGCRN", _FakeAGCRNModel(), ckpt_dir=tmpdir)
            output_path = save_adjacency_snapshot(runner, tag="epoch_005", epoch=5)
            self.assertTrue(os.path.exists(output_path))
            payload = np.load(output_path)
            self.assertIn("adj", payload)
            self.assertEqual(payload["adj"].shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
