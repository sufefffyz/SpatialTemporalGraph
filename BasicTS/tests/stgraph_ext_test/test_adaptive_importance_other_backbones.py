import unittest

import torch

from baselines.AGCRN.arch.adaptive_importance_agcrn_arch import (
    AdaptiveImportanceAGCRN,
    AdaptiveImportanceAVWGCN,
)
from baselines.MTGNN.arch.adaptive_importance_mtgnn_arch import (
    AdaptiveImportanceGraphConstructor,
    AdaptiveImportanceMTGNN,
)


class TestAdaptiveImportanceAGCRN(unittest.TestCase):
    def test_graph_no_relu_uses_raw_embedding_product(self):
        gcn = AdaptiveImportanceAVWGCN(
            dim_in=1,
            dim_out=1,
            cheb_k=2,
            embed_dim=2,
            support_mode="adaptive",
            graph_use_relu=False,
        )
        node_embeddings = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])

        support = gcn.build_support(node_embeddings)
        expected = torch.softmax(torch.mm(node_embeddings, node_embeddings.transpose(0, 1)), dim=1)

        self.assertTrue(torch.allclose(support, expected))

    def test_original_support_keeps_relu_formula(self):
        gcn = AdaptiveImportanceAVWGCN(
            dim_in=1,
            dim_out=1,
            cheb_k=2,
            embed_dim=2,
            support_mode="adaptive",
            graph_use_relu=True,
        )
        node_embeddings = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])

        support = gcn.build_support(node_embeddings)
        expected = torch.softmax(torch.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)

        self.assertTrue(torch.allclose(support, expected))

    def test_identity_support_uses_self_loop_only(self):
        gcn = AdaptiveImportanceAVWGCN(
            dim_in=1,
            dim_out=1,
            cheb_k=2,
            embed_dim=2,
            support_mode="identity",
            graph_use_relu=True,
        )
        node_embeddings = torch.randn(3, 2)

        support = gcn.build_support(node_embeddings)

        self.assertTrue(torch.allclose(support, torch.eye(3)))

    def test_freeze_node_embeddings_only_freezes_agcrn_embedding(self):
        model = AdaptiveImportanceAGCRN(
            num_nodes=4,
            input_dim=1,
            rnn_units=8,
            output_dim=1,
            horizon=12,
            num_layers=1,
            default_graph=True,
            embed_dim=2,
            cheb_k=2,
            freeze_node_embeddings=True,
        )

        self.assertFalse(model.node_embeddings.requires_grad)
        self.assertTrue(model.end_conv.weight.requires_grad)


class TestAdaptiveImportanceMTGNN(unittest.TestCase):
    def test_no_relu_graph_constructor_uses_signed_tanh_scores(self):
        static_feat = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 1.0]])
        graph = AdaptiveImportanceGraphConstructor(
            nnodes=3,
            k=2,
            dim=2,
            alpha=1,
            static_feat=static_feat,
            use_relu=False,
        )
        with torch.no_grad():
            graph.lin1.weight.copy_(torch.eye(2))
            graph.lin1.bias.zero_()
            graph.lin2.weight.copy_(torch.tensor([[1.0, 1.0], [0.0, 1.0]]))
            graph.lin2.bias.zero_()

        idx = torch.arange(3)
        adj = graph.fullA(idx)
        nodevec1 = torch.tanh(graph.lin1(static_feat))
        nodevec2 = torch.tanh(graph.lin2(static_feat))
        expected = torch.tanh(torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0)))

        self.assertTrue(torch.allclose(adj, expected))
        self.assertLess(adj.min().item(), 0.0)

    def test_no_relu_graph_constructor_keeps_topk_mask(self):
        graph = AdaptiveImportanceGraphConstructor(nnodes=4, k=2, dim=3, alpha=1, use_relu=False)

        adj = graph(torch.arange(4))

        self.assertEqual(adj.shape, (4, 4))
        self.assertTrue(torch.all(torch.count_nonzero(adj, dim=1) <= 2))

    def test_freeze_graph_constructor_only_freezes_gc_params(self):
        model = AdaptiveImportanceMTGNN(
            gcn_true=True,
            buildA_true=True,
            gcn_depth=2,
            num_nodes=4,
            predefined_A=None,
            dropout=0.0,
            subgraph_size=2,
            node_dim=4,
            conv_channels=4,
            residual_channels=4,
            skip_channels=8,
            end_channels=8,
            seq_length=12,
            in_dim=2,
            out_dim=12,
            layers=1,
            freeze_graph_constructor=True,
        )

        self.assertTrue(all(not param.requires_grad for param in model.gc.parameters()))
        self.assertTrue(model.start_conv.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
