import torch
import torch.nn as nn
import torch.nn.functional as F

from .mtgnn_arch import MTGNN


class AdaptiveImportanceGraphConstructor(nn.Module):
    def __init__(self, nnodes, k, dim, alpha=3, static_feat=None, use_relu=True):
        super().__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat
        self.use_relu = use_relu

    def _node_vectors(self, idx):
        if self.static_feat is None:
            idx = idx.to(self.emb1.weight.device)
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            idx = idx.to(self.static_feat.device)
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))
        return nodevec1, nodevec2

    def _score(self, idx):
        nodevec1, nodevec2 = self._node_vectors(idx)
        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = torch.tanh(self.alpha * a)
        if self.use_relu:
            adj = F.relu(adj)
        return adj

    def forward(self, idx):
        adj = self._score(idx)
        k = min(self.k, idx.size(0))
        mask = torch.zeros(idx.size(0), idx.size(0), device=adj.device, dtype=adj.dtype)
        _, topk_idx = adj.topk(k, 1)
        mask.scatter_(1, topk_idx, 1)
        return adj * mask

    def fullA(self, idx):
        return self._score(idx)


class AdaptiveImportanceMTGNN(MTGNN):
    def __init__(
        self,
        gcn_true,
        buildA_true,
        gcn_depth,
        num_nodes,
        predefined_A=None,
        static_feat=None,
        dropout=0.3,
        subgraph_size=20,
        node_dim=40,
        dilation_exponential=1,
        conv_channels=32,
        residual_channels=32,
        skip_channels=64,
        end_channels=128,
        seq_length=12,
        in_dim=2,
        out_dim=12,
        layers=3,
        propalpha=0.05,
        tanhalpha=3,
        layer_norm_affline=True,
        adaptive_graph_use_relu=True,
        freeze_graph_constructor=False,
    ):
        super().__init__(
            gcn_true=gcn_true,
            buildA_true=buildA_true,
            gcn_depth=gcn_depth,
            num_nodes=num_nodes,
            predefined_A=predefined_A,
            static_feat=static_feat,
            dropout=dropout,
            subgraph_size=subgraph_size,
            node_dim=node_dim,
            dilation_exponential=dilation_exponential,
            conv_channels=conv_channels,
            residual_channels=residual_channels,
            skip_channels=skip_channels,
            end_channels=end_channels,
            seq_length=seq_length,
            in_dim=in_dim,
            out_dim=out_dim,
            layers=layers,
            propalpha=propalpha,
            tanhalpha=tanhalpha,
            layer_norm_affline=layer_norm_affline,
        )
        if not adaptive_graph_use_relu:
            original_gc = self.gc
            self.gc = AdaptiveImportanceGraphConstructor(
                num_nodes,
                subgraph_size,
                node_dim,
                alpha=tanhalpha,
                static_feat=static_feat,
                use_relu=False,
            )
            self.gc.load_state_dict(original_gc.state_dict())

        if freeze_graph_constructor:
            for param in self.gc.parameters():
                param.requires_grad_(False)
