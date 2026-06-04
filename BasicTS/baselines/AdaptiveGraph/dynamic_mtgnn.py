from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from baselines.MTGNN.arch.mtgnn_arch import MTGNN
from baselines.MTGNN.arch.mtgnn_layers import linear

from .dynamic_support import DynamicThresholdSupport


class DynamicMTGNNNConv(nn.Module):
    """MTGNN nconv variant that accepts static or batch-dynamic adjacency."""

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if adj.dim() == 2:
            x = torch.einsum("bcvl,vw->bcwl", x, adj.to(x.device))
        elif adj.dim() == 3:
            x = torch.einsum("bcvl,bvw->bcwl", x, adj.to(x.device))
        else:
            raise ValueError(f"Unsupported MTGNN adjacency rank {adj.dim()}.")
        return x.contiguous()


class DynamicMTGNNMixProp(nn.Module):
    """MTGNN mixprop with the original self-loop and row-normalization logic."""

    def __init__(self, c_in: int, c_out: int, gdep: int, dropout: float, alpha: float):
        super().__init__()
        self.nconv = DynamicMTGNNNConv()
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        adj = adj.to(device=x.device, dtype=x.dtype)
        if adj.dim() == 2:
            adj = adj + torch.eye(adj.size(0), device=x.device, dtype=x.dtype)
            degree = adj.sum(1)
            transition = adj / degree.view(-1, 1).clamp_min(1e-6)
        elif adj.dim() == 3:
            eye = torch.eye(adj.size(-1), device=x.device, dtype=x.dtype).unsqueeze(0)
            adj = adj + eye
            degree = adj.sum(-1)
            transition = adj / degree.unsqueeze(-1).clamp_min(1e-6)
        else:
            raise ValueError(f"Unsupported MTGNN adjacency rank {adj.dim()}.")

        h = x
        out = [h]
        for _ in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, transition)
            out.append(h)
        return self.mlp(torch.cat(out, dim=1))


class DynamicThresholdMTGNN(MTGNN):
    """MTGNN with its learned graph constructor replaced by dynamic thresholding.

    The temporal convolutions, residual/skip paths, layer normalization, and
    output head stay identical to the local MTGNN implementation. Only the graph
    source used by mixprop is replaced.
    """

    def __init__(
        self,
        dynamic_graph: dict,
        gcn_true: bool,
        buildA_true: bool,
        gcn_depth: int,
        num_nodes: int,
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
    ):
        if buildA_true:
            raise ValueError("DynamicThresholdMTGNN replaces MTGNN buildA_true; set buildA_true=False.")
        if predefined_A is not None:
            raise ValueError("DynamicThresholdMTGNN uses dynamic_graph, not predefined_A.")
        super().__init__(
            gcn_true=gcn_true,
            buildA_true=False,
            gcn_depth=gcn_depth,
            num_nodes=num_nodes,
            predefined_A=None,
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

        dynamic_graph = dict(dynamic_graph)
        if dynamic_graph.get("edge_output", False):
            raise ValueError("DynamicThresholdMTGNN MVP expects dense dynamic adjacency; set edge_output=False.")
        dynamic_graph["edge_output"] = False
        self.dynamic_support = DynamicThresholdSupport(
            num_nodes=num_nodes,
            seq_len=seq_length,
            **dynamic_graph,
        )
        if self.gcn_true:
            self.gconv1 = nn.ModuleList(
                DynamicMTGNNMixProp(conv_channels, residual_channels, gcn_depth, dropout, propalpha)
                for _ in range(self.layers)
            )
            self.gconv2 = nn.ModuleList(
                DynamicMTGNNMixProp(conv_channels, residual_channels, gcn_depth, dropout, propalpha)
                for _ in range(self.layers)
            )

    def _dynamic_adjacency(self, history_data: torch.Tensor, idx: torch.Tensor | None) -> torch.Tensor:
        if idx is None:
            if history_data.shape[2] != self.num_nodes:
                raise ValueError(
                    f"DynamicThresholdMTGNN expected {self.num_nodes} nodes, got {history_data.shape[2]}."
                )
            return self.dynamic_support._masked_weights(history_data)

        idx = idx.to(history_data.device).long()
        if idx.numel() != self.num_nodes:
            raise ValueError(
                "DynamicThresholdMTGNN currently supports MTGNNRunner num_split=1 only; "
                f"got an idx subset with {idx.numel()} of {self.num_nodes} nodes."
            )
        canonical_history = history_data.new_empty(history_data.shape)
        canonical_history.index_copy_(2, idx, history_data)
        canonical_adj = self.dynamic_support._masked_weights(canonical_history)
        return canonical_adj.index_select(1, idx).index_select(2, idx)

    def forward(self, history_data: torch.Tensor, idx: int = None, **kwargs) -> torch.Tensor:
        """Feedforward with batch-conditioned dynamic threshold adjacency."""

        raw_history = history_data
        history_data = history_data.transpose(1, 3).contiguous()
        seq_len = history_data.size(3)
        assert seq_len == self.seq_length, "input sequence length not equal to preset sequence length"

        if self.seq_length < self.receptive_field:
            history_data = F.pad(history_data, (self.receptive_field - self.seq_length, 0, 0, 0))

        if self.gcn_true:
            adp = self._dynamic_adjacency(raw_history, idx)

        x = self.start_conv(history_data)
        skip = self.skip0(F.dropout(history_data, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filter = torch.tanh(self.filter_convs[i](x))
            gate = torch.sigmoid(self.gate_convs[i](x))
            x = F.dropout(filter * gate, self.dropout, training=self.training)

            skip = self.skip_convs[i](x) + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp) + self.gconv2[i](x, adp.transpose(1, 2))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x, self.idx)
            else:
                x = self.norm[i](x, idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        return self.end_conv_2(x)
