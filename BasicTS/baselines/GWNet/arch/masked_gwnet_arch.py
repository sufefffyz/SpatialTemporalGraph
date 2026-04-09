import math

import torch
from torch import nn
import torch.nn.functional as F

from .gwnet_arch import linear, nconv


class masked_gcn(nn.Module):
    """Graph convolution with a single masked attention adjacency."""

    def __init__(self, c_in, c_out, dropout, order=2):
        super().__init__()
        self.nconv = nconv()
        self.order = order
        expanded_c_in = (order + 1) * c_in
        self.mlp = linear(expanded_c_in, c_out)
        self.dropout = dropout

    def forward(self, x, adj):
        out = [x]
        xk = x
        for _ in range(self.order):
            xk = self.nconv(xk, adj)
            out.append(xk)

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class MaskedGraphWaveNet(nn.Module):
    """
    GWNet temporal backbone with a masked learned adjacency.

    The candidate edge set is fixed by a prior thresholded distance graph P.
    The current MVP variant learns adjacency only within the candidate mask:

        S_ij = <e_s_i, e_t_j> / sqrt(d)
        A = softmax(mask(S))

    where mask(S) sets non-candidate entries to -inf.
    """

    def __init__(
        self,
        num_nodes,
        prior_adj,
        dropout=0.3,
        in_dim=3,
        out_dim=12,
        residual_channels=32,
        dilation_channels=32,
        skip_channels=256,
        end_channels=512,
        kernel_size=2,
        blocks=4,
        layers=2,
        attention_embed_dim=10,
        attention_order=2,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.attention_embed_dim = attention_embed_dim

        if isinstance(prior_adj, torch.Tensor):
            prior = prior_adj.detach().clone().float()
        else:
            prior = torch.tensor(prior_adj, dtype=torch.float32)
        if prior.shape != (num_nodes, num_nodes):
            raise ValueError(
                f"prior_adj shape mismatch: expected {(num_nodes, num_nodes)}, got {tuple(prior.shape)}"
            )

        candidate_mask = prior > 0
        empty_rows = candidate_mask.sum(dim=-1) == 0
        if empty_rows.any():
            idx = torch.nonzero(empty_rows, as_tuple=False).flatten()
            prior[idx, idx] = 1.0
            candidate_mask[idx, idx] = True

        self.register_buffer("prior_adj", prior)
        self.register_buffer("candidate_mask", candidate_mask)

        self.source_embeddings = nn.Parameter(
            torch.randn(num_nodes, attention_embed_dim, dtype=torch.float32)
        )
        self.target_embeddings = nn.Parameter(
            torch.randn(num_nodes, attention_embed_dim, dtype=torch.float32)
        )

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=residual_channels,
            kernel_size=(1, 1),
        )

        receptive_field = 1
        for _b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for _i in range(layers):
                self.filter_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation,
                    )
                )
                self.gate_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation,
                    )
                )
                self.residual_convs.append(
                    nn.Conv2d(
                        in_channels=dilation_channels,
                        out_channels=residual_channels,
                        kernel_size=(1, 1),
                    )
                )
                self.skip_convs.append(
                    nn.Conv2d(
                        in_channels=dilation_channels,
                        out_channels=skip_channels,
                        kernel_size=(1, 1),
                    )
                )
                self.bn.append(nn.BatchNorm2d(residual_channels))
                self.gconv.append(
                    masked_gcn(
                        dilation_channels,
                        residual_channels,
                        dropout,
                        order=attention_order,
                    )
                )
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        self.end_conv_1 = nn.Conv2d(
            in_channels=skip_channels,
            out_channels=end_channels,
            kernel_size=(1, 1),
            bias=True,
        )
        self.end_conv_2 = nn.Conv2d(
            in_channels=end_channels,
            out_channels=out_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        self.receptive_field = receptive_field

    def compute_attention_adj(self) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.attention_embed_dim)
        logits = torch.matmul(self.source_embeddings, self.target_embeddings.T) * scale
        logits = logits.masked_fill(~self.candidate_mask, float("-inf"))
        attn = torch.softmax(logits, dim=-1)
        attn = attn * self.candidate_mask.float()
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return attn

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        _ = future_data, batch_seen, epoch, train, kwargs
        input_data = history_data.transpose(1, 3).contiguous()
        in_len = input_data.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input_data, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input_data
        x = self.start_conv(x)
        skip = 0

        attn_adj = self.compute_attention_adj().to(x.device)

        for i in range(self.blocks * self.layers):
            residual = x
            filt = torch.tanh(self.filter_convs[i](residual))
            gate = torch.sigmoid(self.gate_convs[i](residual))
            x = filt * gate

            s = self.skip_convs[i](x)
            try:
                skip = skip[:, :, :, -s.size(3) :]
            except Exception:
                skip = 0
            skip = s + skip

            x = self.gconv[i](x, attn_adj)
            x = x + residual[:, :, :, -x.size(3) :]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
