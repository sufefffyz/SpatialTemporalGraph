from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .dynamic_support import DynamicEdgeSupport, DynamicThresholdSupport, dynamic_edge_support_matmul_4d


class DynamicNConv(nn.Module):
    def forward(self, x: torch.Tensor, support: torch.Tensor | DynamicEdgeSupport) -> torch.Tensor:
        if isinstance(support, DynamicEdgeSupport):
            x = dynamic_edge_support_matmul_4d(support, x)
        elif support.dim() == 2:
            x = torch.einsum("bcvl,vw->bcwl", x, support.to(x.device))
        elif support.dim() == 3:
            x = torch.einsum("bcvl,bvw->bcwl", x, support.to(x.device))
        else:
            raise ValueError(f"Unsupported support rank {support.dim()}.")
        return x.contiguous()


class Linear(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DynamicGCN(nn.Module):
    def __init__(self, c_in: int, c_out: int, dropout: float, support_len: int = 2, order: int = 2):
        super().__init__()
        self.nconv = DynamicNConv()
        self.mlp = Linear((order * support_len + 1) * c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x: torch.Tensor, supports: list[torch.Tensor]) -> torch.Tensor:
        out = [x]
        for support in supports:
            x1 = self.nconv(x, support)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x2 = self.nconv(x1, support)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        return F.dropout(h, self.dropout, training=self.training)


class DynamicThresholdGraphWaveNet(nn.Module):
    """Graph WaveNet using batch-conditioned soft or hard threshold supports."""

    def __init__(
        self,
        num_nodes,
        seq_len,
        dynamic_graph,
        dropout=0.3,
        gcn_bool=True,
        addaptadj=False,
        aptinit=None,
        in_dim=2,
        out_dim=12,
        residual_channels=32,
        dilation_channels=32,
        skip_channels=256,
        end_channels=512,
        kernel_size=2,
        blocks=4,
        layers=2,
    ):
        super().__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.dynamic_support = DynamicThresholdSupport(
            num_nodes=num_nodes,
            seq_len=seq_len,
            normalization="transition",
            **dynamic_graph,
        )

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))

        receptive_field = 1
        self.supports_len = 2
        if gcn_bool and addaptadj:
            if aptinit is None:
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes), requires_grad=True)
            else:
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True)
            self.supports_len += 1

        for _ in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for _ in range(layers):
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
                self.residual_convs.append(nn.Conv2d(dilation_channels, residual_channels, kernel_size=(1, 1)))
                self.skip_convs.append(nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(DynamicGCN(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, out_dim, kernel_size=(1, 1), bias=True)
        self.receptive_field = receptive_field

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor,
        batch_seen: int,
        epoch: int,
        train: bool,
        **kwargs,
    ) -> torch.Tensor:
        input = history_data.transpose(1, 3).contiguous()
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = F.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        active_supports = self.dynamic_support.transition_supports(history_data)
        if self.gcn_bool and self.addaptadj:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            active_supports = active_supports + [adp]

        for i in range(self.blocks * self.layers):
            residual = x
            filter = torch.tanh(self.filter_convs[i](residual))
            gate = torch.sigmoid(self.gate_convs[i](residual))
            x = filter * gate

            s = self.skip_convs[i](x)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except Exception:
                skip = 0
            skip = s + skip

            if self.gcn_bool:
                x = self.gconv[i](x, active_supports)
            else:
                x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x[..., -1:].contiguous()
