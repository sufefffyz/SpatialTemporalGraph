from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import init

from baselines.STGCN.arch.stgcn_layers import Align, OutputBlock, TemporalConvLayer

from .dynamic_support import DynamicThresholdSupport


class DynamicChebGraphConv(nn.Module):
    def __init__(self, c_in, c_out, Ks, bias):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.Ks = Ks
        self.weight = nn.Parameter(torch.FloatTensor(Ks, c_in, c_out))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    @staticmethod
    def _support_mul(gso: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        gso = gso.to(x.device)
        if gso.dim() == 2:
            return torch.einsum("hi,btij->bthj", gso, x)
        if gso.dim() == 3:
            return torch.einsum("bhi,btij->bthj", gso, x)
        raise ValueError(f"Unsupported support rank {gso.dim()}.")

    def forward(self, x, gso):
        x = torch.permute(x, (0, 2, 3, 1))
        if self.Ks - 1 < 0:
            raise ValueError(f"Graph convolution kernel size Ks must be positive, got {self.Ks}.")
        if self.Ks - 1 == 0:
            x_list = [x]
        elif self.Ks - 1 == 1:
            x_0 = x
            x_1 = self._support_mul(gso, x)
            x_list = [x_0, x_1]
        else:
            x_0 = x
            x_1 = self._support_mul(gso, x)
            x_list = [x_0, x_1]
            for k in range(2, self.Ks):
                x_list.append(2 * self._support_mul(gso, x_list[k - 1]) - x_list[k - 2])

        x = torch.stack(x_list, dim=2)
        graph_conv = torch.einsum("btkhi,kij->bthj", x, self.weight)
        if self.bias is not None:
            graph_conv = torch.add(graph_conv, self.bias)
        return graph_conv


class DynamicGraphConv(nn.Module):
    def __init__(self, c_in, c_out, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(c_in, c_out))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, gso):
        x = torch.permute(x, (0, 2, 3, 1))
        if gso.dim() == 2:
            first_mul = torch.einsum("hi,btij->bthj", gso.to(x.device), x)
        elif gso.dim() == 3:
            first_mul = torch.einsum("bhi,btij->bthj", gso.to(x.device), x)
        else:
            raise ValueError(f"Unsupported support rank {gso.dim()}.")
        graph_conv = torch.einsum("bthi,ij->bthj", first_mul, self.weight)
        if self.bias is not None:
            graph_conv = torch.add(graph_conv, self.bias)
        return graph_conv


class DynamicGraphConvLayer(nn.Module):
    def __init__(self, graph_conv_type, c_in, c_out, Ks, bias):
        super().__init__()
        self.graph_conv_type = graph_conv_type
        self.align = Align(c_in, c_out)
        if self.graph_conv_type == "cheb_graph_conv":
            self.cheb_graph_conv = DynamicChebGraphConv(c_out, c_out, Ks, bias)
        elif self.graph_conv_type == "graph_conv":
            self.graph_conv = DynamicGraphConv(c_out, c_out, bias)
        else:
            raise ValueError(f"Unsupported graph_conv_type: {graph_conv_type}")

    def forward(self, x, gso):
        x_gc_in = self.align(x)
        if self.graph_conv_type == "cheb_graph_conv":
            x_gc = self.cheb_graph_conv(x_gc_in, gso)
        else:
            x_gc = self.graph_conv(x_gc_in, gso)
        x_gc = x_gc.permute(0, 3, 1, 2)
        return torch.add(x_gc, x_gc_in)


class DynamicSTConvBlock(nn.Module):
    def __init__(self, Kt, Ks, n_vertex, last_block_channel, channels, act_func, graph_conv_type, bias, droprate):
        super().__init__()
        self.tmp_conv1 = TemporalConvLayer(Kt, last_block_channel, channels[0], n_vertex, act_func)
        self.graph_conv = DynamicGraphConvLayer(graph_conv_type, channels[0], channels[1], Ks, bias)
        self.tmp_conv2 = TemporalConvLayer(Kt, channels[1], channels[2], n_vertex, act_func)
        self.tc2_ln = nn.LayerNorm([n_vertex, channels[2]])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x, gso):
        x = self.tmp_conv1(x)
        x = self.graph_conv(x, gso)
        x = self.relu(x)
        x = self.tmp_conv2(x)
        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(x)


class DynamicThresholdSTGCN(nn.Module):
    """STGCN with batch-conditioned soft or hard threshold graph support."""

    def __init__(self, Kt, Ks, blocks, T, n_vertex, act_func, graph_conv_type, dynamic_graph, bias, droprate):
        super().__init__()
        self.dynamic_support = DynamicThresholdSupport(
            num_nodes=n_vertex,
            seq_len=T,
            normalization="sym",
            **dynamic_graph,
        )
        self.st_blocks = nn.ModuleList()
        for layer in range(len(blocks) - 3):
            self.st_blocks.append(
                DynamicSTConvBlock(
                    Kt,
                    Ks,
                    n_vertex,
                    blocks[layer][-1],
                    blocks[layer + 1],
                    act_func,
                    graph_conv_type,
                    bias,
                    droprate,
                )
            )
        Ko = T - (len(blocks) - 3) * 2 * (Kt - 1)
        self.Ko = Ko
        assert Ko != 0, "Ko = 0."
        self.output = OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, act_func, bias, droprate)

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor,
        batch_seen: int,
        epoch: int,
        train: bool,
        **kwargs,
    ) -> torch.Tensor:
        gso = self.dynamic_support.symmetric_support(history_data)
        x = history_data.permute(0, 3, 1, 2).contiguous()
        for block in self.st_blocks:
            x = block(x, gso)
        x = self.output(x)
        return x.transpose(2, 3)
