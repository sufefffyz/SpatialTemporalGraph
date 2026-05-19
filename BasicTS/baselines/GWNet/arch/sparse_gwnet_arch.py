import warnings

import torch
from torch import nn
import torch.nn.functional as F

from .gwnet_arch import linear


class PackedSparseSupport(nn.Module):
    """Sparse support for Graph WaveNet nconv.

    Dense Graph WaveNet applies ``einsum('ncvl,vw->ncwl')``.  For each target
    node ``w`` it aggregates source nodes ``v`` with ``A[v, w]``.  The default
    backend stores ``A.T`` as sparse CSR and computes ``A.T @ X`` with
    ``torch.sparse.mm``; this is the path that showed actual GPU speedup in the
    SD beta1 microbenchmark.
    """

    def __init__(self, dense_support, include_self_for_empty=False, backend="csr"):
        super().__init__()
        support = torch.as_tensor(dense_support, dtype=torch.float32).detach().cpu()
        if support.ndim != 2 or support.shape[0] != support.shape[1]:
            raise ValueError(f"support must be a square matrix, got {tuple(support.shape)}")

        backend = str(backend).lower()
        if backend not in {"csr", "coo", "gather"}:
            raise ValueError(f"unsupported sparse backend: {backend}")

        num_nodes = int(support.shape[0])
        nonzero = support != 0
        empty_targets = nonzero.sum(dim=0) == 0
        if include_self_for_empty and empty_targets.any():
            idx = torch.nonzero(empty_targets, as_tuple=False).flatten()
            support[idx, idx] = 1.0
            nonzero = support != 0

        degrees = nonzero.sum(dim=0)
        max_degree = max(1, int(degrees.max().item()))
        source_index = torch.zeros((num_nodes, max_degree), dtype=torch.long)
        edge_weight = torch.zeros((num_nodes, max_degree), dtype=torch.float32)

        for target in range(num_nodes):
            sources = torch.nonzero(nonzero[:, target], as_tuple=False).flatten()
            if sources.numel() == 0:
                continue
            k = int(sources.numel())
            source_index[target, :k] = sources
            edge_weight[target, :k] = support[sources, target]

        self.num_nodes = num_nodes
        self.max_degree = max_degree
        self.num_edges = int(nonzero.sum().item())
        self.backend = backend
        if backend == "csr":
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Sparse CSR tensor support is in beta state.*",
                    category=UserWarning,
                )
                support_t = support.T.to_sparse_csr()
            self.register_buffer("support_t", support_t)
        elif backend == "coo":
            self.register_buffer("support_t", support.T.to_sparse_coo().coalesce())
        else:
            self.register_buffer("source_index", source_index)
            self.register_buffer("edge_weight", edge_weight)

    def forward(self, x):
        batch_size, channels, num_nodes, steps = x.shape
        if num_nodes != self.num_nodes:
            raise ValueError(f"x has {num_nodes} nodes, support has {self.num_nodes}")

        if self.backend in {"csr", "coo"}:
            x_flat = x.permute(2, 0, 1, 3).reshape(num_nodes, -1)
            y_flat = torch.sparse.mm(self.support_t, x_flat)
            return (
                y_flat.reshape(num_nodes, batch_size, channels, steps)
                .permute(1, 2, 0, 3)
                .contiguous()
            )

        flat_sources = self.source_index.reshape(-1)
        gathered = x.index_select(2, flat_sources)
        gathered = gathered.view(
            batch_size, channels, self.num_nodes, self.max_degree, steps
        )
        weighted = gathered * self.edge_weight.view(1, 1, self.num_nodes, self.max_degree, 1)
        return weighted.sum(dim=3).contiguous()

    def extra_repr(self):
        return (
            f"backend={self.backend}, num_nodes={self.num_nodes}, num_edges={self.num_edges}, "
            f"max_degree={self.max_degree}"
        )


class sparse_gcn(nn.Module):
    """Graph WaveNet GCN block using sparse physical and optional dense supports."""

    def __init__(self, c_in, c_out, dropout, support_len=2, order=2):
        super().__init__()
        expanded_c_in = (order * support_len + 1) * c_in
        self.mlp = linear(expanded_c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, supports):
        out = [x]
        for support in supports:
            x1 = self._apply_support(x, support)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x2 = self._apply_support(x1, support)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

    @staticmethod
    def _apply_support(x, support):
        if isinstance(support, torch.Tensor):
            return torch.einsum("ncvl,vw->ncwl", (x, support.to(x.device))).contiguous()
        return support(x)


class SparseGraphWaveNet(nn.Module):
    """
    Graph WaveNet with sparse fixed supports and optional dense adaptive support.

    Fixed physical supports are applied with CSR sparse matmul.  When
    ``addaptadj=True``, the learned adaptive adjacency follows the original
    Graph WaveNet dense formulation so the model remains semantically aligned
    with the adaptive baseline.
    """

    def __init__(
        self,
        num_nodes,
        dropout=0.3,
        supports=None,
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
        self.num_nodes = num_nodes

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

        if supports is None:
            self.supports = None
            self.supports_len = 0
        else:
            self.supports = nn.ModuleList([PackedSparseSupport(s, backend="csr") for s in supports])
            self.supports_len = len(self.supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                self.nodevec1 = nn.Parameter(
                    torch.randn(num_nodes, 10), requires_grad=True
                )
                self.nodevec2 = nn.Parameter(
                    torch.randn(10, num_nodes), requires_grad=True
                )
            else:
                aptinit = torch.as_tensor(aptinit, dtype=torch.float32)
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True)
            self.supports_len += 1

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
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool and self.supports_len > 0:
                    self.gconv.append(
                        sparse_gcn(
                            dilation_channels,
                            residual_channels,
                            dropout,
                            support_len=self.supports_len,
                        )
                    )

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

    def sparse_stats(self):
        if self.supports is None:
            stats = []
        else:
            stats = [
                {
                    "num_nodes": support.num_nodes,
                    "num_edges": support.num_edges,
                    "max_degree": support.max_degree,
                }
                for support in self.supports
            ]
        if self.addaptadj:
            stats.append(
                {
                    "num_nodes": self.num_nodes,
                    "num_edges": self.num_nodes * self.num_nodes,
                    "max_degree": self.num_nodes,
                    "adaptive_dense": True,
                }
            )
        return stats

    def forward(self, history_data, future_data, batch_seen, epoch, train, **kwargs):
        _ = future_data, batch_seen, epoch, train, kwargs
        input_data = history_data.transpose(1, 3).contiguous()
        in_len = input_data.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input_data, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input_data
        x = self.start_conv(x)
        skip = 0

        active_supports = None
        if self.gcn_bool and self.supports_len > 0:
            active_supports = list(self.supports) if self.supports is not None else []
            if self.addaptadj:
                adp = F.softmax(
                    F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1
                )
                active_supports.append(adp)

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

            if self.gcn_bool and active_supports is not None:
                x = self.gconv[i](x, active_supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3) :]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
