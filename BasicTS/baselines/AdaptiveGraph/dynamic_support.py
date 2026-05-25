from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def load_distance_matrix(path: str, num_nodes: int, normalize: str = "max") -> torch.Tensor:
    matrix_path = Path(path).expanduser()
    if not matrix_path.exists():
        raise FileNotFoundError(f"Dynamic graph distance matrix not found: {matrix_path}")
    dist = np.load(matrix_path).astype("float32")
    if dist.shape[0] < num_nodes or dist.shape[1] < num_nodes:
        raise ValueError(f"Distance matrix shape {dist.shape} is smaller than num_nodes={num_nodes}.")
    dist = dist[:num_nodes, :num_nodes]
    finite = np.isfinite(dist)
    if not finite.all():
        max_finite = float(np.nanmax(dist[finite])) if finite.any() else 1.0
        dist = np.where(finite, dist, max_finite)
    dist = np.maximum(dist, 0.0)
    np.fill_diagonal(dist, 0.0)

    normalize = str(normalize or "none").lower()
    positive = dist[dist > 0]
    if positive.size and normalize not in {"none", "identity", "raw"}:
        if normalize == "p95":
            scale = float(np.quantile(positive, 0.95))
        elif normalize == "median":
            scale = float(np.median(positive))
        else:
            scale = float(np.max(positive))
        if scale > 0:
            dist = dist / scale
    return torch.from_numpy(dist.astype("float32"))


def load_candidate_mask(path: str | None, num_nodes: int) -> torch.Tensor:
    if not path:
        return torch.ones((num_nodes, num_nodes), dtype=torch.float32) - torch.eye(num_nodes, dtype=torch.float32)
    matrix_path = Path(path).expanduser()
    if not matrix_path.exists():
        raise FileNotFoundError(f"Dynamic graph candidate adjacency not found: {matrix_path}")
    with matrix_path.open("rb") as f:
        adj = pickle.load(f, encoding="latin1")
    if isinstance(adj, (list, tuple)):
        if len(adj) == 3:
            adj = adj[2]
        elif len(adj) == 1:
            adj = adj[0]
    adj = np.asarray(adj, dtype="float32")
    if adj.shape[0] < num_nodes or adj.shape[1] < num_nodes:
        raise ValueError(f"Candidate adjacency shape {adj.shape} is smaller than num_nodes={num_nodes}.")
    adj = adj[:num_nodes, :num_nodes]
    mask = (adj > 0).astype("float32")
    np.fill_diagonal(mask, 0.0)
    return torch.from_numpy(mask)


class DynamicEdgeSupport:
    """Batch-conditioned sparse support on a fixed candidate edge set.

    ``edge_index[0]`` and ``edge_index[1]`` store the row and column indices of
    the support matrix. ``edge_weight`` is batch-dependent with shape ``[B, E]``.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        num_nodes: int,
        chunk_size: int = 2048,
    ) -> None:
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.num_nodes = int(num_nodes)
        self.chunk_size = int(chunk_size)


def dynamic_edge_support_matmul_3d(support: DynamicEdgeSupport, x: torch.Tensor) -> torch.Tensor:
    """Apply a dynamic sparse support to ``x`` with shape ``[B, N, F]``."""

    if x.dim() != 3:
        raise ValueError(f"Expected x rank 3, got {x.dim()}.")
    batch_size, num_nodes, features = x.shape
    if num_nodes != support.num_nodes:
        raise ValueError(f"x has {num_nodes} nodes, support has {support.num_nodes}.")

    edge_index = support.edge_index.to(x.device)
    edge_weight = support.edge_weight.to(x.device)
    row, col = edge_index[0], edge_index[1]
    out = x.new_zeros(batch_size, num_nodes, features)
    chunk_size = max(1, support.chunk_size)
    for start in range(0, row.numel(), chunk_size):
        end = min(start + chunk_size, row.numel())
        row_chunk = row[start:end]
        col_chunk = col[start:end]
        msg = x.index_select(1, col_chunk) * edge_weight[:, start:end].unsqueeze(-1)
        dst_index = row_chunk.view(1, -1, 1).expand(batch_size, -1, features)
        out.scatter_add_(1, dst_index, msg)
    return out


def dynamic_edge_support_matmul_4d(support: DynamicEdgeSupport, x: torch.Tensor) -> torch.Tensor:
    """Apply a dynamic sparse support to ``x`` with shape ``[B, C, N, T]``."""

    if x.dim() != 4:
        raise ValueError(f"Expected x rank 4, got {x.dim()}.")
    batch_size, channels, num_nodes, steps = x.shape
    if num_nodes != support.num_nodes:
        raise ValueError(f"x has {num_nodes} nodes, support has {support.num_nodes}.")

    edge_index = support.edge_index.to(x.device)
    edge_weight = support.edge_weight.to(x.device)
    src, dst = edge_index[0], edge_index[1]
    out = x.new_zeros(batch_size, channels, num_nodes, steps)
    chunk_size = max(1, support.chunk_size)
    for start in range(0, src.numel(), chunk_size):
        end = min(start + chunk_size, src.numel())
        src_chunk = src[start:end]
        dst_chunk = dst[start:end]
        msg = x.index_select(2, src_chunk) * edge_weight[:, start:end].view(batch_size, 1, -1, 1)
        dst_index = dst_chunk.view(1, 1, -1, 1).expand(batch_size, channels, -1, steps)
        out.scatter_add_(2, dst_index, msg)
    return out


class DynamicThresholdSupport(nn.Module):
    """FlowNet-style node-wise dynamic radius converted into graph supports.

    The soft mode keeps a dense differentiable mask. The hard mode uses a hard
    distance-threshold mask in the forward pass with a straight-through sigmoid
    gradient, so it is the candidate that can later be paired with sparse
    gather/CSR kernels.
    """

    def __init__(
        self,
        num_nodes: int,
        seq_len: int,
        dist_mtx_path: str,
        candidate_adj_path: str | None = None,
        mode: str = "hard",
        d_model: int = 32,
        target_avg_degree: float = 24.1885,
        init_radius: float | None = None,
        radius_scale: float = 1.0,
        radius_param: str = "exp_tanh",
        temperature: float = 0.05,
        dist_norm: str = "max",
        gaussian_sigma: float | None = None,
        weight_mode: str = "binary",
        normalization: str = "transition",
        self_loops: bool = True,
        straight_through: bool = True,
        edge_chunk_size: int = 2048,
    ) -> None:
        super().__init__()
        if mode not in {"soft", "hard"}:
            raise ValueError(f"Unsupported dynamic threshold mode: {mode}")
        if weight_mode not in {"binary", "gaussian"}:
            raise ValueError(f"Unsupported dynamic threshold weight mode: {weight_mode}")
        if normalization not in {"transition", "sym"}:
            raise ValueError(f"Unsupported support normalization: {normalization}")
        radius_param = str(radius_param or "exp_tanh").lower()
        if radius_param not in {"exp_tanh", "softplus"}:
            raise ValueError(f"Unsupported radius parameterization: {radius_param}")

        self.num_nodes = int(num_nodes)
        self.seq_len = int(seq_len)
        self.mode = mode
        self.radius_scale = float(radius_scale)
        self.radius_param = radius_param
        if self.radius_param == "softplus" and self.radius_scale <= 0:
            raise ValueError("radius_scale must be positive when radius_param='softplus'.")
        self.temperature = float(temperature)
        self.weight_mode = weight_mode
        self.normalization = normalization
        self.self_loops = bool(self_loops)
        self.straight_through = bool(straight_through)
        self.edge_chunk_size = int(edge_chunk_size)
        self.edge_output = candidate_adj_path is not None

        dist = load_distance_matrix(dist_mtx_path, self.num_nodes, dist_norm)
        self.register_buffer("distance", dist)
        self.register_buffer("eye", torch.eye(self.num_nodes, dtype=torch.float32))
        candidate_mask = load_candidate_mask(candidate_adj_path, self.num_nodes)
        self.register_buffer("candidate_mask", candidate_mask)
        candidate_src, candidate_dst = torch.nonzero(candidate_mask > 0, as_tuple=True)
        self.register_buffer("candidate_src", candidate_src.long())
        self.register_buffer("candidate_dst", candidate_dst.long())
        self.register_buffer("self_loop_nodes", torch.arange(self.num_nodes, dtype=torch.long))

        if init_radius is None:
            init_radius = self._degree_to_radius(dist, target_avg_degree, candidate_mask)
        self.init_radius = float(init_radius)
        if self.weight_mode == "gaussian":
            sigma = float(gaussian_sigma) if gaussian_sigma is not None else max(self.init_radius, 1e-3)
            base_weight = torch.exp(-torch.square(dist / max(sigma, 1e-6)))
        else:
            base_weight = torch.ones_like(dist)
        base_weight = base_weight * (1.0 - self.eye) + self.eye
        self.register_buffer("base_weight", base_weight.float())

        self.history_proj = nn.Linear(self.seq_len, d_model)
        self.node_embed = nn.Parameter(torch.randn(self.num_nodes, d_model) * 0.02)
        self.radius_head = nn.Linear(d_model * 2, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.history_proj.weight)
        nn.init.zeros_(self.history_proj.bias)
        nn.init.zeros_(self.radius_head.weight)
        if self.radius_param == "softplus":
            nn.init.constant_(self.radius_head.bias, math.log(math.expm1(1.0)) / self.radius_scale)
        else:
            nn.init.zeros_(self.radius_head.bias)

    @staticmethod
    def _degree_to_radius(
        distance: torch.Tensor,
        target_avg_degree: float,
        candidate_mask: torch.Tensor | None = None,
    ) -> float:
        n = int(distance.shape[0])
        target_edges = int(round(float(target_avg_degree) * n))
        if target_edges <= 0:
            return 1e-3
        if candidate_mask is None:
            mask = ~torch.eye(n, dtype=torch.bool, device=distance.device)
        else:
            mask = candidate_mask.to(device=distance.device, dtype=torch.bool)
        values = distance[mask]
        values = values[torch.isfinite(values)]
        values = values[values > 0]
        if values.numel() == 0:
            return 1.0
        target_edges = max(1, min(target_edges, int(values.numel())))
        sorted_values = torch.sort(values).values
        return float(sorted_values[target_edges - 1].item())

    def _node_radius(self, history_data: torch.Tensor) -> torch.Tensor:
        if history_data.dim() == 4:
            history = history_data[..., 0]
        else:
            history = history_data
        if history.shape[1] != self.seq_len:
            raise ValueError(f"Expected history length {self.seq_len}, got {history.shape[1]}.")
        node_history = history.transpose(1, 2).contiguous()
        state = torch.tanh(self.history_proj(node_history))
        node_embed = self.node_embed.unsqueeze(0).expand(state.shape[0], -1, -1)
        score = self.radius_head(torch.cat([state, node_embed], dim=-1)).squeeze(-1)
        if self.radius_param == "softplus":
            multiplier = F.softplus(self.radius_scale * score)
        else:
            delta = self.radius_scale * torch.tanh(score)
            multiplier = torch.exp(delta)
        return self.init_radius * multiplier

    def _masked_weights(self, history_data: torch.Tensor) -> torch.Tensor:
        radius = self._node_radius(history_data)
        logits = (radius.unsqueeze(-1) - self.distance.unsqueeze(0)) / max(self.temperature, 1e-6)
        soft_mask = torch.sigmoid(logits)
        if self.mode == "hard":
            hard_mask = (logits >= 0).to(soft_mask.dtype)
            if self.straight_through:
                mask = hard_mask + soft_mask - soft_mask.detach()
            else:
                mask = hard_mask
        else:
            mask = soft_mask

        mask = mask * self.candidate_mask.unsqueeze(0)
        if self.self_loops:
            mask = mask * (1.0 - self.eye.unsqueeze(0)) + self.eye.unsqueeze(0)
        else:
            mask = mask * (1.0 - self.eye.unsqueeze(0))
        return mask * self.base_weight.unsqueeze(0)

    def _candidate_edge_weights(self, history_data: torch.Tensor) -> torch.Tensor:
        radius = self._node_radius(history_data)
        src = self.candidate_src
        dst = self.candidate_dst
        logits = (radius[:, src] - self.distance[src, dst].unsqueeze(0)) / max(self.temperature, 1e-6)
        soft_mask = torch.sigmoid(logits)
        if self.mode == "hard":
            hard_mask = (logits >= 0).to(soft_mask.dtype)
            if self.straight_through:
                mask = hard_mask + soft_mask - soft_mask.detach()
            else:
                mask = hard_mask
        else:
            mask = soft_mask
        return mask * self.base_weight[src, dst].unsqueeze(0)

    def _transition_edge_supports(self, history_data: torch.Tensor) -> list[DynamicEdgeSupport]:
        edge_weight = self._candidate_edge_weights(history_data)
        batch_size = edge_weight.shape[0]
        src = self.candidate_src
        dst = self.candidate_dst

        row_sum = edge_weight.new_zeros(batch_size, self.num_nodes)
        row_sum.scatter_add_(1, src.unsqueeze(0).expand(batch_size, -1), edge_weight)
        col_sum = edge_weight.new_zeros(batch_size, self.num_nodes)
        col_sum.scatter_add_(1, dst.unsqueeze(0).expand(batch_size, -1), edge_weight)

        if self.self_loops:
            loop_weight = edge_weight.new_ones(batch_size, self.num_nodes)
            row_sum = row_sum + loop_weight
            col_sum = col_sum + loop_weight
            loop_nodes = self.self_loop_nodes
            forward_edge_index = torch.stack(
                [torch.cat([dst, loop_nodes]), torch.cat([src, loop_nodes])],
                dim=0,
            )
            backward_edge_index = torch.stack(
                [torch.cat([src, loop_nodes]), torch.cat([dst, loop_nodes])],
                dim=0,
            )
            forward_weight = torch.cat(
                [
                    edge_weight / row_sum[:, src].clamp_min(1e-6),
                    loop_weight / row_sum.clamp_min(1e-6),
                ],
                dim=1,
            )
            backward_weight = torch.cat(
                [
                    edge_weight / col_sum[:, dst].clamp_min(1e-6),
                    loop_weight / col_sum.clamp_min(1e-6),
                ],
                dim=1,
            )
        else:
            forward_edge_index = torch.stack([dst, src], dim=0)
            backward_edge_index = torch.stack([src, dst], dim=0)
            forward_weight = edge_weight / row_sum[:, src].clamp_min(1e-6)
            backward_weight = edge_weight / col_sum[:, dst].clamp_min(1e-6)

        return [
            DynamicEdgeSupport(forward_edge_index, forward_weight, self.num_nodes, self.edge_chunk_size),
            DynamicEdgeSupport(backward_edge_index, backward_weight, self.num_nodes, self.edge_chunk_size),
        ]

    @staticmethod
    def _row_normalize(weights: torch.Tensor) -> torch.Tensor:
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def transition_supports(self, history_data: torch.Tensor) -> list[torch.Tensor | DynamicEdgeSupport]:
        if self.edge_output:
            return self._transition_edge_supports(history_data)
        weights = self._masked_weights(history_data)
        forward_adj = self._row_normalize(weights)
        backward_adj = self._row_normalize(weights.transpose(-1, -2))
        return [forward_adj.transpose(-1, -2).contiguous(), backward_adj.transpose(-1, -2).contiguous()]

    def symmetric_support(self, history_data: torch.Tensor) -> torch.Tensor:
        weights = self._masked_weights(history_data)
        weights = torch.maximum(weights, weights.transpose(-1, -2))
        degree = weights.sum(dim=-1).clamp_min(1e-6)
        inv_sqrt = torch.rsqrt(degree)
        return weights * inv_sqrt.unsqueeze(-1) * inv_sqrt.unsqueeze(-2)

    def symmetric_laplacian(self, history_data: torch.Tensor) -> torch.Tensor:
        sym_adj = self.symmetric_support(history_data)
        return self.eye.unsqueeze(0).to(sym_adj.device) - sym_adj

    def forward(self, history_data: torch.Tensor) -> list[torch.Tensor] | torch.Tensor:
        if self.normalization == "sym":
            return self.symmetric_support(history_data)
        return self.transition_supports(history_data)
