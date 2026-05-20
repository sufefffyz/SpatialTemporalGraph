from __future__ import annotations

import math
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
        self.temperature = float(temperature)
        self.weight_mode = weight_mode
        self.normalization = normalization
        self.self_loops = bool(self_loops)
        self.straight_through = bool(straight_through)

        dist = load_distance_matrix(dist_mtx_path, self.num_nodes, dist_norm)
        self.register_buffer("distance", dist)
        self.register_buffer("eye", torch.eye(self.num_nodes, dtype=torch.float32))

        if init_radius is None:
            init_radius = self._degree_to_radius(dist, target_avg_degree)
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
        nn.init.zeros_(self.radius_head.bias)

    @staticmethod
    def _degree_to_radius(distance: torch.Tensor, target_avg_degree: float) -> float:
        n = int(distance.shape[0])
        target_edges = int(round(float(target_avg_degree) * n))
        if target_edges <= 0:
            return 1e-3
        mask = ~torch.eye(n, dtype=torch.bool, device=distance.device)
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
            offset = math.log(math.expm1(1.0))
            multiplier = F.softplus(self.radius_scale * score + offset)
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

        if self.self_loops:
            mask = mask * (1.0 - self.eye.unsqueeze(0)) + self.eye.unsqueeze(0)
        else:
            mask = mask * (1.0 - self.eye.unsqueeze(0))
        return mask * self.base_weight.unsqueeze(0)

    @staticmethod
    def _row_normalize(weights: torch.Tensor) -> torch.Tensor:
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def transition_supports(self, history_data: torch.Tensor) -> list[torch.Tensor]:
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

    def forward(self, history_data: torch.Tensor) -> list[torch.Tensor] | torch.Tensor:
        if self.normalization == "sym":
            return self.symmetric_support(history_data)
        return self.transition_supports(history_data)
