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
        matmul_mode: str = "scatter",
    ) -> None:
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.num_nodes = int(num_nodes)
        self.chunk_size = int(chunk_size)
        self.matmul_mode = str(matmul_mode or "scatter").lower()


def dynamic_edge_support_matmul_3d(support: DynamicEdgeSupport, x: torch.Tensor) -> torch.Tensor:
    """Apply a dynamic sparse support to ``x`` with shape ``[B, N, F]``."""

    if x.dim() != 3:
        raise ValueError(f"Expected x rank 3, got {x.dim()}.")
    batch_size, num_nodes, features = x.shape
    if num_nodes != support.num_nodes:
        raise ValueError(f"x has {num_nodes} nodes, support has {support.num_nodes}.")

    edge_index = support.edge_index.to(x.device)
    edge_weight = support.edge_weight.to(x.device)
    if support.matmul_mode in {"block_sparse_mm", "torch_sparse"}:
        src, dst = edge_index[0], edge_index[1]
        x_flat = x.reshape(batch_size * num_nodes, features)
        out = _batched_edge_sparse_mm(
            src=src,
            dst=dst,
            edge_weight=edge_weight,
            x_flat=x_flat,
            batch_size=batch_size,
            num_nodes=num_nodes,
            matmul_mode=support.matmul_mode,
        )
        return out.view(batch_size, num_nodes, features)
    if support.matmul_mode == "sparse_mm":
        src, dst = edge_index[0], edge_index[1]
        sparse_index = torch.stack([dst, src], dim=0)
        outputs = []
        for batch_idx in range(batch_size):
            adj = torch.sparse_coo_tensor(
                sparse_index,
                edge_weight[batch_idx],
                size=(num_nodes, num_nodes),
                device=x.device,
            ).coalesce()
            outputs.append(torch.sparse.mm(adj, x[batch_idx]))
        return torch.stack(outputs, dim=0)
    if support.matmul_mode != "scatter":
        raise ValueError(f"Unsupported dynamic edge matmul mode: {support.matmul_mode}")
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
    if support.matmul_mode in {"block_sparse_mm", "torch_sparse"}:
        src, dst = edge_index[0], edge_index[1]
        features = channels * steps
        x_flat = x.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, features)
        out = _batched_edge_sparse_mm(
            src=src,
            dst=dst,
            edge_weight=edge_weight,
            x_flat=x_flat,
            batch_size=batch_size,
            num_nodes=num_nodes,
            matmul_mode=support.matmul_mode,
        )
        return out.view(batch_size, num_nodes, channels, steps).permute(0, 2, 1, 3).contiguous()
    if support.matmul_mode == "sparse_mm":
        src, dst = edge_index[0], edge_index[1]
        sparse_index = torch.stack([dst, src], dim=0)
        outputs = []
        for batch_idx in range(batch_size):
            adj = torch.sparse_coo_tensor(
                sparse_index,
                edge_weight[batch_idx],
                size=(num_nodes, num_nodes),
                device=x.device,
            ).coalesce()
            x_batch = x[batch_idx].permute(1, 0, 2).reshape(num_nodes, channels * steps)
            out_batch = torch.sparse.mm(adj, x_batch)
            outputs.append(out_batch.view(num_nodes, channels, steps).permute(1, 0, 2))
        return torch.stack(outputs, dim=0)
    if support.matmul_mode != "scatter":
        raise ValueError(f"Unsupported dynamic edge matmul mode: {support.matmul_mode}")
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


def _batched_edge_sparse_mm(
    src: torch.Tensor,
    dst: torch.Tensor,
    edge_weight: torch.Tensor,
    x_flat: torch.Tensor,
    batch_size: int,
    num_nodes: int,
    matmul_mode: str,
) -> torch.Tensor:
    """Apply batch-conditioned edges as one block-diagonal sparse matmul."""

    edge_count = int(src.numel())
    batch_offset = torch.arange(batch_size, device=x_flat.device, dtype=src.dtype).view(batch_size, 1) * num_nodes
    src_flat = (src.view(1, edge_count) + batch_offset).reshape(-1)
    dst_flat = (dst.view(1, edge_count) + batch_offset).reshape(-1)
    value_flat = edge_weight.reshape(-1)
    sparse_size = (batch_size * num_nodes, batch_size * num_nodes)
    if matmul_mode == "torch_sparse":
        try:
            from torch_sparse import SparseTensor
        except ImportError as exc:
            raise ImportError("DYNAMIC_GRAPH_EDGE_MATMUL=torch_sparse requires torch_sparse.") from exc
        adj = SparseTensor(
            row=dst_flat,
            col=src_flat,
            value=value_flat,
            sparse_sizes=sparse_size,
        )
        return adj.matmul(x_flat)
    if matmul_mode == "block_sparse_mm":
        adj = torch.sparse_coo_tensor(
            torch.stack([dst_flat, src_flat], dim=0),
            value_flat,
            size=sparse_size,
            device=x_flat.device,
        )
        return torch.sparse.mm(adj, x_flat)
    raise ValueError(f"Unsupported dynamic edge matmul mode: {matmul_mode}")


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
        init_mode: str = "scalar",
        init_degree_min: int = 8,
        init_degree_max: int = 64,
        init_seed: int = 2023,
        radius_scale: float = 1.0,
        radius_param: str = "exp_tanh",
        quantile_min: float = 0.05,
        quantile_max: float = 1.0,
        quantile_init: float | None = None,
        degree_min: float = 8.0,
        degree_max: float = 64.0,
        degree_init: float | None = None,
        state_act: str = "tanh",
        quantile_temperature: float = 1.0,
        temperature: float = 0.05,
        dist_norm: str = "max",
        gaussian_sigma: float | None = None,
        weight_mode: str = "binary",
        normalization: str = "transition",
        self_loops: bool = True,
        straight_through: bool = True,
        edge_chunk_size: int = 2048,
        edge_matmul_mode: str = "scatter",
        edge_output: bool = False,
    ) -> None:
        super().__init__()
        if mode not in {"soft", "hard"}:
            raise ValueError(f"Unsupported dynamic threshold mode: {mode}")
        if weight_mode not in {"binary", "gaussian", "learned"}:
            raise ValueError(f"Unsupported dynamic threshold weight mode: {weight_mode}")
        if normalization not in {"transition", "sym"}:
            raise ValueError(f"Unsupported support normalization: {normalization}")
        radius_param = str(radius_param or "exp_tanh").lower()
        if radius_param not in {"exp_tanh", "softplus", "quantile", "degree_quantile"}:
            raise ValueError(f"Unsupported radius parameterization: {radius_param}")
        init_mode = str(init_mode or "scalar").lower()
        if init_mode not in {"scalar", "random_degree"}:
            raise ValueError(f"Unsupported dynamic threshold init mode: {init_mode}")
        state_act = str(state_act or "tanh").lower()
        if state_act not in {"tanh", "identity"}:
            raise ValueError(f"Unsupported dynamic threshold state_act: {state_act}")
        if not 0.0 <= quantile_min <= quantile_max <= 1.0:
            raise ValueError("quantile_min and quantile_max must satisfy 0 <= min <= max <= 1.")
        if degree_min <= 0 or degree_max < degree_min:
            raise ValueError("degree_min and degree_max must satisfy 0 < min <= max.")
        if quantile_temperature <= 0:
            raise ValueError("quantile_temperature must be positive.")

        self.num_nodes = int(num_nodes)
        self.seq_len = int(seq_len)
        self.mode = mode
        self.init_mode = init_mode
        self.radius_scale = float(radius_scale)
        self.radius_param = radius_param
        if self.radius_param == "softplus" and self.radius_scale <= 0:
            raise ValueError("radius_scale must be positive when radius_param='softplus'.")
        self.quantile_min = float(quantile_min)
        self.quantile_max = float(quantile_max)
        self.quantile_init = None if quantile_init is None else float(quantile_init)
        self.degree_min = float(degree_min)
        self.degree_max = float(degree_max)
        self.degree_init = None if degree_init is None else float(degree_init)
        self.state_act = state_act
        self.quantile_temperature = float(quantile_temperature)
        self.temperature = float(temperature)
        self.weight_mode = weight_mode
        self.normalization = normalization
        self.self_loops = bool(self_loops)
        self.straight_through = bool(straight_through)
        self.edge_chunk_size = int(edge_chunk_size)
        self.edge_matmul_mode = str(edge_matmul_mode or "scatter").lower()
        if self.edge_matmul_mode not in {"scatter", "sparse_mm", "block_sparse_mm", "torch_sparse"}:
            raise ValueError(f"Unsupported dynamic edge matmul mode: {self.edge_matmul_mode}")
        self.edge_output = bool(edge_output)

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
            init_radius_node = self._init_node_radius(
                dist,
                target_avg_degree=target_avg_degree,
                candidate_mask=candidate_mask,
                init_mode=init_mode,
                init_degree_min=init_degree_min,
                init_degree_max=init_degree_max,
                init_seed=init_seed,
            )
        else:
            init_radius_node = torch.full((self.num_nodes,), float(init_radius), dtype=torch.float32)
        self.init_radius = float(init_radius_node.mean().item())
        self.register_buffer("init_radius_node", init_radius_node.float())
        quantile_distances, quantile_counts = self._build_quantile_table(dist, candidate_mask)
        self.register_buffer("quantile_distances", quantile_distances.float())
        self.register_buffer("quantile_counts", quantile_counts.long())
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
        if self.weight_mode == "learned":
            self.edge_src_proj = nn.Linear(d_model, d_model, bias=False)
            self.edge_dst_proj = nn.Linear(d_model, d_model, bias=False)
            self.edge_weight_bias = nn.Parameter(torch.zeros(1))
        else:
            self.edge_src_proj = None
            self.edge_dst_proj = None
            self.edge_weight_bias = None
        self.reset_parameters()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Older DynamicThreshold checkpoints do not contain newer buffers. Reuse
        # the config-built values so strict loading remains backward compatible.
        for name in ("init_radius_node", "quantile_distances", "quantile_counts"):
            key = prefix + name
            if key not in state_dict:
                state_dict[key] = getattr(self, name)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @classmethod
    def _init_node_radius(
        cls,
        distance: torch.Tensor,
        target_avg_degree: float,
        candidate_mask: torch.Tensor,
        init_mode: str,
        init_degree_min: int,
        init_degree_max: int,
        init_seed: int,
    ) -> torch.Tensor:
        if init_mode == "scalar":
            radius = cls._degree_to_radius(distance, target_avg_degree, candidate_mask)
            return torch.full((int(distance.shape[0]),), float(radius), dtype=torch.float32)
        return cls._random_degree_to_node_radius(
            distance,
            init_degree_min=init_degree_min,
            init_degree_max=init_degree_max,
            candidate_mask=candidate_mask,
            init_seed=init_seed,
        )

    @staticmethod
    def _build_quantile_table(
        distance: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = int(distance.shape[0])
        mask = candidate_mask.to(device=distance.device, dtype=torch.bool)
        sorted_rows: list[torch.Tensor] = []
        counts = torch.empty(n, dtype=torch.long, device=distance.device)
        max_count = 1
        for node in range(n):
            values = distance[node][mask[node]]
            values = values[torch.isfinite(values)]
            values = values[values > 0]
            if values.numel() == 0:
                values = torch.ones(1, dtype=distance.dtype, device=distance.device)
            values = torch.sort(values).values
            sorted_rows.append(values)
            counts[node] = int(values.numel())
            max_count = max(max_count, int(values.numel()))

        table = torch.empty((n, max_count), dtype=distance.dtype, device=distance.device)
        for node, values in enumerate(sorted_rows):
            count = int(values.numel())
            table[node, :count] = values
            if count < max_count:
                table[node, count:] = values[-1]
        return table.cpu(), counts.cpu()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.history_proj.weight)
        nn.init.zeros_(self.history_proj.bias)
        nn.init.zeros_(self.radius_head.weight)
        if self.edge_src_proj is not None:
            nn.init.xavier_uniform_(self.edge_src_proj.weight)
            nn.init.xavier_uniform_(self.edge_dst_proj.weight)
            nn.init.zeros_(self.edge_weight_bias)
        if self.radius_param == "softplus":
            nn.init.constant_(self.radius_head.bias, math.log(math.expm1(1.0)) / self.radius_scale)
        elif self.radius_param == "quantile":
            if self.quantile_init is None:
                target = 0.5
            elif self.quantile_max > self.quantile_min:
                target = (self.quantile_init - self.quantile_min) / (self.quantile_max - self.quantile_min)
            else:
                target = 0.5
            nn.init.constant_(self.radius_head.bias, self._safe_logit(target))
        elif self.radius_param == "degree_quantile":
            if self.degree_init is None:
                target_degree = 0.5 * (self.degree_min + self.degree_max)
            else:
                target_degree = self.degree_init
            target = (target_degree - self.degree_min) / max(self.degree_max - self.degree_min, 1e-6)
            nn.init.constant_(self.radius_head.bias, self._safe_logit(target))
        else:
            nn.init.zeros_(self.radius_head.bias)

    @staticmethod
    def _safe_logit(value: float) -> float:
        value = min(max(float(value), 1e-4), 1.0 - 1e-4)
        return math.log(value / (1.0 - value))

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

    @staticmethod
    def _random_degree_to_node_radius(
        distance: torch.Tensor,
        init_degree_min: int,
        init_degree_max: int,
        candidate_mask: torch.Tensor,
        init_seed: int,
    ) -> torch.Tensor:
        n = int(distance.shape[0])
        k_min = max(1, int(init_degree_min))
        k_max = max(k_min, int(init_degree_max))
        mask = candidate_mask.to(device=distance.device, dtype=torch.bool)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(init_seed))
        target_degrees = torch.randint(k_min, k_max + 1, (n,), generator=generator)
        radii = torch.empty(n, dtype=distance.dtype, device=distance.device)
        for node in range(n):
            values = distance[node][mask[node]]
            values = values[torch.isfinite(values)]
            values = values[values > 0]
            if values.numel() == 0:
                radii[node] = 1e-3
                continue
            k = min(int(target_degrees[node].item()), int(values.numel()))
            radii[node] = torch.sort(values).values[k - 1]
        return radii.cpu().float()

    def _radius_from_quantile(self, quantile: torch.Tensor) -> torch.Tensor:
        counts = self.quantile_counts.to(quantile.device).float().unsqueeze(0)
        position = quantile.clamp(0.0, 1.0) * (counts - 1.0).clamp_min(0.0)
        return self._radius_from_position(position)

    def _radius_from_degree(self, degree: torch.Tensor) -> torch.Tensor:
        counts = self.quantile_counts.to(degree.device).float().unsqueeze(0)
        degree = torch.minimum(degree, counts).clamp_min(1.0)
        return self._radius_from_position(degree - 1.0)

    def _radius_from_position(self, position: torch.Tensor) -> torch.Tensor:
        table = self.quantile_distances.to(position.device)
        counts = self.quantile_counts.to(position.device).long().unsqueeze(0)
        max_position = (counts - 1).clamp_min(0)
        position = torch.minimum(position.clamp_min(0.0), max_position.float())
        lower = torch.floor(position).long()
        upper = torch.minimum(lower + 1, max_position)
        frac = position - lower.float()
        expanded_table = table.unsqueeze(0).expand(position.shape[0], -1, -1)
        lower_radius = torch.gather(expanded_table, 2, lower.unsqueeze(-1)).squeeze(-1)
        upper_radius = torch.gather(expanded_table, 2, upper.unsqueeze(-1)).squeeze(-1)
        return lower_radius + (upper_radius - lower_radius) * frac

    def _node_state(self, history_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if history_data.dim() == 4:
            history = history_data[..., 0]
        else:
            history = history_data
        if history.shape[1] != self.seq_len:
            raise ValueError(f"Expected history length {self.seq_len}, got {history.shape[1]}.")
        node_history = history.transpose(1, 2).contiguous()
        state = self.history_proj(node_history)
        if self.state_act == "tanh":
            state = torch.tanh(state)
        node_embed = self.node_embed.unsqueeze(0).expand(state.shape[0], -1, -1)
        return state, node_embed

    def _node_radius(self, history_data: torch.Tensor) -> torch.Tensor:
        state, node_embed = self._node_state(history_data)
        score = self.radius_head(torch.cat([state, node_embed], dim=-1)).squeeze(-1)
        quantile_score = score / self.quantile_temperature
        if self.radius_param == "quantile":
            quantile = self.quantile_min + (self.quantile_max - self.quantile_min) * torch.sigmoid(quantile_score)
            return self._radius_from_quantile(quantile)
        if self.radius_param == "degree_quantile":
            degree = self.degree_min + (self.degree_max - self.degree_min) * torch.sigmoid(quantile_score)
            return self._radius_from_degree(degree)
        if self.radius_param == "softplus":
            multiplier = F.softplus(self.radius_scale * score)
        else:
            delta = self.radius_scale * torch.tanh(score)
            multiplier = torch.exp(delta)
        return self.init_radius_node.unsqueeze(0).to(multiplier.device) * multiplier

    def _learned_dense_weights(self, history_data: torch.Tensor) -> torch.Tensor:
        state, node_embed = self._node_state(history_data)
        edge_state = state + node_embed
        q = self.edge_src_proj(edge_state)
        k = self.edge_dst_proj(edge_state)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        logits = logits + self.edge_weight_bias
        return torch.sigmoid(logits)

    def _learned_edge_weights(self, history_data: torch.Tensor) -> torch.Tensor:
        state, node_embed = self._node_state(history_data)
        edge_state = state + node_embed
        q = self.edge_src_proj(edge_state)
        k = self.edge_dst_proj(edge_state)
        src = self.candidate_src
        dst = self.candidate_dst
        logits = (q[:, src, :] * k[:, dst, :]).sum(dim=-1) / math.sqrt(q.shape[-1])
        logits = logits + self.edge_weight_bias
        return torch.sigmoid(logits)

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
        if self.weight_mode == "learned":
            mask = mask * self._learned_dense_weights(history_data)
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
        if self.weight_mode == "learned":
            mask = mask * self._learned_edge_weights(history_data)
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
            DynamicEdgeSupport(
                forward_edge_index,
                forward_weight,
                self.num_nodes,
                self.edge_chunk_size,
                self.edge_matmul_mode,
            ),
            DynamicEdgeSupport(
                backward_edge_index,
                backward_weight,
                self.num_nodes,
                self.edge_chunk_size,
                self.edge_matmul_mode,
            ),
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
