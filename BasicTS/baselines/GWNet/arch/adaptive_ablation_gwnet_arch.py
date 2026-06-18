import math

import torch
from torch import nn
import torch.nn.functional as F

from .gwnet_arch import GraphWaveNet


class AdaptiveAblationGraphWaveNet(GraphWaveNet):
    """Graph WaveNet with test-time adaptive-adjacency interventions.

    The parameters intentionally match ``GraphWaveNet``. The extra controls
    only alter the adaptive support behavior, so a checkpoint trained with the
    original adaptive GWNet can be loaded strictly.
    """

    VALID_EVAL_MODES = {"learned", "topk", "prune", "shuffle", "off", "zero"}

    def __init__(
        self,
        num_nodes,
        *args,
        adaptive_eval_mode="learned",
        adaptive_keep_ratio=1.0,
        adaptive_shuffle_seed=2023,
        freeze_adaptive_params=False,
        adaptive_use_relu=True,
        **kwargs,
    ):
        super().__init__(num_nodes=num_nodes, *args, **kwargs)
        mode = str(adaptive_eval_mode).lower()
        if mode not in self.VALID_EVAL_MODES:
            raise ValueError(
                f"adaptive_eval_mode must be one of {sorted(self.VALID_EVAL_MODES)}, got {adaptive_eval_mode}"
            )

        self.num_nodes = int(num_nodes)
        self.adaptive_eval_mode = mode
        self.adaptive_keep_ratio = float(adaptive_keep_ratio)
        self.adaptive_use_relu = bool(adaptive_use_relu)
        if bool(freeze_adaptive_params) and self.addaptadj:
            self.nodevec1.requires_grad_(False)
            self.nodevec2.requires_grad_(False)

        generator = torch.Generator()
        generator.manual_seed(int(adaptive_shuffle_seed))
        perm = torch.randperm(self.num_nodes, generator=generator)
        self.register_buffer("_adaptive_shuffle_perm", perm, persistent=False)

    def build_adaptive_adj(self) -> torch.Tensor:
        logits = torch.mm(self.nodevec1, self.nodevec2)
        if self.adaptive_use_relu:
            logits = F.relu(logits)
        return F.softmax(logits, dim=1)

    def transform_adaptive_adj(self, adp: torch.Tensor) -> torch.Tensor:
        """Apply the configured eval-time intervention to adaptive adjacency."""

        if self.training or self.adaptive_eval_mode == "learned":
            return adp

        if self.adaptive_eval_mode in {"off", "zero"}:
            return torch.zeros_like(adp)

        if self.adaptive_eval_mode == "shuffle":
            perm = self._adaptive_shuffle_perm.to(adp.device)
            return adp.index_select(0, perm).index_select(1, perm)

        if self.adaptive_eval_mode in {"topk", "prune"}:
            num_nodes = adp.size(-1)
            keep_ratio = min(max(self.adaptive_keep_ratio, 0.0), 1.0)
            keep_k = max(1, min(num_nodes, int(math.ceil(num_nodes * keep_ratio))))
            if keep_k >= num_nodes:
                return adp

            _, indices = torch.topk(adp, keep_k, dim=-1)
            mask = torch.zeros_like(adp)
            mask.scatter_(-1, indices, 1.0)
            return adp * mask

        raise RuntimeError(f"Unhandled adaptive_eval_mode: {self.adaptive_eval_mode}")

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """Feedforward function with optional eval-time adaptive graph ablation."""

        input = history_data.transpose(1, 3).contiguous()
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = self.build_adaptive_adj()
            adp = self.transform_adaptive_adj(adp)
            new_supports = self.supports + [adp]

        for i in range(self.blocks * self.layers):
            residual = x
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except Exception:
                skip = 0
            skip = s + skip

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x


class SignalMLPGraphWaveNet(GraphWaveNet):
    """Graph WaveNet with a batch-specific adaptive graph from shared signal MLPs."""

    def __init__(
        self,
        num_nodes,
        *args,
        signal_input_len=12,
        signal_feature_dim=2,
        signal_hidden_dim=32,
        signal_embedding_dim=10,
        signal_graph_relu=True,
        **kwargs,
    ):
        super().__init__(num_nodes=num_nodes, *args, **kwargs)
        if hasattr(self, "nodevec1"):
            del self.nodevec1
        if hasattr(self, "nodevec2"):
            del self.nodevec2

        signal_input_dim = int(signal_input_len) * int(signal_feature_dim)
        self.signal_input_len = int(signal_input_len)
        self.signal_feature_dim = int(signal_feature_dim)
        self.signal_graph_relu = bool(signal_graph_relu)
        self.signal_query_mlp = nn.Sequential(
            nn.Linear(signal_input_dim, int(signal_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(signal_hidden_dim), int(signal_embedding_dim)),
        )
        self.signal_key_mlp = nn.Sequential(
            nn.Linear(signal_input_dim, int(signal_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(signal_hidden_dim), int(signal_embedding_dim)),
        )

    def build_signal_adaptive_adj(self, history_data: torch.Tensor) -> torch.Tensor:
        batch_size, input_len, num_nodes, num_features = history_data.shape
        if input_len != self.signal_input_len or num_features != self.signal_feature_dim:
            raise ValueError(
                "Signal MLP graph expected history_data shape "
                f"[B, {self.signal_input_len}, N, {self.signal_feature_dim}], "
                f"got {tuple(history_data.shape)}"
            )

        node_signal = history_data.permute(0, 2, 1, 3).contiguous()
        node_signal = node_signal.view(batch_size, num_nodes, input_len * num_features)
        query = self.signal_query_mlp(node_signal)
        key = self.signal_key_mlp(node_signal)
        logits = torch.matmul(query, key.transpose(1, 2))
        if self.signal_graph_relu:
            logits = F.relu(logits)
        return F.softmax(logits, dim=-1)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """Feedforward function with a batch-specific signal-derived support."""

        input = history_data.transpose(1, 3).contiguous()
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = self.build_signal_adaptive_adj(history_data)
            new_supports = self.supports + [adp]

        for i in range(self.blocks * self.layers):
            residual = x
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except Exception:
                skip = 0
            skip = s + skip

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
