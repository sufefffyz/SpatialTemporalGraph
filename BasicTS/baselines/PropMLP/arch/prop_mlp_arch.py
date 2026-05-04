from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class PropMLP(nn.Module):
    """
    Minimal graph-propagation MLP baseline.

    Input:
        history_data: [B, L, N, C], where C is expected to be 1 after runner feature selection.

    Output:
        prediction: [B, L_out, N, 1]
    """

    def __init__(
        self,
        num_nodes: int,
        seq_len: int,
        pred_len: int,
        supports: Iterable[torch.Tensor] | None = None,
        max_order: int = 1,
        include_original: bool = True,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.max_order = int(max_order)
        self.include_original = bool(include_original)

        if supports is None:
            supports = []
        supports = [torch.as_tensor(support, dtype=torch.float32) for support in supports]
        self.support_count = len(supports)

        for idx, support in enumerate(supports):
            self.register_buffer(f"support_{idx}", support)

        feature_blocks = (1 if self.include_original else 0) + self.support_count * self.max_order
        if feature_blocks <= 0:
            raise ValueError("PropMLP needs at least one feature block. Set include_original=True or provide supports.")

        in_dim = self.seq_len * feature_blocks
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.pred_len),
        )

    def _supports(self) -> list[torch.Tensor]:
        return [getattr(self, f"support_{idx}") for idx in range(self.support_count)]

    @staticmethod
    def _apply_support(x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        # x: [B, L, N], support: [N, N]
        return torch.einsum("bln,nm->blm", x, support)

    def _build_feature_stack(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, N]
        features = []
        if self.include_original:
            features.append(x.unsqueeze(-1))

        for support in self._supports():
            propagated = x
            for _ in range(self.max_order):
                propagated = self._apply_support(propagated, support.to(x.device))
                features.append(propagated.unsqueeze(-1))

        stacked = torch.cat(features, dim=-1)  # [B, L, N, F]
        return stacked

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor,
        batch_seen: int,
        epoch: int,
        train: bool,
        **kwargs,
    ) -> torch.Tensor:
        assert history_data.shape[-1] == 1, "PropMLP expects one selected input channel."
        x = history_data[..., 0]  # [B, L, N]
        feats = self._build_feature_stack(x)  # [B, L, N, F]
        feats = feats.permute(0, 2, 1, 3).contiguous()  # [B, N, L, F]
        feats = feats.reshape(feats.shape[0], feats.shape[1], -1)  # [B, N, L*F]
        out = self.mlp(feats)  # [B, N, pred_len]
        return out.permute(0, 2, 1).unsqueeze(-1)  # [B, pred_len, N, 1]
