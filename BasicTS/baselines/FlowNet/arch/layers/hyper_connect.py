import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.normalization import LayerNorm


class HyperConnection(nn.Module):
    def __init__(self, dim, rate, layer_id, dynamic=True, device=None):
        super(HyperConnection, self).__init__()
        self.rate = rate
        self.layer_id = layer_id
        self.dynamic = dynamic
        self.static_beta = nn.Parameter(torch.ones((rate,), device=device))
        init_alpha0 = torch.zeros((rate, 1), device=device)
        init_alpha0[layer_id % rate, 0] = 1.0
        self.static_alpha = nn.Parameter(
            torch.cat([init_alpha0, torch.eye((rate), device=device)], dim=1)
        )
        if self.dynamic:
            self.dynamic_alpha_fn = nn.Parameter(
                torch.zeros((dim, rate + 1), device=device)
            )
            self.dynamic_alpha_scale = nn.Parameter(torch.ones(1, device=device) * 0.01)
            self.dynamic_beta_fn = nn.Parameter(torch.zeros((dim,), device=device))
            self.dynamic_beta_scale = nn.Parameter(torch.ones(1, device=device) * 0.01)
            self.layer_norm = LayerNorm(dim)

    def width_connection(self, h):
        # get alpha and beta
        if self.dynamic:
            norm_h = self.layer_norm(h)

        if self.dynamic:
            wc_weight = norm_h @ self.dynamic_alpha_fn
            wc_weight = F.tanh(wc_weight)
            dynamic_alpha = wc_weight * self.dynamic_alpha_scale
            alpha = dynamic_alpha + self.static_alpha[None, None, ...]
        else:
            alpha = self.static_alpha[None, None, ...]

        if self.dynamic:
            dc_weight = norm_h @ self.dynamic_beta_fn
            dc_weight = F.tanh(dc_weight)
            dynamic_beta = dc_weight * self.dynamic_beta_scale
            beta = dynamic_beta + self.static_beta[None, None, ...]
        else:
            beta = self.static_beta[None, None, ...]

        # width connection
        mix_h = alpha.transpose(-1, -2) @ h

        return mix_h, beta

    def depth_connection(self, mix_h, h_o, beta):
        h = torch.einsum("blh,bln->blnh", h_o, beta) + mix_h[..., 1:, :]
        return h
