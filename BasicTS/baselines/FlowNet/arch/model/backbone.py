import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F

from ..blocks.hyper_flow import FlowBlock
from ..layers.patch import FeatPatchEmbedding


class ConvEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size, stride, padding)

    def forward(self, x):
        B, T, N = x.shape
        result = rearrange(x, "B T N -> (B N) 1 T")
        result = self.conv(result)
        result = rearrange(result, "(B N) F T -> B T N F", N=N)
        return result + x.unsqueeze(-1)


class FlowNet(nn.Module):

    def __init__(self, config, dist_mtx: torch.Tensor) -> None:
        super().__init__()
        self.seq_embed = FeatPatchEmbedding(
            d_model=config.d_model,
            patch_len=config.patch_len,
            stride=config.stride,
            freq=config.freq,
        )
        self.num_patches = (config.seq_len - config.patch_len) // config.stride + 1
        # self.seq_embed = ConvEmbedding(1, config.d_model, config.moving_avg, 1, int(config.moving_avg // 2))
        # num_patches = config.seq_len
        self.feat_est = FlowBlock(
            self.num_patches,
            config.pred_len,
            config.d_model,
            1,
            config.nhead,
            config.ffn_dim,
            config.n_layer,
            config.dropout,
            num_node=config.enc_in,
            n_expert=config.n_expert,
            rate=config.rate,
        )
        self.node_embed = nn.Parameter(torch.randn(config.enc_in, config.d_model))
        self.register_buffer("dist_mtx", dist_mtx.float())
        self.dist_est = nn.Linear(config.d_model * 2, 1)
        self.dist_threshold = []
        self.mask_record = []
        self.initialize()

    def initialize(self):
        nn.init.zeros_(self.dist_est.weight)
        nn.init.constant_(self.dist_est.bias, float(self.dist_mtx.mean().item()))

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec=None,
        x_mark_dec=None,
    ):
        B, T, N = x_enc.shape
        x_feat = self.seq_embed(x_enc, x_mark_enc)
        dist_feat = torch.concat(
            [
                x_feat,
                repeat(self.node_embed, "N f -> B P N f", B=B, P=self.num_patches),
            ],
            dim=-1,
        )
        dist_threshold = F.softplus(self.dist_est(dist_feat))
        delta_radius = dist_threshold - self.dist_mtx.unsqueeze(0).unsqueeze(0)
        mask = F.sigmoid(delta_radius)
        feat_est = self.feat_est(x_feat, mask)
        # return seq_est + feat_est.squeeze(-1)
        return feat_est.squeeze(-1)
