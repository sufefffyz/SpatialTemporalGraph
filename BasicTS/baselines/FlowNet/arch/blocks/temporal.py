import torch
from einops import rearrange, repeat
from torch import nn
from .encoder import TransformerEncoderBlock


class TemporalEstimator(nn.Module):
    def __init__(
        self,
        his_len,
        pred_len,
        in_feat,
        out_feat,
        nhead,
        ffn_dim,
        n_layer,
        dropout,
        is_causal=True,
        use_RoPE=True,
    ):
        super().__init__()
        self.affine = nn.Linear(in_feat, in_feat)
        self.transformer = nn.Sequential(
            *(
                TransformerEncoderBlock(
                    in_feat,
                    nhead,
                    ffn_dim,
                    dropout=dropout,
                    is_causal=is_causal,
                    use_RoPE=use_RoPE,
                )
                for _ in range(n_layer)
            )
        )
        self.pred_proj = nn.Linear(his_len, pred_len)
        self.feat_proj = nn.Linear(in_feat, out_feat)
        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)
        nn.init.xavier_uniform_(self.pred_proj.weight)
        nn.init.zeros_(self.pred_proj.bias)
        nn.init.xavier_uniform_(self.feat_proj.weight)
        nn.init.zeros_(self.feat_proj.bias)

    def forward(self, x_feat):
        B, T, N, F = x_feat.shape
        x_feat = rearrange(x_feat, "b t n f -> (b n) t f")
        x_feat = self.affine(x_feat)
        x_feat = self.transformer(x_feat)
        x_feat = rearrange(x_feat, "b t f -> b f t")
        x_feat = self.pred_proj(x_feat)
        x_feat = rearrange(x_feat, "(b n) f l -> b l n f", n=N)
        x_feat = self.feat_proj(x_feat)
        return x_feat
