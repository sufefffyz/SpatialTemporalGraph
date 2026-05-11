import torch as th
from torch import nn
from ..functions.attention import attention
from einops import rearrange, repeat


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        nhead,
        ffn_dim,
        dropout=0.1,
        use_RoPE=False,
        is_causal=False,
        use_q_embed=False,
        num_q=None,
        use_kv_embed=False,
        num_kv=None,
    ):
        super().__init__()
        assert (
            in_dim % nhead == 0
        ), f"dim {in_dim} should be divided by num_heads {nhead}."
        self.nhead = nhead
        self.head_dim = in_dim // nhead
        ffn_dim = ffn_dim

        self.use_ROPE = use_RoPE
        self.is_causal = is_causal

        self.use_q_embed = use_q_embed
        if self.use_q_embed:
            assert num_q is not None, "num_q should be provided if use_node_embed."
            self.q_embed = nn.Parameter(th.randn(num_q, self.head_dim))
        self.use_kv_embed = use_kv_embed
        if self.use_kv_embed:
            assert num_kv is not None, "num_kv should be provided if use_node_embed."
            self.kv_embed = nn.Parameter(th.randn(num_kv, self.head_dim))

        self.qkv = nn.Linear(in_dim, in_dim * 3)
        self.mixer = nn.Linear(in_dim, in_dim)
        self.norm1 = nn.LayerNorm(in_dim)
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, ffn_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(ffn_dim, ffn_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(ffn_dim, in_dim),
        )
        self.norm2 = nn.LayerNorm(in_dim)
        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.zeros_(self.qkv.bias)

    def _self_attn(self, x: th.Tensor):
        B, L, F = x.shape
        qkv = self.qkv(x)
        qkv = rearrange(qkv, "b l (h f) -> b h l f", h=self.nhead)
        q, k, v = qkv.chunk(3, dim=-1)
        if self.use_q_embed:
            q = q + repeat(self.q_embed, "n f -> 1 1 n f")
        if self.use_kv_embed:
            k, v = [i + repeat(self.kv_embed, "n f -> 1 1 n f") for i in (k, v)]
        result, att_scores = attention(q, k, v, self.use_ROPE, self.is_causal)
        result = rearrange(result, "b h l f -> b l (h f)")
        result = self.mixer(result)
        return result

    def _ffn(self, x: th.Tensor):
        B, L, F = x.shape
        return self.ffn(x)

    def forward(self, x: th.Tensor):
        # x: (B, L, F)
        x = self.norm1(self._self_attn(x) + x)
        x = self.norm2(self._ffn(x) + x)
        return x
