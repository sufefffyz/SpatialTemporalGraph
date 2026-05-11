import torch
from einops import einsum, rearrange, reduce, repeat
from torch import nn

from ..functions.attention import attention
from ..layers.hyper_connect import HyperConnection
from ..layers.mo_linear import MoLinear


class FlowEstimator(nn.Module):
    def __init__(
        self,
        in_dim,
        nhead,
        use_RoPE=True,
        is_causal=True,
    ):
        super().__init__()
        assert (
            in_dim % nhead == 0
        ), f"dim {in_dim} should be divided by num_heads {nhead}."
        self.nhead = nhead
        self.head_dim = in_dim // nhead

        self.use_ROPE = use_RoPE
        self.is_causal = is_causal
        self.norm = nn.LayerNorm(in_dim)
        self.qkv = nn.Linear(in_dim, in_dim * 3)
        self.initialize()

    def initialize(self):
        pass

    def _self_attn(self, x: torch.Tensor):
        B, L, F = x.shape
        x = self.norm(x)
        qkv = self.qkv(x)
        qkv = rearrange(qkv, "b l (h f) -> b h l f", h=self.nhead)
        q, k, v = qkv.chunk(3, dim=-1)
        result, att_scores = attention(q, k, v, self.use_ROPE, self.is_causal)
        return result

    def forward(self, x: torch.Tensor):
        return self._self_attn(x)


class Allocator(nn.Module):
    def __init__(
        self,
        in_dim,
        nhead,
        n_expert,
        use_q_embed=False,
        num_q=None,
        use_k_embed=False,
        num_k=None,
    ):
        super().__init__()
        self.scale = in_dim**-0.5
        assert (
            in_dim % nhead == 0
        ), f"dim {in_dim} should be divided by num_heads {nhead}."
        self.nhead = nhead
        self.head_dim = in_dim // nhead
        self.norm = nn.LayerNorm(in_dim)

        self.use_q_embed = use_q_embed
        if self.use_q_embed:
            assert num_q is not None, "num_q should be provided if use_node_embed."
            self.q_embed = nn.Parameter(torch.randn(num_q, self.head_dim))
        self.use_k_embed = use_k_embed
        if self.use_k_embed:
            assert num_k is not None, "num_k should be provided if use_node_embed."
            self.k_embed = nn.Parameter(torch.randn(num_k, self.head_dim))

        self.qk = MoLinear(in_dim, in_dim * 2, n_expert)
        self.initialize()

    def initialize(self):
        pass

    def _get_allocate_mtx(self, x: torch.Tensor):
        B, N, F = x.shape
        qk = self.qk(self.norm(x))
        qk = rearrange(qk, "b l (h f) -> b h l f", h=self.nhead)
        q, k = qk.chunk(2, dim=-1)
        if self.use_q_embed:
            q = q + repeat(self.q_embed, "n f -> 1 1 n f")
        if self.use_k_embed:
            k = k + repeat(self.k_embed, "m f -> 1 1 m f")
        att_logits = einsum(q, k, "b h n f, b h m f -> b h n m") * self.scale
        return att_logits
        # allocate_mtx = att_logits.softmax(dim=2)
        # return allocate_mtx

    def forward(self, x: torch.Tensor):
        return self._get_allocate_mtx(x)


class FlowBlock(nn.Module):
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
        num_node,
        n_expert,
        rate,
    ):
        super().__init__()
        self.in_feat = in_feat
        self.temporal_dim = in_feat
        self.spatial_dim = in_feat
        self.expand_spatial_dim = in_feat * his_len
        self.nhead = nhead
        self.ffn_dim = ffn_dim
        self.num_node = num_node
        self.dropout = dropout
        self.n_expert = n_expert
        self.rate = rate
        self.pred_len = pred_len

        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn_hyper_connection": HyperConnection(
                            dim=in_feat, rate=rate, layer_id=i
                        ),
                        "attn_norm": nn.LayerNorm(self.in_feat),
                        "t_affine": nn.Linear(self.temporal_dim, self.temporal_dim),
                        "s_affine": nn.Linear(
                            self.expand_spatial_dim, self.expand_spatial_dim
                        ),
                        "inner_flow": self.build_flow_block(),
                        "extra_flow": self.build_flow_block(),
                        "allocator": self.build_allocator_block(),
                        "attn_dropout": nn.Dropout(self.dropout),
                        # FFN
                        "ffn_hyper_connection": HyperConnection(
                            dim=in_feat, rate=rate, layer_id=i
                        ),
                        "ffn_norm": nn.LayerNorm(self.in_feat),
                        "ffn": self.build_mixer(),
                        "ffn_dropout": nn.Dropout(self.dropout),
                    }
                )
                for i in range(n_layer)
            ]
        )
        # self.pred_proj = nn.Linear(his_len, pred_len)
        # self.feat_proj = nn.Linear(in_feat, out_feat)
        self.pred_proj = nn.Linear(in_feat * his_len, pred_len * out_feat)
        self.mtx_records = []
        self.initialize()

    def build_flow_block(self):
        return FlowEstimator(
            in_dim=self.temporal_dim,
            nhead=self.nhead,
            is_causal=True,
            use_RoPE=True,
        )

    def build_allocator_block(self):
        return Allocator(
            in_dim=self.spatial_dim,
            nhead=self.nhead,
            n_expert=self.n_expert,
            use_q_embed=True,
            num_q=self.num_node,
            use_k_embed=True,
            num_k=self.num_node,
        )

    def build_mixer(self):
        return nn.Sequential(
            MoLinear(self.in_feat, self.ffn_dim, self.n_expert),
            nn.Dropout(self.dropout),
            nn.GELU(),
            MoLinear(self.ffn_dim, self.ffn_dim, self.n_expert),
            nn.Dropout(self.dropout),
            nn.GELU(),
            MoLinear(self.ffn_dim, self.in_feat, self.n_expert),
        )

    def initialize(self):
        for block in self.blocks:
            nn.init.xavier_uniform_(block["t_affine"].weight)
            nn.init.zeros_(block["t_affine"].bias)
            nn.init.xavier_uniform_(block["s_affine"].weight)
            nn.init.zeros_(block["s_affine"].bias)
        nn.init.xavier_uniform_(self.pred_proj.weight)
        nn.init.zeros_(self.pred_proj.bias)

    def forward(self, x_feat, mask):
        B, T, N, F = x_feat.shape
        h = repeat(x_feat, "b t n F -> b t n r F", r=self.rate)
        h = rearrange(h, "b t n r F -> (b n) t r F")
        for block in self.blocks:
            mix_h, beta = block["attn_hyper_connection"].width_connection(h)
            h = block["attn_norm"](mix_h[..., 0, :])  # B, T, N, F

            h_t = block["t_affine"](h)
            inner_flow = block["inner_flow"](h_t)
            inner_flow = rearrange(inner_flow, "(b n) h t f -> b t h n f", n=N)
            extra_flow = block["extra_flow"](h_t)
            extra_flow = rearrange(extra_flow, "(b n) h t f -> b t h n f", n=N)

            h_s = rearrange(h, "(b n) t F -> b n (t F)", n=N)
            h_s = block["s_affine"](h_s)
            h_s = rearrange(h_s, "b n (t F) -> (b t) n F", t=T)
            allocate_mtx = block["allocator"](h_s)
            allocate_mtx = rearrange(allocate_mtx, "(b t) h n m -> b t h n m", t=T)
            allocate_mtx = allocate_mtx * mask.unsqueeze(2).float()
            allocate_mtx = allocate_mtx.softmax(dim=-2)
            extra_in = einsum(
                extra_flow, allocate_mtx, "b t h n f, b t h n m -> b t h n f"
            )
            # result = einsum(x_t, allocate_mtx, "b t h n f, b h n m -> b t h n f")
            h = extra_in - extra_flow + inner_flow
            h = rearrange(h, "b t h n f -> (b n) t (h f)")
            h = block["attn_hyper_connection"].depth_connection(
                mix_h, block["attn_dropout"](h), beta
            )

            mix_h, beta = block["ffn_hyper_connection"].width_connection(h)
            h = block["ffn_norm"](mix_h[..., 0, :])
            h = block["ffn"](h)
            h = block["ffn_hyper_connection"].depth_connection(
                mix_h, block["ffn_dropout"](h), beta
            )
        h = rearrange(h, "(b n) t r F -> b t n r F", n=N)
        h = reduce(h, "b t n r F -> b t n F", "sum")

        h = rearrange(h, "b t n F -> b n (F t)")
        h = self.pred_proj(h)
        h = rearrange(h, "b n (F l) -> b l n F", l=self.pred_len)
        # h = self.feat_proj(h)
        return h
