import torch
from einops import einsum, rearrange
from torch import nn


class MoLinear(nn.Module):
    def __init__(self, in_dim, out_dim, n_experts):
        super().__init__()
        self.router = nn.Linear(in_dim, n_experts)
        self.weight = nn.Parameter(torch.randn(n_experts, in_dim, out_dim))
        self.bias = nn.Parameter(torch.randn(n_experts, out_dim))
        self.n_experts = n_experts
        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        for i in range(self.n_experts):
            nn.init.xavier_uniform_(self.weight[i])
            nn.init.zeros_(self.bias[i])

    def forward(self, x):
        # x: [..., F]
        router_weight = self.router(x).softmax(dim=-1).unsqueeze(-1)
        # router_weight: [..., n_model, 1]
        output = einsum(x, self.weight, "... f, n f m -> ... n m") + self.bias
        # output: [..., n_model, F]
        output = output.transpose(-1, -2)
        output = torch.matmul(output, router_weight).squeeze(-1)
        return output
