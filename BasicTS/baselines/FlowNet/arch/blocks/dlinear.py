import torch
from torch import nn
from ..layers.decomp import SeriesDecompose


class IndividualDLinear(nn.Module):
    def __init__(self, seq_len, pred_len, kernel_size=5):
        super(IndividualDLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.decompsition = SeriesDecompose(kernel_size)
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        self.initialize()

    def initialize(self):
        self.Linear_Seasonal.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
        )
        self.Linear_Trend.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
        )

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(
            0, 2, 1
        )
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # to [Batch, Output length, Channel]


class MoDLinear(nn.Module):
    def __init__(self, his_len, pred_len, n_model, kernel_size=5):
        super(MoDLinear, self).__init__()
        self.router = nn.Linear(his_len, n_model)
        self.n_model = n_model
        self.models = nn.ModuleList(
            [IndividualDLinear(his_len, pred_len, kernel_size) for _ in range(n_model)]
        )
        self.pred_len = pred_len
        self.initialize()

    def initialize(self):
        nn.init.xavier_normal_(self.router.weight)
        nn.init.zeros_(self.router.bias)

    def forward(self, x):
        # x: [B, L, N]
        router_weight = self.router(x.transpose(1, 2)).softmax(dim=-1)
        # router_weight: [B, N, n_model]
        output = []
        for i in range(self.n_model):
            output.append(self.models[i](x))
        output = torch.stack(output, dim=-1)
        # output: [B, L, N, n_model]
        output = torch.einsum("blnm,bnm->bln", output, router_weight)
        return output
