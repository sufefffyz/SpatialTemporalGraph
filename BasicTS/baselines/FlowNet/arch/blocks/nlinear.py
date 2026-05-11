import torch
from torch import nn
from ..layers.mo_linear import MoLinear


class IndividualNLinear(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, his_len, pred_len):
        super(IndividualNLinear, self).__init__()
        self.his_len = his_len
        self.pred_len = pred_len

        # Use this line if you want to visualize the weights
        self.Linear = nn.Linear(self.his_len, self.pred_len)
        self.initialize()

    def initialize(self):
        self.Linear.weight = nn.Parameter(
            (1 / self.his_len) * torch.ones([self.pred_len, self.his_len])
        )

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return x  # [Batch, Output length, Channel]


class MoNLinear(nn.Module):
    def __init__(self, his_len, pred_len, n_model):
        super(MoNLinear, self).__init__()
        self.linear = MoLinear(his_len, pred_len, n_model)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        x = self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return x  # [Batch, Output length, Channel]
