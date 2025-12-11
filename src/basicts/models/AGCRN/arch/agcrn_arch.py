import torch
import torch.nn as nn

from .agcrn_cell import AGCRNCell
from ..config.agcrn_config import AGCRNConfig
from typing import List, Optional, Tuple
class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(
            AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(
                AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](
                    current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(
                self.dcrnn_cells[i].init_hidden_state(batch_size))
        # (num_layers, B, N, hidden_dim)
        return torch.stack(init_states, dim=0)


class AGCRN(nn.Module):
    """
    Paper: Adaptive Graph Convolutional Recurrent Network for Trafﬁc Forecasting
    Official Code: https://github.com/LeiBAI/AGCRN
    Link: https://arxiv.org/abs/2007.02842
    Venue: NeurIPS 2020
    Task: Spatial-Temporal Forecasting
    """

    def __init__(self, config: AGCRNConfig):
        super(AGCRN, self).__init__()
        self.num_node = config.num_nodes
        self.input_dim = config.input_dim
        self.hidden_dim = config.rnn_units
        self.output_dim = config.output_dim
        self.horizon = config.horizon
        self.num_layers = config.num_layers

        self.default_graph = config.default_graph
        self.node_embeddings = nn.Parameter(torch.randn(
            self.num_node, config.embed_dim), requires_grad=True)

        self.encoder = AVWDCRNN(config.num_nodes, config.input_dim, config.rnn_units, config.cheb_k,
                                config.embed_dim, config.num_layers)

        # predictor
        self.end_conv = nn.Conv2d(
            1, config.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

        self.init_param()

    def init_param(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, inputs: torch.tensor, inputs_timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Feedforward function of AGCRN.


        Args:
            inputs (Tensor): Input data with shape: [batch_size, input_len, num_features]
            inputs_timestamps (Tensor): Input timestamps with shape: [batch_size, input_len, num_time_stamps]

        Returns:
            torch.Tensor: outputs with shape [batch_size, output_len, num_features]
        """

        init_state = self.encoder.init_hidden(inputs.shape[0])
        inputs = inputs.unsqueeze(-1)  # B, T, N, D
        output, _ = self.encoder(
            inputs, init_state, self.node_embeddings)  # B, T, N, hidden
        output = output[:, -1:, :, :]  # B, 1, N, hidden

        # CNN based predictor
        output = self.end_conv((output))  # B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon,
                                            self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)  # B, T, N, C

        return output.squeeze(-1)
