import torch
from torch import nn
import numpy as np

from .dcrnn_cell import DCGRUCell
from ..config.dcrnn_config import DCRNNConfig
from typing import Optional

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# class Seq2SeqAttrs:
#     def __init__(self, adj_mx, **model_kwargs):
#         self.adj_mx = adj_mx
#         self.max_diffusion_step = int(
#             model_kwargs.get("max_diffusion_step", 2))
#         self.cl_decay_steps = int(model_kwargs.get("cl_decay_steps", 1000))
#         self.filter_type = model_kwargs.get("filter_type", "laplacian")
#         self.num_nodes = int(model_kwargs.get("num_nodes", 1))
#         self.num_rnn_layers = int(model_kwargs.get("num_rnn_layers", 1))
#         self.rnn_units = int(model_kwargs.get("rnn_units"))
#         self.hidden_state_size = self.num_nodes * self.rnn_units

class Seq2SeqAttrs:
    def __init__(self, adj_mx, config):
        self.adj_mx = adj_mx
        
        self.max_diffusion_step = int(getattr(config, "max_diffusion_step", 2))
        self.cl_decay_steps = int(getattr(config, "cl_decay_steps", 1000))
        self.filter_type = getattr(config, "filter_type", "laplacian")
        self.num_nodes = int(getattr(config, "num_nodes", 1))
        self.num_rnn_layers = int(getattr(config, "num_rnn_layers", 1))
        self.rnn_units = int(getattr(config, "rnn_units"))
        self.hidden_state_size = self.num_nodes * self.rnn_units



# class EncoderModel(nn.Module, Seq2SeqAttrs):
#     def __init__(self, adj_mx, **model_kwargs):
#         nn.Module.__init__(self)
#         Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
#         self.input_dim = int(model_kwargs.get("input_dim", 1))
#         self.seq_len = int(model_kwargs.get("seq_len"))  # for the encoder
#         self.dcgru_layers = nn.ModuleList(
#             [DCGRUCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes) for _ in range(self.num_rnn_layers)])
class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, config):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, config)
        
        self.input_dim = int(getattr(config, "input_dim", 1))
        self.seq_len = int(getattr(config, "seq_len", 12))  # 使用默认值 10，如果 config 中没有 seq_len

        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes) for _ in range(self.num_rnn_layers)]
        )


    def forward(self, inputs, hidden_state=None):
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros(
                (self.num_rnn_layers, batch_size, self.hidden_state_size)).to(inputs.device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        # runs in O(num_layers) so not too slow
        return output, torch.stack(hidden_states)


# class DecoderModel(nn.Module, Seq2SeqAttrs):
#     def __init__(self, adj_mx, **model_kwargs):
#         # super().__init__(is_training, adj_mx, **model_kwargs)
#         nn.Module.__init__(self)
#         Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
#         self.output_dim = int(model_kwargs.get("output_dim", 1))
#         self.horizon = int(model_kwargs.get("horizon", 1))  # for the decoder
#         self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
#         self.dcgru_layers = nn.ModuleList(
#             [DCGRUCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes) for _ in range(self.num_rnn_layers)])
class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, config):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, config)
        
        self.output_dim = int(getattr(config, "output_dim", 1))
        self.horizon = int(getattr(config, "horizon", 1))
        
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes) for _ in range(self.num_rnn_layers)]
        )


    def forward(self, inputs, hidden_state=None):
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)

        return output, torch.stack(hidden_states)


class DCRNN(nn.Module, Seq2SeqAttrs):
    """
    Paper: Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting
    Link: https://arxiv.org/abs/1707.01926
    Official Code:
        https://github.com/chnsh/DCRNN_PyTorch/blob/pytorch_scratch/model/pytorch/dcrnn_cell.py,
        https://github.com/chnsh/DCRNN_PyTorch/blob/pytorch_scratch/model/pytorch/dcrnn_model.py
    Venue: ICLR 2018
    Task: Spatial-Temporal Forecasting
    """

    def __init__(self, config: DCRNNConfig):
        super().__init__()

        # graph adjacency
        adj_mx = config.adj_mx
        
        # init Seq2Seq and models
        Seq2SeqAttrs.__init__(self, adj_mx, config)
        self.encoder_model = EncoderModel(adj_mx, config)
        self.decoder_model = DecoderModel(adj_mx, config)

        self.if_time_in_day = config.if_time_in_day
        self.if_day_in_week = config.if_day_in_week
        # self.num_time_in_day = config.num_time_in_day
        # self.num_day_in_week = config.num_day_in_week

        self.time_idx = 0
        if self.if_time_in_day:
            self.time_idx += 1
        if self.if_day_in_week:
            self.time_idx += 1
        # directly use config
        self.cl_decay_steps = config.cl_decay_steps
        self.use_curriculum_learning = config.use_curriculum_learning
        


    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
            self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs):
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(
                inputs[t], encoder_hidden_state)

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, labels=None, batches_seen=None):
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros(
            (batch_size, self.num_nodes * self.decoder_model.output_dim)).to(encoder_hidden_state.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(
                decoder_input, decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, inputs: torch.Tensor, inputs_timestamps: torch.Tensor, targets: torch.Tensor, step: int) -> torch.Tensor:
        """Feedforward function of DCRNN.

        Args:
            inputs (Tensor): Input data with shape: [batch_size, input_len, num_features]

        Returns:
            torch.Tensor: outputs with shape [batch_size, output_len, num_features]
        """

        # reshape data
        # batch_size, length, num_nodes, channels = history_data.shape
        # history_data = history_data.reshape(batch_size, length, num_nodes * channels)      # [B, L, N*C]
        # history_data = history_data.transpose(0, 1)         # [L, B, N*C]
        batch_size, length, num_nodes = inputs.shape
        inputs_timestamps = inputs_timestamps[:,:,:self.time_idx].unsqueeze(2).expand(-1, -1, num_nodes,-1)
        inputs = inputs.unsqueeze(-1) # [B, L, N, 1]
        inputs_all = torch.cat([inputs, inputs_timestamps], dim=-1) # [B, L, N, 1+time_idx]
        inputs_all = inputs_all.reshape(batch_size, length, num_nodes * (1 + self.time_idx))  # [B, L, N*(1+time_idx)]

        inputs_all = inputs_all.transpose(0, 1)         # [L, B, N*C]

        if targets is not None:
            # future_data = future_data[..., [0]]     # teacher forcing only use the first dimension.
            # batch_size, length, num_nodes, channels = future_data.shape
            # future_data = future_data.reshape(batch_size, length, num_nodes * channels)      # [B, L, N*C]
            # future_data = future_data.transpose(0, 1)         # [L, B, N*C]
            targets = targets.transpose(0, 1)         # [L, B, N*C]

        # DCRNN
        encoder_hidden_state = self.encoder(inputs_all)
        outputs = self.decoder(encoder_hidden_state, targets,
                               batches_seen=step)      # [L, B, N*C_out]

        # reshape to B, L, N, C
        L, B, _ = outputs.shape
        outputs = outputs.transpose(0, 1)  # [B, L, N*C_out]
        outputs = outputs.view(B, L, self.num_nodes,
                               self.decoder_model.output_dim)

        if step == 0:
            print("Warning: decoder only takes the first dimension as groundtruth.")
            print("Parameter Number: ".format(count_parameters(self)))
            print(count_parameters(self))
        return outputs.squeeze(-1)  # [B, L, N]
