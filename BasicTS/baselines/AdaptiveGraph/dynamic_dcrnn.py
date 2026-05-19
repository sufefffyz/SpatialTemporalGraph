from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .dynamic_support import DynamicThresholdSupport


class LayerParams:
    def __init__(self, rnn_network: nn.Module, layer_type: str):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = nn.Parameter(torch.empty(*shape))
            nn.init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter(f"{self._type}_weight_{shape}", nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = nn.Parameter(torch.empty(length))
            nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter(f"{self._type}_biases_{length}", biases)
        return self._biases_dict[length]


class DynamicDCGRUCell(nn.Module):
    def __init__(self, num_units, max_diffusion_step, num_nodes, nonlinearity="tanh", use_gc_for_ru=True):
        super().__init__()
        self._activation = torch.tanh if nonlinearity == "tanh" else torch.relu
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._use_gc_for_ru = use_gc_for_ru
        self._fc_params = LayerParams(self, "fc")
        self._gconv_params = LayerParams(self, "gconv")

    def forward(self, inputs, hx, supports):
        output_size = 2 * self._num_units
        fn = self._gconv if self._use_gc_for_ru else self._fc
        if self._use_gc_for_ru:
            value = torch.sigmoid(fn(inputs, hx, output_size, supports, bias_start=1.0))
        else:
            value = torch.sigmoid(fn(inputs, hx, output_size, bias_start=1.0))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(value, self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))

        c = self._gconv(inputs, r * hx, self._num_units, supports)
        if self._activation is not None:
            c = self._activation(c)
        return u * hx + (1.0 - u) * c

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        input_size = inputs_and_state.shape[-1]
        weights = self._fc_params.get_weights((input_size, output_size)).to(inputs_and_state.device)
        value = torch.matmul(inputs_and_state, weights)
        value += self._fc_params.get_biases(output_size, bias_start).to(value.device)
        return value

    @staticmethod
    def _support_mul(support: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        support = support.to(x.device)
        if support.dim() == 2:
            return torch.einsum("ij,bjf->bif", support, x)
        if support.dim() == 3:
            return torch.einsum("bij,bjf->bif", support, x)
        raise ValueError(f"Unsupported support rank {support.dim()}.")

    def _gconv(self, inputs, state, output_size, supports, bias_start=0.0):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x0 = inputs_and_state
        x_list = [x0]
        if self._max_diffusion_step > 0:
            for support in supports:
                x1 = self._support_mul(support, x0)
                x_list.append(x1)
                prev0, prev1 = x0, x1
                for _ in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * self._support_mul(support, prev1) - prev0
                    x_list.append(x2)
                    prev0, prev1 = prev1, x2

        x = torch.stack(x_list, dim=-1)
        x = torch.reshape(x, (batch_size * self._num_nodes, input_size * len(x_list)))
        weights = self._gconv_params.get_weights((input_size * len(x_list), output_size)).to(x.device)
        x = torch.matmul(x, weights)
        x += self._gconv_params.get_biases(output_size, bias_start).to(x.device)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])


class Seq2SeqAttrs:
    def __init__(self, **model_kwargs):
        self.max_diffusion_step = int(model_kwargs.get("max_diffusion_step", 2))
        self.cl_decay_steps = int(model_kwargs.get("cl_decay_steps", 1000))
        self.num_nodes = int(model_kwargs.get("num_nodes", 1))
        self.num_rnn_layers = int(model_kwargs.get("num_rnn_layers", 1))
        self.rnn_units = int(model_kwargs.get("rnn_units"))
        self.hidden_state_size = self.num_nodes * self.rnn_units


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.input_dim = int(model_kwargs.get("input_dim", 1))
        self.seq_len = int(model_kwargs.get("seq_len"))
        self.dcgru_layers = nn.ModuleList(
            [
                DynamicDCGRUCell(self.rnn_units, self.max_diffusion_step, self.num_nodes)
                for _ in range(self.num_rnn_layers)
            ]
        )

    def forward(self, inputs, supports, hidden_state=None):
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size)).to(inputs.device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], supports)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state
        return output, torch.stack(hidden_states)


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.output_dim = int(model_kwargs.get("output_dim", 1))
        self.horizon = int(model_kwargs.get("horizon", 1))
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.dcgru_layers = nn.ModuleList(
            [
                DynamicDCGRUCell(self.rnn_units, self.max_diffusion_step, self.num_nodes)
                for _ in range(self.num_rnn_layers)
            ]
        )

    def forward(self, inputs, supports, hidden_state=None):
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], supports)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)
        return output, torch.stack(hidden_states)


class DynamicThresholdDCRNN(nn.Module, Seq2SeqAttrs):
    """DCRNN with batch-conditioned soft or hard threshold diffusion supports."""

    def __init__(self, dynamic_graph, **model_kwargs):
        super().__init__()
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.encoder_model = EncoderModel(**model_kwargs)
        self.decoder_model = DecoderModel(**model_kwargs)
        self.cl_decay_steps = int(model_kwargs.get("cl_decay_steps", 2000))
        self.use_curriculum_learning = bool(model_kwargs.get("use_curriculum_learning", False))
        self.dynamic_support = DynamicThresholdSupport(
            num_nodes=self.num_nodes,
            seq_len=int(model_kwargs.get("seq_len")),
            normalization="transition",
            **dynamic_graph,
        )

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs, supports):
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[t], supports, encoder_hidden_state)
        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, supports, labels=None, batches_seen=None):
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.decoder_model.output_dim)).to(encoder_hidden_state.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol
        outputs = []

        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input, supports, decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        return torch.stack(outputs)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None, **kwargs) -> torch.Tensor:
        supports = self.dynamic_support.transition_supports(history_data)
        batch_size, length, num_nodes, channels = history_data.shape
        history_data = history_data.reshape(batch_size, length, num_nodes * channels).transpose(0, 1)

        if future_data is not None:
            future_data = future_data[..., [0]]
            batch_size, length, num_nodes, channels = future_data.shape
            future_data = future_data.reshape(batch_size, length, num_nodes * channels).transpose(0, 1)

        encoder_hidden_state = self.encoder(history_data, supports)
        outputs = self.decoder(encoder_hidden_state, supports, future_data, batches_seen=batch_seen)
        length, batch, _ = outputs.shape
        outputs = outputs.transpose(0, 1).view(batch, length, self.num_nodes, self.decoder_model.output_dim)
        if batch_seen == 0:
            print("Warning: decoder only takes the first dimension as groundtruth.")
        return outputs
