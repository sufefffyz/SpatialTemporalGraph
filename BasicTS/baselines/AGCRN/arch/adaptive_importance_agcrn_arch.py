import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveImportanceAVWGCN(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        cheb_k,
        embed_dim,
        support_mode="adaptive",
        graph_use_relu=True,
    ):
        super().__init__()
        if support_mode not in {"adaptive", "identity"}:
            raise ValueError(f"Unknown AGCRN support_mode: {support_mode}")
        self.cheb_k = cheb_k
        self.support_mode = support_mode
        self.graph_use_relu = graph_use_relu
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

    def build_support(self, node_embeddings):
        node_num = node_embeddings.shape[0]
        if self.support_mode == "identity":
            return torch.eye(node_num, device=node_embeddings.device, dtype=node_embeddings.dtype)

        score = torch.mm(node_embeddings, node_embeddings.transpose(0, 1))
        if self.graph_use_relu:
            score = F.relu(score)
        return F.softmax(score, dim=1)

    def forward(self, x, node_embeddings):
        node_num = node_embeddings.shape[0]
        supports = self.build_support(node_embeddings)
        support_set = [torch.eye(node_num, device=supports.device, dtype=supports.dtype), supports]
        for _ in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)

        weights = torch.einsum("nd,dkio->nkio", node_embeddings, self.weights_pool)
        bias = torch.matmul(node_embeddings, self.bias_pool)
        x_g = torch.einsum("knm,bmc->bknc", supports, x)
        x_g = x_g.permute(0, 2, 1, 3)
        return torch.einsum("bnki,nkio->bno", x_g, weights) + bias


class AdaptiveImportanceAGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, support_mode, graph_use_relu):
        super().__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AdaptiveImportanceAVWGCN(
            dim_in + self.hidden_dim,
            2 * dim_out,
            cheb_k,
            embed_dim,
            support_mode=support_mode,
            graph_use_relu=graph_use_relu,
        )
        self.update = AdaptiveImportanceAVWGCN(
            dim_in + self.hidden_dim,
            dim_out,
            cheb_k,
            embed_dim,
            support_mode=support_mode,
            graph_use_relu=graph_use_relu,
        )

    def forward(self, x, state, node_embeddings):
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        return r * state + (1 - r) * hc

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class AdaptiveImportanceAVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers, support_mode, graph_use_relu):
        super().__init__()
        assert num_layers >= 1, "At least one DCRNN layer in the Encoder."
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(
            AdaptiveImportanceAGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim, support_mode, graph_use_relu)
        )
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(
                AdaptiveImportanceAGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim, support_mode, graph_use_relu)
            )

    def forward(self, x, init_state, node_embeddings):
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(x.shape[1]):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        return torch.stack([cell.init_hidden_state(batch_size) for cell in self.dcrnn_cells], dim=0)


class AdaptiveImportanceAGCRN(nn.Module):
    def __init__(
        self,
        num_nodes,
        input_dim,
        rnn_units,
        output_dim,
        horizon,
        num_layers,
        default_graph,
        embed_dim,
        cheb_k,
        support_mode="adaptive",
        graph_use_relu=True,
        freeze_node_embeddings=False,
    ):
        super().__init__()
        self.num_node = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = rnn_units
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers
        self.default_graph = default_graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, embed_dim), requires_grad=True)
        self.encoder = AdaptiveImportanceAVWDCRNN(
            num_nodes,
            input_dim,
            rnn_units,
            cheb_k,
            embed_dim,
            num_layers,
            support_mode,
            graph_use_relu,
        )
        self.end_conv = nn.Conv2d(1, horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

        self.init_param()
        if freeze_node_embeddings:
            self.node_embeddings.requires_grad_(False)

    def init_param(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs):
        init_state = self.encoder.init_hidden(history_data.shape[0])
        output, _ = self.encoder(history_data, init_state, self.node_embeddings)
        output = output[:, -1:, :, :]
        output = self.end_conv(output)
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        return output.permute(0, 1, 3, 2)
