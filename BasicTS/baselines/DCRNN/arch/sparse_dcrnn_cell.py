import warnings

import torch

from .dcrnn_cell import DCGRUCell


class SparseMatrixSupport(torch.nn.Module):
    """CSR sparse matrix wrapper for DCRNN diffusion supports."""

    def __init__(self, dense_support):
        super().__init__()
        support = torch.as_tensor(dense_support, dtype=torch.float32).detach().cpu()
        if support.ndim != 2 or support.shape[0] != support.shape[1]:
            raise ValueError(f"support must be a square matrix, got {tuple(support.shape)}")
        self.num_nodes = int(support.shape[0])
        self.num_edges = int((support != 0).sum().item())
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Sparse CSR tensor support is in beta state.*",
                category=UserWarning,
            )
            support_csr = support.to_sparse_csr()
        self.register_buffer("support", support_csr)

    def forward(self, x):
        return torch.sparse.mm(self.support, x)

    def extra_repr(self):
        return f"num_nodes={self.num_nodes}, num_edges={self.num_edges}"


class SparseDCGRUCell(DCGRUCell):
    """DCGRU cell with CSR sparse diffusion support multiplication."""

    def __init__(
        self,
        num_units,
        adj_mx,
        max_diffusion_step,
        num_nodes,
        nonlinearity="tanh",
        use_gc_for_ru=True,
    ):
        super().__init__(
            num_units,
            [],
            max_diffusion_step,
            num_nodes,
            nonlinearity=nonlinearity,
            use_gc_for_ru=use_gc_for_ru,
        )
        self._supports = torch.nn.ModuleList([SparseMatrixSupport(s) for s in adj_mx])

    def _gconv(self, inputs, state, output_size, bias_start=0.0):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state
        x0 = x.permute(1, 2, 0)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)

        if self._max_diffusion_step != 0:
            for support in self._supports:
                x1 = support(x0)
                x = self._concat(x, x1)

                for _ in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * support(x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        num_matrices = len(self._supports) * self._max_diffusion_step + 1
        x = torch.reshape(
            x, shape=[num_matrices, self._num_nodes, input_size, batch_size]
        )
        x = x.permute(3, 1, 2, 0)
        x = torch.reshape(
            x, shape=[batch_size * self._num_nodes, input_size * num_matrices]
        )
        weights = self._gconv_params.get_weights(
            (input_size * num_matrices, output_size)
        ).to(x.device)
        x = torch.matmul(x, weights)

        biases = self._gconv_params.get_biases(output_size, bias_start).to(x.device)
        x += biases
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])
