from torch import nn

from .dcrnn_arch import DCRNN, DecoderModel, EncoderModel, Seq2SeqAttrs
from .sparse_dcrnn_cell import SparseDCGRUCell


class SparseEncoderModel(EncoderModel):
    def __init__(self, adj_mx, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.input_dim = int(model_kwargs.get("input_dim", 1))
        self.seq_len = int(model_kwargs.get("seq_len"))
        self.dcgru_layers = nn.ModuleList(
            [
                SparseDCGRUCell(
                    self.rnn_units,
                    adj_mx,
                    self.max_diffusion_step,
                    self.num_nodes,
                )
                for _ in range(self.num_rnn_layers)
            ]
        )


class SparseDecoderModel(DecoderModel):
    def __init__(self, adj_mx, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.output_dim = int(model_kwargs.get("output_dim", 1))
        self.horizon = int(model_kwargs.get("horizon", 1))
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.dcgru_layers = nn.ModuleList(
            [
                SparseDCGRUCell(
                    self.rnn_units,
                    adj_mx,
                    self.max_diffusion_step,
                    self.num_nodes,
                )
                for _ in range(self.num_rnn_layers)
            ]
        )


class SparseDCRNN(DCRNN):
    """DCRNN with CSR sparse diffusion support multiplication."""

    def __init__(self, adj_mx, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.encoder_model = SparseEncoderModel(adj_mx, **model_kwargs)
        self.decoder_model = SparseDecoderModel(adj_mx, **model_kwargs)
        self.cl_decay_steps = int(model_kwargs.get("cl_decay_steps", 2000))
        self.use_curriculum_learning = bool(
            model_kwargs.get("use_curriculum_learning", False)
        )

    def sparse_stats(self):
        stats = []
        for cell in self.encoder_model.dcgru_layers:
            for support in cell._supports:
                stats.append(
                    {
                        "num_nodes": support.num_nodes,
                        "num_edges": support.num_edges,
                    }
                )
        return stats
