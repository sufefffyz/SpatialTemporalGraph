import torch
import torch.nn as nn

from .stgcn_layers import STConvBlock, OutputBlock
from ..config.stgcn_config import STGCNConfig

class STGCNChebGraphConv(nn.Module):
    """
    Paper: Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Trafﬁc Forecasting
    Official Code: https://github.com/VeritasYin/STGCN_IJCAI-18 (tensorflow)
    Ref Code: https://github.com/hazdzz/STGCN
    Venue: IJCAI 2018
    Task: Spatial-Temporal Forecasting
    Note:  
        https://github.com/hazdzz/STGCN/issues/9
    Link: https://arxiv.org/abs/1709.04875
    """

    # STGCNChebGraphConv contains 'TGTND TGTND TNFF' structure
    # ChebGraphConv is the graph convolution from ChebyNet.
    # Using the Chebyshev polynomials of the first kind as a graph filter.

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normalization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(self, config:STGCNConfig):
        super(STGCNChebGraphConv, self).__init__()
        Kt = config.Kt
        Ks = config.Ks
        blocks = config.blocks
        T = config.T
        n_vertex = config.n_vertex
        act_func = config.act_func
        graph_conv_type = config.graph_conv_type
        gso = config.gso
        bias = config.bias
        droprate = config.droprate
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(STConvBlock(
                Kt, Ks, n_vertex, blocks[l][-1], blocks[l+1], act_func, graph_conv_type, gso, bias, droprate))
        self.st_blocks = nn.Sequential(*modules)
        Ko = T - (len(blocks) - 3) * 2 * (Kt - 1)
        self.Ko = Ko
        assert Ko != 0, "Ko = 0."
        self.output = OutputBlock(
            Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, act_func, bias, droprate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """feedforward function of STGCN.

        Args:
            inputs (Tensor): Input data with shape: [batch_size, input_len, num_features]

        Returns:
            torch.Tensor: outputs with shape [batch_size, output_len, num_features]
        """
        x = inputs.unsqueeze(-1).permute(0, 3, 1, 2).contiguous()

        x = self.st_blocks(x)
        x = self.output(x)

        x = x.transpose(2, 3).squeeze(-1)
        return x
