from dataclasses import dataclass, field
from basicts.configs import BasicTSModelConfig
import torch


@dataclass
class GWNETConfig(BasicTSModelConfig):
    """
    Config class for Graph WaveNet.
    """

    num_nodes: int = field(default=None, metadata={"help": "Number of graph nodes."})

    supports: list = field(default=None, metadata={"help": "Support adjacency matrices (as tensors)."})

    dropout: float = field(default=0.3, metadata={"help": "Dropout rate."})
    gcn_bool: bool = field(default=True, metadata={"help": "Whether to use GCN layers."})
    addaptadj: bool = field(default=True, metadata={"help": "Whether to use adaptive adjacency."})
    aptinit: any = field(default=None, metadata={"help": "Adaptive adjacency matrix initialization."})

    in_dim: int = field(default=2, metadata={"help": "Input dimension (channels)."})
    out_dim: int = field(default=12, metadata={"help": "Output dimension (prediction horizon)."})
    
    residual_channels: int = field(default=32, metadata={"help": "Number of residual channels."})
    dilation_channels: int = field(default=32, metadata={"help": "Number of dilation channels."})
    skip_channels: int = field(default=256, metadata={"help": "Number of skip channels."})
    end_channels: int = field(default=512, metadata={"help": "Number of end channels."})

    kernel_size: int = field(default=2, metadata={"help": "Kernel size of temporal convolution."})

    blocks: int = field(default=4, metadata={"help": "Number of WaveNet residual blocks."})
    layers: int = field(default=2, metadata={"help": "Number of layers per block."})
