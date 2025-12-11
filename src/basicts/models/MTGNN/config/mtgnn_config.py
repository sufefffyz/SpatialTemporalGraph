from dataclasses import dataclass, field
from basicts.configs import BasicTSModelConfig


@dataclass
class MTGNNConfig(BasicTSModelConfig):
    """
    Config class for MTGNN model.
    """

    # Graph model and adjacency matrix parameters
    gcn_true: bool = field(default=True, metadata={"help": "Whether to use GCN layers."})
    buildA_true: bool = field(default=True, metadata={"help": "Whether to build adjacency matrix."})
    gcn_depth: int = field(default=2, metadata={"help": "Depth of GCN layers."})
    num_nodes: int = field(default=None, metadata={"help": "Number of graph nodes."})
    predefined_A: any = field(default=None, metadata={"help": "Predefined adjacency matrix."})
    static_feat: any = field(default=None, metadata={"help": "Static feature matrix."})

    # General model parameters
    dropout: float = field(default=0.3, metadata={"help": "Dropout rate."})
    subgraph_size: int = field(default=20, metadata={"help": "Size of subgraph."})
    node_dim: int = field(default=40, metadata={"help": "Dimensionality of nodes."})
    dilation_exponential: int = field(default=1, metadata={"help": "Exponential factor for dilation."})

    # Convolution channels for different parts
    conv_channels: int = field(default=32, metadata={"help": "Number of convolution channels."})
    residual_channels: int = field(default=32, metadata={"help": "Number of residual channels."})
    skip_channels: int = field(default=64, metadata={"help": "Number of skip channels."})
    end_channels: int = field(default=128, metadata={"help": "Number of end channels."})

    # Sequence parameters
    seq_length: int = field(default=12, metadata={"help": "Length of the input sequence."})
    in_dim: int = field(default=2, metadata={"help": "Input feature dimension."})
    out_dim: int = field(default=12, metadata={"help": "Output feature dimension."})

    # Layer and normalization parameters
    layers: int = field(default=3, metadata={"help": "Number of layers in the model."})
    propalpha: float = field(default=0.05, metadata={"help": "Propagation alpha."})
    tanhalpha: int = field(default=3, metadata={"help": "Tanh alpha parameter."})
    layer_norm_affline: bool = field(default=True, metadata={"help": "Whether to apply affine transformation in layer normalization."})
