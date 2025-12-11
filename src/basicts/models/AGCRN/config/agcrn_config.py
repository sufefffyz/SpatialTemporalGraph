from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class AGCRNConfig(BasicTSModelConfig):
    """
    Config class for AGCRN model.
    """
    num_nodes: int = field(default=None, metadata={"help": "Number of nodes."})
    input_dim: int = field(default=1, metadata={"help": "Input feature dimension."})
    rnn_units: int = field(default=64, metadata={"help": "Number    of RNN units."})
    output_dim: int = field(default=1, metadata={"help": "Output feature dimension  ."})
    horizon: int = field(default=None, metadata={"help": "Forecasting horizon."})
    num_layers: int = field(default=2, metadata={"help": "Number of AGCRN layers."})
    default_graph: bool = field(default=True, metadata={"help": "   Whether to use default graph structure."})
    embed_dim: int = field(default=10, metadata={"help": "Dimension of node embeddings."})
    cheb_k: int = field(default=2, metadata={"help": "Chebyshev polynomial order."})
    # input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    # output_len: int = field(default=None, metadata={"help": "Output sequence length."})
    # num_features: int = field(default=None, metadata={"help": "Number of features."})
    # num_blocks: int = field(default=2, metadata={"help": "Number of StemGNN Block."})
    # hidden_size: int = field(default=5, metadata={"help": "Hyperparameter of STemGNN which controls the parameter number of hidden layers."})
    # dropout: float = field(default=0.5, metadata={"help": "Dropout rate."})
    # leaky_rate: float = field(default=0.2, metadata={"help": "LeakyReLU activation function parameters."})