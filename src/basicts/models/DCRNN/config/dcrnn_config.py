from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class DCRNNConfig(BasicTSModelConfig):
    """
    Config class for DCRNN model.
    """

    cl_decay_steps: int = field(default=1000, metadata={"help": "Curriculum learning decay steps."})
    horizon: int = field(default=None, metadata={"help": "Prediction horizon."})
    input_dim: int = field(default=None, metadata={"help": "Input feature dimension."})
    max_diffusion_step: int = field(default=2, metadata={"help": "Maximum diffusion step."})
    num_nodes: int = field(default=None, metadata={"help": "Number of graph nodes."})
    num_rnn_layers: int = field(default=1, metadata={"help": "Number of RNN layers."})
    output_dim: int = field(default=None, metadata={"help": "Output feature dimension."})
    rnn_units: int = field(default=None, metadata={"help": "Number of RNN units."})
    seq_len: int = field(default=None, metadata={"help": "Input sequence length."})
    adj_mx: any = field(default=None, metadata={"help": "Adjacency matrix of the graph."})
    use_curriculum_learning: bool = field(default=False, metadata={"help": "Whether to use curriculum learning."})
    if_time_in_day: bool = field(default=False, metadata={"help": "Whether to use time of day as feature."})
    if_day_in_week: bool = field(default=False, metadata={"help": "Whether to use day of week as feature."})
    # input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    # output_len: int = field(default=None, metadata={"help": "Output sequence length."})
    # num_features: int = field(default=None, metadata={"help": "Number of features."})
    # num_blocks: int = field(default=2, metadata={"help": "Number of StemGNN Block."})
    # hidden_size: int = field(default=5, metadata={"help": "Hyperparameter of STemGNN which controls the parameter number of hidden layers."})
    # dropout: float = field(default=0.5, metadata={"help": "Dropout rate."})
    # leaky_rate: float = field(default=0.2, metadata={"help": "LeakyReLU activation function parameters."})