from dataclasses import dataclass, field
from typing import Any, List
from basicts.configs import BasicTSModelConfig

@dataclass
class STGCNConfig(BasicTSModelConfig):
    """ Config class for STGCN model. """

    Kt: int = field(default=None, metadata={"help": "Temporal kernel size (order of temporal convolution)."})
    Ks: int = field(default=None, metadata={"help": "Spatial kernel size (order of spatial graph convolution)."})
    
    blocks: List[Any] = field(default=None,
                              metadata={"help": "Block structure definition of STGCN, nested list format."})
    
    T: int = field(default=None, metadata={"help": "Input time step length."})
    n_vertex: int = field(default=None, metadata={"help": "Number of graph nodes."})

    act_func: str = field(default="relu", metadata={"help": "Activation function type: e.g., relu / glu / sigmoid."})

    graph_conv_type: str = field(default="cheb_graph_conv", 
                                 metadata={"help": "Graph convolution type, such as cheb_graph_conv."})

    gso: Any = field(default=None, metadata={"help": "Graph shift operator, adjacency matrix (numpy array or tensor)."})

    bias: bool = field(default=True, metadata={"help": "Whether to use bias in convolution layers."})
    droprate: float = field(default=0.5, metadata={"help": "Dropout rate for model regularization."})

