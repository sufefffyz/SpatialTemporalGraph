from .get_model import get_model

try:
    from .onenn_regr import OneNNRegr
except ModuleNotFoundError:
    OneNNRegr = None
