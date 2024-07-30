from .linear import Linear
from .attn import Qwen2SdpaAttention
import torch.nn as nn


def get_all_class_pairs():
    pairs = {}

    for name, obj in globals().items():
        if isinstance(obj, type) and issubclass(obj, nn.Module) and hasattr(obj, "raw_nn_class"):
            pairs[obj.raw_nn_class] = obj
    return pairs
