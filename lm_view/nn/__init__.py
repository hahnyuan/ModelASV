from .linear import Linear
import torch.nn as nn


def get_all_class_pairs():
    pairs = {}

    for name, obj in globals().items():
        print(name, obj)
        if isinstance(obj, type) and issubclass(obj, nn.Module) and hasattr(obj, "raw_nn_class"):
            pairs[obj.raw_nn_class] = obj
    return pairs
