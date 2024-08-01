from . import attn, linear, norm, conv, embedding
import torch.nn as nn
from .register import analyze_classes


def get_all_class_pairs():
    pairs = {}

    for obj in analyze_classes:
        if isinstance(obj, type) and issubclass(obj, nn.Module) and hasattr(obj, "raw_nn_class"):
            pairs[obj.raw_nn_class] = obj
    print(f"Pairs: {pairs}")
    return pairs
