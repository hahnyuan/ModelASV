import torch
import torch.nn as nn
from .nn import get_all_class_pairs
import types


class LMViewAnalyzer:
    def __init__(self, model) -> None:
        self.raw_model = model
        self.class_pairs = get_all_class_pairs()
        print(self.class_pairs)
        self.warp_model("", model)

    def warp_model(self, prefix_name, model):
        # replace all modules in model with their corresponding LMView modules
        for name, module in model.named_children():
            full_name = prefix_name + f".{name}"
            if type(module) in self.class_pairs:
                module.__class__ = self.class_pairs[type(module)]
                print(f"Replaced {full_name} {type(module)}")
            else:
                print(f"Skipping {full_name} {type(module)}")
            self.warp_model(full_name, module)
