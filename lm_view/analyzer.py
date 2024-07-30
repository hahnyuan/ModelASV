import torch
import torch.nn as nn


class LMViewAnalyzer:
    def __init__(self, model) -> None:
        self.raw_model = model
        self.model = self.transform_model(model)

    def warp_model(self, model):
        pass

    def analyze_inference(self, x):
        pass
