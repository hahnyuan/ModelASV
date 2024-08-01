import torch
import torch.nn as nn
from .nn import get_all_class_pairs
import types
import numpy as np


class LMViewAnalyzer:
    def __init__(self, verbose=False) -> None:
        self.class_pairs = get_all_class_pairs()
        self.verbose = verbose
        if self.verbose:
            print("Class Pairs:", self.class_pairs)

    def _warp_model(self, model, reset=True):
        self._unwarp_model(model)

        warp_info = {}

        def warp_module(prefix_name, module):
            # replace all modules in model with their corresponding LMView modules

            for name, module in module.named_children():
                full_name = prefix_name + f".{name}"
                if type(module) not in warp_info:
                    warp_info[type(module)] = 0
                if type(module) in self.class_pairs:
                    warp_info[type(module)] += 1
                    module.__class__ = self.class_pairs[type(module)]
                    if reset:
                        module.analyze_report = {}
                    # print(f"Replaced {full_name} {type(module)}")
                else:
                    # print(f"Skipping {full_name} {type(module)}")
                    warp_info[type(module)] -= 1
                warp_module(full_name, module)

        warp_module("", model)
        if self.verbose:
            print(f"warp_info (>0 means wrapped layers, <0 means not wrapped layers):")
            for k, v in warp_info.items():
                print(f"{k}: {v}")
        return warp_info

    def _unwarp_model(self, model):
        for name, module in model.named_modules():
            if hasattr(module, "raw_nn_class"):
                module.__class__ = module.raw_nn_class

    def analyze(self, model, reset=True):
        class WarpContext:
            def __init__(self, analyzer):
                self.analyzer = analyzer

            def __enter__(self):
                self.analyzer._warp_model(model, reset)

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.analyzer._unwarp_model(model)

        return WarpContext(self)

    def accumulate_report(self, model):
        def accumulate(module):
            if not hasattr(module, "analyze_report"):
                module.analyze_report = {}
            for name, sub_module in module.named_children():
                accumulate(sub_module)
                if hasattr(sub_module, "analyze_report"):
                    for k, v in sub_module.analyze_report.items():
                        if k == "":
                            subname = name
                        else:
                            subname = f"{name}.{k}"
                        module.analyze_report[subname] = v

        accumulate(model)

        tot_operations = 0
        tot_weights = 0
        tot_inputs = 0
        tot_outputs = 0
        for layer_name, layer_reports in model.analyze_report.items():
            for i, report in enumerate(layer_reports):
                tot_operations += report["operations"]
                n_weights = sum([np.prod(x) for x in report["weights_shape"].values()])
                if i == 0:
                    tot_weights += n_weights
                n_inputs = sum([np.prod(x) for x in report["inputs_shape"].values()])
                tot_inputs += n_inputs
                n_outputs = sum([np.prod(x) for x in report["outputs_shape"].values()])
                tot_outputs += n_outputs
        rst = {
            "tot_operations": tot_operations,
            "tot_weights": tot_weights,
            "tot_inputs": tot_inputs,
            "tot_outputs": tot_outputs,
            "layerwise_report": model.analyze_report,
        }
        return rst
