import torch
import torch.nn as nn
from .nn import get_all_class_pairs
import types
import numpy as np


class LMViewAnalyzer:
    def __init__(self, model) -> None:
        self.model = model
        self.class_pairs = get_all_class_pairs()
        print(self.class_pairs)
        self.skip_classes = []
        self.warp_model("", model)
        print(f"Skipped classes: {self.skip_classes}")

    def warp_model(self, prefix_name, model):
        # replace all modules in model with their corresponding LMView modules

        for name, module in model.named_children():
            full_name = prefix_name + f".{name}"
            if type(module) in self.class_pairs:
                module.__class__ = self.class_pairs[type(module)]
                # print(f"Replaced {full_name} {type(module)}")
            else:
                # print(f"Skipping {full_name} {type(module)}")
                if type(module) not in self.skip_classes:
                    self.skip_classes.append(type(module))
            self.warp_model(full_name, module)

    def unwarp_model(self):
        for name, module in self.model.named_children():
            if hasattr(module, "raw_nn_class"):
                module.__class__ = self.class_pairs[type(module)].raw_nn_class
            self.unwarp_model(module)

    def accumulate_report(self):
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

        accumulate(self.model)

        tot_operations = 0
        tot_weights = 0
        tot_inputs = 0
        tot_outputs = 0
        for layer_name, layer_reports in self.model.analyze_report.items():
            for i, report in enumerate(layer_reports):
                tot_operations += report["operations"]
                n_weights = sum([np.prod(x) for x in report["weights_shape"].values()])
                if i == 0:
                    tot_weights += n_weights
                n_inputs = sum([np.prod(x) for x in report["inputs_shape"].values()])
                tot_inputs += n_inputs
                n_outputs = sum([np.prod(x) for x in report["outputs_shape"].values()])
                tot_outputs += n_outputs
        tot_info = {
            "operations": tot_operations,
            "weights": tot_weights,
            "inputs": tot_inputs,
            "outputs": tot_outputs,
        }
        return self.model.analyze_report, tot_info
