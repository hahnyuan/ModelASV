import copy
import numpy as np
import re


class LMSimulator:
    def __init__(self, hardware_model, verbose=False):
        self.hardware_model = hardware_model
        self.verbose = verbose

    def get_hardware_model(self, layer_name):
        return self.hardware_model

    def simulate(self, analyze_result, w_bit, a_bit):

        layerwise_report = copy.deepcopy(analyze_result["layerwise_report"])
        tot_latency = 0

        tot_operations = 0
        tot_weights_bytes = 0
        tot_inputs_bytes = 0
        tot_outputs_bytes = 0
        tot_mem_access_bytes = 0

        for layer_name, layer_reports in layerwise_report.items():
            for i, report in enumerate(layer_reports):
                operations = report["operations"]
                tot_operations += operations
                n_weights = sum([np.prod(x) for x in report["weights_shape"].values()])
                if i == 0:
                    tot_weights_bytes += n_weights * w_bit / 8
                n_inputs = sum([np.prod(x) for x in report["inputs_shape"].values()])
                tot_inputs_bytes += n_inputs * a_bit / 8
                n_outputs = sum([np.prod(x) for x in report["outputs_shape"].values()])
                tot_outputs_bytes += n_outputs * a_bit / 8
                weight_access_offset = report["info"].get("weight_access_offset", 0)
                memory_access_bytes = (n_inputs + n_outputs) * a_bit / 8 + (
                    n_weights + weight_access_offset
                ) * w_bit / 8
                tot_mem_access_bytes += memory_access_bytes

                # roofline model compute
                compute_bit = max(w_bit, a_bit)
                hardware_model = self.get_hardware_model(layer_name)
                inference_time, simulate_info = hardware_model.run(
                    operations, memory_access_bytes, compute_bit, report["info"]
                )

                tot_latency += inference_time
                if "memory_access_bytes" not in report:
                    report["memory_access_bytes"] = 0
                    report["operations"] = 0
                    report["inference_time"] = 0
                    report["arithmetic_intensity"] = 0
                    report["simulate_info"] = ""
                report["memory_access_bytes"] += memory_access_bytes
                report["operations"] += operations
                report["inference_time"] += inference_time
                report["arithmetic_intensity"] += operations / memory_access_bytes
                report["simulate_info"] += str(simulate_info) + " "
        tot_info = {
            "operations": tot_operations,
            "weights_bytes": tot_weights_bytes,
            "inputs_bytes": tot_inputs_bytes,
            "outputs_bytes": tot_outputs_bytes,
            "mem_access": tot_mem_access_bytes,
            "latency": tot_latency,
        }
        return layerwise_report, tot_info


class HeterogeneousSimulator(LMSimulator):
    def __init__(self, models_mapping, default_hardware_model, verbose=False):
        """
        models_mapping: a dict with keys as the hardware model
             and values as the list of str
        """
        self.models_mapping = models_mapping
        self.default_hardware_model = default_hardware_model
        self.verbose = verbose

    def get_hardware_model(self, layer_name):
        for hardware_model, match_list in self.models_mapping.items():
            for s in match_list:
                if s in layer_name:
                    if self.verbose:
                        print(f"Matched {s} in {layer_name}, using {hardware_model}")
                    return hardware_model
        if self.verbose:
            print(f"Using default hardware model for {layer_name}")
        return self.default_hardware_model
