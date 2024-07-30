import numpy as np
import copy


class RooflineModel:
    def __init__(self, bandwidth, onchip_buffer, compute_capacity, w_bit=32, a_bit=32, operations_multiplier=2):
        """
        operations_multiplier: the number of operations that a MAC counts as.
            For example, if a MAC counts as 2 operations, then operations_multiplier=2
        compute_capacity: the compute capacity of the accelerator, a dict with keys as bitwidth and values as FLOPS/OPS
            The bitwidth is the bitwidth of the data type, e.g., 32 for float32 and 16 for float16
        """
        self.bandwidth = bandwidth
        self.onchip_buffer = onchip_buffer
        self.compute_capacity = compute_capacity
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.operations_multiplier = operations_multiplier

    def simulate(self, analyze_report):

        analyze_report = copy.deepcopy(analyze_report)
        tot_latency = 0

        tot_operations = 0
        tot_weights_bytes = 0
        tot_inputs_bytes = 0
        tot_outputs_bytes = 0
        tot_mem_access = 0

        for layer_name, layer_reports in analyze_report.items():
            for i, report in enumerate(layer_reports):
                operations = report["operations"] * self.operations_multiplier
                tot_operations += operations
                n_weights = sum([np.prod(x) for x in report["weights_shape"].values()])
                if i == 0:
                    tot_weights_bytes += n_weights * self.w_bit / 8
                n_inputs = sum([np.prod(x) for x in report["inputs_shape"].values()])
                tot_inputs_bytes += n_inputs * self.a_bit / 8
                n_outputs = sum([np.prod(x) for x in report["outputs_shape"].values()])
                tot_outputs_bytes += n_outputs * self.a_bit / 8
                tot_mem_access += n_inputs + n_outputs + n_weights

                # roofline model compute
                max_OPS = self.compute_capacity[max(self.w_bit, self.a_bit)]
                y_max = max_OPS
                memory_access_bytes = n_inputs + n_outputs + n_weights
                turning_point = y_max / self.bandwidth
                arithmetic_intensity = operations / memory_access_bytes
                if arithmetic_intensity < turning_point:
                    bound = "memory"
                    performance = arithmetic_intensity * self.bandwidth
                else:
                    bound = "compute"
                    performance = y_max
                if performance == 0:
                    inference_time = memory_access_bytes / self.bandwidth
                else:
                    inference_time = operations / performance
                tot_latency += inference_time
                report["inference_time"] = inference_time
                report["arithmetic_intensity"] = arithmetic_intensity
                report["bound"] = bound
        tot_info = {
            "operations": tot_operations,
            "weights_bytes": tot_weights_bytes,
            "inputs_bytes": tot_inputs_bytes,
            "outputs_bytes": tot_outputs_bytes,
            "mem_access": tot_mem_access,
            "latency": tot_latency,
        }
        return tot_info, analyze_report
