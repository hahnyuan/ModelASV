import numpy as np
import copy


class RooflineModel:
    def __init__(self, bandwidth, compute_capacity):
        """
        bandwidth: the bandwidth of the accelerator
        compute_capacity: the compute capacity of the accelerator, a dict with keys as bitwidth and values as FLOPS/OPS
            The bitwidth is the bitwidth of the data type, e.g., 32 for float32 and 16 for float16
        """
        self.bandwidth = bandwidth
        self.compute_capacity = compute_capacity

    def run(self, operations, memory_access_bytes, compute_bit, info):
        max_OPS = self.compute_capacity[compute_bit]
        y_max = max_OPS
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
        simulate_info = {
            "bound": bound,
        }
        return inference_time, simulate_info
