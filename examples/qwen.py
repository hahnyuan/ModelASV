import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Config
from modelasv import LMViewAnalyzer, RooflineModel, LMSimulator
from transformers.modeling_utils import no_init_weights

model_id = "Qwen/Qwen1.5-7B"
device = "cuda"
with no_init_weights():
    config = Qwen2Config.from_pretrained(model_id)
    model = Qwen2ForCausalLM(config).half().to(device).eval()


analyzer = LMViewAnalyzer()
bandwidth = 100e9
compute_capacity = {8: 100e12}
# compute_capacity: the compute capacity of the accelerator, a dict with keys as bitwidth and values as FLOPS/OPS
#    The bitwidth is the bitwidth of the data type, e.g., 32 for float32 and 16 for float16
rf = RooflineModel(bandwidth, compute_capacity)
simulator = LMSimulator(rf)

with torch.no_grad():
    print("=== prefill ===")
    seqlen = 1024
    x = torch.ones(1, seqlen).long().to(device)
    with analyzer.analyze(model):
        y = model(x)
    analyze_result = analyzer.accumulate_report(model)
    report, tot_report = simulator.simulate(analyze_result, a_bit=8, w_bit=8)
    print(tot_report)

    print("=== decode ===")
    kv_cache = y.past_key_values
    x = torch.ones(1, 1).long().to(device)
    with analyzer.analyze(model):
        y = model(x, past_key_values=kv_cache)
    analyze_result = analyzer.accumulate_report(model)
    report, tot_report = simulator.simulate(analyze_result, a_bit=8, w_bit=8)
    print(tot_report)
