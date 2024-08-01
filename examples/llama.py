import torch
from transformers import AutoModelForCausalLM, AutoConfig
from modelasv import LMViewAnalyzer, RooflineModel, LMSimulator
from transformers.modeling_utils import no_init_weights

model_id = "meta-llama/Llama-2-7b-hf"
device = "cuda"
with no_init_weights():
    config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_config(config).half().to(device).eval()


analyzer = LMViewAnalyzer(verbose=True)
bandwidth = 10e9
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
