import torch
from transformers import AutoModelForCausalLM, AutoConfig
from modelasv import LMViewAnalyzer, RooflineModel, LMSimulator
from transformers.modeling_utils import no_init_weights

model_id = "meta-llama/Llama-2-7b-hf"
device = "cuda"
with no_init_weights():
    config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_config(config).half().to(device).eval()


analyzer = LMViewAnalyzer()
bandwidth = 100e9
compute = 100e12
rf = RooflineModel(bandwidth, 27648e3, {8: compute})
simulator = LMSimulator(rf)

with torch.no_grad():
    print("=== prefill ===")
    analyzer.warp_model(model)
    seqlen = 1024
    x = torch.ones(1, seqlen).long().to(device)
    y = model(x)
    analyzer.unwarp_model(model)
    analyze_result = analyzer.accumulate_report(model)
    report, tot_report = simulator.simulate(analyze_result, a_bit=8, w_bit=8)
    print(tot_report)

    print("=== decode ===")
    analyzer.warp_model(model)
    kv_cache = y.past_key_values
    x = torch.ones(1, 1).long().to(device)
    y = model(x, past_key_values=kv_cache)
    analyzer.unwarp_model(model)
    analyze_result = analyzer.accumulate_report(model)
    report, tot_report = simulator.simulate(analyze_result, a_bit=8, w_bit=8)
    print(tot_report)
