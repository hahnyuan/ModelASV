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
