import torch
from diffusers import StableDiffusionPipeline
from modelasv import LMViewAnalyzer, RooflineModel, LMSimulator


model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")


analyzer = LMViewAnalyzer()
bandwidth = 100e9  # 100GB/s
compute_capacity = {8: 100e12}  # 100TOPS
# compute_capacity: the compute capacity of the accelerator, a dict with keys as bitwidth and values as FLOPS/OPS
#    The bitwidth is the bitwidth of the data type, e.g., 32 for float32 and 16 for float16
rf = RooflineModel(bandwidth, compute_capacity)
simulator = LMSimulator(rf)

with torch.no_grad():
    for model in [pipe.text_encoder, pipe.unet, pipe.vae]:
        prompt = "a photo of an astronaut riding a horse on mars"
        with analyzer.analyze(model):
            image = pipe(prompt, num_inference_steps=20).images[0]
        analyze_result = analyzer.accumulate_report(model)
        report, tot_report = simulator.simulate(analyze_result, a_bit=8, w_bit=8)
        print(tot_report)
