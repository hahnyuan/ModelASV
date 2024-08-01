import torch
from diffusers import StableDiffusionPipeline
from modelasv import LMViewAnalyzer, RooflineModel, LMSimulator


model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")


analyzer = LMViewAnalyzer()
bandwidth = 100e9
compute = 100e12
rf = RooflineModel(bandwidth, 27648e3, {8: compute})
simulator = LMSimulator(rf)

with torch.no_grad():
    for model in [pipe.text_encoder, pipe.unet, pipe.vae]:
        analyzer.warp_model(model)
        prompt = "a photo of an astronaut riding a horse on mars"
        image = pipe(prompt, num_inference_steps=20).images[0]
        analyzer.unwarp_model(model)
        analyze_result = analyzer.accumulate_report(model)
        report, tot_report = simulator.simulate(analyze_result, a_bit=8, w_bit=8)
        print(tot_report)
