# ModelASV

This is a project in developing. 

ModelASV is a tool for analyzing models including large language models, multi-modality models and image/video generation models.
It has two main functions: Analyze model, Simulate model, and Visualize model (web browser).

Because the training analyzing is becoming more and more important, ModelASV targets analyzing both the training and the infernce of models.

## Install

```bash
pip install modelasv
```

## Usage

You can see example in `examples` folder. Here is a simple example:

```python
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
```


## API description

### Analyze model

ModelASV uses torch.nn module wrappers to wrap all kinds of structures. And it generates the `analyze_report` on each module after run inference.

The `analyze_report` is a dict (if module itself "", else sub-module name) contains list of dict with these information:
- operations
- weights_shape: dict
- inputs_shape: dict
- outputs_shape: dict
- info: dict, some useful information need to show, such as `num_key_value_groups`, `load_kv_cache`, `store_kv_cache`
If there are sub-module, there are more than one dict in the analyze_report dict.
If we execute the module several times, there should be multiple dict with these information.

### Simulate model

ModelASV provides the `Simulator` to estimate how fast a module runs at a given hardware device.

- `RooflineModelSimulator`

### Visualize model

Comming soon.