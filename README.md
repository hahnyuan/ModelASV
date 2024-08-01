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

You can see example in `examples` folder. 

To analyze the performance of a deep learning model, you can use the `LMViewAnalyzer`, `RooflineModel`, and `LMSimulator` classes from the modelasv module.

The `LMViewAnalyzer` is used to collect information, such as the number of operations, and input and output shapes, for a given model. The `RooflineModel` is used to estimate the theoretical performance limit of a operation based on the hardware specifications, such as memory bandwidth and compute capability. The `LMSimulator` is then used to simulate the performance of the model with different bit-widths, taking into account the performance data collected by the analyzer and the theoretical performance limit estimated by the roofline model.

Here's an example of how to use these classes to analyze the performance of the Stable Diffusion model:

```python
import torch
from diffusers import StableDiffusionPipeline
from modelasv import LMViewAnalyzer, RooflineModel, LMSimulator

# Load the Stable Diffusion model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Initialize the performance analysis tools
analyzer = LMViewAnalyzer()
bandwidth = 100e9  # Memory bandwidth of hardware in bytes/s
compute_capacity = {8: 100e12}  
# compute_capacity: the compute capacity of the accelerator, a dict with keys as bitwidth and values as FLOPS/OPS
#    The bitwidth is the bitwidth of the data type, e.g., 32 for float32 and 16 for float16
rf = RooflineModel(bandwidth, compute_capacity)  # Create a roofline model

simulator = LMSimulator(rf)

# Analyze the performance of different sub-model of the Stable Diffusion model
with torch.no_grad():
    for model in [pipe.text_encoder, pipe.unet, pipe.vae]:
        # Generate an image using the Stable Diffusion model
        prompt = "a photo of an astronaut riding a horse on mars"

        # Use this context to enable analyze
        with analyzer.analyze(model):
            image = pipe(prompt, num_inference_steps=20).images[0]

        # Collect performance data and simulate the model with different bit-widths
        analyze_result = analyzer.accumulate_report(model)
        report, tot_report = simulator.simulate(analyze_result, a_bit=8, w_bit=8)
        print(tot_report)
```

In this example, we first load the Stable Diffusion model using the StableDiffusionPipeline class from the diffusers library. We then create an LMViewAnalyzer to collect performance data, a RooflineModel to estimate the theoretical performance limit, and an LMSimulator to simulate the model's performance with different bit-widths.

We loop through the key components of the Stable Diffusion model (the text encoder, U-Net, and VAE) and generate an image using the model. During the image generation, the LMViewAnalyzer collects performance data, which is then passed to the LMSimulator to simulate the model's performance with 8-bit activations and weights.

The tot_report variable contains the simulated performance results, which can be used to analyze the model's efficiency and identify potential bottlenecks.


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