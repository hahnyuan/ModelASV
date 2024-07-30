# LM-View

This is a new project in developing. Before 1.0 version is finished, it is recommended to use [LLM-Viewer](https://github.com/hahnyuan/LLM-Viewer).

LM-View (Large Models View) is a tool for analyzing models including large language models, multi-modality models and image/video generation models.
It has two main functions: Analyze model, Simulate model, and Visualize model (web browser).

Because the training analyzing is becoming more and more important, LM-View targets analyzing both the training and the infernce of models.

## Analyze model

LM-View uses torch.nn module wrappers to wrap all kinds of structures. And it generates the `analyze_report` on each module after run inference.

The `analyze_report` is a dict (if module itself "", else sub-module name) contains list of dict with these information:
- operations
- weights_shape: dict
- inputs_shape: dict
- outputs_shape: dict
- info: dict, some useful information need to show, such as `num_key_value_groups`, `load_kv_cache`, `store_kv_cache`
If there are sub-module, there are more than one dict in the analyze_report dict.
If we execute the module several times, there should be multiple dict with these information.

## Simulate model

LM-View provides the `Simulator` to estimate how fast a module runs at a given hardware device.

- `RooflineModelSimulator`

## Visualize model

Not implemented yet.