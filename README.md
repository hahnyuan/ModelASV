# LM-View

This is a new project in developing. Before 1.0 version is finished, it is recommended to use [LLM-Viewer](https://github.com/hahnyuan/LLM-Viewer).

LM-View (Large Models View) is a tool for analyzing models including large language models, multi-modality models and image/video generation models.
It has two main functions: Analyze model, Simulate model, and Visualize model (web browser).

Because the training analyzing is becoming more and more important, LM-View targets analyzing both the training and the infernce of models.

## Analyze model

LM-View uses torch.nn module wrappers to wrap all kinds of structures. And it generates the `analyze_report` on each module after run inference.

The `analyze_report` is a EasyDict contains these information:
- operations
- weights_shape
- inputs_shape
- outputs_shape

## Simulate model

LM-View provides the `Simulator` to estimate how fast a module runs at a given hardware device.

- `RooflineModelSimulator`

## Visualize model

Not implemented yet.