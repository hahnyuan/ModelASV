# LM-View

This is a new project in developing.

LM-View (Large Models View) is a tool for analyzing models including large language models, multi-modality models and image/video generation models.
It has two main functions: Analyze model, Simulate model, and Visualize model (web browser).

Because the training analyzing is becoming more and more important, LM-View targets analyzing both the training and the infernce of models.

## Analyze model

LM-View uses torch.nn module wrappers to wrap all kinds of structures. And it generates the `analyze_report` on each module after run inference.

The `analyze_report` contains these information:
- operations
- load_memory
- store_memory

## Simulate model

LM-View provides the `Simulator` to estimate how fast a module runs at a given hardware device.

- `RooflineModelSimulator`

## Visualize model

Not implemented yet, please use [LLM-Viewer](https://github.com/hahnyuan/LLM-Viewer) now.