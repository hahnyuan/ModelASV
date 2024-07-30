def update_analyze_report(
    self, sub_module_name="", operations=None, weights_shape=None, inputs_shape=None, outputs_shape=None, info=None
):
    if not hasattr(self, "analyze_report") or self.analyze_report is None:
        self.analyze_report = {}
    if sub_module_name not in self.analyze_report:
        self.analyze_report[sub_module_name] = []
    if weights_shape is None:
        weights_shape = {k: v.shape for k, v in self.named_parameters() if v is not None}
    self.analyze_report[sub_module_name] += [
        dict(
            operations=operations,
            weights_shape=weights_shape,
            inputs_shape=inputs_shape,
            outputs_shape=outputs_shape,
            info=info,
        )
    ]
