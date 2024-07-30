def update_analyze_report(self, operations=None, weights_shape=None, inputs_shape=None, outputs_shape=None, info=None):
    if not hasattr(self, "analyze_report") or self.analyze_report is None:
        self.analyze_report = []
    self.analyze_report += [
        dict(
            operations=operations,
            weights_shape=weights_shape,
            inputs_shape=inputs_shape,
            outputs_shape=outputs_shape,
            info=info,
        )
    ]
