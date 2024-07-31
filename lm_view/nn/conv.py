from torch.nn import Conv2d as _Conv2d
from .utils import update_analyze_report
from torch import Tensor
from .register import register_class
import numpy as np


@register_class
class Conv2d(_Conv2d):
    raw_nn_class = _Conv2d

    def forward(self, input: Tensor) -> Tensor:
        output = super().forward(input)
        operations = output.numel() * np.prod(self.weight.shape[1:]) * 2  # mul, add
        inputs_shape = {"x": input.shape}
        outputs_shape = {"y": output.shape}
        update_analyze_report(
            self,
            operations=operations,
            weights_shape={k: v.shape for k, v in self.named_parameters() if v is not None},
            inputs_shape=inputs_shape,
            outputs_shape=outputs_shape,
        )
        return output
