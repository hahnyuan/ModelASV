from torch.nn import Linear as _Linear
from .utils import update_analyze_report
from torch import Tensor
from .register import register_class


@register_class
class Linear(_Linear):
    raw_nn_class = _Linear

    def forward(self, input: Tensor) -> Tensor:
        output = super().forward(input)
        operations = output.numel() * self.in_features * 2  # mul, add
        inputs_shape = {"x": input.shape}
        outputs_shape = {"y": output.shape}
        # print(
        #     f"Linear: {operations} operations, weights shape: {weights_shape}, inputs shape: {inputs_shape}, outputs shape: {outputs_shape}"
        # )
        update_analyze_report(
            self,
            operations=operations,
            weights_shape={k: v.shape for k, v in self.named_parameters() if v is not None},
            inputs_shape=inputs_shape,
            outputs_shape=outputs_shape,
        )
        return output
