from torch.nn import Linear as _Linear
from easydict import EasyDict
from torch import Tensor


class Linear(_Linear):
    raw_nn_class = _Linear

    def forward(self, input: Tensor) -> Tensor:
        output = super().forward(input)
        operations = output.numel() * self.in_features
        weights_shape = [self.weight.shape]
        if self.bias is not None:
            operations += output.numel()
            weights_shape.append(self.bias.shape)
        inputs_shape = {"x1": input.shape}
        outputs_shape = {"y1": output.shape}
        # print(
        #     f"Linear: {operations} operations, weights shape: {weights_shape}, inputs shape: {inputs_shape}, outputs shape: {outputs_shape}"
        # )
        self.analyze_report = EasyDict(
            operations=operations,
            weights_shape=weights_shape,
            inputs_shape=inputs_shape,
            outputs_shape=outputs_shape,
        )
        return output
