from torch.nn.modules.sparse import Embedding as _Embedding

from .utils import update_analyze_report
from torch import Tensor
from .register import register_class


@register_class
class Embedding(_Embedding):
    raw_nn_class = _Embedding

    def forward(self, input: Tensor) -> Tensor:
        output = super().forward(input)
        inputs_shape = {"x": input.shape}
        outputs_shape = {"y": output.shape}

        # Embedding should not load all of the weights, so we need to specify the memory access offset
        update_analyze_report(
            self,
            operations=0,
            weights_shape={k: v.shape for k, v in self.named_parameters() if v is not None},
            inputs_shape=inputs_shape,
            outputs_shape=outputs_shape,
            info={"weight_access_offset": output.numel() - self.weight.data.numel()},
        )
        return output
