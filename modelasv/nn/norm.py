import torch.nn as nn
from torch.nn import GroupNorm as _GroupNorm
from torch.nn import LayerNorm as _LayerNorm
import torch
from typing import List, Optional, Tuple, Union
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm as _Qwen2RMSNorm

from .utils import update_analyze_report
from .register import register_class
from transformers.models.llama.modeling_llama import LlamaRMSNorm as _LlamaRMSNorm


@register_class
class Qwen2RMSNorm(_Qwen2RMSNorm):
    raw_nn_class = _Qwen2RMSNorm

    def forward(self, hidden_states):
        output = super().forward(hidden_states)
        # pow2, mean, add, rsqrt, mul
        operations = hidden_states.numel() * 5
        inputs_shape = {"x": hidden_states.shape}
        outputs_shape = {"y": output.shape}
        update_analyze_report(
            self,
            operations=operations,
            weights_shape={k: v.shape for k, v in self.named_parameters() if v is not None},
            inputs_shape=inputs_shape,
            outputs_shape=outputs_shape,
        )
        return output


@register_class
class LlamaRMSNorm(_LlamaRMSNorm):
    raw_nn_class = _LlamaRMSNorm

    def forward(self, hidden_states):
        output = super().forward(hidden_states)
        # pow2, mean, add, rsqrt, mul
        operations = hidden_states.numel() * 5
        inputs_shape = {"x": hidden_states.shape}
        outputs_shape = {"y": output.shape}
        update_analyze_report(
            self,
            operations=operations,
            weights_shape={k: v.shape for k, v in self.named_parameters() if v is not None},
            inputs_shape=inputs_shape,
            outputs_shape=outputs_shape,
        )
        return output


@register_class
class GroupNorm(_GroupNorm):
    raw_nn_class = _GroupNorm

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        # mean, var, sub, div, mul, add
        if self.affine:
            operations = input.numel() * 6
        else:
            operations = input.numel() * 4
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


@register_class
class LayerNorm(_LayerNorm):
    raw_nn_class = _LayerNorm

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        # mean, var, sub, div, mul, add
        if self.elementwise_affine:
            operations = input.numel() * 6
        else:
            operations = input.numel() * 4
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
