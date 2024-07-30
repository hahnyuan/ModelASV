import torch.nn as nn
import torch
from typing import List, Optional, Tuple, Union
from transformers.models.qwen2.modeling_qwen2 import Qwen2SdpaAttention as _Qwen2SdpaAttention
from .utils import update_analyze_report
from .register import register_class


def analyze_attn(self, B, Q, KV, H, D, n_cache, GQA):
    # matmul qk
    operations = B * H * D * Q * KV
    inputs_shape = {"q": (B, H, Q, D), "k": (B, H // GQA, KV, D)}
    outputs_shape = {"s": (B, H, Q, KV)}

    update_analyze_report(
        self,
        "qk",
        operations=operations,
        inputs_shape=inputs_shape,
        outputs_shape=outputs_shape,
        info={"GQA": GQA, "load_kv_cache": n_cache * H // GQA * D},
    )

    # softmax
    operations = B * H * Q * KV * 5
    update_analyze_report(
        self,
        "softmax",
        operations=operations,
        inputs_shape={"s": (B, H, Q, KV)},
        outputs_shape={"p": (B, H, Q, KV)},
    )

    # matmul pv
    operations = B * H * KV * Q * D
    update_analyze_report(
        self,
        "pv",
        operations=operations,
        inputs_shape={"s": (B, H, Q, KV), "v": (B, H // GQA, KV, D)},
        outputs_shape={"y": (B, H, Q, D)},
        info={"GQA": GQA, "load_kv_cache": n_cache * H // GQA * D},
    )


@register_class
class Qwen2SdpaAttention(_Qwen2SdpaAttention):
    raw_nn_class = _Qwen2SdpaAttention

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output = super().forward(
            hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs
        )
        GQA = self.num_heads
        B, Q, _ = hidden_states.size()
        H = self.num_heads
        D = self.head_dim
        if past_key_value is not None:
            n_cache = past_key_value.get_usable_length(Q, self.layer_idx)
            KV = Q + n_cache
        else:
            KV = Q
        analyze_attn(self, B, Q, KV, H, D, n_cache, GQA)

        return output
