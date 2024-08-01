import torch.nn as nn
import torch
from typing import List, Optional, Tuple, Union
from transformers.models.qwen2.modeling_qwen2 import Qwen2SdpaAttention as _Qwen2SdpaAttention
from transformers.models.llama.modeling_llama import LlamaSdpaAttention as _LlamaSdpaAttention
from transformers.models.bert.modeling_bert import BertAttention as _BertAttention
from diffusers.models.attention_processor import Attention as _Attention
from .utils import update_analyze_report
from .register import register_class


def analyze_attn(self, B, Q, KV, H, D, n_cache, GQA):
    """
    Args:
        B (int): The batch size.
        Q (int): The number of query seq len.
        KV (int): The number of key-value seq len.
        H (int): The number of attention heads.
        D (int): The head dimension of the query, key, and value.
        n_cache (int): The number of KV cache elements.
        GQA (int): The number of groups (group query attention).
    """
    # matmul qk
    operations = B * H * D * Q * KV * 2
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
    operations = B * H * KV * Q * D * 2
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

        GQA = self.num_key_value_groups
        B, Q, _ = hidden_states.size()
        H = self.num_heads
        D = self.head_dim
        if past_key_value is not None:
            # this should run before forward because it will change the shape of KV
            n_cache = past_key_value.get_usable_length(Q, self.layer_idx)
            KV = Q + n_cache
        else:
            n_cache = 0
            KV = Q
        analyze_attn(self, B, Q, KV, H, D, n_cache, GQA)

        output = super().forward(
            hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs
        )

        return output


@register_class
class LlamaSdpaAttention(_LlamaSdpaAttention):
    raw_nn_class = _LlamaSdpaAttention

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        GQA = self.num_key_value_groups
        B, Q, _ = hidden_states.size()
        H = self.num_heads
        D = self.head_dim
        if past_key_value is not None:
            # this should run before forward because it will change the shape of KV
            n_cache = past_key_value.get_usable_length(Q, self.layer_idx)
            KV = Q + n_cache
        else:
            n_cache = 0
            KV = Q
        analyze_attn(self, B, Q, KV, H, D, n_cache, GQA)
        # print(f"Q={Q}, KV={KV}, n_cache={n_cache}, past_key_value={len(past_key_value)}")

        output = super().forward(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
            **kwargs,
        )
        return output


@register_class
class Attention(_Attention):
    raw_nn_class = _Attention

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:

        GQA = 1
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            B, C, height, width = hidden_states.shape
            Q = height * width
        else:
            B, Q = hidden_states.shape[:2]

        H = self.heads
        D = self.inner_dim // H
        n_cache = 0
        if encoder_hidden_states is not None:
            # this should run before forward because it will change the shape of KV
            assert encoder_hidden_states.ndim == 3
            KV = encoder_hidden_states.size(1)
        else:
            KV = Q
        analyze_attn(self, B, Q, KV, H, D, n_cache, GQA)
        # print(f"Q={Q}, KV={KV}, n_cache={n_cache}, past_key_value={len(past_key_value)}")

        output = super().forward(hidden_states, encoder_hidden_states, attention_mask, **cross_attention_kwargs)
        return output


@register_class
class BertAttention(_BertAttention):
    raw_nn_class = _BertAttention

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:

        GQA = 1
        B, Q, _ = hidden_states.size()
        H = self.self.num_attention_heads
        D = self.self.attention_head_size
        if past_key_value is not None:
            # this should run before forward because it will change the shape of KV
            n_cache = past_key_value[0].size(2)
            KV = Q + n_cache
        else:
            n_cache = 0
            KV = Q
        analyze_attn(self, B, Q, KV, H, D, n_cache, GQA)

        output = super().forward(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )

        return output
