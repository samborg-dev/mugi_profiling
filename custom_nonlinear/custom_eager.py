import torch
import torch.nn as nn
from typing import Optional
from transformers.models.llama.modeling_llama import repeat_kv


def VITEager(nonlinear_object):
    def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling

        # Use the captured nonlinear_object (no self needed)
        attn_weights = nonlinear_object.forward(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

        if attention_mask is not None:
            attn_weights = attn_weights * attention_mask

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights
    return eager_attention_forward

def WhisperEager(nonlinear_object):
    def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: Optional[float] = None,
        dropout: float = 0.0,
        head_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if scaling is None:
            scaling = query.size(-1) ** -0.5

        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        if attention_mask is not None and attention_mask.ndim == 4:
            attn_weights = attn_weights + attention_mask[:, :, :, : key.shape[-2]]

        attn_weights = nonlinear_object(attn_weights, dim=-1, dtype=attn_weights.dtype)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask.view(1, -1, 1, 1)

        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights
    return eager_attention_forward

def LlamaEager(nonlinear_object):
    def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nonlinear_object(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights
    return eager_attention_forward

def VivitEager(nonlinear_object):
    def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling

        # Normalize the attention scores to probabilities.
        attn_weights = nonlinear_object(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

        # Mask heads if we want to
        if attention_mask is not None:
            attn_weights = attn_weights * attention_mask

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights

    return eager_attention_forward