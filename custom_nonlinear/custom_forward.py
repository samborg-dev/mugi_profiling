import torch
import math
from typing import Optional, Callable, Unpack, Union
from transformers.cache_utils import Cache, EncoderDecoderCache

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from transformers.models.llama.modeling_llama import ALL_ATTENTION_FUNCTIONS as LLAMA_ATTENTION_FUNCTIONS

from transformers.models.whisper.modeling_whisper import FlashAttentionKwargs
from transformers.models.whisper.modeling_whisper import ALL_ATTENTION_FUNCTIONS as WHISPER_ATTENTION_FUNCTIONS

from transformers.models.vivit.modeling_vivit import ALL_ATTENTION_FUNCTIONS as VIVIT_ATTENTION_FUNCTIONS

def llama_forward(llama_eager):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = llama_eager
        if self.config._attn_implementation != "eager":
            attention_interface = LLAMA_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    return forward

def swin_forward(nonlinear_operation):
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = False,
        ) -> tuple[torch.Tensor]:
            batch_size, dim, num_channels = hidden_states.shape
            query_layer = (
                self.query(hidden_states)
                .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
                .transpose(1, 2)
            )
            key_layer = (
                self.key(hidden_states)
                .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
                .transpose(1, 2)
            )
            value_layer = (
                self.value(hidden_states)
                .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
                .transpose(1, 2)
            )

            # cosine attention
            attention_scores = torch.nn.functional.normalize(query_layer, dim=-1) @ torch.nn.functional.normalize(
                key_layer, dim=-1
            ).transpose(-2, -1)
            logit_scale = torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()
            attention_scores = attention_scores * logit_scale
            relative_position_bias_table = self.continuous_position_bias_mlp(self.relative_coords_table).view(
                -1, self.num_attention_heads
            )
            # [window_height*window_width,window_height*window_width,num_attention_heads]
            relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
            )
            # [num_attention_heads,window_height*window_width,window_height*window_width]
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
            attention_scores = attention_scores + relative_position_bias.unsqueeze(0)

            if attention_mask is not None:
                # Apply the attention mask is (precomputed for all layers in Swinv2Model forward() function)
                mask_shape = attention_mask.shape[0]
                attention_scores = attention_scores.view(
                    batch_size // mask_shape, mask_shape, self.num_attention_heads, dim, dim
                ) + attention_mask.unsqueeze(1).unsqueeze(0)
                attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(0)
                attention_scores = attention_scores.view(-1, self.num_attention_heads, dim, dim)

            # Normalize the attention scores to probabilities.
            attention_probs = nonlinear_operation(attention_scores, dim=-1)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.dropout(attention_probs)

            # Mask heads if we want to
            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(new_context_layer_shape)

            outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

            return outputs
    return forward

def vivit_forward(vivit_eager):
    def forward(
            self,
            hidden_states,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
        ) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor]]:
            batch_size, seq_length, _ = hidden_states.shape
            key_layer = (
                self.key(hidden_states)
                .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
                .transpose(1, 2)
            )
            value_layer = (
                self.value(hidden_states)
                .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
                .transpose(1, 2)
            )
            query_layer = (
                self.query(hidden_states)
                .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
                .transpose(1, 2)
            )

            attention_interface: Callable = vivit_eager
            if self.config._attn_implementation != "eager":
                if not(self.config._attn_implementation == "sdpa" and output_attentions):
                    attention_interface = VIVIT_ATTENTION_FUNCTIONS[self.config._attn_implementation]

            context_layer, attention_probs = attention_interface(
                self,
                query_layer,
                key_layer,
                value_layer,
                head_mask,
                is_causal=self.is_causal,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.dropout_prob,
            )

            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.reshape(new_context_layer_shape)

            outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

            return outputs
    return forward

def whisper_forward(whisper_eager):
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        # TODO: we need a refactor so that the different attention modules can get their specific kwargs
        # ATM, we have mixed things encoder, decoder, and encoder-decoder attn
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        # determine input shapes
        bsz, tgt_len = hidden_states.shape[:-1]
        q_input_shape = (bsz, tgt_len, -1, self.head_dim)

        # Scaling is susceptible to floating point arithmetics' inprecisions
        # which can lead to different results (this is dependent from model
        # to model, e.g. whisper is one such case). We therefore keep the
        # original order of scaling to follow the original implementation
        # and enforce no scaling (1.0) in the attention call below.
        query_states = self.q_proj(hidden_states) * self.scaling
        query_states = query_states.view(*q_input_shape)
        query_states = query_states.transpose(1, 2).contiguous()

        # Check is encoder-decoder model is being used. Otherwise we'll get `DynamicCache`
        if past_key_value is not None and isinstance(past_key_value, EncoderDecoderCache):
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                past_key_value.is_updated[self.layer_idx] = True
                past_key_value = past_key_value.cross_attention_cache
            else:
                past_key_value = past_key_value.self_attention_cache

        # use key_value_states if cross attention
        current_states = key_value_states if key_value_states is not None else hidden_states
        if is_cross_attention and past_key_value and is_updated:
            # reuse k,v, cross_attentions
            key_states = past_key_value.layers[self.layer_idx].keys
            value_states = past_key_value.layers[self.layer_idx].values
        else:
            key_states = self.k_proj(current_states).view(bsz, -1, self.num_heads, self.head_dim)
            value_states = self.v_proj(current_states).view(bsz, -1, self.num_heads, self.head_dim)
            key_states = key_states.transpose(1, 2).contiguous()
            value_states = value_states.transpose(1, 2).contiguous()
            if past_key_value is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )

        attention_interface: Callable = whisper_eager
        if self.config._attn_implementation != "eager":
            attention_interface = WHISPER_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=1.0,
            output_attentions=output_attentions,
            head_mask=layer_head_mask,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights, past_key_value
    return forward