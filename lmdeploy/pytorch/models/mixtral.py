# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from ..kernels import apply_rotary_pos_emb, fill_kv_cache, paged_attention_fwd
from ..kernels.fused_moe import fused_moe
from ..weight_loader.dist_utils import (colwise_parallelize_linear,
                                        rowwise_parallelize_linear)


class PatchedMixtralAttention(nn.Module):
    """Rewrite module of MixtralAttention."""

    def _load_weights(self, loader, rank: int, world_size: int,
                      device: torch.device):
        """load weights."""
        for mod_name in ['q_proj', 'k_proj', 'v_proj']:
            colwise_parallelize_linear(getattr(self, mod_name),
                                       loader,
                                       rank=rank,
                                       world_size=world_size,
                                       prefix=mod_name)
        rowwise_parallelize_linear(self.o_proj,
                                   loader,
                                   rank=rank,
                                   world_size=world_size,
                                   prefix='o_proj')

    @classmethod
    def _distribute_output_fn(cls, outputs, **kwargs):
        """Distribution output hook."""
        dist.all_reduce(outputs[0])
        return outputs

    def _contiguous_batching_forward_impl(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        world_size: int = 1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """default rewrite."""

        context = self.context.context
        kv_seq_length = context.kv_seq_length
        q_seq_length = context.q_seq_length
        q_start_loc = context.q_start_loc
        block_offsets = context.block_offsets
        max_q_seq_length = context.max_q_seq_length
        max_kv_seq_length = context.max_kv_seq_length

        num_heads = self.num_heads // world_size
        num_kv_heads = self.num_key_value_heads // world_size
        hidden_size = num_heads * self.head_dim

        def __qkv_proj(hidden_states):
            """qkv proj."""
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            return query_states, key_states, value_states

        def __rotary_emb_fn(query_states, key_states, value_states):
            if hasattr(self, 'rotary_emb'):
                if not hasattr(context, '_cos'):
                    cos, sin = self.rotary_emb(value_states,
                                               seq_len=max_kv_seq_length)
                    context._cos = cos
                    context._sin = sin
                else:
                    cos = context._cos
                    sin = context._sin
                query_states, key_states = apply_rotary_pos_emb(
                    query_states,
                    key_states,
                    cos,
                    sin,
                    position_ids,
                    context.position_ids_1d,
                    q_embed=query_states,
                    k_embed=key_states)
            return query_states, key_states, value_states

        query_states, key_states, value_states = __qkv_proj(hidden_states)

        query_states = query_states.view(-1, num_heads, self.head_dim)
        key_states = key_states.view(-1, num_kv_heads, self.head_dim)
        value_states = value_states.view(-1, num_kv_heads, self.head_dim)

        query_states, key_states, value_states = __rotary_emb_fn(
            query_states, key_states, value_states)
        # fill kv cache
        fill_kv_cache(
            key_states,
            value_states,
            past_key_value[0],
            past_key_value[1],
            q_start_loc,
            q_seq_length,
            kv_seq_length=kv_seq_length,
            max_q_seq_length=max_q_seq_length,
            block_offsets=block_offsets,
        )
        # page attention
        attn_output = query_states
        window_size = self.config.sliding_window or -1
        paged_attention_fwd(
            query_states,
            past_key_value[0],
            past_key_value[1],
            attn_output,
            block_offsets,
            q_start_loc=q_start_loc,
            q_seqlens=q_seq_length,
            kv_seqlens=kv_seq_length,
            max_seqlen=max_q_seq_length,
            window_size=window_size,
        )

        attn_output = attn_output.reshape(*hidden_states.shape[:-1],
                                          hidden_size)

        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """Rewrite of MistralAttention.forward."""
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()
        return self._contiguous_batching_forward_impl(
            hidden_states,
            position_ids,
            past_key_value,
            output_attentions,
            attention_mask=attention_mask,
            world_size=world_size,
        )


class PatchedMixtralBLockSparseTop2MLP(nn.Module):

    def _load_weights(self, loader, rank: int, world_size: int,
                      device: torch.device):
        """load weights."""
        for mod_name in ['w1', 'w3']:
            colwise_parallelize_linear(getattr(self, mod_name),
                                       loader,
                                       rank=rank,
                                       world_size=world_size,
                                       prefix=mod_name)
        rowwise_parallelize_linear(self.w2,
                                   loader,
                                   rank=rank,
                                   world_size=world_size,
                                   prefix='w2')

    @classmethod
    def _distribute_output_fn(cls, outputs, **kwargs):
        """Distribution output hook."""
        dist.all_reduce(outputs)
        return outputs


class PatchedMixtralSparseMoeBlock(nn.Module):

    def _update_model_fn(self):
        """update model."""
        num_experts = self.num_experts

        def __get_meta():
            exp = self.experts[0]
            ffn_dim = exp.w1.weight.size(0)
            hidden_dim = exp.w2.weight.size(0)
            dtype = exp.w1.weight.dtype
            device = exp.w1.weight.device
            return ffn_dim, hidden_dim, dtype, device

        def __copy_assign_param(param, weight):
            """copy assign."""
            weight.copy_(param.data)
            param.data = weight

        ffn_dim, hidden_dim, dtype, device = __get_meta()

        gate_up_weights = torch.empty(num_experts,
                                      ffn_dim * 2,
                                      hidden_dim,
                                      device=device,
                                      dtype=dtype)
        down_weights = torch.empty(num_experts,
                                   hidden_dim,
                                   ffn_dim,
                                   device=device,
                                   dtype=dtype)
        for exp_id, exp in enumerate(self.experts):
            __copy_assign_param(exp.w1.weight,
                                gate_up_weights[exp_id, :ffn_dim])
            __copy_assign_param(exp.w3.weight, gate_up_weights[exp_id,
                                                               ffn_dim:])
            __copy_assign_param(exp.w2.weight, down_weights[exp_id])

        torch.cuda.empty_cache()

        self.register_buffer('gate_up_weights', gate_up_weights)
        self.register_buffer('down_weights', down_weights)

    @classmethod
    def _distribute_output_fn(cls, outputs, **kwargs):
        """Distribution output hook."""
        dist.all_reduce(outputs[0])
        return outputs

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """rewrite moe forward."""

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights = torch.softmax(router_logits,
                                        dim=-1,
                                        dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(routing_weights,
                                            self.top_k,
                                            dim=-1)
        del routing_weights
        out_states = fused_moe(hidden_states,
                               self.gate_up_weights,
                               self.down_weights,
                               topk_weights,
                               topk_ids,
                               topk=self.top_k,
                               renormalize=True)

        out_states = out_states.reshape(batch_size, sequence_length, -1)
        return out_states, router_logits


class PatchedMixtralModel(nn.Module):

    def _continuous_batching_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """Rewrite implementation of LlamaModel.forward."""

        from transformers.modeling_outputs import MoeModelOutputWithPast

        output_attentions = (output_attentions if output_attentions is not None
                             else self.config.output_attentions)
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)

        if use_cache is None:
            use_cache = self.config.use_cache

        return_dict = (return_dict if return_dict is not None else
                       self.config.use_return_dict)

        assert (
            position_ids is not None
        ), 'position_ids can not be none when using continuous batching mode.'
        assert position_ids.dim() == 2

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Attention mask is not necessary in continuous batching
        attention_mask = None

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states, )

            past_key_value = (past_key_values[idx]
                              if past_key_values is not None else None)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (
                    layer_outputs[2 if output_attentions else 1], )

            if output_attentions:
                all_self_attns += (layer_outputs[1], )

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states, )

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v for v in
                [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None)

        return MoeModelOutputWithPast(last_hidden_state=hidden_states,
                                      past_key_values=next_cache,
                                      hidden_states=all_hidden_states,
                                      attentions=all_self_attns,
                                      router_logits='')

    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs) -> Union[Tuple, BaseModelOutputWithPast]:
        """Rewrite of LlamaModel.forward."""
        return self._continuous_batching_forward(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
