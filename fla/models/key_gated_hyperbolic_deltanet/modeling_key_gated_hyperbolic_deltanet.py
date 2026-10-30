# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Adapted from HDN at 46c54ef0d57609d66a5c158fa4a8cce124ca1483.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils.deprecation import deprecate_kwarg

from fla.layers.key_gated_hyperbolic_deltanet import KeyGatedHyperbolicDeltaNet
from fla.models.generation_utils import FLAUnsupportedCacheGenerationMixin
from fla.models.key_gated_hyperbolic_deltanet.configuration_key_gated_hyperbolic_deltanet import (
    KeyGatedHyperbolicDeltaNetConfig,
)
from fla.models.modeling_utils import FLAForCausalLM, FLAModel, FLAPreTrainedModel, get_attention
from fla.modules import GatedMLP, RMSNorm
from fla.ops.attnres import fused_attnres

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.cache_utils import Cache


class KeyGatedHyperbolicDeltaNetBlock(GradientCheckpointingLayer):

    def __init__(self, config: KeyGatedHyperbolicDeltaNetConfig, layer_idx: int):
        super().__init__()
        self.config, self.layer_idx = config, layer_idx
        norm = RMSNorm if config.fuse_norm else nn.RMSNorm
        self.attn_norm = norm(config.hidden_size, eps=config.norm_eps)
        self.attn = get_attention(config=config, layer_idx=layer_idx)
        if self.attn is None:
            self.attn = KeyGatedHyperbolicDeltaNet(
                hidden_size=config.hidden_size,
                expand_v=config.expand_v,
                head_dim=config.head_dim,
                num_heads=config.num_heads,
                num_v_heads=config.num_v_heads,
                mode=config.attn_mode,
                curvature=config.curvature,
                learnable_curvature=config.learnable_curvature,
                value_margin=config.value_margin,
                group_size=config.group_size,
                chunk_size=config.chunk_size,
                carry_mode=config.carry_mode,
                precision_policy=config.precision_policy,
                decay_mode=config.decay_mode,
                decay_rank=config.decay_rank,
                decay_function=config.decay_function,
                readout=config.readout,
                use_gate=config.use_gate,
                use_short_conv=config.use_short_conv,
                conv_size=config.conv_size,
                conv_bias=config.conv_bias,
                fuse_norm=config.fuse_norm,
                norm_eps=config.norm_eps,
                layer_idx=layer_idx,
            )
        self.mlp_norm = norm(config.hidden_size, eps=config.norm_eps)
        self.mlp = GatedMLP(
            hidden_size=config.hidden_size, hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size, hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )
        self.use_attnres = config.attnres_block_size is not None
        if self.use_attnres:
            self.attn_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
            self.attn_res_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.mlp_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
            self.mlp_res_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.attnres_is_attn_boundary = (2 * layer_idx) % config.attnres_block_size == 0
            self.attnres_is_mlp_boundary = (2 * layer_idx + 1) % config.attnres_block_size == 0
            self.attn_res_proj._is_attnres_proj = True
            self.mlp_res_proj._is_attnres_proj = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        attnres_states: list[torch.Tensor] | None = None,
        **kwargs: Unpack[dict],
    ):
        if self.use_attnres:
            prefix_sum = hidden_states
            if attnres_states is None:
                hidden_states = self.attn_norm(prefix_sum)
                attnres_states = [prefix_sum]
                prefix_sum = None
            else:
                residuals = [*attnres_states, prefix_sum]
                if self.attnres_is_attn_boundary:
                    attnres_states, prefix_sum = residuals, None
                hidden_states = fused_attnres(
                    query=self.attn_res_proj.weight, residuals=residuals,
                    rms_weight=self.attn_res_norm.weight, output_rms_weight=self.attn_norm.weight,
                    rms_eps=self.attn_res_norm.eps,
                )
        else:
            residual = hidden_states
            hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states, attention_mask=attention_mask,
            past_key_values=past_key_values, use_cache=use_cache,
            output_attentions=output_attentions, **kwargs,
        )
        if self.use_attnres:
            prefix_sum = hidden_states if prefix_sum is None else prefix_sum + hidden_states
            residuals = [*attnres_states, prefix_sum]
            if self.attnres_is_mlp_boundary:
                attnres_states, prefix_sum = residuals, None
            hidden_states = fused_attnres(
                query=self.mlp_res_proj.weight, residuals=residuals,
                rms_weight=self.mlp_res_norm.weight, output_rms_weight=self.mlp_norm.weight,
                rms_eps=self.mlp_res_norm.eps,
            )
        elif self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        if self.use_attnres:
            hidden_states = hidden_states if prefix_sum is None else prefix_sum + hidden_states
        else:
            hidden_states = residual + hidden_states
        return hidden_states, attentions, past_key_values, attnres_states


class KeyGatedHyperbolicDeltaNetPreTrainedModel(FLAPreTrainedModel):
    config_class = KeyGatedHyperbolicDeltaNetConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['KeyGatedHyperbolicDeltaNetBlock']
    _supports_cache_class = True
    _keep_in_fp32_modules = ['A_log', 'dt_bias', 'curvature_raw']
    _keep_in_fp32_modules_strict = ['A_log', 'dt_bias', 'curvature_raw']

    def __init__(self, config: KeyGatedHyperbolicDeltaNetConfig, *args, **kwargs):
        if not isinstance(config, KeyGatedHyperbolicDeltaNetConfig):
            raise TypeError('KHDN requires KeyGatedHyperbolicDeltaNetConfig; legacy HDN configs are incompatible.')
        super().__init__(config, *args, **kwargs)

    def _init_weights(self, module: nn.Module, **kwargs):
        super()._init_weights(module, **kwargs)
        if isinstance(module, KeyGatedHyperbolicDeltaNet):
            module._init_gate_parameters(respect_hf_initialized=True)


class KeyGatedHyperbolicDeltaNetModel(KeyGatedHyperbolicDeltaNetPreTrainedModel, FLAModel):

    def __init__(self, config: KeyGatedHyperbolicDeltaNetConfig):
        super().__init__(config)
        self.padding_idx, self.vocab_size = config.pad_token_id, config.vocab_size
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            KeyGatedHyperbolicDeltaNetBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.use_attnres = config.attnres_block_size is not None
        if self.use_attnres:
            self.res_proj = nn.Linear(config.hidden_size, 1, bias=False)
            self.res_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.res_proj._is_attnres_proj = True
        self.gradient_checkpointing = False
        self.post_init()


class KeyGatedHyperbolicDeltaNetForCausalLM(
    KeyGatedHyperbolicDeltaNetPreTrainedModel, FLAForCausalLM, FLAUnsupportedCacheGenerationMixin,
):
    _tied_weights_keys = {'lm_head.weight': 'model.embeddings.weight'}

    def __init__(self, config: KeyGatedHyperbolicDeltaNetConfig):
        super().__init__(config)
        self.model = KeyGatedHyperbolicDeltaNetModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None
        self.post_init()

    @deprecate_kwarg('num_logits_to_keep', version='4.50', new_name='logits_to_keep')
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | torch.Tensor | None = 0,
        num_items_in_batch: int | torch.Tensor | None = None,
        **kwargs: Unpack[dict],
    ) -> tuple | CausalLMOutputWithPast:
        cu_seqlens = kwargs.get('cu_seqlens')
        if labels is not None and cu_seqlens is not None:
            ignore_index = self.criterion.ignore_index if self.criterion is not None else -100
            labels = labels.clone()
            starts = cu_seqlens[1:-1].to(device=labels.device, dtype=torch.long)
            # The shared LM loss shifts labels left. Do not predict across packed boundaries.
            labels.reshape(-1).index_fill_(0, starts[starts < labels.numel()], ignore_index)
        return super().forward(
            input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds,
            past_key_values=past_key_values, labels=labels, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict, logits_to_keep=logits_to_keep,
            num_items_in_batch=num_items_in_batch, **kwargs,
        )
