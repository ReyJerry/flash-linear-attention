# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

from fla.layers.utils import get_layer_cache, repad_hidden_states, unpad_hidden_states, update_layer_cache
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.ops.hyperbolic_delta_rule import (
    chunk_hyperbolic_delta_rule,
    fused_recurrent_hyperbolic_delta_rule,
    naive_recurrent_hyperbolic_delta_rule,
)
from fla.ops.hyperbolic_delta_rule.naive import prepare_key_value
from fla.ops.hyperbolic_delta_rule.protected import protected_recurrent_hyperbolic_delta_rule

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.cache_utils import Cache


class HyperbolicDeltaNet(nn.Module):
    r"""Hyperbolic KV projection memory with canonical state S=M/z.

    q/k/v projections, optional short convolutions, output normalization and cache
    handling follow the other FLA layers. Keys are normalized and values bounded
    so that ||k||^2 - curvature*v^2 stays positive. This layer implements the base
    projection rule, NOT the optional protected-old-association subspace variant.

    Args:
        mode: 'chunk', 'fused_recurrent', or the slow PyTorch oracle 'naive'.
        curvature: Fixed nonnegative scalar. Zero gives normalized Delta updates.
        value_margin: Bound relative to the admissible value radius, in (0, 1).
        chunk_size: 16 or 32; also the backward checkpoint interval.
        num_v_heads: Optional grouped-value head count, divisible by num_heads.
        fuse_norm: Use FLA fused norms; disable for the unfused reference path.

    The state is FP32 and has shape [batch_or_num_sequences, H_v, K, V].
    Curvature is fixed across tokens/layers unless configured per layer externally;
    it is not a trained tensor. No extra forgetting or negative-eigenvalue gate is
    introduced, since those would change the derived geometry.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1.0,
        head_dim: int = 128,
        num_heads: int = 16,
        num_v_heads: int | None = None,
        mode: str = 'chunk',
        curvature: float = 1.0,
        learnable_curvature: bool = False,
        protection_rank: int = 0,
        value_margin: float = 0.5,
        chunk_size: int = 16,
        use_gate: bool = True,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        fuse_norm: bool = True,
        norm_eps: float = 1e-6,
        layer_idx: int | None = None,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size, self.head_dim, self.num_heads = hidden_size, head_dim, num_heads
        self.num_v_heads = num_heads if num_v_heads is None else num_v_heads
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim, self.value_dim = num_heads * head_dim, self.num_v_heads * self.head_v_dim
        self.mode, self.value_margin = mode, value_margin
        self.learnable_curvature = learnable_curvature
        self.protection_rank = protection_rank
        if learnable_curvature:
            if curvature <= 0:
                raise ValueError('Learnable curvature requires an initial curvature > 0.')
            self.curvature_raw = nn.Parameter(torch.tensor(math.log(math.expm1(curvature)), dtype=torch.float32))
        else:
            self.curvature = float(curvature)
        if protection_rank < 0:
            raise ValueError('protection_rank must be nonnegative.')
        self.chunk_size, self.layer_idx = chunk_size, layer_idx
        self.use_gate, self.use_short_conv, self.fuse_norm = use_gate, use_short_conv, fuse_norm
        if min(hidden_size, head_dim, num_heads, self.num_v_heads, self.head_v_dim) <= 0 or head_dim > 256:
            raise ValueError('Dimensions must be positive and head_dim must be <=256.')
        if not math.isclose(head_dim * expand_v, self.head_v_dim):
            raise ValueError('head_dim * expand_v must be an integer.')
        if self.num_v_heads % num_heads:
            raise ValueError('num_v_heads must be divisible by num_heads.')
        if mode not in ('chunk', 'fused_recurrent', 'naive') or chunk_size not in (16, 32):
            raise ValueError('Unsupported mode or chunk_size (expected 16 or 32).')
        if not math.isfinite(curvature) or curvature < 0 or not 0 < value_margin < 1:
            raise ValueError('Require finite curvature >= 0 and 0 < value_margin < 1.')
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        if use_short_conv:
            self.q_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, bias=conv_bias, activation='silu')
            self.k_conv1d = ShortConvolution(self.key_dim, kernel_size=conv_size, bias=conv_bias, activation='silu')
            self.v_conv1d = ShortConvolution(self.value_dim, kernel_size=conv_size, bias=conv_bias, activation='silu')
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if fuse_norm and use_gate:
            self.o_norm = FusedRMSNormGated(self.head_v_dim, activation='sigmoid', eps=norm_eps)
        else:
            self.o_norm = (RMSNorm if fuse_norm else nn.RMSNorm)(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ):
        batch_size, q_len, _ = hidden_states.shape
        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape[0] != batch_size or attention_mask.shape[1] < q_len:
                raise ValueError('attention_mask must be a [B,L>=T] padding mask; dense attention masks are unsupported.')
        last_state = get_layer_cache(self, past_key_values)
        cached_recurrent = None if last_state is None else last_state['recurrent_state']
        if isinstance(cached_recurrent, dict):
            initial_state = cached_recurrent['state']
            protected_keys = cached_recurrent.get('protected_keys')
            protected_values = cached_recurrent.get('protected_values')
            protected_count = cached_recurrent.get('protected_count')
            protected_scores = cached_recurrent.get('protected_scores')
        else:
            initial_state = cached_recurrent
            protected_keys = protected_values = protected_count = protected_scores = None
        curvature = F.softplus(self.curvature_raw) if self.learnable_curvature else self.curvature
        conv_states = (None, None, None)
        if last_state is not None and last_state.get('conv_state') is not None:
            conv_states = last_state['conv_state']
        hidden_states, indices, cu_seqlens = unpad_hidden_states(
            hidden_states, kwargs.get('cu_seqlens'), attention_mask if q_len else None, q_len,
        )
        projections = (self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states))
        if self.use_short_conv and hidden_states.shape[1] > 0:
            convolved = [
                conv(x=x, cache=cache, output_final_state=use_cache, cu_seqlens=cu_seqlens)
                for x, conv, cache in zip(projections, (self.q_conv1d, self.k_conv1d, self.v_conv1d), conv_states)
            ]
            (q, k, raw_v), conv_states = tuple(zip(*convolved))
        else:
            q, k, raw_v = (F.silu(x) for x in projections)
        q, k = (rearrange(x, 'b t (h d) -> b t h d', d=self.head_dim) for x in (q, k))
        raw_v = rearrange(raw_v, 'b t (h d) -> b t h d', d=self.head_v_dim)
        if self.num_v_heads != self.num_heads:
            q, k = (repeat(x, 'b t h d -> b t (h g) d', g=self.num_v_heads // self.num_heads) for x in (q, k))
        q = F.normalize(q.float(), dim=-1, eps=1e-6).to(hidden_states.dtype)
        k, v = prepare_key_value(k, raw_v, curvature, self.value_margin)
        beta = self.b_proj(hidden_states).float().sigmoid()
        mode = self.mode
        if mode != 'naive' and not self.training and q_len <= 64:
            mode = 'fused_recurrent'
        op = {
            'chunk': chunk_hyperbolic_delta_rule,
            'fused_recurrent': fused_recurrent_hyperbolic_delta_rule,
            'naive': naive_recurrent_hyperbolic_delta_rule,
        }[mode]
        extra = {'chunk_size': self.chunk_size} if mode == 'chunk' else {}
        if mode == 'fused_recurrent':
            extra['checkpoint_size'] = self.chunk_size
        if self.protection_rank:
            if cu_seqlens is not None or attention_mask is not None:
                raise ValueError('Protected memory currently requires dense, unpadded batches.')
            o, final_state, protected_keys, protected_values, protected_scores, protected_count = protected_recurrent_hyperbolic_delta_rule(
                q, k, v, beta, curvature=curvature, scale=1.0, initial_state=initial_state,
                output_final_state=use_cache, protected_keys=protected_keys,
                protected_values=protected_values, protected_count=protected_count,
                protected_scores=protected_scores, mode='triton' if q.is_cuda else 'naive',
                protection_rank=self.protection_rank, checkpoint_size=max(64, self.chunk_size),
            )
            recurrent_state = {
                'state': final_state,
                'protected_keys': protected_keys,
                'protected_values': protected_values,
                'protected_scores': protected_scores,
                'protected_count': protected_count,
            }
        else:
            o, final_state = op(
                q, k, v, beta, curvature=curvature, scale=1.0,
                initial_state=initial_state, output_final_state=use_cache, cu_seqlens=cu_seqlens, **extra,
            )
            recurrent_state = final_state
        if use_cache:
            update_layer_cache(
                self, past_key_values, recurrent_state=recurrent_state,
                conv_state=conv_states if self.use_short_conv else None, offset=q_len,
            )
        if o.numel():
            if self.use_gate:
                gate = rearrange(self.g_proj(hidden_states), 'b t (h d) -> b t h d', d=self.head_v_dim)
                o = self.o_norm(o, gate) if self.fuse_norm else self.o_norm(o) * gate.sigmoid()
            else:
                o = self.o_norm(o)
        o = self.o_proj(rearrange(o, 'b t h d -> b t (h d)'))
        return repad_hidden_states(o, indices, batch_size, q_len), None, past_key_values
