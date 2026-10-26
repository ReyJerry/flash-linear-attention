# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Adapted from HDN at 46c54ef0d57609d66a5c158fa4a8cce124ca1483.

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.utils import get_layer_cache, repad_hidden_states, unpad_hidden_states, update_layer_cache
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.causal_conv1d.ops import causal_conv1d
from fla.ops.hyperbolic_delta_rule.naive import prepare_key_value

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.cache_utils import Cache


class KeyGatedHyperbolicDeltaNet(nn.Module):
    """Key-channel decay followed by grouped Lorentz projection and inclusive readout.

    The public state is canonical S, shaped [N,H,K,V], with curvature c/group_size
    per Value group. Optimized routes require their corresponding operator entry;
    only explicit ``mode='naive'`` selects the dense-solve reference.
    """

    def __setattr__(self, name, value):
        if name in ('A_log', 'dt_bias', 'curvature_raw') and isinstance(value, nn.Parameter):
            # assignment and inference-mode construction must retain FP32, version-tracked controls
            if value.dtype != torch.float32 or value.is_inference():
                attributes = value.__dict__.copy()
                grad = value.grad
                with torch.inference_mode(False), torch.no_grad():
                    value = nn.Parameter(value.detach().to(dtype=torch.float32).clone(), requires_grad=value.requires_grad)
                    if grad is not None:
                        value.grad = grad.to(dtype=torch.float32).clone()
                value.__dict__.update(attributes)
            if name in ('A_log', 'dt_bias'):
                value._no_weight_decay = True
        super().__setattr__(name, value)

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
        value_margin: float = 0.5,
        group_size: int = 32,
        chunk_size: int = 32,
        carry_mode: str = 'no_carry',
        decay_mode: str = 'key_channel',
        decay_rank: int | None = None,
        decay_function: str = 'negative_exp_softplus',
        readout: str = 'inclusive',
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
        if kwargs:
            raise ValueError(f'Unsupported KHDN parameters: {sorted(kwargs)}.')
        self.hidden_size, self.head_dim, self.num_heads = hidden_size, head_dim, num_heads
        self.num_v_heads = num_heads if num_v_heads is None else num_v_heads
        self.head_v_dim = int(head_dim * expand_v)
        if min(hidden_size, head_dim, num_heads, self.head_v_dim) <= 0 or head_dim > 256:
            raise ValueError('Dimensions must be positive and head_dim must be <=256.')
        if not math.isclose(head_dim * expand_v, self.head_v_dim):
            raise ValueError('head_dim * expand_v must be an integer.')
        if self.num_v_heads != num_heads:
            raise ValueError('KHDN requires identical q/k/v head counts; GVA is unsupported.')
        if (isinstance(group_size, bool) or not isinstance(group_size, int)
                or group_size not in (1, 16, 32) or self.head_v_dim % group_size):
            raise ValueError('group_size must be 1, 16 or 32 and divide head_v_dim.')
        if mode not in ('chunk', 'fused_recurrent', 'naive') or chunk_size not in (16, 32, 64):
            raise ValueError('Unsupported mode or chunk_size (expected 16, 32 or 64).')
        if carry_mode not in ('no_carry', 'y_carry'):
            raise ValueError('carry_mode must be no_carry or y_carry.')
        if decay_mode != 'key_channel' or decay_function != 'negative_exp_softplus' or readout != 'inclusive':
            raise ValueError('KHDN requires key_channel negative_exp_softplus decay and inclusive readout.')
        self.decay_rank = self.head_v_dim if decay_rank is None else decay_rank
        if isinstance(self.decay_rank, bool) or not isinstance(self.decay_rank, int) or self.decay_rank < 1:
            raise ValueError('decay_rank must be a positive integer.')
        if not math.isfinite(curvature) or curvature < 0 or not 0 < value_margin < 1:
            raise ValueError('Require finite curvature >= 0 and 0 < value_margin < 1.')
        if learnable_curvature and curvature <= 0:
            raise ValueError('Learnable curvature requires initial curvature > 0.')
        self.key_dim, self.value_dim = num_heads * head_dim, num_heads * self.head_v_dim
        self.mode, self.value_margin = mode, value_margin
        self.group_size, self.chunk_size, self.carry_mode = group_size, chunk_size, carry_mode
        self.decay_mode, self.decay_function, self.readout = decay_mode, decay_function, readout
        self.layer_idx = layer_idx
        self.use_gate, self.use_short_conv, self.fuse_norm = use_gate, use_short_conv, fuse_norm
        self.curvature = float(curvature)
        self.learnable_curvature = learnable_curvature
        if learnable_curvature:
            self.curvature_raw = nn.Parameter(torch.empty((), dtype=torch.float32))
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.f_proj = nn.Sequential(
            nn.Linear(hidden_size, self.decay_rank, bias=False),
            nn.Linear(self.decay_rank, self.key_dim, bias=False),
        )
        self.A_log = nn.Parameter(torch.empty(num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(self.key_dim, dtype=torch.float32))
        self._init_gate_parameters()
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

    @torch.no_grad()
    def _init_gate_parameters(self, respect_hf_initialized: bool = False):
        for name in ('A_log', 'dt_bias', 'curvature_raw'):
            param = getattr(self, name, None)
            if param is None:
                continue
            if name in ('A_log', 'dt_bias'):
                param._no_weight_decay = True
            if param.is_meta or (respect_hf_initialized and getattr(param, '_is_hf_initialized', False)):
                continue
            if name == 'A_log':
                param.uniform_(1, 16).log_()
            elif name == 'dt_bias':
                dt = torch.empty_like(param).uniform_(math.log(0.001), math.log(0.1)).exp_()
                param.copy_(dt + torch.log(-torch.expm1(-dt)))
            else:
                param.fill_(self.curvature + math.log(-math.expm1(-self.curvature)))
            if respect_hf_initialized:
                param._is_hf_initialized = True

    def _apply(self, fn, recurse=True):
        originals = {
            name: (param.data, None if param.grad is None else param.grad.data)
            for name in ('A_log', 'dt_bias', 'curvature_raw')
            if (param := getattr(self, name, None)) is not None
        }
        result = super()._apply(fn, recurse=recurse)
        # keep the decay exponent and bias in FP32 even under whole-model dtype conversion
        for name, (original, grad) in originals.items():
            param = getattr(self, name)
            with torch.inference_mode(False), torch.no_grad():
                param.data = param.data.float() if original.is_meta else original.to(device=param.device, dtype=torch.float32)
                if param.grad is not None:
                    param.grad.data = (param.grad.data.float() if grad.is_meta
                                       else grad.to(device=param.device, dtype=torch.float32))
            setattr(self, name, param)
        self._curvature_snapshot_key = None
        return result

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        for name in ('A_log', 'dt_bias', 'curvature_raw'):
            param = getattr(self, name, None)
            if param is not None and prefix + name in state_dict:
                param._is_hf_initialized = True
        self._curvature_snapshot_key = None

    @staticmethod
    def _prepare_cache_tensor(tensor):
        # inference history has no autograd graph; ordinary cached tensors retain theirs
        if tensor is not None and torch.is_grad_enabled() and tensor.is_inference():
            with torch.inference_mode(False):
                return tensor.clone()
        return tensor

    def _cache_metadata(self, curvature):
        if self.learnable_curvature:
            key = (id(self.curvature_raw), self.curvature_raw._version, self.curvature_raw.device)
            if getattr(self, '_curvature_snapshot_key', None) != key:
                self._curvature_snapshot_value = float(curvature.detach().item())
                self._curvature_snapshot_key = key
            c_value = self._curvature_snapshot_value
        else:
            c_value = self.curvature
        if not math.isfinite(c_value) or c_value < 0:
            raise ValueError('Cannot cache a nonfinite or negative curvature.')
        return dict(
            model_type='key_gated_hyperbolic_deltanet',
            state_format='canonical_S_v1',
            group_size=self.group_size,
            curvature=c_value,
            decay_mode=self.decay_mode,
            decay_rank=self.decay_rank,
            decay_function=self.decay_function,
            readout=self.readout,
        )

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
        curvature = F.softplus(self.curvature_raw) if self.learnable_curvature else self.curvature
        metadata = self._cache_metadata(curvature) if use_cache or last_state is not None else None
        initial_state = None
        conv_states = (None, None, None)
        if last_state is not None:
            recurrent = last_state.get('recurrent_state')
            if not isinstance(recurrent, dict) or recurrent.get('metadata') != metadata:
                raise ValueError('Incompatible or stale KHDN cache: identity, group, curvature or gate semantics changed.')
            initial_state = recurrent.get('state')
            if not isinstance(initial_state, torch.Tensor):
                raise ValueError('KHDN cache must contain the canonical S tensor.')
            initial_state = self._prepare_cache_tensor(initial_state)
            if last_state.get('conv_state') is not None:
                conv_states = tuple(self._prepare_cache_tensor(state) for state in last_state['conv_state'])
        hidden_states, indices, cu_seqlens = unpad_hidden_states(
            hidden_states, kwargs.get('cu_seqlens'), attention_mask if q_len else None, q_len,
        )
        n_sequences = batch_size if cu_seqlens is None else cu_seqlens.numel() - 1
        if initial_state is not None and initial_state.shape != (n_sequences, self.num_heads, self.head_dim, self.head_v_dim):
            raise ValueError('KHDN cache state shape does not match the sequence/head/Key/Value dimensions.')
        projections = (self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states))
        if self.use_short_conv and hidden_states.shape[1] > 0:
            convolved = [
                causal_conv1d(
                    x=x,
                    weight=conv.weight[:, 0, :],
                    bias=conv.bias,
                    initial_state=cache,
                    output_final_state=use_cache,
                    activation=conv.activation,
                    backend=conv.backend,
                    cu_seqlens=cu_seqlens,
                )
                for x, conv, cache in zip(projections, (self.q_conv1d, self.k_conv1d, self.v_conv1d), conv_states)
            ]
            (q, k, raw_v), conv_states = tuple(zip(*convolved))
        else:
            q, k, raw_v = (F.silu(x) for x in projections)
        q, k = (rearrange(x, 'b t (h d) -> b t h d', d=self.head_dim) for x in (q, k))
        raw_v = rearrange(raw_v, 'b t (h d) -> b t h d', d=self.head_v_dim)
        q = F.normalize(q.float(), dim=-1, eps=1e-6).to(hidden_states.dtype)
        k, v = prepare_key_value(k, raw_v, curvature=curvature, value_margin=self.value_margin)
        beta = self.b_proj(hidden_states).float().sigmoid()
        raw_decay = rearrange(self.f_proj(hidden_states).float() + self.dt_bias, 'b t (h d) -> b t h d', h=self.num_heads)
        log_decay = -self.A_log.float().exp()[None, None, :, None] * F.softplus(raw_decay)
        mode = self.mode
        if mode != 'naive' and not self.training and not torch.is_grad_enabled() and q_len <= 64:
            mode = 'fused_recurrent'
        extra = {}
        if mode == 'naive':
            from fla.ops.key_gated_hyperbolic_delta_rule import naive_recurrent_key_gated_hyperbolic_delta_rule as op
        elif mode == 'chunk':
            from fla.ops.key_gated_hyperbolic_delta_rule import chunk_key_gated_hyperbolic_delta_rule as op
            extra = dict(chunk_size=self.chunk_size, carry_mode=self.carry_mode)
        else:
            from fla.ops.key_gated_hyperbolic_delta_rule import fused_recurrent_key_gated_hyperbolic_delta_rule as op
            extra = dict(checkpoint_size=self.chunk_size)
        o, final_state = op(
            q=q,
            k=k,
            v=v,
            beta=beta,
            log_decay=log_decay,
            curvature=curvature,
            initial_state=initial_state,
            output_final_state=use_cache,
            scale=1.0,
            cu_seqlens=cu_seqlens,
            group_size=self.group_size,
            check_inputs=mode == 'naive',
            **extra,
        )
        if use_cache:
            update_layer_cache(
                self,
                past_key_values,
                recurrent_state=dict(state=final_state, metadata=metadata),
                conv_state=conv_states if self.use_short_conv else None,
                offset=q_len,
            )
        if o.numel():
            if self.use_gate:
                gate = rearrange(self.g_proj(hidden_states), 'b t (h d) -> b t h d', d=self.head_v_dim)
                o = self.o_norm(o, gate) if self.fuse_norm else self.o_norm(o) * gate.sigmoid()
            else:
                o = self.o_norm(o)
        o = self.o_proj(rearrange(o, 'b t h d -> b t (h d)'))
        return repad_hidden_states(o, indices, batch_size, q_len), None, past_key_values
