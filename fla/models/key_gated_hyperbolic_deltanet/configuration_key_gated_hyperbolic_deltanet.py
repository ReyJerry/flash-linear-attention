# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Adapted from HDN at 46c54ef0d57609d66a5c158fa4a8cce124ca1483.

import math

from transformers.configuration_utils import PretrainedConfig

from fla.models.configuration_utils import (
    AttentionConfig,
    AttentionConfigBase,
    validate_attnres_config,
    validate_loss_config,
)


class KeyGatedHyperbolicDeltaNetConfig(AttentionConfigBase, PretrainedConfig):
    model_type = 'key_gated_hyperbolic_deltanet'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        attn_mode: str = 'chunk',
        hidden_size: int = 2048,
        expand_v: float = 1.0,
        head_dim: int = 128,
        num_heads: int = 16,
        num_v_heads: int | None = None,
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
        max_position_embeddings: int = 8192,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        hidden_act: str = 'swish',
        num_hidden_layers: int = 20,
        norm_eps: float = 1e-6,
        attn: AttentionConfig = None,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        initializer_range: float = 0.02,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        use_l2warp: bool = False,
        vocab_size: int = 32000,
        attnres_block_size: int | None = None,
        precision_policy: str = 'strict',
        **kwargs,
    ):
        identity = kwargs.pop('model_type', self.model_type)
        if identity != self.model_type:
            raise ValueError(f'Incompatible model_type {identity!r}; expected {self.model_type!r}.')
        unsupported = {'protection_rank', 'protected_checkpoint', 'tensor_core', 'safe_gate', 'lower_bound'} & kwargs.keys()
        if unsupported:
            raise ValueError(f'Unsupported KHDN parameters: {sorted(unsupported)}.')
        head_v_dim = int(head_dim * expand_v)
        if min(hidden_size, head_dim, num_heads, head_v_dim) <= 0 or head_dim > 256:
            raise ValueError('Dimensions must be positive and head_dim must be <=256.')
        if not math.isclose(head_dim * expand_v, head_v_dim):
            raise ValueError('head_dim * expand_v must be an integer.')
        if num_v_heads is not None and num_v_heads != num_heads:
            raise ValueError('KHDN requires identical q/k/v head counts; GVA is unsupported.')
        if (isinstance(group_size, bool) or not isinstance(group_size, int)
                or group_size not in (1, 16, 32) or head_v_dim % group_size):
            raise ValueError('group_size must be 1, 16 or 32 and divide head_v_dim.')
        if chunk_size not in (16, 32, 64) or attn_mode not in ('chunk', 'fused_recurrent', 'naive'):
            raise ValueError('Unsupported attn_mode or chunk_size (expected 16, 32 or 64).')
        if precision_policy not in ('strict', 'mixed_readout'):
            raise ValueError('precision_policy must be strict or mixed_readout.')
        if precision_policy != 'strict' and attn_mode != 'chunk':
            raise ValueError('mixed_readout precision requires attn_mode=chunk.')
        self.precision_policy = precision_policy
        if carry_mode not in ('no_carry', 'y_carry'):
            raise ValueError('carry_mode must be no_carry or y_carry.')
        if decay_mode != 'key_channel' or decay_function != 'negative_exp_softplus' or readout != 'inclusive':
            raise ValueError('KHDN requires key_channel negative_exp_softplus decay and inclusive readout.')
        decay_rank = head_v_dim if decay_rank is None else decay_rank
        if isinstance(decay_rank, bool) or not isinstance(decay_rank, int) or decay_rank < 1:
            raise ValueError('decay_rank must be a positive integer.')
        if not math.isfinite(curvature) or curvature < 0 or not 0 < value_margin < 1:
            raise ValueError('Require finite curvature >= 0 and 0 < value_margin < 1.')
        if learnable_curvature and curvature <= 0:
            raise ValueError('Learnable curvature requires initial curvature > 0.')
        self.group_size = group_size
        self.carry_mode = carry_mode
        self.decay_mode = decay_mode
        self.decay_rank = decay_rank
        self.decay_function = decay_function
        self.readout = readout
        self.attn_mode = attn_mode
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads
        self.curvature = curvature
        self.learnable_curvature = learnable_curvature
        self.value_margin = value_margin
        self.chunk_size = chunk_size
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.max_position_embeddings = max_position_embeddings
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.norm_eps = norm_eps
        self.attn = attn
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size
        self.attnres_block_size = attnres_block_size
        validate_loss_config(fuse_cross_entropy=fuse_cross_entropy, fuse_linear_cross_entropy=fuse_linear_cross_entropy)
        validate_attnres_config(attnres_block_size=attnres_block_size)
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
