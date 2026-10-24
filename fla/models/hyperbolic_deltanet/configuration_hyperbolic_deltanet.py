# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers.configuration_utils import PretrainedConfig

from fla.models.configuration_utils import (
    AttentionConfig,
    AttentionConfigBase,
    validate_attnres_config,
    validate_loss_config,
)


class HyperbolicDeltaNetConfig(AttentionConfigBase, PretrainedConfig):
    model_type = 'hyperbolic_deltanet'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        attn_mode: str = 'chunk',
        tensor_core: bool = False,
        hidden_size: int = 2048,
        expand_v: float = 1.0,
        head_dim: int = 128,
        num_heads: int = 16,
        num_v_heads: int | None = None,
        curvature: float = 1.0,
        learnable_curvature: bool = False,
        protection_rank: int = 0,
        value_margin: float = 0.5,
        chunk_size: int = 16,
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
        **kwargs,
    ):
        self.attn_mode = attn_mode
        self.tensor_core = tensor_core
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads
        self.curvature = curvature
        self.learnable_curvature = learnable_curvature
        self.protection_rank = protection_rank
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
