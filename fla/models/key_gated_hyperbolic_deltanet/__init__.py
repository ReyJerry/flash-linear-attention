# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.key_gated_hyperbolic_deltanet.configuration_key_gated_hyperbolic_deltanet import (
    KeyGatedHyperbolicDeltaNetConfig,
)
from fla.models.key_gated_hyperbolic_deltanet.modeling_key_gated_hyperbolic_deltanet import (
    KeyGatedHyperbolicDeltaNetBlock,
    KeyGatedHyperbolicDeltaNetForCausalLM,
    KeyGatedHyperbolicDeltaNetModel,
    KeyGatedHyperbolicDeltaNetPreTrainedModel,
)

AutoConfig.register(KeyGatedHyperbolicDeltaNetConfig.model_type, KeyGatedHyperbolicDeltaNetConfig, exist_ok=True)
AutoModel.register(KeyGatedHyperbolicDeltaNetConfig, KeyGatedHyperbolicDeltaNetModel, exist_ok=True)
AutoModelForCausalLM.register(KeyGatedHyperbolicDeltaNetConfig, KeyGatedHyperbolicDeltaNetForCausalLM, exist_ok=True)

__all__ = [
    'KeyGatedHyperbolicDeltaNetBlock',
    'KeyGatedHyperbolicDeltaNetConfig',
    'KeyGatedHyperbolicDeltaNetForCausalLM',
    'KeyGatedHyperbolicDeltaNetModel',
    'KeyGatedHyperbolicDeltaNetPreTrainedModel',
]
