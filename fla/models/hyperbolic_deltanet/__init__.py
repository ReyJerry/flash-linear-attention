# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_hyperbolic_deltanet import HyperbolicDeltaNetConfig
from .modeling_hyperbolic_deltanet import HyperbolicDeltaNetForCausalLM, HyperbolicDeltaNetModel

AutoConfig.register(HyperbolicDeltaNetConfig.model_type, HyperbolicDeltaNetConfig, exist_ok=True)
AutoModel.register(HyperbolicDeltaNetConfig, HyperbolicDeltaNetModel, exist_ok=True)
AutoModelForCausalLM.register(HyperbolicDeltaNetConfig, HyperbolicDeltaNetForCausalLM, exist_ok=True)

__all__ = ['HyperbolicDeltaNetConfig', 'HyperbolicDeltaNetForCausalLM', 'HyperbolicDeltaNetModel']
