# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# ruff: noqa: E402
"""CPU lifecycle and dispatch checks; spies do not claim mixed GPU arithmetic."""
import json

import pytest
import torch
import torch.nn.functional as F

# autopep8: off
pytest.importorskip('triton', reason='Actual FLA imports require the Triton runtime.')
pytest.importorskip('transformers')

import fla.modules.mlp as mlp
import fla.ops.key_gated_hyperbolic_delta_rule as ops
from fla.layers import KeyGatedHyperbolicDeltaNet
from fla.models import KeyGatedHyperbolicDeltaNetConfig, KeyGatedHyperbolicDeltaNetForCausalLM

# autopep8: on


def configuration(**kwargs):
    values = dict(hidden_size=64, head_dim=32, num_heads=2, num_hidden_layers=2,
                  intermediate_size=128, vocab_size=97, group_size=32, curvature=.6,
                  use_short_conv=False, fuse_norm=False, fuse_swiglu=False,
                  fuse_cross_entropy=False, attn_mode='chunk', use_cache=False)
    values.update(kwargs)
    return KeyGatedHyperbolicDeltaNetConfig(**values)


def test_default_legacy_and_new_configuration_roundtrip(tmp_path):
    default = configuration()
    assert default.precision_policy == 'strict'
    old = default.to_dict()
    old.pop('precision_policy')
    assert KeyGatedHyperbolicDeltaNetConfig.from_dict(old).precision_policy == 'strict'
    chosen = configuration(precision_policy='mixed_readout')
    chosen.save_pretrained(tmp_path)
    assert json.loads((tmp_path/'config.json').read_text())['precision_policy'] == 'mixed_readout'
    loaded = KeyGatedHyperbolicDeltaNetConfig.from_pretrained(tmp_path)
    model = KeyGatedHyperbolicDeltaNetForCausalLM(loaded)
    assert all(block.attn.precision_policy == 'mixed_readout' for block in model.model.layers)


@pytest.mark.parametrize('policy', ['unknown', True, None])
def test_invalid_policy_rejected_by_both_entries(policy):
    with pytest.raises(ValueError, match='precision_policy'):
        configuration(precision_policy=policy)
    with pytest.raises(ValueError, match='precision_policy'):
        KeyGatedHyperbolicDeltaNet(precision_policy=policy)


@pytest.mark.parametrize('mode', ['naive', 'fused_recurrent'])
def test_explicit_unsupported_mode_is_not_silently_treated_as_mixed(mode):
    with pytest.raises(ValueError, match='requires'):
        configuration(attn_mode=mode, precision_policy='mixed_readout')
    with pytest.raises(ValueError, match='requires'):
        KeyGatedHyperbolicDeltaNet(mode=mode, precision_policy='mixed_readout')


@pytest.mark.parametrize('policy', ['strict', 'mixed_readout'])
def test_real_model_layers_forward_explicit_policy_and_preserve_parameter_gradients(monkeypatch, policy):
    calls = []

    def observed(*args, **kwargs):
        calls.append(kwargs.copy())
        kwargs.pop('precision_policy', None)
        kwargs.pop('chunk_size')
        kwargs.pop('carry_mode')
        return ops.naive_recurrent_key_gated_hyperbolic_delta_rule(*args, **kwargs)

    monkeypatch.setattr(ops, 'chunk_key_gated_hyperbolic_delta_rule', observed)
    # The non-fused MLP still calls a CUDA-only activation; this test observes CPU policy wiring.
    monkeypatch.setattr(mlp, 'swiglu', lambda gate, value: F.silu(gate) * value)
    torch.manual_seed(12)
    model = KeyGatedHyperbolicDeltaNetForCausalLM(configuration(precision_policy=policy)).train()
    token = torch.randint(0, 97, (2, 5))
    result = model(input_ids=token, labels=token)
    result.loss.backward()
    assert len(calls) == 2
    for call in calls:
        assert call.get('precision_policy', 'strict') == policy
        if policy == 'strict':
            assert 'precision_policy' not in call
    assert all(param.grad is not None and torch.isfinite(param.grad).all() for param in model.parameters())


def test_mixed_prefill_configuration_uses_strict_recurrent_decode(monkeypatch):
    calls = []

    def observed(*args, **kwargs):
        calls.append(kwargs.copy())
        kwargs.pop('checkpoint_size')
        return ops.naive_recurrent_key_gated_hyperbolic_delta_rule(*args, **kwargs)

    monkeypatch.setattr(ops, 'fused_recurrent_key_gated_hyperbolic_delta_rule', observed)
    layer = KeyGatedHyperbolicDeltaNet(hidden_size=64, head_dim=32, num_heads=2, group_size=32,
                                       mode='chunk', precision_policy='mixed_readout',
                                       use_short_conv=False, fuse_norm=False, layer_idx=0).eval()
    with torch.no_grad():
        result = layer(torch.randn(1, 1, 64))[0]
    assert result.shape == (1, 1, 64) and len(calls) == 1
    assert 'precision_policy' not in calls[0]
