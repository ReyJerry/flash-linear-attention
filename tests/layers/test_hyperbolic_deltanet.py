# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# ruff: noqa: E402
import copy

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip('CUDA and the full FLA runtime are required for layer integration.', allow_module_level=True)

from fla.layers import HyperbolicDeltaNet
from fla.models.cache_utils import Cache


@pytest.mark.parametrize('conv', [False, True])
@pytest.mark.parametrize('gva', [False, True])
def test_layer_backward_matches_naive(conv, gva):
    torch.manual_seed(51)
    layer = HyperbolicDeltaNet(
        hidden_size=32,
        head_dim=16,
        num_heads=2,
        num_v_heads=4 if gva else 2,
        mode='chunk',
        use_short_conv=conv,
        fuse_norm=False,
        layer_idx=0,
    ).cuda().train()
    oracle = copy.deepcopy(layer)
    oracle.mode = 'naive'
    x = torch.randn(2, 35, 32, device='cuda', requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    out = layer(x)[0]
    expected = oracle(xr)[0]
    torch.testing.assert_close(out, expected, atol=3e-4, rtol=3e-3)
    do = torch.randn_like(out)
    out.backward(do)
    expected.backward(do)
    torch.testing.assert_close(x.grad, xr.grad, atol=1e-3, rtol=1e-2)
    for (name, param), (_, reference) in zip(layer.named_parameters(), oracle.named_parameters()):
        assert param.grad is not None, name
        torch.testing.assert_close(param.grad, reference.grad, atol=2e-3, rtol=2e-2, msg=name)


@pytest.mark.parametrize('conv', [False, True])
@torch.no_grad()
def test_layer_padding_and_cache(conv):
    torch.manual_seed(51)
    layer = HyperbolicDeltaNet(
        hidden_size=32, head_dim=16, num_heads=2, use_short_conv=conv, layer_idx=0,
    ).cuda().eval()
    x = torch.randn(2, 19, 32, device='cuda')
    mask = torch.ones(2, 19, device='cuda', dtype=torch.long)
    mask[1, :6] = 0
    cache = Cache()
    actual, _, cache = layer(x, attention_mask=mask, past_key_values=cache, use_cache=True)
    expected = layer(x[1:2, 6:])[0]
    torch.testing.assert_close(actual[1:2, 6:], expected, atol=3e-4, rtol=3e-3)
    torch.testing.assert_close(actual[1, :6], torch.zeros_like(actual[1, :6]))
    state = cache[0]['recurrent_state'].clone()
    zeros = torch.zeros(2, 2, device='cuda', dtype=torch.long)
    empty_out, _, cache = layer(x[:, :2], attention_mask=zeros, past_key_values=cache, use_cache=True)
    torch.testing.assert_close(empty_out, torch.zeros_like(empty_out))
    torch.testing.assert_close(cache[0]['recurrent_state'], state)


def test_layer_learnable_curvature_and_protected_cache():
    torch.manual_seed(91)
    learned = HyperbolicDeltaNet(
        hidden_size=32, head_dim=16, num_heads=2, curvature=0.5,
        learnable_curvature=True, use_short_conv=False, fuse_norm=False,
        mode='chunk', layer_idx=0,
    ).cuda().train()
    x = torch.randn(1, 7, 32, device='cuda')
    learned(x)[0].square().mean().backward()
    assert learned.curvature_raw.grad is not None
    assert torch.isfinite(learned.curvature_raw.grad).all()
    assert learned.curvature_raw.grad.abs() > 0

    protected = HyperbolicDeltaNet(
        hidden_size=32, head_dim=16, num_heads=2, curvature=0.5,
        protection_rank=2, use_short_conv=False, fuse_norm=False,
        mode='fused_recurrent', layer_idx=0,
    ).cuda().eval()
    cache = Cache()
    with torch.no_grad():
        first, _, cache = protected(x, past_key_values=cache, use_cache=True)
        state = cache[0]['recurrent_state']
        assert state['protected_keys'].shape == (1, 2, 16, 2, 16)
        assert state['protected_values'].shape == (1, 2, 16, 2)
        assert state['protected_count'].shape == (1, 2, 16)
        second, _, cache = protected(x[:, :1], past_key_values=cache, use_cache=True)
        assert torch.isfinite(first).all() and torch.isfinite(second).all()
