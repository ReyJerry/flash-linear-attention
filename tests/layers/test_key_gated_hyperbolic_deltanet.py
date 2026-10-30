# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# ruff: noqa: E402
import copy
import io

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip('triton', reason='Actual FLA import requires the Triton runtime, including on CPU.')
pytest.importorskip('transformers')

from fla.layers import KeyGatedHyperbolicDeltaNet
from fla.models.cache_utils import Cache
from fla.ops.hyperbolic_delta_rule.naive import prepare_key_value


def small_layer(**kwargs):
    options = dict(
        hidden_size=64, head_dim=32, num_heads=2, group_size=32, mode='naive',
        curvature=0.6, use_short_conv=False, fuse_norm=False, layer_idx=0,
    )
    options.update(kwargs)
    return KeyGatedHyperbolicDeltaNet(**options)


@pytest.mark.parametrize('kwargs', [
    dict(protection_rank=0), dict(protected_checkpoint=False), dict(tensor_core=False),
    dict(safe_gate=True), dict(lower_bound=-5), dict(num_v_heads=4), dict(group_size=64),
    dict(head_dim=16), dict(group_size=32.0), dict(chunk_size=8), dict(decay_rank=0), dict(decay_mode='scalar'),
    dict(decay_function='sigmoid'), dict(readout='exclusive'),
])
def test_layer_rejects_unsupported_contract(kwargs):
    with pytest.raises(ValueError):
        small_layer(**kwargs)


def test_raw_gate_preprocessing_and_external_gradients():
    from fla.ops.key_gated_hyperbolic_delta_rule import naive_recurrent_key_gated_hyperbolic_delta_rule

    torch.manual_seed(51)
    layer = small_layer(learnable_curvature=True, curvature=1.7).train()
    reference = copy.deepcopy(layer)
    x = torch.randn(2, 5, 64, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    actual = layer(x)[0]
    shape = (2, 5, 2, 32)
    q = F.normalize(F.silu(reference.q_proj(xr)).reshape(shape).float(), dim=-1, eps=1e-6)
    raw_k = F.silu(reference.k_proj(xr)).reshape(shape)
    raw_v = F.silu(reference.v_proj(xr)).reshape(shape)
    curvature = F.softplus(reference.curvature_raw)
    k, v = prepare_key_value(raw_k, raw_v, curvature=curvature, value_margin=reference.value_margin)
    raw_decay = reference.f_proj(xr).float().reshape(shape)
    ell = -reference.A_log.exp()[None, None, :, None] * F.softplus(raw_decay + reference.dt_bias.reshape(2, 32))
    beta = reference.b_proj(xr).float().sigmoid()
    raw_decay.retain_grad()
    raw_k.retain_grad()
    raw_v.retain_grad()
    out, _ = naive_recurrent_key_gated_hyperbolic_delta_rule(
        q=q, k=k, v=v, beta=beta, log_decay=ell, curvature=curvature, group_size=32, scale=1.0,
    )
    expected = reference.o_proj((reference.o_norm(out) * reference.g_proj(xr).reshape(shape).sigmoid()).flatten(2))
    torch.testing.assert_close(actual, expected)
    cotangent = torch.randn_like(actual)
    actual.backward(cotangent)
    expected.backward(cotangent)
    torch.testing.assert_close(x.grad, xr.grad)
    for name, param in layer.named_parameters():
        ref = reference.get_parameter(name)
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum() > 0, name
        torch.testing.assert_close(param.grad, ref.grad, msg=name)
    for value in (raw_decay, raw_k, raw_v):
        assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0
    assert ell.shape == q.shape and beta.shape == q.shape[:3]
    assert not torch.equal(ell[..., 0], ell[..., 1])


def test_preprocessing_zero_key_bound_and_total_curvature_chain():
    layer = small_layer(curvature=2.0, learnable_curvature=True)
    k = torch.zeros(1, 2, 2, 32, requires_grad=True)
    raw_v = torch.randn_like(k, requires_grad=True)
    c = F.softplus(layer.curvature_raw)
    key, value = prepare_key_value(k, raw_v, curvature=c, value_margin=layer.value_margin)
    torch.testing.assert_close(key[..., 0], torch.ones_like(key[..., 0]))
    assert torch.count_nonzero(key[..., 1:]) == 0
    assert ((c / 32) * value.square().sum(-1) < key.square().sum(-1)).all()
    value.square().sum().backward()
    assert layer.curvature_raw.grad is not None and layer.curvature_raw.grad.abs() > 0
    assert raw_v.grad is not None and torch.isfinite(raw_v.grad).all()


@torch.no_grad()
def test_cache_prefill_decode_restore_reorder_beam_and_empty():
    torch.manual_seed(51)
    layer = small_layer().eval()
    x = torch.randn(2, 9, 64)
    full = layer(x)[0]
    cache = Cache()
    first = layer(x[:, :5], past_key_values=cache, use_cache=True)[0]
    stream = io.BytesIO()
    torch.save(cache.to_legacy_cache(), stream)
    stream.seek(0)
    cache = Cache.from_legacy_cache(torch.load(stream, weights_only=True), seen_tokens=5)
    outputs = [first]
    for t in range(5, 9):
        outputs.append(layer(x[:, t:t + 1], past_key_values=cache, use_cache=True)[0])
    torch.testing.assert_close(torch.cat(outputs, 1), full, atol=3e-5, rtol=3e-4)
    assert cache.get_seq_length() == 9
    state = cache[0]['recurrent_state']['state'].clone()
    metadata = cache[0]['recurrent_state']['metadata'].copy()
    assert all(isinstance(value, (str, int, float, bool)) for value in metadata.values())
    assert set(cache[0]['recurrent_state']) == {'state', 'metadata'}
    empty = layer(x[:, :0], past_key_values=cache, use_cache=True)[0]
    assert empty.shape == (2, 0, 64) and cache.get_seq_length() == 9
    torch.testing.assert_close(cache[0]['recurrent_state']['state'], state)
    cache.reorder_cache(torch.tensor([1, 0, 1]))
    torch.testing.assert_close(cache[0]['recurrent_state']['state'], state[[1, 0, 1]])
    cache.batch_repeat_interleave(2)
    assert cache[0]['recurrent_state']['state'].shape[0] == 6
    cache.batch_select_indices(torch.tensor([5, 0]))
    assert cache[0]['recurrent_state']['metadata'] == metadata
    assert torch.isfinite(layer(x[:, :1], past_key_values=cache, use_cache=True)[0]).all()


def test_cross_call_gradient_and_future_causality():
    torch.manual_seed(41)
    layer = small_layer().train()
    other = copy.deepcopy(layer)
    x = torch.randn(1, 7, 64, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    full = layer(x)[0]
    cache = Cache()
    first = other(xr[:, :3], past_key_values=cache, use_cache=True)[0]
    second = other(xr[:, 3:], past_key_values=cache, use_cache=True)[0]
    joined = torch.cat((first, second), 1)
    torch.testing.assert_close(joined, full, atol=3e-5, rtol=3e-4)
    do = torch.randn_like(full)
    full.backward(do)
    joined.backward(do)
    torch.testing.assert_close(x.grad, xr.grad, atol=3e-5, rtol=3e-4)
    for name, param in layer.named_parameters():
        torch.testing.assert_close(param.grad, other.get_parameter(name).grad, atol=3e-5, rtol=3e-4, msg=name)
    x = torch.randn(1, 7, 64, requires_grad=True)
    out = layer(x)[0]
    grad, = torch.autograd.grad(out[:, :3].square().sum(), x)
    assert torch.count_nonzero(grad[:, 3:]) == 0
    perturbed = x.detach().clone()
    perturbed[:, 3:] += 10
    torch.testing.assert_close(out[:, :3], layer(perturbed)[0][:, :3])


@pytest.mark.parametrize('learned', [False, True])
@torch.no_grad()
def test_reject_legacy_stale_or_different_group_cache(learned):
    layer = small_layer(learnable_curvature=learned)
    x = torch.randn(1, 2, 64)
    legacy = Cache()
    legacy.update(recurrent_state=torch.zeros(1, 2, 32, 32), layer_idx=0, offset=0)
    with pytest.raises(ValueError, match='cache'):
        layer(x, past_key_values=legacy, use_cache=True)
    cache = Cache()
    layer(x, past_key_values=cache, use_cache=True)
    other = small_layer(group_size=16, learnable_curvature=learned)
    with pytest.raises(ValueError, match='cache'):
        other(x, past_key_values=cache, use_cache=True)
    if learned:
        layer.curvature_raw.add_(0.1)
    else:
        layer.curvature += 0.1
    with pytest.raises(ValueError, match='stale'):
        layer(x[:, :1], past_key_values=cache, use_cache=True)


@pytest.mark.parametrize('group_size', [16, 32])
def test_main_dimensions_and_empty_initial_cache(group_size):
    layer = small_layer(head_dim=128, group_size=group_size)
    cache = Cache()
    output = layer(torch.empty(2, 0, 64), past_key_values=cache, use_cache=True)[0]
    assert output.shape == (2, 0, 64)
    assert cache[0]['recurrent_state']['state'].shape == (2, 2, 128, 128)
    assert torch.count_nonzero(cache[0]['recurrent_state']['state']) == 0
    assert torch.isfinite(layer(torch.randn(2, 2, 64), past_key_values=cache, use_cache=True)[0]).all()


@pytest.mark.parametrize('conv', [False, True])
@torch.no_grad()
def test_padding_packed_empty_sequences_and_convolution(conv):
    if conv and not torch.cuda.is_available():
        pytest.skip('Actual short-convolution kernels require CUDA.')
    device = 'cuda' if conv else 'cpu'
    layer = small_layer(use_short_conv=conv).to(device).eval()
    x = torch.randn(2, 9, 64, device=device)
    mask = torch.ones(2, 9, dtype=torch.long, device=device)
    mask[0, :3] = 0
    mask[1] = 0
    cache = Cache()
    actual = layer(x, attention_mask=mask, past_key_values=cache, use_cache=True)[0]
    expected = layer(x[:1, 3:])[0]
    torch.testing.assert_close(actual[:1, 3:], expected, atol=3e-4, rtol=3e-3)
    assert torch.count_nonzero(actual[1]) == 0 and torch.count_nonzero(actual[0, :3]) == 0
    state = cache[0]['recurrent_state']['state'].clone()
    masked = layer(x[:, :1], attention_mask=mask[:, :1] * 0, past_key_values=cache, use_cache=True)[0]
    assert torch.count_nonzero(masked) == 0
    torch.testing.assert_close(state, cache[0]['recurrent_state']['state'])
    cu = torch.tensor([0, 3, 3, 9], dtype=torch.int64, device=device)
    packed_cache = Cache()
    packed = layer(x[:1], cu_seqlens=cu, past_key_values=packed_cache, use_cache=True)[0]
    separate = torch.cat((layer(x[:1, :3])[0], layer(x[:1, 3:])[0]), 1)
    torch.testing.assert_close(packed, separate, atol=3e-4, rtol=3e-3)
    assert torch.count_nonzero(packed_cache[0]['recurrent_state']['state'][1]) == 0
    full = layer(x)[0]
    cache = Cache()
    split = [layer(x[:, :5], past_key_values=cache, use_cache=True)[0]]
    split.extend(layer(x[:, t:t + 1], past_key_values=cache, use_cache=True)[0] for t in range(5, 9))
    torch.testing.assert_close(torch.cat(split, 1), full, atol=3e-4, rtol=3e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Optimized production routes require CUDA.')
@pytest.mark.parametrize('mode', ['chunk', 'fused_recurrent'])
@pytest.mark.parametrize('chunk_size', [16, 32, 64])
def test_real_optimized_layer_output_and_all_gradients(mode, chunk_size):
    import fla.ops.key_gated_hyperbolic_delta_rule as ops

    name = mode + '_key_gated_hyperbolic_delta_rule'
    if not hasattr(ops, name):
        pytest.skip(f'Production entry {name} is not implemented yet.')
    torch.manual_seed(51)
    layer = small_layer(mode=mode, chunk_size=chunk_size, learnable_curvature=True).cuda().train()
    oracle = copy.deepcopy(layer)
    oracle.mode = 'naive'
    x = torch.randn(1, 35, 64, device='cuda', requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    actual, expected = layer(x)[0], oracle(xr)[0]
    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-3)
    do = torch.randn_like(actual)
    actual.backward(do)
    expected.backward(do)
    torch.testing.assert_close(x.grad, xr.grad, atol=1e-3, rtol=1e-2)
    for name, param in layer.named_parameters():
        assert param.grad is not None, name
        torch.testing.assert_close(param.grad, oracle.get_parameter(name).grad, atol=2e-3, rtol=2e-2, msg=name)
        reference = oracle.get_parameter(name).grad
        assert torch.isfinite(param.grad).all() and torch.isfinite(reference).all(), name
        assert param.grad.count_nonzero() > 0 and reference.count_nonzero() > 0, name
        if name.startswith('f_proj.') or name in ('A_log', 'dt_bias', 'curvature_raw'):
            relative_error = (param.grad.double() - reference.double()).norm() / reference.double().norm()
            assert relative_error <= 2e-2, name


def test_inference_constructed_layer_tracks_curvature_after_conversion():
    with torch.inference_mode():
        layer = small_layer(learnable_curvature=True).eval()
        x = torch.randn(1, 3, 64)
        assert torch.isfinite(layer(x, past_key_values=Cache(), use_cache=True)[0]).all()
        layer.bfloat16()
        cache = Cache()
        x = x.bfloat16()
        assert torch.isfinite(layer(x, past_key_values=cache, use_cache=True)[0]).all()
        for name in ('A_log', 'dt_bias', 'curvature_raw'):
            param = getattr(layer, name)
            assert param.dtype == torch.float32 and not param.is_inference()
            assert isinstance(param._version, int)
        layer.curvature_raw.add_(0.1)
        with pytest.raises(ValueError, match='stale'):
            layer(x[:, :1], past_key_values=cache, use_cache=True)


@pytest.mark.parametrize('shape', [(2, 2, 32, 32), (2, 64, 4)])
def test_cache_tensor_adoption_clones_only_inference_history(shape):
    source = torch.randn(shape, requires_grad=True)
    regular = source * 2
    assert KeyGatedHyperbolicDeltaNet._prepare_cache_tensor(regular) is regular
    KeyGatedHyperbolicDeltaNet._prepare_cache_tensor(regular).sum().backward()
    torch.testing.assert_close(source.grad, torch.full_like(source, 2))
    with torch.inference_mode():
        cached = torch.randn(shape)
        assert KeyGatedHyperbolicDeltaNet._prepare_cache_tensor(cached) is cached
    with torch.no_grad():
        assert KeyGatedHyperbolicDeltaNet._prepare_cache_tensor(cached) is cached
    adopted = KeyGatedHyperbolicDeltaNet._prepare_cache_tensor(cached)
    assert not adopted.is_inference() and not adopted.requires_grad
    assert adopted.data_ptr() != cached.data_ptr()
    torch.testing.assert_close(adopted, cached, atol=0, rtol=0)
    (adopted * source).sum().backward()
    torch.testing.assert_close(source.grad, cached + 2)


@pytest.mark.parametrize('conv', [False, True])
@pytest.mark.parametrize('prefill_mode', ['inference', 'no_grad', 'grad'])
def test_cache_reentry_preserves_available_history(prefill_mode, conv):
    if conv and not torch.cuda.is_available():
        pytest.skip('Actual short-convolution reentry requires CUDA.')
    device = 'cuda' if conv else 'cpu'
    torch.manual_seed(71)
    layer = small_layer(learnable_curvature=True, use_short_conv=conv).to(device).train()
    reference = copy.deepcopy(layer)
    prefix = torch.randn(1, 4, 64, device=device, requires_grad=True)
    suffix = torch.randn(1, 3, 64, device=device, requires_grad=True)
    prefix_ref = prefix.detach().clone().requires_grad_()
    suffix_ref = suffix.detach().clone().requires_grad_()
    cache, reference_cache = Cache(), Cache()
    context = {'inference': torch.inference_mode, 'no_grad': torch.no_grad, 'grad': torch.enable_grad}[prefill_mode]
    with context():
        layer(prefix, past_key_values=cache, use_cache=True)
    with (torch.enable_grad() if prefill_mode == 'grad' else torch.no_grad()):
        reference(prefix_ref, past_key_values=reference_cache, use_cache=True)
    state = cache[0]['recurrent_state']['state']
    assert state.is_inference() == (prefill_mode == 'inference')
    if conv:
        assert all(t.is_inference() == (prefill_mode == 'inference') for t in cache[0]['conv_state'])
    actual = layer(suffix, past_key_values=cache, use_cache=True)[0]
    expected = reference(suffix_ref, past_key_values=reference_cache, use_cache=True)[0]
    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-3)
    cotangent = torch.randn_like(actual)
    actual.backward(cotangent)
    expected.backward(cotangent)
    torch.testing.assert_close(suffix.grad, suffix_ref.grad, atol=1e-3, rtol=1e-2)
    for name, param in layer.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
        torch.testing.assert_close(param.grad, reference.get_parameter(name).grad, atol=2e-3, rtol=2e-2, msg=name)
    if prefill_mode == 'grad':
        assert prefix.grad is not None and prefix.grad.count_nonzero() > 0
        torch.testing.assert_close(prefix.grad, prefix_ref.grad, atol=1e-3, rtol=1e-2)
    else:
        assert prefix.grad is None and prefix_ref.grad is None
    assert not cache[0]['recurrent_state']['state'].is_inference()
    if conv:
        assert all(not t.is_inference() for t in cache[0]['conv_state'])
