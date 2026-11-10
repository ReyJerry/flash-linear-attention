# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""CPU coverage of the untimed model-reference dtype and autograd boundary."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from benchmarks import hyperbolic_training_distributed as training


@pytest.fixture
def reference():
    path = Path(__file__).resolve().parents[2] / 'fla/ops/key_gated_hyperbolic_delta_rule/naive.py'
    spec = importlib.util.spec_from_file_location('khdn_model_oracle_cpu', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.naive_recurrent_key_gated_hyperbolic_delta_rule


def sources(dtype, packed):
    generator = torch.Generator().manual_seed(73)
    shape = (1, 6, 1, 3) if packed else (2, 3, 1, 3)
    q = torch.randn(shape, generator=generator)
    k = torch.nn.functional.normalize(torch.randn(shape, generator=generator), dim=-1)
    v = .02 * torch.randn(*shape[:3], 2, generator=generator)
    beta = .3 + .2 * torch.rand(shape[:3], generator=generator)
    ell = -.2 - .1 * torch.rand(shape, generator=generator)
    state = .003 * torch.randn(3 if packed else 2, 1, 3, 2, generator=generator)
    curvature = torch.tensor(.6)
    kinds = [torch.float32] * 7
    if dtype == 'bf16':
        kinds[:3] = [torch.bfloat16] * 3
    elif dtype == 'fp64':
        kinds = [torch.float64] * 7
    elif dtype == 'mixed_state':
        kinds[5] = torch.float64
    elif dtype == 'mixed_curvature':
        kinds[6] = torch.float64
    values = tuple(value.to(kind).requires_grad_() for value, kind in zip((q, k, v, beta, ell, state, curvature), kinds))
    bounds = torch.tensor([0, 2, 2, 6], dtype=torch.int32) if packed else None
    return values, bounds


def call(entry, values, bounds, final=True):
    q, k, v, beta, ell, state, curvature = values
    return entry(q=q, k=k, v=v, beta=beta, log_decay=ell, initial_state=state, curvature=curvature,
                 output_final_state=final, cu_seqlens=bounds, group_size=2, scale=.7)


def loss(outputs):
    output, state = outputs
    weights = torch.linspace(.1, .8, output.numel(), dtype=torch.float64).reshape(output.shape)
    result = (output.double() * weights).sum()
    return result if state is None else result + .27 * state.double().sum()


@pytest.mark.parametrize('dtype', ['fp32', 'bf16', 'fp64', 'mixed_state', 'mixed_curvature'])
@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('final', [False, True])
def test_model_oracle_dtypes_sources_and_checkpoint(reference, dtype, packed, final):
    values, bounds = sources(dtype, packed)
    expected_state_dtype = torch.float64 if dtype in ('fp64', 'mixed_state', 'mixed_curvature') else torch.float32

    def adapted(*values):
        return call(lambda **kwargs: training.khdn_model_oracle_call(reference, **kwargs), values, bounds, final)

    with torch.autocast('cpu', dtype=torch.bfloat16):
        direct = call(reference, tuple(value.double() for value in values), bounds, final)
        expected = direct[0].to(values[0].dtype), None if direct[1] is None else direct[1].to(expected_state_dtype)
        actual = adapted(*values)
        replay = checkpoint(adapted, *values, use_reentrant=False)
    assert actual[0].dtype == values[0].dtype
    assert (actual[1] is None) == (not final)
    if final:
        assert actual[1].dtype == expected_state_dtype
    if packed:
        assert bounds.dtype == torch.int32 and bounds.tolist() == [0, 2, 2, 6]
    gradients = [torch.autograd.grad(loss(result), values, retain_graph=True) for result in (expected, actual, replay)]
    for result in (actual, replay):
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
    for baseline, candidate, replayed, value in zip(*gradients, values):
        assert candidate.dtype == value.dtype and torch.isfinite(candidate).all() and candidate.abs().max() > 0
        torch.testing.assert_close(candidate, baseline, atol=0, rtol=0)
        torch.testing.assert_close(replayed, baseline, atol=0, rtol=0)


def test_model_oracle_fp64_gradcheck(reference):
    values, bounds = sources('fp64', False)
    assert torch.autograd.gradcheck(
        lambda *values: call(lambda **kwargs: training.khdn_model_oracle_call(reference, **kwargs), values, bounds),
        values, fast_mode=True,
    )


@pytest.mark.parametrize('dtype', ['fp32', 'bf16', 'fp64'])
@pytest.mark.parametrize('final', [False, True])
def test_model_oracle_default_state_and_python_curvature(reference, dtype, final):
    values, _ = sources(dtype, False)
    q, k, v, beta, ell = values[:5]
    output, state = training.khdn_model_oracle_call(
        reference, q, k, v, beta, ell, curvature=.6, initial_state=None, group_size=2, output_final_state=final,
    )
    assert output.dtype == q.dtype and torch.isfinite(output).all()
    assert (state is None) == (not final)
    if final:
        assert state.dtype == (torch.float64 if dtype == 'fp64' else torch.float32) and torch.isfinite(state).all()
    gradients = torch.autograd.grad(loss((output, state)), values[:5])
    assert all(torch.isfinite(value).all() and value.abs().max() > 0 for value in gradients)


@pytest.mark.parametrize('fail', [False, True])
def test_model_oracle_scope_restores_after_checkpoint_and_exception(reference, monkeypatch, fail):
    package = ModuleType('fla.ops.key_gated_hyperbolic_delta_rule')
    calls = []

    def original(**kwargs):
        calls.append(tuple(value.dtype for value in kwargs.values() if isinstance(
            value, torch.Tensor) and value.is_floating_point()))
        return reference(**kwargs)

    package.naive_recurrent_key_gated_hyperbolic_delta_rule = original
    monkeypatch.setitem(sys.modules, package.__name__, package)
    values, bounds = sources('fp32', True)
    expected = pytest.raises(RuntimeError, match='scope probe') if fail else training.contextlib.nullcontext()
    with expected, training.khdn_model_oracle_precision() as evidence:
        assert package.naive_recurrent_key_gated_hyperbolic_delta_rule is not original
        result = checkpoint(lambda *values: call(package.naive_recurrent_key_gated_hyperbolic_delta_rule, values, bounds),
                            *values, use_reentrant=False)
        torch.autograd.grad(loss(result), values)
        assert len(calls) >= 2 and all(kind == torch.float64 for row in calls for kind in row)
        assert evidence['calls'] and len(evidence['reference_sha256']) == 64
        assert evidence['calls'][0]['source_dtypes']['curvature'] == 'torch.float32'
        assert evidence['calls'][0]['output_dtype'] == evidence['calls'][0]['state_dtype'] == 'torch.float32'
        if fail:
            raise RuntimeError('scope probe')
    assert package.naive_recurrent_key_gated_hyperbolic_delta_rule is original
