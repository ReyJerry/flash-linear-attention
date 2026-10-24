# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""CPU checks of the HDN measurement harness; no CUDA results are implied."""

import importlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from benchmarks.hyperbolic_benchmark_utils import (
    Method,
    _ObservedKernel,
    _prepare,
    attest,
    compare,
    environment,
    evaluate,
    summarize,
)


def test_environment_is_restored_after_failure(monkeypatch):
    monkeypatch.setenv('FLA_HYPERBOLIC_SHARED_INVERSE', '0')
    monkeypatch.delenv('FLA_HYPERBOLIC_PROTECTED_STAGED', raising=False)
    with pytest.raises(RuntimeError), environment({
        'FLA_HYPERBOLIC_SHARED_INVERSE': '1', 'FLA_HYPERBOLIC_PROTECTED_STAGED': '1',
    }):
        raise RuntimeError('failed correctness gate')
    assert os.environ['FLA_HYPERBOLIC_SHARED_INVERSE'] == '0'
    assert 'FLA_HYPERBOLIC_PROTECTED_STAGED' not in os.environ


def test_timing_summary_keeps_distribution():
    result = summarize([4, 1, 3, 2])
    assert result['median'] == 2.5
    assert result['p95'] == pytest.approx(3.85)
    assert result['min'] == 1 and result['max'] == 4
    assert result['stdev'] == pytest.approx(1.118033988749895)


def test_backward_measurement_prepares_forward_outside_callable():
    calls = []

    def forward(inputs):
        calls.append('forward')
        return (inputs[0] * inputs[1],)

    method = Method('example', forward, {}, 'example', 'example')
    inputs = (torch.tensor([2.0]), torch.tensor([3.0]))
    operation = _prepare(method, inputs, 'bwd')
    assert calls == ['forward']
    result = operation()
    assert calls == ['forward']
    torch.testing.assert_close(result[0], torch.tensor([36.0]))
    torch.testing.assert_close(result[1], torch.tensor([24.0]))


def test_reference_cotangents_are_identical_across_dtypes():
    inputs = (torch.tensor([0.125, 0.75, 1.5]), torch.tensor([0.25, 1.0, -0.5]))

    def run(values):
        return (values[0] * values[1],)

    actual, _ = evaluate(run, inputs, 77)
    expected, _ = evaluate(run, tuple(x.double() for x in inputs), 77)
    compare(actual, expected, (0, 0), (0, 0))
    actual['gradients'] = (actual['gradients'][0] + 0.1, actual['gradients'][1])
    with pytest.raises(AssertionError):
        compare(actual, expected, (0, 0), (0, 0))


def test_route_attestation_checks_actual_launch_flags():
    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: args[0] + 1

    calls = Counter()
    kernel = _ObservedKernel(Kernel(), 'example.protected_value_tile_fwd_kernel', calls)
    assert kernel[(1,)](4, SHARED_INVERSE=True, BK=16) == 5
    records = [{'kernel': name, 'flags': dict(flags), 'launches': count} for (name, flags), count in calls.items()]
    method = Method('shared', None, {}, 'protected', 'protected_value_tile_fwd_kernel', {'SHARED_INVERSE': True})
    attest(method, records)
    method.expected_flags = {'SHARED_INVERSE': False}
    with pytest.raises(AssertionError):
        attest(method, records)
    with pytest.raises(AssertionError):
        attest(method, [])


@pytest.mark.parametrize('arguments,code', [
    (['--dry-run', '--modes', 'fwd', 'bwd', 'fwd_bwd'], 0),
    (['--dry-run', '--backward', '--compare-strategies'], 0),
    (['--dry-run', '--repeats', '1'], 2),
    (['--dry-run', '--rank', '17'], 2),
    (['--dry-run', '--backward', '--modes', 'fwd'], 2),
    (['--dry-run', '--temperature', 'nan'], 2),
    (['--dry-run', '--temperature', 'inf'], 2),
    (['--dry-run', '--compare-trajectories'], 0),
])
def test_protected_benchmark_cpu_cli(arguments, code):
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, 'benchmarks/benchmark_protected_hyperbolic_delta_rule.py', *arguments],
        cwd=root, capture_output=True, text=True, check=False,
    )
    assert result.returncode == code, result.stderr
    if code == 0:
        payload = json.loads(result.stdout)
        assert payload['status'] == 'DRY RUN; GPU NOT RUN'
        shared = next(x for x in payload['methods'] if x['name'] == 'protected_shared_inverse')
        assert shared['expected_flags'] == {'SHARED_INVERSE': True}
        assert shared['baseline'] == 'protected_value_parallel'
        trajectory = [x for x in payload['methods'] if x['name'] == 'protected_trajectory']
        assert bool(trajectory) == ('--compare-trajectories' in arguments)


def test_benchmark_no_gpu_writes_unrun_status(monkeypatch, tmp_path):
    module = importlib.import_module('benchmarks.benchmark_protected_hyperbolic_delta_rule')
    path = tmp_path / 'no-gpu.json'
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(sys, 'argv', ['benchmark_protected_hyperbolic_delta_rule', '--check-only', '--output', str(path)])
    assert module.main() == 2
    result = json.loads(path.read_text())
    assert result['status'] == 'GPU NOT RUN: CUDA device unavailable'
    assert 'correctness' not in result and 'measurements' not in result


@pytest.mark.parametrize('checkpoint_enabled', [False, True])
def test_decomposed_benchmark_matches_checkpoint_policy(monkeypatch, checkpoint_enabled):
    from torch.utils import checkpoint

    from benchmarks import benchmark_protected_hyperbolic_delta_rule as benchmark

    calls = []
    original = checkpoint.checkpoint

    def observed(function, *inputs, **kwargs):
        calls.append(kwargs['use_reentrant'])
        return original(function, *inputs, **kwargs)

    def chunk(args, q, k, v, beta, selection, state, bank_k, bank_v, bank_scores, c):
        return (q * v + beta, state + k.sum(), bank_k + selection.sum(), bank_v + c,
                bank_scores + selection.sum(), torch.ones(1, 1, dtype=torch.int32))

    monkeypatch.setattr(checkpoint, 'checkpoint', observed)
    monkeypatch.setattr(benchmark, '_decomposed_chunk', chunk)
    base = [torch.ones(1, 3, 1, 1) for _ in range(4)]
    base += [torch.ones(1, 1, 1, 1) for _ in range(3)]
    base += [torch.zeros(1, 1, 1), torch.zeros(1, 3, 1), torch.tensor(0.5)]

    def execute(no_checkpoint):
        inputs = tuple(x.clone().requires_grad_() for x in base)
        args = SimpleNamespace(chunk_size=2, no_checkpoint=no_checkpoint)
        output = benchmark._decomposed(args, inputs)
        gradients = torch.autograd.grad(sum(x.sum() for x in output[:5]), inputs)
        return output, gradients

    expected, reference_gradients = execute(True)
    actual, actual_gradients = execute(not checkpoint_enabled)
    assert calls == ([False, False] if checkpoint_enabled else [])
    for actual_value, expected_value in zip((*actual, *actual_gradients), (*expected, *reference_gradients)):
        torch.testing.assert_close(actual_value, expected_value, atol=0, rtol=0)


def test_protected_training_attestation_requires_matching_backward():
    from benchmarks.benchmark_protected_hyperbolic_delta_rule import _attest_training

    method = Method('shared', None, {'FLA_HYPERBOLIC_SHARED_INVERSE': '1'}, 'protected',
                    'protected_value_tile_fwd_kernel', {'SHARED_INVERSE': True})
    trace = [
        {'kernel': 'protected_value_tile_fwd_kernel', 'flags': {'SHARED_INVERSE': True}, 'launches': 1},
        {'kernel': 'shared_key_gram_fwd_kernel', 'flags': {}, 'launches': 1},
        {'kernel': 'soft_bank_update_bwd_kernel', 'flags': {}, 'launches': 1},
    ]
    with pytest.raises(AssertionError):
        _attest_training(method, trace)
    trace.append({'kernel': 'protected_value_tile_bwd_kernel', 'flags': {'SHARED_INVERSE': False}, 'launches': 1})
    with pytest.raises(AssertionError):
        _attest_training(method, trace)
    trace[-1]['flags']['SHARED_INVERSE'] = True
    _attest_training(method, trace)


def test_protected_trajectory_attestation_requires_reverse_stages():
    from benchmarks.benchmark_protected_hyperbolic_delta_rule import _attest_training

    method = Method('trajectory', None, {}, 'protected', 'protected_value_trajectory_fwd_kernel')
    names = ('protected_value_trajectory_fwd_kernel', 'protected_shared_trajectory_fwd_kernel',
             'shared_key_gram_fwd_kernel', 'protected_value_trajectory_bwd_kernel',
             'protected_shared_trajectory_bwd_kernel', 'protected_value_partials_bwd_kernel',
             'protected_bank_partials_bwd_kernel')
    trace = [{'kernel': name, 'flags': {}, 'launches': 1} for name in names]
    _attest_training(method, trace)
    with pytest.raises(AssertionError):
        _attest_training(method, trace[:-1])
