# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""CPU CLI and route-attestation checks for the staged TC benchmark."""

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from benchmarks.hyperbolic_benchmark_utils import Method


@pytest.mark.parametrize('arguments,code', [
    (['--dry-run', '--dtype', 'fp32', '--layer-inputs', '--modes', 'fwd', 'bwd', 'fwd_bwd'], 0),
    (['--dry-run', '--dtype', 'bf16', '--legacy-fast'], 0),
    (['--dry-run', '--dtype', 'fp32', '--legacy-fast'], 2),
    (['--dry-run', '--key-dim', '160'], 0),
    (['--dry-run', '--key-dim', '257'], 2),
    (['--dry-run', '--backward', '--modes', 'bwd'], 2),
])
def test_tensor_core_benchmark_cpu_cli(arguments, code):
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, 'benchmarks/benchmark_hyperbolic_tensor_core.py', *arguments],
        cwd=root, capture_output=True, text=True, check=False,
    )
    assert result.returncode == code, result.stderr
    if code == 0:
        payload = json.loads(result.stdout)
        assert payload['status'] == 'DRY RUN; GPU NOT RUN'
        names = [method['name'] for method in payload['methods']]
        if '--key-dim' in arguments:
            assert 'staged_fallback' in names and 'staged_wy' not in names
        else:
            assert 'staged_wy' in names
        assert ('tensor_core_legacy_fast' in names) == ('--legacy-fast' in arguments)


def test_tensor_core_benchmark_no_gpu_writes_unrun_status(monkeypatch, tmp_path):
    module = importlib.import_module('benchmarks.benchmark_hyperbolic_tensor_core')
    path = tmp_path / 'no-gpu.json'
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(sys, 'argv', ['benchmark_hyperbolic_tensor_core', '--check-only', '--output', str(path)])
    assert module.main() == 2
    result = json.loads(path.read_text())
    assert result['status'] == 'GPU NOT RUN: CUDA device unavailable'
    assert 'correctness' not in result and 'measurements' not in result


def test_tensor_core_training_attestation_preserves_backward_precision():
    from benchmarks.benchmark_hyperbolic_tensor_core import _attest_training

    method = Method('tc', None, {}, 'unprotected', 'hyperbolic_delta_tc_chunk_fwd', {'FAST': True})
    trace = [{'kernel': 'hyperbolic_delta_tc_chunk_fwd', 'flags': {'FAST': True}, 'launches': 1},
             {'kernel': 'hyperbolic_delta_tc_chunk_bwd', 'flags': {'FAST': False}, 'launches': 1}]
    with pytest.raises(AssertionError):
        _attest_training(method, trace)
    trace[-1]['flags']['FAST'] = True
    _attest_training(method, trace)


def test_staged_training_attestation_requires_all_seven_kernels():
    from benchmarks.benchmark_hyperbolic_tensor_core import _attest_training

    method = Method('staged_wy', None, {}, 'unprotected', 'hyperbolic_staged_prepare_fwd_kernel')
    names = ('hyperbolic_staged_prepare_fwd_kernel', 'hyperbolic_staged_state_fwd_kernel',
             'hyperbolic_staged_output_fwd_kernel', 'hyperbolic_staged_output_bwd_kernel',
             'hyperbolic_staged_state_bwd_kernel', 'hyperbolic_staged_values_bwd_kernel',
             'hyperbolic_staged_parameters_bwd_kernel')
    trace = [{'kernel': name, 'flags': {}, 'launches': 1} for name in names]
    _attest_training(method, trace)
    with pytest.raises(AssertionError):
        _attest_training(method, trace[:-1])


@pytest.mark.parametrize('dtype', ['fp16', 'bf16'])
def test_legacy_state_gate_keeps_existing_fp32_tolerance(dtype):
    from benchmarks.benchmark_hyperbolic_tensor_core import _compare_reference

    method = Method('old_chunk', None, {}, 'unprotected', '')
    oracle = {'outputs': (torch.zeros(1), torch.zeros(1)), 'gradients': (), 'discrete_outputs': ()}
    actual = {**oracle, 'outputs': (torch.zeros(1), torch.tensor([4e-4])),
              'output_dtypes': (torch.float16, torch.float32), 'input_dtypes': ()}
    with pytest.raises(AssertionError):
        _compare_reference(SimpleNamespace(dtype=dtype, key_dim=64), method, actual, oracle)
    actual['outputs'] = (torch.zeros(1), torch.tensor([1e-4]))
    assert _compare_reference(SimpleNamespace(dtype=dtype, key_dim=64), method, actual, oracle)['outputs_1']['atol'] == 2e-4
