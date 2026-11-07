# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.layers.phba import PHBALinear
from fla.ops.phba.backends.cublaslt import exact_backend
from fla.utils import IS_NVIDIA, device

pytestmark = pytest.mark.skipif(not IS_NVIDIA, reason='PHBA optional NVIDIA projection path')


def _compare_linear(out_features):
    torch.manual_seed(42)
    module = PHBALinear(2048, out_features, bias=False).to(device).train()
    x = torch.randn(1, 8192, 2048, device=device, requires_grad=True)
    baseline_x = x.detach().clone().requires_grad_()
    baseline_w = module.weight.detach().clone().requires_grad_()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        output = module(x)
        expected = F.linear(baseline_x, baseline_w)
    grad = torch.randn_like(output)
    actual_grads = torch.autograd.grad(output, (x, module.weight), grad)
    expected_grads = torch.autograd.grad(expected, (baseline_x, baseline_w), grad)
    assert torch.equal(output, expected)
    for actual, reference in zip(actual_grads, expected_grads):
        assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))


def test_linear_unscreened_fallback():
    torch.manual_seed(42)
    module = PHBALinear(32, 17, bias=True).to(device)
    x = torch.randn(3, 33, 32, device=device, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    weight = module.weight.detach().clone().requires_grad_()
    bias = module.bias.detach().clone().requires_grad_()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        actual = module(x)
        expected = F.linear(y, weight, bias)
    grad = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (x, module.weight, module.bias), grad)
    expected_grads = torch.autograd.grad(expected, (y, weight, bias), grad)
    assert torch.equal(actual, expected)
    for dx, dy in zip(actual_grads, expected_grads):
        assert torch.equal(dx, dy)


def _compare_swiglu_down():
    from fla.modules.activations import swiglu_linear
    from fla.ops.phba.linear import _SwiGLULinear

    torch.manual_seed(42)
    x = torch.randn(1, 8192, 4272, device=device, dtype=torch.bfloat16, requires_grad=True)
    y = torch.randn_like(x, requires_grad=True)
    weight = torch.randn(2048, 4272, device=device, requires_grad=True)
    reference_inputs = [tensor.detach().clone().requires_grad_() for tensor in (x, y, weight)]
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        actual = _SwiGLULinear.apply(x, y, weight, None)
        expected = swiglu_linear(*reference_inputs, None)
    grad = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(actual, (x, y, weight), grad)
    expected_gradients = torch.autograd.grad(expected, reference_inputs, grad)
    assert torch.equal(actual, expected)
    for result, reference in zip(actual_gradients, expected_gradients):
        assert torch.equal(result.view(torch.uint8), reference.view(torch.uint8))


def _compare_projection(geometry):
    if geometry == 'ffn-down':
        _compare_swiglu_down()
    else:
        _compare_linear(2048 if geometry == 'square' else 4272)


@pytest.mark.parametrize('geometry', ['square', 'ffn-up', 'ffn-down'])
def test_projection_native_bytes(geometry):
    _compare_projection(geometry)


def _exact_backend_or_skip():
    import os

    backend = exact_backend(torch.device('cuda', torch.cuda.current_device()))
    if backend is None:
        if os.environ.get('PHBA_REQUIRE_EXACT_GEMM') == '1':
            pytest.fail('The pinned H20 gate requires a working exact backend')
        pytest.skip('Exact backend runtime, library fingerprint or build prerequisites do not match')
    return backend


@pytest.mark.parametrize('geometry', ['square', 'ffn-up', 'ffn-down'])
def test_exact_backend_executes_geometry(geometry, monkeypatch):
    backend = _exact_backend_or_skip()
    calls = []
    native = backend.library.run_algo

    def observe(*args):
        calls.append(args[0])
        return native(*args)

    monkeypatch.setattr(backend.library, 'run_algo', observe)
    _compare_projection(geometry)
    assert len(calls) == 1, f'{geometry} did not execute its selected cuBLASLt algorithm'


@pytest.mark.parametrize('condition', ['cuda-version', 'library-fingerprint', 'build-failure'])
def test_runtime_fallback_gradients(condition, monkeypatch):
    import json
    from pathlib import Path

    from fla.ops.phba.backends import cublaslt

    if condition != 'cuda-version':
        _exact_backend_or_skip()

    monkeypatch.setattr(cublaslt, '_INITIALIZED', False)
    monkeypatch.setattr(cublaslt, '_BACKEND', None)
    if condition == 'cuda-version':
        monkeypatch.setattr(torch.version, 'cuda', 'unsupported')
    elif condition == 'library-fingerprint':
        original = Path.read_text

        def read(path, *args, **kwargs):
            text = original(path, *args, **kwargs)
            if path.name == 'exact_gemm_algorithms.json':
                data = json.loads(text)
                data['library_sha256'] = '0' * 64
                return json.dumps(data)
            return text

        monkeypatch.setattr(Path, 'read_text', read)
    else:
        def unavailable(*args, **kwargs):
            raise RuntimeError('test compiler unavailable')

        monkeypatch.setattr(cublaslt, '_ExactBackend', unavailable)
        with pytest.warns(UserWarning, match='test compiler unavailable'):
            assert exact_backend(torch.device('cuda', torch.cuda.current_device())) is None
    assert exact_backend(torch.device('cuda', torch.cuda.current_device())) is None
    _compare_linear(2048)
