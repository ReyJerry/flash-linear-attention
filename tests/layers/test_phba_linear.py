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


@pytest.mark.parametrize('out_features', [2048, 4272], ids=['square', 'ffn-up'])
def test_linear_native_bytes(out_features):
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
    if torch.cuda.get_device_name() == 'NVIDIA H20' and torch.__version__.split('+')[0] == '2.9.0':
        backend = exact_backend(x.device)
        assert backend is not None, 'The H20 test must execute the built exact backend'
        assert backend.sessions, 'The exact weight-gradient path was not used'


def test_linear_unscreened_fallback():
    torch.manual_seed(42)
    module = PHBALinear(32, 17, bias=True).to(device)
    x = torch.randn(3, 33, 32, device=device, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        actual = module(x)
        expected = F.linear(y, module.weight, module.bias)
    grad = torch.randn_like(actual)
    dx, = torch.autograd.grad(actual, (x,), grad)
    dy, = torch.autograd.grad(expected, (y,), grad)
    assert torch.equal(actual, expected)
    assert torch.equal(dx, dy)


def test_swiglu_down_native_bytes():
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
