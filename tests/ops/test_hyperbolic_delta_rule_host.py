# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""CPU-only launch-contract checks; these do not compile or execute Triton kernels.

Run with pytest --noconftest -q tests/ops/test_hyperbolic_delta_rule_host.py.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


class _Kernel:

    def __init__(self, launch):
        self.launch = launch

    def __getitem__(self, grid):
        return self.launch


@pytest.fixture
def tensor_core_host(monkeypatch):
    package_name = '_hyperbolic_host_test'
    path = Path(__file__).resolve().parents[2] / 'fla/ops/hyperbolic_delta_rule'
    package = types.ModuleType(package_name)
    package.__path__ = [str(path)]
    monkeypatch.setitem(sys.modules, package_name, package)
    triton = types.ModuleType('triton')
    triton.cdiv = lambda x, y: (x + y - 1) // y
    triton.next_power_of_2 = lambda x: 1 << (x - 1).bit_length()
    monkeypatch.setitem(sys.modules, 'triton', triton)
    kernels = types.ModuleType(f'{package_name}.kernels')
    kernels.hyperbolic_delta_rule_chunk_fwd_kernel = None
    kernels.hyperbolic_delta_rule_fwd_kernel = None
    kernels.hyperbolic_delta_rule_bwd_kernel = None
    monkeypatch.setitem(sys.modules, kernels.__name__, kernels)
    launches = []

    def forward(q, k, v, beta, h0, cu, out, final, checkpoints, curvature, **kwargs):
        launches.append(('forward', kwargs['FAST']))
        out.fill_(2)
        final.fill_(3)
        checkpoints.fill_(4)

    def backward(q, k, v, beta, cu, checkpoints, dout, dfinal, curvature,
                 dq, dk, dv, dbeta, dh0, dc, **kwargs):
        launches.append(('backward', kwargs['FAST']))
        assert dout.shape == v.shape and dfinal.shape == dh0.shape
        assert torch.all(dout == 1) and torch.all(dfinal == 2)
        assert torch.all(checkpoints == 4)
        for gradient in (dq, dk, dv, dbeta, dh0, dc):
            gradient.fill_(1)

    kernels = types.ModuleType(f'{package_name}.tensor_core_kernels')
    kernels.hyperbolic_delta_tc_chunk_fwd = _Kernel(forward)
    kernels.hyperbolic_delta_tc_chunk_bwd = _Kernel(backward)
    monkeypatch.setitem(sys.modules, kernels.__name__, kernels)
    for name in ('naive', 'fused_recurrent'):
        spec = importlib.util.spec_from_file_location(f'{package_name}.{name}', path / f'{name}.py')
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
    return module.HyperbolicDeltaRuleFunction, launches


@pytest.mark.parametrize('fast', [False, True], ids=['accurate', 'fast'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16], ids=['fp32', 'fp16', 'bf16'])
def test_tensor_core_forward_backward_launch(tensor_core_host, monkeypatch, dtype, fast):
    function, launches = tensor_core_host
    monkeypatch.setenv('FLA_HYPERBOLIC_TC_FAST', str(int(fast)))
    q = torch.ones((2, 17, 2, 16), dtype=dtype, requires_grad=True)
    k = q.detach().clone().requires_grad_()
    v = torch.ones((2, 17, 2, 17), dtype=dtype, requires_grad=True)
    beta = torch.ones_like(v, dtype=torch.float32, requires_grad=True)
    state = torch.zeros((2, 2, 16, 17), requires_grad=True)
    curvature = torch.tensor(0.5, requires_grad=True)
    cu = torch.tensor([0, 17, 34], dtype=torch.int64)
    output, final, _ = function.apply(q, k, v, beta, state, cu, curvature, 0.37, 16, 'chunk_tc', True, None)
    monkeypatch.setenv('FLA_HYPERBOLIC_TC_FAST', str(int(not fast)))
    gradients = torch.autograd.grad(
        outputs=(output, final),
        inputs=(q, k, v, beta, state, curvature),
        grad_outputs=(torch.ones_like(output), torch.full_like(final, 2)),
    )
    expected_fast = fast and dtype != torch.float32
    assert launches == [('forward', expected_fast), ('backward', expected_fast)]
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
