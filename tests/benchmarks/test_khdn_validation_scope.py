# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import importlib.util
from pathlib import Path

import pytest
import torch

_path = Path(__file__).resolve().parents[2] / 'fla/ops/key_gated_hyperbolic_delta_rule/validation.py'
_spec = importlib.util.spec_from_file_location('khdn_validation_scope', _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
FiniteValidationScope = _module.FiniteValidationScope


def test_finite_scope_blocks_optimizer_before_parameter_or_state_update():
    parameter = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float32))
    parameter.grad = torch.ones_like(parameter)
    optimizer = torch.optim.SGD([parameter], lr=.1, momentum=.9)
    scope = FiniteValidationScope(optimizer)
    with pytest.raises(RuntimeError, match='nonfinite KHDN training'), scope:
        scope.add(torch.tensor(False))
        optimizer.step()
    assert parameter.item() == 1 and not optimizer.state
    assert not scope.open and not scope.flags and scope._hook is None


def test_finite_scope_commits_successful_update_and_removes_hook():
    parameter = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float32))
    parameter.grad = torch.ones_like(parameter)
    optimizer = torch.optim.SGD([parameter], lr=.1)
    with FiniteValidationScope(optimizer) as scope:
        scope.add(torch.tensor(True))
        scope.add(torch.tensor(True))
        optimizer.step()
        assert scope.committed and scope.flushes == 1 and scope.checks == 2
    torch.testing.assert_close(parameter, torch.tensor([.9]), atol=1e-7, rtol=0)
    assert scope._hook is None
    with pytest.raises(RuntimeError, match='cannot be reused'), scope:
        pass


def test_finite_scope_exit_checks_pending_forward_without_optimizer():
    with pytest.raises(RuntimeError, match='nonfinite KHDN training'), FiniteValidationScope() as scope:
        scope.add(torch.tensor(True))
        scope.add(torch.tensor(False))
    assert not scope.flags and not scope.open


def test_finite_scope_flush_keeps_backward_collection_open_and_preserves_source_error():
    scope = FiniteValidationScope()
    with pytest.raises(ValueError, match='input domain'), scope:
        scope.add(torch.tensor(True))
        scope.flush()
        assert scope.accepting and scope.flushes == 1
        scope.add(torch.tensor(False))
        raise ValueError('input domain')
    assert not scope.flags and not scope.open
    with pytest.raises(RuntimeError, match='cannot be reused'):
        scope.__enter__()
