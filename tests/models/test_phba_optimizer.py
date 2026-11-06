# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.models.phba.optim import PHBAAdamW
from fla.utils import IS_NVIDIA, device


def _same_bytes(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))


@pytest.mark.skipif(not IS_NVIDIA, reason='PHBA optimizer CUDA packet path')
@pytest.mark.parametrize(
    ('size', 'amsgrad', 'maximize'),
    [(312 * 65536 + 4, False, False), (257, False, False), (4096, True, True)],
    ids=['packet-boundary', 'native-tail-fallback', 'native-amsgrad-fallback'],
)
def test_optimizer_states(size, amsgrad, maximize):
    torch.manual_seed(42)
    original_dispatch = torch._fused_adamw_
    source = torch.randn(size, device=device)
    left, right = (torch.nn.Parameter(source.clone()) for _ in range(2))
    base = torch.optim.AdamW([left], lr=1e-4, betas=(.9, .95), fused=True, amsgrad=amsgrad, maximize=maximize)
    candidate = PHBAAdamW([right], lr=1e-4, betas=(.9, .95), fused=True, amsgrad=amsgrad, maximize=maximize)
    for learning_rate in (0., 1e-4, 3e-4, 0.):
        grad = torch.randn_like(source)
        for parameter, optimizer in ((left, base), (right, candidate)):
            parameter.grad = grad.clone()
            optimizer.param_groups[0]['lr'] = learning_rate
            torch.nn.utils.clip_grad_norm_([parameter], 1.)
            optimizer.step()
        assert torch._fused_adamw_ is original_dispatch
        _same_bytes(left, right)
        _same_bytes(left.grad, right.grad)
        for key in base.state[left]:
            _same_bytes(base.state[left][key], candidate.state[right][key])
