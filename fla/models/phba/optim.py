# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from functools import cache

import torch

_CHUNK_ELEMENTS = 65536
_PACKET_BLOCKS = 312
_MAX_TENSORS = 36


@cache
def _supported_device(index: int) -> bool:
    properties = torch.cuda.get_device_properties(index)
    return properties.name == 'NVIDIA H20' and properties.multi_processor_count == 78


def _packet_update(params, grads, exp_avgs, exp_avg_sqs, steps, **kwargs):
    arrays = (params, grads, exp_avgs, exp_avg_sqs)
    if not all(
        tensor.is_contiguous() and tensor.numel() % 4 == 0 and tensor.data_ptr() % 16 == 0
        for group in arrays for tensor in group
    ):
        torch._fused_adamw_(*arrays, [], steps, **kwargs)
        return
    packet = [[] for _ in arrays]
    packet_steps = []
    blocks = 0
    for index, parameter in enumerate(params):
        flat = [group[index].view(-1) for group in arrays]
        offset = 0
        while offset < parameter.numel():
            if blocks == _PACKET_BLOCKS or len(packet_steps) == _MAX_TENSORS:
                torch._fused_adamw_(*packet, [], packet_steps, **kwargs)
                packet = [[] for _ in arrays]
                packet_steps = []
                blocks = 0
            length = min(parameter.numel() - offset, (_PACKET_BLOCKS - blocks) * _CHUNK_ELEMENTS)
            for outputs, tensor in zip(packet, flat):
                outputs.append(tensor.narrow(0, offset, length))
            packet_steps.append(steps[index])
            blocks += (length + _CHUNK_ELEMENTS - 1) // _CHUNK_ELEMENTS
            offset += length
    if packet_steps:
        torch._fused_adamw_(*packet, [], packet_steps, **kwargs)


class PHBAAdamW(torch.optim.AdamW):
    """Native AdamW with H20 packet scheduling and an unchanged native fallback.

    The optimized path retains the installed CUDA kernel and FP32 state arithmetic.
    Views and launch metadata are rebuilt every step. Native dispatcher functions
    and the optimizer behavior of other models are never replaced.
    """

    def _can_use_packets(self) -> bool:
        if torch.__version__.split('+')[0] != '2.9.0' or torch.version.cuda != '12.8':
            return False
        if getattr(self, 'grad_scale', None) is not None or getattr(self, 'found_inf', None) is not None:
            return False
        devices = set()
        for group in self.param_groups:
            if (not group['fused'] or group['amsgrad'] or group['differentiable'] or group['capturable']
                    or not all(isinstance(value, float) for value in (group['lr'], *group['betas']))):
                return False
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                if parameter.dtype != torch.float32 or not parameter.is_cuda or parameter.grad.is_sparse:
                    return False
                devices.add(parameter.device.index)
        return len(devices) == 1 and _supported_device(next(iter(devices)))

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None or not self._can_use_packets():
            return super().step(closure)
        self._cuda_graph_capture_health_check()
        for group in self.param_groups:
            params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, steps = [], [], [], [], [], []
            self._init_group(group, params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, steps)
            if not params:
                continue
            torch._foreach_add_(steps, 1)
            _packet_update(
                params=params,
                grads=grads,
                exp_avgs=exp_avgs,
                exp_avg_sqs=exp_avg_sqs,
                steps=steps,
                amsgrad=False,
                lr=group['lr'],
                beta1=group['betas'][0],
                beta2=group['betas'][1],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                maximize=group['maximize'],
                grad_scale=None,
                found_inf=None,
            )
        return None
