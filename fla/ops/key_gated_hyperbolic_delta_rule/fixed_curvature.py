# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Bounded immutable device constants shared by preparation and the chunk operator."""
from functools import lru_cache

import torch


@lru_cache(maxsize=16)
def _constant(device, value):
    with torch.inference_mode(False), torch.no_grad():
        tensor = torch.tensor(float.fromhex(value), dtype=torch.float32, device=device)
        ready = None
        if device.type == 'cuda':
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(device))
    return tensor, tensor._version, ready


def curvature_tensor(value, device):
    if isinstance(value, torch.Tensor):
        return value
    device = torch.device(device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    key = float(value).hex()
    tensor, version, ready = _constant(device, key)
    if tensor._version != version or tensor.requires_grad:
        _constant.cache_clear()
        tensor, version, ready = _constant(device, key)
    if ready is not None:
        stream = torch.cuda.current_stream(device)
        stream.wait_event(ready)
        tensor.record_stream(stream)
    return tensor
