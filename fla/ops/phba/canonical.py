# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Fixed-capacity GPU canonicalization of (expert, query) route edges."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.phba.errors import fail


@triton.jit(do_not_specialize=['T', 'K', 'RH', 'RT', 'CAPACITY'])
def _route_keys_kernel(route, keys, T, K, RH, RT, CAPACITY, H: tl.constexpr, BT: tl.constexpr):
    SENTINEL: tl.constexpr = 2147483647 if keys.dtype.element_ty == tl.int32 else 9223372036854775807
    i = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    slot = i % K
    token = (i // K) % T
    head = i // (tl.cast(T, tl.int64) * K)
    block = tl.load(route + head * RH + token * RT + slot, i < CAPACITY, other=-1).to(tl.int64)
    row_stride = tl.cast(T, tl.int64) * H
    expert = tl.maximum(block, 0) * H + head
    key = expert * row_stride + token * H + head
    sentinel = tl.full((BT,), SENTINEL, tl.int64)
    tl.store(keys + i, tl.where(block >= 0, key, sentinel), i < CAPACITY)


@triton.jit(do_not_specialize=['CAPACITY', 'ROW_STRIDE'])
def _unpack_rho_kernel(keys, rho, CAPACITY, ROW_STRIDE, BT: tl.constexpr,
                       permutation=None, slots=None, K: tl.constexpr = 1, STORE_SLOTS: tl.constexpr = False):
    SENTINEL: tl.constexpr = 2147483647 if keys.dtype.element_ty == tl.int32 else 9223372036854775807
    i = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    key = tl.load(keys + i, i < CAPACITY, other=SENTINEL)
    row = tl.where(key != SENTINEL, key % tl.cast(ROW_STRIDE, tl.int64), -1)
    tl.store(rho + i, row.to(tl.int32), i < CAPACITY)
    if STORE_SLOTS:
        source = tl.load(permutation + i, i < CAPACITY, other=0)
        tl.store(slots + i, (source % K).to(tl.uint8), i < CAPACITY)


@triton.jit
def _sorted_bounds_kernel(keys, count, offset, CAP: tl.constexpr, ROW: tl.constexpr,
                          NE: tl.constexpr, B: tl.constexpr, STEPS: tl.constexpr):
    SENTINEL: tl.constexpr = 2147483647 if keys.dtype.element_ty == tl.int32 else 9223372036854775807
    e = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B).to(tl.int64)
    lo = tl.full((B,), 0, tl.int64)
    hi = tl.full((B,), CAP, tl.int64)
    lo2 = tl.full((B,), 0, tl.int64)
    hi2 = tl.full((B,), CAP, tl.int64)
    target = e * ROW
    target2 = (e + 1) * ROW
    for _ in range(STEPS):
        mid = lo + (hi - lo) // 2
        mid2 = lo2 + (hi2 - lo2) // 2
        v = tl.load(keys + mid, (e <= NE) & (mid < CAP), other=SENTINEL)
        v2 = tl.load(keys + mid2, (e < NE) & (mid2 < CAP), other=SENTINEL)
        left = v < target
        left2 = v2 < target2
        next_lo = tl.where((lo < hi) & left, mid + 1, lo)
        hi = tl.where((lo < hi) & ~left, mid, hi)
        next_lo2 = tl.where((lo2 < hi2) & left2, mid2 + 1, lo2)
        hi2 = tl.where((lo2 < hi2) & ~left2, mid2, hi2)
        lo = next_lo
        lo2 = next_lo2
    tl.store(offset + e, lo.to(tl.int32), e <= NE)
    tl.store(count + e, (lo2 - lo).to(tl.int32), e < NE)


def canonical_rho(route: torch.Tensor, nc: int, *, return_slots: bool = False, return_counts: bool = False):
    """Sort full capacity, including invalid sentinels; never read a device scalar on the host."""
    h, t, k = route.shape
    if k < 1 or t < 1 or h < 1:
        raise fail('invalid_shape', 'canonical metadata requires a nonempty branch')
    if (nc * h) * (t * h) >= 2**63 - 1:
        raise fail('int32_overflow', 'canonical expert/query sort key exceeds int64')
    capacity = t * h * k
    key_dtype = torch.int32 if (nc * h) * (t * h) < 2**31 - 1 else torch.int64
    keys = torch.empty(capacity, dtype=key_dtype, device=route.device)
    _route_keys_kernel[(triton.cdiv(capacity, 256),)](
        route, keys, t, k, route.stride(0), route.stride(1), capacity, H=h, BT=256,
        num_warps=4, num_stages=1)
    sorted_keys, permutation = torch.sort(keys)
    rho = torch.empty(capacity, dtype=torch.int32, device=route.device)
    slots = torch.empty(capacity, dtype=torch.uint8, device=route.device) if return_slots else None
    _unpack_rho_kernel[(triton.cdiv(capacity, 256),)](
        sorted_keys, rho, capacity, t * h, BT=256, permutation=permutation if return_slots else None,
        slots=slots, K=k, STORE_SLOTS=return_slots, num_warps=4, num_stages=1)
    if return_counts:
        count = torch.empty(nc * h, dtype=torch.int32, device=route.device)
        offset = torch.empty(nc * h + 1, dtype=torch.int32, device=route.device)
        _sorted_bounds_kernel[(triton.cdiv(nc * h + 1, 128),)](
            sorted_keys, count, offset, CAP=capacity, ROW=t * h, NE=nc * h, B=128,
            STEPS=(capacity + 1).bit_length(), num_warps=4, num_stages=1)
        return rho, slots, count, offset
    return (rho, slots) if return_slots else rho
