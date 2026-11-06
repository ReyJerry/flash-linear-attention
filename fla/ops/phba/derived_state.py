# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

'Derive state-route metadata from canonical token routes.'

import torch
import triton
import triton.language as tl


@triton.jit
def state_counts(tc, bp, pv, sc, N: tl.constexpr, H: tl.constexpr, B: tl.constexpr):
    e = tl.program_id(0).to(tl.int64)*B+tl.arange(0, B).to(tl.int64)
    block = e//H
    valid = e < N
    n = tl.load(tc+e, valid, other=0)
    length = tl.load(bp+block+1, valid, other=0)-tl.load(bp+block, valid, other=0)
    active = tl.load(pv+block, valid, other=0)
    tl.store(sc+e, tl.where(active, n+length, 0), valid)


@triton.jit
def state_rows(to, tr, bp, pv, so, sr, H: tl.constexpr, B: tl.constexpr):
    e = tl.program_id(0).to(tl.int64)
    block = e//H
    head = e % H
    if tl.load(pv+block) == 0:
        return
    start = tl.load(bp+block).to(tl.int64)
    end = tl.load(bp+block+1).to(tl.int64)
    first = tl.load(to+e).to(tl.int64)
    last = tl.load(to+e+1).to(tl.int64)
    dst = tl.load(so+e).to(tl.int64)
    cur = end-start
    total = cur+last-first
    lane = tl.arange(0, B).to(tl.int64)
    for base in range(0, total, B):
        ix = base+lane
        hist = tl.load(tr+first+ix-cur, (ix >= cur) & (ix < total), other=0)
        row = tl.where(ix < cur, (start+ix)*H+head, hist)
        tl.store(sr+dst+ix, row.to(tl.int32), ix < total)


def derive_state(tok, layout, h, capacity):
    from fla.ops.phba.metadata import BranchMetadata
    if capacity > 2147483647 or tok.count.numel()+1 > 2147483647:
        raise ValueError('state metadata exceeds int32 capacity')
    count = torch.empty_like(tok.count)
    offset = torch.zeros_like(tok.offset)
    state_counts[(triton.cdiv(count.numel(), 128),)](tok.count, layout.block_ptr,
                                                     layout.prefix_valid, count, N=count.numel(), H=h, B=128)
    torch.cumsum(count, 0, dtype=torch.int32, out=offset[1:])
    rho = torch.full((capacity,), -1, device=count.device, dtype=torch.int32)
    state_rows[(count.numel(),)](tok.offset, tok.rho, layout.block_ptr, layout.prefix_valid, offset, rho, H=h, B=128)
    return BranchMetadata(count, offset, rho, capacity)
