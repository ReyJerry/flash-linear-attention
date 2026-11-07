# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Deterministic query-owner dQ fallback without atomics or an edge-by-D workspace."""
import torch
import triton
import triton.language as tl

OWNER_NUM_WARPS = 4
OWNER_NUM_STAGES = 2


@triton.jit(do_not_specialize=['T', 'K', 'RH', 'RT'])
def query_owner_dq_kernel(
    q, kmem, vmem, dout, lse, delta, route, block_ptr, prefix_valid, dq,
    T, K, RH, RT, SQT, SQH, SKT, SKH, SVT, SVH, SDT, SDH, SPB, SPM, SPH, SCALE,
    H: tl.constexpr, D: tl.constexpr, M: tl.constexpr, C: tl.constexpr,
    BQ: tl.constexpr, BC: tl.constexpr, BD: tl.constexpr, KP: tl.constexpr, BRANCH: tl.constexpr,
):
    start = tl.program_id(0).to(tl.int64) * BQ
    head = tl.program_id(1).to(tl.int64)
    lane = tl.arange(0, BQ)
    rows = start + lane
    qvalid = rows < T
    d = tl.arange(0, BD)
    qm = qvalid[:, None] & (d[None, :] < D)
    qt = tl.load(q + rows[:, None] * SQT + head * SQH + d[None, :], qm, other=0)
    dot = tl.load(dout + rows[:, None] * SDT + head * SDH + d[None, :], qm, other=0)
    joint_l = tl.load(lse + head * T + rows, qvalid, other=0)
    de = tl.load(delta + head * T + rows, qvalid, other=0)
    acc = tl.zeros((BQ, BD), tl.float32)

    flat = tl.arange(0, BQ * KP)
    flat_row = start + flat // KP
    flat_slot = flat % KP
    ids = tl.load(route + head * RH + flat_row * RT + flat_slot,
                  (flat_row < T) & (flat_slot < K), other=-1)
    remaining = ids
    position = tl.min(tl.where(remaining >= 0, flat, BQ * KP), 0)
    memory_lane = tl.arange(0, BC)
    while position < BQ * KP:
        candidate = tl.load(route + head * RH + (start + position // KP) * RT + position % KP)
        leader = candidate >= 0
        safe = tl.where(leader, candidate, 0)
        if BC <= 32 or D == BD:
            member = tl.sum((tl.reshape(ids, (BQ, KP)) == safe).to(tl.int32), axis=1) > 0
        else:
            member = tl.full((BQ,), False, tl.int1)
            for slot in tl.static_range(KP):
                row_id = tl.load(route + head * RH + rows * RT + slot, qvalid & (slot < K), other=-1)
                member = member | (row_id == safe)
        row_valid = qvalid & member & leader
        permitted = leader
        if BRANCH == 1:
            permitted = permitted & (tl.load(prefix_valid + safe) != 0)
        if permitted:
            if BRANCH == 0:
                begin = tl.load(block_ptr + safe)
                end = tl.load(block_ptr + safe + 1)
                extent = C
            else:
                begin = 0
                end = M
                extent = M
            for m0 in range(0, extent, BC):
                memory = m0 + memory_lane
                ix = tl.cast(begin, tl.int64) + memory
                valid_memory = ix < end
                mask = valid_memory[:, None] & (d[None, :] < D)
                if BRANCH == 0:
                    kaddr = ix[:, None] * SKT + head * SKH + d[None, :]
                    vaddr = ix[:, None] * SVT + head * SVH + d[None, :]
                else:
                    kaddr = safe.to(tl.int64) * SPB + memory.to(tl.int64)[:, None] * SPM + head * SPH + d[None, :]
                    vaddr = kaddr
                kt = tl.load(kmem + kaddr, mask, other=0)
                vt = tl.load(vmem + vaddr, mask, other=0)
                score = tl.dot(qt, tl.trans(kt), out_dtype=tl.float32) * SCALE
                score = tl.where(row_valid[:, None] & valid_memory[None, :], score, float('-inf'))
                probability = tl.exp(score - joint_l[:, None])
                dp = tl.dot(dot, tl.trans(vt), out_dtype=tl.float32)
                ds = probability * (dp - de[:, None])
                acc += tl.dot(ds.to(qt.dtype), kt, out_dtype=tl.float32) * SCALE
        remaining = tl.where(remaining == candidate, -1, remaining)
        position = tl.min(tl.where(remaining >= 0, flat, BQ * KP), 0)
    tl.store(dq + (rows[:, None] * H + head) * D + d[None, :], acc, qm)


def query_owner_dq(q, k, v, dout, lse, delta, route, block_ptr, prefix_valid, *,
                   chunk_size, num_slots, branch, bc=32, out=None):
    t, h, d = q.shape
    if out is None:
        out = torch.empty((t, h, d), device=q.device, dtype=torch.float32)
    width = route.shape[-1]
    if not width:
        return out.zero_()
    prefix_strides = k.stride()[:3] if branch else (0, 0, 0)
    query_owner_dq_kernel[(triton.cdiv(t, 16), h)](
        q, k, v, dout, lse, delta, route, block_ptr, prefix_valid.view(torch.uint8), out,
        t, width, route.stride(0), route.stride(1), q.stride(0), q.stride(1),
        k.stride(0), k.stride(1), v.stride(0), v.stride(1), dout.stride(0), dout.stride(1),
        *prefix_strides, d ** -0.5,
        H=h, D=d, M=num_slots, C=chunk_size, BQ=16, BC=bc, BD=max(16, triton.next_power_of_2(d)),
        KP=triton.next_power_of_2(width), BRANCH=branch, num_warps=OWNER_NUM_WARPS, num_stages=OWNER_NUM_STAGES)
    return out


def deterministic_sparse_dq(q_rope, q_raw, k_rope, value, pk, pv, dout, lse, delta,
                            rtok, rstate, block_ptr, prefix_valid, *, chunk_size, bc=32, outputs=None):
    results = []
    for branch, q, k, v, route in ((0, q_rope, k_rope, value, rtok), (1, q_raw, pk, pv, rstate)):
        results.append(query_owner_dq(q, k, v, dout, lse, delta, route, block_ptr, prefix_valid,
                                      chunk_size=chunk_size, num_slots=pk.shape[1], branch=branch, bc=bc,
                                      out=outputs[branch] if outputs is not None else None))
    return tuple(results)
