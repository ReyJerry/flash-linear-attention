# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Algorithm 7 (§4.24–4.26): two branch launches, fp32 dQ atomics, single-writer dK/dV.

RoPE and chunkwise Prefix backward remain outside the attention Function.
Every output is zero-initialized, including unrouted blocks and invalid prefixes.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from fla.ops.phba.errors import fail

NUM_WARPS = 4
NUM_STAGES = 1


@triton.jit(do_not_specialize=['T'])
def _delta_kernel(do, out, delta, T, sot, soh, sdt, sdh,
                  H: tl.constexpr, D: tl.constexpr, BD: tl.constexpr, BT: tl.constexpr):
    rows = tl.program_id(0) * BT + tl.arange(0, BT)
    h = tl.program_id(1).to(tl.int64)
    d = tl.arange(0, BD)
    mask = (rows[:, None] < T) & (d[None, :] < D)
    r = rows.to(tl.int64)
    a = tl.load(do + r[:, None] * sdt + h * sdh + d[None, :], mask, other=0).to(tl.float32)
    b = tl.load(out + r[:, None] * sot + h * soh + d[None, :], mask, other=0).to(tl.float32)
    tl.store(delta + h * T + r, tl.sum(a * b, 1), rows < T)


@triton.jit(do_not_specialize=['T'])
def _sparse_bwd_kernel(
    q, kmem, vmem, do, lse, delta, count, offset, rho, block_ptr, prefix_valid,
    dq, dk, dv, T, scale,
    sqt, sqh, skt, skh, svt, svh, sdt, sdh,
    spb, spm, sph, sgb, sgm, sgh,
    H: tl.constexpr, D: tl.constexpr, M: tl.constexpr,
    BC: tl.constexpr, BQ: tl.constexpr, BD: tl.constexpr, BRANCH: tl.constexpr, WRITE_DQ: tl.constexpr = True,
):
    e = tl.program_id(0)
    block = e // H
    h = (e % H).to(tl.int64)
    tile = tl.program_id(1)
    n = tl.load(count + e)
    if n == 0:
        return
    mem = tile * BC + tl.arange(0, BC)
    d = tl.arange(0, BD)
    if BRANCH == 0:
        start = tl.load(block_ptr + block)
        end = tl.load(block_ptr + block + 1)
        if start + tile * BC >= end:
            return
        ix = (start + mem).to(tl.int64)
        valid_mem = ix < end
        kaddr = ix[:, None] * skt + h * skh + d[None, :]
        vaddr = ix[:, None] * svt + h * svh + d[None, :]
        grad_addr = (ix[:, None] * H + h) * D + d[None, :]
    else:
        if tl.load(prefix_valid + block) == 0:
            return
        valid_mem = mem < M
        kaddr = block.to(tl.int64) * spb + mem.to(tl.int64)[:, None] * spm + h * sph + d[None, :]
        vaddr = kaddr
        grad_addr = block.to(tl.int64) * sgb + mem.to(tl.int64)[:, None] * sgm + h * sgh + d[None, :]
    mm = valid_mem[:, None] & (d[None, :] < D)
    k = tl.load(kmem + kaddr, mm, other=0)
    v = tl.load(vmem + vaddr, mm, other=0)
    dk_acc = tl.zeros((BC, BD), tl.float32)
    dv_acc = tl.zeros((BC, BD), tl.float32)
    begin = tl.load(offset + e)
    finish = tl.load(offset + e + 1)
    lane = tl.arange(0, BQ)
    for p in range(begin, finish, BQ):
        pos = p + lane
        valid_q = pos < finish
        r = tl.load(rho + pos.to(tl.int64), valid_q, other=0)
        valid_q = valid_q & (r % H == h)
        t = (r // H).to(tl.int64)
        qm = valid_q[:, None] & (d[None, :] < D)
        qt = tl.load(q + t[:, None] * sqt + h * sqh + d[None, :], qm, other=0)
        dot = tl.load(do + t[:, None] * sdt + h * sdh + d[None, :], qm, other=0)
        l = tl.load(lse + h * T + t, valid_q, other=0)
        de = tl.load(delta + h * T + t, valid_q, other=0)
        score = tl.dot(qt, tl.trans(k), out_dtype=tl.float32) * scale
        score = tl.where(valid_q[:, None] & valid_mem[None, :], score, float('-inf'))
        prob = tl.exp(score - l[:, None])
        dp = tl.dot(dot, tl.trans(v), out_dtype=tl.float32)
        ds = prob * (dp - de[:, None])
        ds_st = ds.to(qt.dtype)
        dk_acc += tl.dot(tl.trans(ds_st), qt, out_dtype=tl.float32) * scale
        dv_acc += tl.dot(tl.trans(prob.to(dot.dtype)), dot, out_dtype=tl.float32)
        if WRITE_DQ:
            dq_tile = tl.dot(ds_st, k, out_dtype=tl.float32) * scale
            tl.atomic_add(dq + (t[:, None] * H + h) * D + d[None, :], dq_tile, qm)
    tl.store(dk + grad_addr, dk_acc, mm)
    tl.store(dv + grad_addr, dv_acc, mm)


@triton.jit(do_not_specialize=['ROWS'])
def _merge_kernel(local, sparse, out, ROWS, D: tl.constexpr, BLOCK: tl.constexpr):
    N = tl.cast(ROWS, tl.int64) * D
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(local + i, i < N, other=0).to(tl.float32)
    b = tl.load(sparse + i, i < N, other=0).to(tl.float32)
    tl.store(out + i, a + b, i < N)


def compute_delta(dout, out):
    T, H, D = out.shape
    delta = torch.empty((H, T), device=out.device, dtype=torch.float32)
    _delta_kernel[(triton.cdiv(T, 32), H)](
        dout, out, delta, T, out.stride(0), out.stride(1), dout.stride(0), dout.stride(1),
        H=H, D=D, BD=max(16, triton.next_power_of_2(D)), BT=32, num_warps=4, num_stages=1)
    return delta


def merge_gradient(local, sparse, *, output_dtype=torch.float32):
    if not local.is_contiguous() or not sparse.is_contiguous():
        raise fail('non_contiguous_input', 'gradient merge requires contiguous buffers')
    out = torch.empty(local.shape, device=local.device, dtype=output_dtype)
    n, d = local.numel(), local.shape[-1]
    _merge_kernel[(triton.cdiv(n, 256),)](local, sparse, out, n // d, D=d, BLOCK=256, num_warps=4, num_stages=1)
    return out


@dataclass
class SparseGradients:
    dq_rope: torch.Tensor
    dk_rope: torch.Tensor
    dv: torch.Tensor
    dq_raw: torch.Tensor
    dpk: torch.Tensor
    dpv: torch.Tensor


def launch_branch(q, k, v, dout, lse, delta, meta, block_ptr, prefix_valid, dq, dk, dv,
                  *, branch, chunk_size, num_slots, bc=32, bq=16, write_dq=True):
    T, H, D = q.shape
    extent = chunk_size if branch == 0 else num_slots
    ps = k.stride()[:3] if branch else (0, 0, 0)
    gs = dk.stride()[:3] if branch else (0, 0, 0)
    _sparse_bwd_kernel[(meta.count.numel(), triton.cdiv(extent, bc), 1)](
        q, k, v, dout, lse, delta, meta.count, meta.offset, meta.rho, block_ptr, prefix_valid.view(torch.uint8),
        dq, dk, dv, T, D ** -0.5,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
        dout.stride(0), dout.stride(1), *ps, *gs,
        H=H, D=D, M=num_slots, BC=bc, BQ=bq, BD=max(16, triton.next_power_of_2(D)), BRANCH=branch, WRITE_DQ=write_dq,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES)


def direct_sparse_backward(q_rope, q_raw, k_rope, v, pk, pv, dout, lse, delta, metadata,
                           block_ptr, prefix_valid, *, chunk_size, bc=32, bq=16, n_query_split=1,
                           backward_mode='atomic', rtok=None, rstate=None, tape_plan=None):
    if n_query_split != 1:
        raise fail('n_query_split_unsupported', 'Algorithm 7 phase 1 requires n_query_split=1')
    if bq not in (16, 32) or bc not in (16, 32, 64):
        raise fail('invalid_config', 'unsupported Algorithm 7 tiles')
    if not lse.is_contiguous() or not delta.is_contiguous():
        raise fail('non_contiguous_input', 'L/Delta must be contiguous [H,T]')
    if backward_mode not in ('atomic', 'query_owner_deterministic'):
        raise fail('invalid_config', f'unknown backward mode: {backward_mode}')
    deterministic = backward_mode == 'query_owner_deterministic'
    if deterministic and (rtok is None or rstate is None):
        raise fail('invalid_config', 'query-owner backward requires its saved forward routes')
    T, H, D = q_rope.shape

    def z():
        return torch.zeros((T, H, D), device=q_rope.device, dtype=torch.float32)

    def q_buffer():
        return torch.empty((T, H, D), device=q_rope.device, dtype=torch.float32) if deterministic else z()

    grads = SparseGradients(q_buffer(), z(), z(), q_buffer(), torch.zeros_like(pk, dtype=torch.float32),
                            torch.zeros_like(pv, dtype=torch.float32))
    if deterministic:
        from fla.ops.phba.plans import owner_tape_plan
        if tape_plan is None:
            tape_plan = owner_tape_plan(T, H, pk.shape[0], chunk_size, pk.shape[1], rtok.shape[-1],
                                        rstate.shape[-1], bc, q_rope.element_size())
        if tape_plan.enabled:
            from fla.ops.phba.tape import backward_with_tape
            return backward_with_tape(q_rope, q_raw, k_rope, v, pk, pv, dout, lse, delta, metadata,
                                      block_ptr, prefix_valid, rtok, rstate, grads, tape_plan,
                                      chunk_size=chunk_size, bc=bc, bq=bq)
    args = (dout, lse, delta)
    if metadata.tok is not None:
        launch_branch(q_rope, k_rope, v, *args, metadata.tok, block_ptr, prefix_valid,
                      grads.dq_rope, grads.dk_rope, grads.dv, branch=0, chunk_size=chunk_size,
                      num_slots=pk.shape[1], bc=bc, bq=bq, write_dq=not deterministic)
    launch_branch(q_raw, pk, pv, *args, metadata.state, block_ptr, prefix_valid,
                  grads.dq_raw, grads.dpk, grads.dpv, branch=1, chunk_size=chunk_size,
                  num_slots=pk.shape[1], bc=bc, bq=bq, write_dq=not deterministic)
    if deterministic:
        from fla.ops.phba.query_owner import deterministic_sparse_dq
        deterministic_sparse_dq(q_rope, q_raw, k_rope, v, pk, pv, dout, lse, delta, rtok, rstate,
                                block_ptr, prefix_valid, chunk_size=chunk_size, bc=bc,
                                outputs=(grads.dq_rope, grads.dq_raw))
    return grads


__all__ = ['SparseGradients', 'compute_delta', 'direct_sparse_backward', 'launch_branch', 'merge_gradient']
