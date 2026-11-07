# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

'Deterministic PHBA attention backward with tape and bounded-memory query owners.'

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from fla.ops.phba.utils import fail

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
    dk, dv, T, scale,
    sqt, sqh, skt, skh, svt, svh, sdt, sdh,
    spb, spm, sph, sgb, sgm, sgh,
    H: tl.constexpr, D: tl.constexpr, M: tl.constexpr,
    BC: tl.constexpr, BQ: tl.constexpr, BD: tl.constexpr, BRANCH: tl.constexpr,
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


def launch_branch(q, k, v, dout, lse, delta, meta, block_ptr, prefix_valid, dk, dv,
                  *, branch, chunk_size, num_slots, bc=32, bq=16):
    T, H, D = q.shape
    extent = chunk_size if branch == 0 else num_slots
    ps = k.stride()[:3] if branch else (0, 0, 0)
    gs = dk.stride()[:3] if branch else (0, 0, 0)
    _sparse_bwd_kernel[(meta.count.numel(), triton.cdiv(extent, bc), 1)](
        q, k, v, dout, lse, delta, meta.count, meta.offset, meta.rho, block_ptr, prefix_valid.view(torch.uint8),
        dk, dv, T, D ** -0.5,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
        dout.stride(0), dout.stride(1), *ps, *gs,
        H=H, D=D, M=num_slots, BC=bc, BQ=bq, BD=max(16, triton.next_power_of_2(D)), BRANCH=branch,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES)


def direct_sparse_backward(q_rope, q_raw, k_rope, v, pk, pv, dout, lse, delta, metadata,
                           block_ptr, prefix_valid, *, chunk_size, bc=32, bq=16, n_query_split=1,
                           rtok=None, rstate=None, tape_plan=None):
    if n_query_split != 1:
        raise fail('n_query_split_unsupported', 'Sparse backward requires n_query_split=1')
    if bq not in (16, 32) or bc not in (16, 32, 64):
        raise fail('invalid_config', 'unsupported PHBA backward tiles')
    if not lse.is_contiguous() or not delta.is_contiguous():
        raise fail('non_contiguous_input', 'L/Delta must be contiguous [H,T]')
    if rtok is None or rstate is None:
        raise fail('invalid_config', 'query-owner backward requires its saved forward routes')
    T, H, D = q_rope.shape

    def z():
        return torch.zeros((T, H, D), device=q_rope.device, dtype=torch.float32)

    def q_buffer():
        return torch.empty((T, H, D), device=q_rope.device, dtype=torch.float32)

    grads = SparseGradients(q_buffer(), z(), z(), q_buffer(), torch.zeros_like(pk, dtype=torch.float32),
                            torch.zeros_like(pv, dtype=torch.float32))
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
                      grads.dk_rope, grads.dv, branch=0, chunk_size=chunk_size,
                      num_slots=pk.shape[1], bc=bc, bq=bq)
    launch_branch(q_raw, pk, pv, *args, metadata.state, block_ptr, prefix_valid,
                  grads.dpk, grads.dpv, branch=1, chunk_size=chunk_size,
                  num_slots=pk.shape[1], bc=bc, bq=bq)
    deterministic_sparse_dq(q_rope, q_raw, k_rope, v, pk, pv, dout, lse, delta, rtok, rstate,
                            block_ptr, prefix_valid, chunk_size=chunk_size, bc=bc,
                            outputs=(grads.dq_rope, grads.dq_raw))
    return grads


def attention_backward(ctx, dout, _dlse):
    from fla.ops.phba.flash_attention import local_backward
    from fla.ops.phba.metadata import BranchMetadata, DirectMetadata
    from fla.ops.phba.utils import prepare_grad_output

    qr, qw, kr, v, pk, pv, out, lse, bp, valid, sc, so, sr, *token = ctx.saved_tensors
    c, bc, bq = ctx.settings
    rtok, rstate = token[-2:]
    token = token[:-2]
    state = BranchMetadata(sc, so, sr, sr.numel())
    tok = BranchMetadata(*token, token[-1].numel()) if ctx.has_token else None
    metadata = DirectMetadata(tok, state, int(ctx.has_token))
    dout, _copy_bytes = prepare_grad_output(dout, out)
    dout = dout.contiguous()
    delta = compute_delta(dout, out)
    local = local_backward(dout, qr, kr, v, out, lse, bp, ctx.batches, deterministic=True)
    sparse = direct_sparse_backward(qr, qw, kr, v, pk, pv, dout, lse, delta, metadata, bp, valid,
                                    chunk_size=c, bc=bc, bq=bq, rtok=rtok, rstate=rstate,
                                    tape_plan=getattr(ctx, 'owner_tape_plan', None))
    dq = merge_gradient(local[0], sparse.dq_rope, output_dtype=qr.dtype)
    dk = merge_gradient(local[1], sparse.dk_rope, output_dtype=kr.dtype)
    dv = merge_gradient(local[2], sparse.dv, output_dtype=v.dtype)
    return dq.to(qr.dtype), sparse.dq_raw.to(qw.dtype), dk.to(kr.dtype), dv.to(v.dtype), \
        sparse.dpk.to(pk.dtype), sparse.dpv.to(pv.dtype), \
        None, None, None, None, None


__all__ = [
    'SparseGradients',
    'attention_backward',
    'compute_delta',
    'deterministic_sparse_dq',
    'direct_sparse_backward',
    'launch_branch',
    'merge_gradient',
]
