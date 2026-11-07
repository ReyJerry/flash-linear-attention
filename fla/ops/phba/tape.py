# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Backward-local storage-dtype dS tape, bounded by shape and reused across head groups."""
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from fla.ops.phba.chunk_bwd import OWNER_NUM_STAGES, OWNER_NUM_WARPS
from fla.ops.phba.plans import (
    OwnerBranchTapePlan,
    OwnerTapePlan,
    owner_tape_index32_safe,
    owner_tape_producer_stages,
    owner_tape_uses_tile_major,
)


@triton.jit(do_not_specialize=['NC', 'HEAD_START'])
def _group_count_kernel(count, group_count, group_offset, NC, HEAD_START,
                        H: tl.constexpr, GROUP_SIZE: tl.constexpr, BT: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT).to(tl.int64)
    block = index // GROUP_SIZE
    head = tl.cast(HEAD_START, tl.int64) + index % GROUP_SIZE
    valid = (block < NC) & (head < H)
    value = tl.load(count + block * H + head, valid, other=0)
    tl.store(group_count + index, value, index < tl.cast(NC, tl.int64) * GROUP_SIZE)
    if tl.program_id(0) == 0:
        tl.store(group_offset, 0)


@triton.jit(do_not_specialize=['N'])
def _group_offset_kernel(count, offset, N, BT: tl.constexpr):
    start = tl.program_id(0).to(tl.int64) * BT
    lane = tl.arange(0, BT).to(tl.int64)
    carry = tl.full((), 0, tl.int32)
    previous = tl.full((), 0, tl.int64)
    while previous < start:
        value = tl.load(count + previous + lane, previous + lane < N, other=0)
        carry += tl.sum(value, axis=0)
        previous += BT
    current = tl.load(count + start + lane, start + lane < N, other=0)
    inclusive = tl.cumsum(current, axis=0) + carry
    tl.store(offset + start + lane + 1, inclusive, start + lane < N)


@triton.jit(do_not_specialize=['T', 'K', 'RH', 'RT', 'HEAD_START', 'SIZE'])
def _route_edge_map_kernel(route, offset, rho, group_offset, edge_map,
                           T, K, RH, RT, HEAD_START, SIZE,
                           H: tl.constexpr, GROUP_SIZE: tl.constexpr, MAP_WIDTH: tl.constexpr, BT: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT).to(tl.int64)
    slot = index % MAP_WIDTH
    row = (index // MAP_WIDTH) % T
    local_head = index // (tl.cast(T, tl.int64) * MAP_WIDTH)
    head = tl.cast(HEAD_START, tl.int64) + local_head
    valid = (index < SIZE) & (head < H) & (slot < K)
    block = tl.load(route + head * RH + row * RT + slot, valid, other=-1).to(tl.int64)
    valid = valid & (block >= 0)
    safe_block = tl.maximum(block, 0)
    expert = safe_block * H + head
    begin = tl.load(offset + expert, valid, other=0).to(tl.int64)
    finish = tl.load(offset + expert + 1, valid, other=0).to(tl.int64)
    lower = begin
    upper = finish
    target = row * H + head
    while tl.sum((lower < upper).to(tl.int32), axis=0) > 0:
        middle = (lower + upper) // 2
        searching = lower < upper
        value = tl.load(rho + middle, searching, other=2147483647).to(tl.int64)
        right = searching & (value < target)
        lower = tl.where(right, middle + 1, lower)
        upper = tl.where(searching & ~right, middle, upper)
    found_value = tl.load(rho + lower, valid & (lower < finish), other=-1).to(tl.int64)
    found = valid & (lower < finish) & (found_value == target)
    group_base = tl.load(group_offset + safe_block * GROUP_SIZE + local_head, found, other=0).to(tl.int64)
    edge = tl.where(found, group_base + lower - begin, -1)
    tl.store(edge_map + index, edge.to(tl.int32), index < SIZE)


@triton.jit(do_not_specialize=['T', 'HEAD_START', 'TAPE_EDGES'])
def _sparse_bwd_tape_kernel(
    q, kmem, vmem, do, lse, delta, count, offset, rho, block_ptr, prefix_valid,
    dq, dk, dv, tape, group_offset, HEAD_START, T, scale,
    sqt, sqh, skt, skh, svt, svh, sdt, sdh,
    spb, spm, sph, sgb, sgm, sgh, TAPE_EDGES,
    H: tl.constexpr, D: tl.constexpr, M: tl.constexpr,
    BC: tl.constexpr, BQ: tl.constexpr, BD: tl.constexpr, BRANCH: tl.constexpr,
    GROUP_SIZE: tl.constexpr, TAPE_STRIDE: tl.constexpr, INDEX32: tl.constexpr,
    WRITE_DK: tl.constexpr = True, TILE_MAJOR: tl.constexpr = True,
    TASK_ORDER=None, ORDERED: tl.constexpr = False,
):
    if ORDERED:
        group_expert = tl.load(TASK_ORDER + tl.program_id(1).to(tl.int64)).to(tl.int64)
    else:
        group_expert = tl.program_id(0).to(tl.int64)
    block = group_expert // GROUP_SIZE
    h = tl.cast(HEAD_START, tl.int64) + group_expert % GROUP_SIZE
    if h >= H:
        return
    e = block * H + h
    tile_id = tl.program_id(0) if ORDERED else tl.program_id(1)
    if INDEX32:
        tile = tile_id.to(tl.int32)
    else:
        tile = tile_id.to(tl.int64)
    n = tl.load(count + e)
    if n == 0:
        return
    mem = tile * BC + tl.arange(0, BC)
    d = tl.arange(0, BD)
    if BRANCH == 0:
        start = tl.load(block_ptr + block)
        end = tl.load(block_ptr + block + 1)
        if start.to(tl.int64) + tile.to(tl.int64) * BC >= end:
            return
        ix = start.to(tl.int64) + mem.to(tl.int64)
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
    if WRITE_DK:
        dk_acc = tl.zeros((BC, BD), tl.float32)
    dv_acc = tl.zeros((BC, BD), tl.float32)
    group_begin = tl.load(group_offset + group_expert).to(tl.int64)
    begin = tl.load(offset + e)
    finish = tl.load(offset + e + 1)
    if not INDEX32:
        begin = begin.to(tl.int64)
        finish = finish.to(tl.int64)
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
        edge = group_begin + pos.to(tl.int64) - tl.cast(begin, tl.int64)
        if TILE_MAJOR:
            tape_address = (tile.to(tl.int64) * tl.cast(TAPE_EDGES, tl.int64) + edge[:, None]) * BC
            tape_address += tl.arange(0, BC).to(tl.int64)[None, :]
        else:
            tape_address = edge[:, None] * TAPE_STRIDE + mem[None, :]
        tl.store(tape + tape_address, ds_st, valid_q[:, None] & valid_mem[None, :])
        if WRITE_DK:
            dk_acc += tl.dot(tl.trans(ds_st), qt, out_dtype=tl.float32) * scale
        dv_acc += tl.dot(tl.trans(prob.to(dot.dtype)), dot, out_dtype=tl.float32)
    if WRITE_DK:
        tl.store(dk + grad_addr, dk_acc, mm)
    tl.store(dv + grad_addr, dv_acc, mm)


@triton.jit(do_not_specialize=['T', 'K', 'RH', 'RT', 'HEAD_START', 'TAPE_EDGES'])
def _owner_from_tape_kernel(
    q, kmem, route, edge_map, tape, block_ptr, prefix_valid, dq,
    T, K, RH, RT, HEAD_START, SKT, SKH, SPB, SPM, SPH, SCALE, TAPE_EDGES,
    H: tl.constexpr, D: tl.constexpr, M: tl.constexpr, C: tl.constexpr,
    BQ: tl.constexpr, BC: tl.constexpr, BD: tl.constexpr, KP: tl.constexpr, BRANCH: tl.constexpr,
    MAP_WIDTH: tl.constexpr, TAPE_STRIDE: tl.constexpr, TILE_MAJOR: tl.constexpr = True,
):
    start = tl.program_id(0).to(tl.int64) * BQ
    local_head = tl.program_id(1).to(tl.int64)
    head = tl.cast(HEAD_START, tl.int64) + local_head
    if head >= H:
        return
    lane = tl.arange(0, BQ)
    rows = start + lane
    qvalid = rows < T
    d = tl.arange(0, BD)
    qm = qvalid[:, None] & (d[None, :] < D)
    acc = tl.zeros((BQ, BD), tl.float32)
    flat = tl.arange(0, BQ * KP)
    flat_row = start + flat // KP
    flat_slot = flat % KP
    ids = tl.load(route + head * RH + flat_row * RT + flat_slot,
                  (flat_row < T) & (flat_slot < K), other=-1)
    remaining = ids
    position = tl.min(tl.where(remaining >= 0, flat, BQ * KP), 0)
    memory_lane = tl.arange(0, BC)
    slots = tl.arange(0, KP)
    while position < BQ * KP:
        selected = position.to(tl.int64)
        candidate = tl.load(route + head * RH + (start + selected // KP) * RT + selected % KP)
        leader = candidate >= 0
        safe = tl.where(leader, candidate, 0)
        matches = tl.reshape(ids, (BQ, KP)) == safe
        first_slot = tl.min(tl.where(matches, slots[None, :], KP), axis=1)
        member = first_slot < K
        edge_addr = (local_head * T + rows) * MAP_WIDTH + first_slot.to(tl.int64)
        edge = tl.load(edge_map + edge_addr, qvalid & member & leader, other=-1).to(tl.int64)
        row_valid = qvalid & member & leader & (edge >= 0)
        permitted = leader
        if BRANCH == 1:
            permitted = permitted & (tl.load(prefix_valid + safe.to(tl.int64)) != 0)
        if permitted:
            if BRANCH == 0:
                begin = tl.load(block_ptr + safe.to(tl.int64))
                end = tl.load(block_ptr + safe.to(tl.int64) + 1)
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
                else:
                    kaddr = safe.to(tl.int64) * SPB + memory.to(tl.int64)[:, None] * SPM + head * SPH + d[None, :]
                kt = tl.load(kmem + kaddr, mask, other=0)
                if TILE_MAJOR:
                    tape_address = (tl.cast(m0, tl.int64) // BC * tl.cast(TAPE_EDGES, tl.int64)
                                    + tl.maximum(edge, 0)[:, None]) * BC + memory_lane.to(tl.int64)[None, :]
                else:
                    tape_address = tl.maximum(edge, 0)[:, None] * TAPE_STRIDE + memory[None, :]
                ds_st = tl.load(tape + tape_address, row_valid[:, None] & valid_memory[None, :], other=0)
                acc += tl.dot(ds_st, kt, out_dtype=tl.float32) * SCALE
        remaining = tl.where(remaining == candidate, -1, remaining)
        position = tl.min(tl.where(remaining >= 0, flat, BQ * KP), 0)
    tl.store(dq + (rows[:, None] * H + head) * D + d[None, :], acc, qm)


@dataclass(frozen=True)
class TapeWorkspace:
    tape: torch.Tensor
    edge_map: torch.Tensor
    count: torch.Tensor
    offset: torch.Tensor

    @property
    def nbytes(self):
        return sum(value.numel() * value.element_size() for value in (self.tape, self.edge_map, self.count, self.offset))


def allocate_tape_workspace(q, plan: OwnerBranchTapePlan):
    if not plan.enabled:
        raise ValueError('disabled owner tape plan must use the original algorithm')
    workspace = TapeWorkspace(
        torch.empty(plan.tape_elements, dtype=q.dtype, device=q.device),
        torch.empty(plan.map_elements, dtype=torch.int32, device=q.device),
        torch.empty(plan.count_elements, dtype=torch.int32, device=q.device),
        torch.empty(plan.offset_elements, dtype=torch.int32, device=q.device),
    )
    if workspace.nbytes != plan.workspace_bytes:
        raise ValueError('owner tape dtype/allocation differs from the forward-bound byte plan')
    return workspace


def build_group_metadata(route, meta, workspace, plan: OwnerTapePlan, *, head_start):
    h, t, width = route.shape
    if width == 0:
        raise ValueError('zero-width token branch must not build tape metadata')
    nc = meta.count.numel() // h
    _group_count_kernel[(triton.cdiv(plan.count_elements, 128),)](
        meta.count, workspace.count, workspace.offset, nc, head_start,
        H=h, GROUP_SIZE=plan.group_size, BT=128, num_warps=4, num_stages=1,
    )
    _group_offset_kernel[(triton.cdiv(plan.count_elements, 128),)](
        workspace.count, workspace.offset, plan.count_elements, BT=128, num_warps=4, num_stages=1,
    )
    _route_edge_map_kernel[(triton.cdiv(plan.map_elements, 256),)](
        route, meta.offset, meta.rho, workspace.offset, workspace.edge_map,
        t, width, route.stride(0), route.stride(1), head_start, plan.map_elements,
        H=h, GROUP_SIZE=plan.group_size, MAP_WIDTH=plan.map_width, BT=256, num_warps=4, num_stages=1,
    )


def run_tape_group(q, k, v, dout, lse, delta, route, meta, block_ptr, prefix_valid, dq, dk, dv,
                   workspace, plan: OwnerTapePlan, *, head_start, branch, chunk_size, num_slots, bc, bq,
                   index32: bool | None = None):
    """Compute one deterministic producer/consumer head group using its saved forward plan."""
    producer_stages = owner_tape_producer_stages(plan, bc)
    tile_major = owner_tape_uses_tile_major(plan, branch)
    t, h, d = q.shape
    extent = chunk_size if branch == 0 else num_slots
    tape_stride = plan.token_extent if branch == 0 else plan.state_extent
    if meta.capacity != t * h * route.shape[-1]:
        raise ValueError('canonical capacity differs from forward-bound route geometry')
    safe32 = owner_tape_index32_safe(meta.capacity, tape_stride, bq)
    if index32 is None:
        index32 = safe32
    elif index32 and not safe32:
        raise ValueError('32-bit producer indices require a padding-safe capacity')
    build_group_metadata(route, meta, workspace, plan, head_start=head_start)
    prefix_strides = k.stride()[:3] if branch else (0, 0, 0)
    gradient_strides = dk.stride()[:3] if branch else (0, 0, 0)
    nc = meta.count.numel() // h
    # reorder only independent CTAs for the validated 8K geometry
    # retain empty CTAs and each CTA's canonical row traversal
    ordered = (t == 8192 and h == 16 and d == 128 and num_slots == 128 and bq == 16 and bc == 32
               and chunk_size in (64, 128, 256, 512) and nc * plan.group_size <= 2048
               and workspace.nbytes <= 255 * 1024**2)
    task_order = (torch.argsort(workspace.count[:nc * plan.group_size], descending=True, stable=True)
                  if ordered else None)
    producer_grid = ((triton.cdiv(extent, bc), nc * plan.group_size, 1) if ordered
                     else (nc * plan.group_size, triton.cdiv(extent, bc), 1))
    _sparse_bwd_tape_kernel[producer_grid](
        q, k, v, dout, lse, delta, meta.count, meta.offset, meta.rho, block_ptr, prefix_valid.view(torch.uint8),
        dq, dk, dv, workspace.tape, workspace.offset, head_start, t, d ** -0.5,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
        dout.stride(0), dout.stride(1), *prefix_strides, *gradient_strides,
        H=h, D=d, M=num_slots, BC=bc, BQ=bq, BD=max(16, triton.next_power_of_2(d)), BRANCH=branch,
        GROUP_SIZE=plan.group_size, TAPE_STRIDE=tape_stride, INDEX32=index32,
        TAPE_EDGES=t * plan.group_size * route.shape[-1],
        WRITE_DK=True, TILE_MAJOR=tile_major, TASK_ORDER=task_order, ORDERED=ordered,
        num_warps=4, num_stages=producer_stages,
    )
    _owner_from_tape_kernel[(triton.cdiv(t, 16), plan.group_size)](
        q, k, route, workspace.edge_map, workspace.tape, block_ptr, prefix_valid.view(torch.uint8), dq,
        t, route.shape[-1], route.stride(0), route.stride(1), head_start, k.stride(0), k.stride(1),
        *prefix_strides, d ** -0.5,
        H=h, D=d, M=num_slots, C=chunk_size, BQ=16, BC=bc, BD=max(16, triton.next_power_of_2(d)),
        KP=triton.next_power_of_2(route.shape[-1]), BRANCH=branch, MAP_WIDTH=plan.map_width,
        TAPE_STRIDE=tape_stride, TAPE_EDGES=t * plan.group_size * route.shape[-1],
        TILE_MAJOR=tile_major, num_warps=OWNER_NUM_WARPS, num_stages=OWNER_NUM_STAGES,
    )


def backward_with_tape(qr, qw, kr, value, pk, pv, dout, lse, delta, metadata, block_ptr, prefix_valid,
                       rtok, rstate, grads, plan: OwnerBranchTapePlan, *, chunk_size, bc, bq):
    t, h, _ = qr.shape
    geometry = (t, h, pk.shape[0], chunk_size, pk.shape[1], rtok.shape[-1], rstate.shape[-1], bc, qr.element_size())
    if plan.geometry != geometry:
        raise ValueError('owner tape invocation differs from forward-bound geometry')
    workspace = allocate_tape_workspace(qr, plan)
    branches = ((0, qr, kr, value, rtok, metadata.tok, grads.dq_rope, grads.dk_rope, grads.dv),
                (1, qw, pk, pv, rstate, metadata.state, grads.dq_raw, grads.dpk, grads.dpv))
    for branch, q, k, v, route, meta, dq, dk, dv in branches:
        if route.shape[-1] == 0:
            dq.zero_()
            continue
        branch_plan = plan.token if branch == 0 else plan.state
        if not branch_plan.enabled:
            raise ValueError('enabled owner plan lacks a required branch')
        for head_start in range(0, h, branch_plan.group_size):
            run_tape_group(q, k, v, dout, lse, delta, route, meta, block_ptr, prefix_valid, dq, dk, dv,
                           workspace, branch_plan, head_start=head_start, branch=branch,
                           chunk_size=chunk_size, num_slots=pk.shape[1], bc=bc, bq=bq)
    return grads
