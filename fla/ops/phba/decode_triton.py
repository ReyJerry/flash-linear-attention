# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Inference-only PHBA kernels with direct cache addressing and bounded workspaces."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["BLOCK"])
def _boundary_update(Raw, Value, Log, Write, SK, SV, PK, PV, BLOCK,
                     CAP: tl.constexpr, NB: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                     M: tl.constexpr, C: tl.constexpr, BM: tl.constexpr, BD: tl.constexpr):
    bh = tl.program_id(0).to(tl.int64)
    b, h = bh // H, bh % H
    slot = tl.program_id(1).to(tl.int64) * BM + tl.arange(0, BM)
    dim = tl.program_id(2).to(tl.int64) * BD + tl.arange(0, BD)
    tok = tl.arange(0, C).to(tl.int64)
    gate_addr = ((b * C + tok[:, None]) * H + h) * M + slot[None, :]
    g = tl.load(Log + gate_addr, slot[None, :] < M, other=0).to(tl.float32)
    cumulative = tl.cumsum(g, axis=0)
    last = tl.sum(tl.where(tok[:, None] == C - 1, cumulative, 0.0), axis=0)
    s = tl.load(Write + gate_addr, slot[None, :] < M, other=0).to(tl.float32)
    w = (s * tl.exp(last[None, :] - cumulative)).to(Raw.dtype.element_ty)
    raw_addr = ((b * C + tok[:, None]) * H + h) * D + dim[None, :]
    val_addr = ((b * CAP + tl.cast(BLOCK, tl.int64) * C + tok[:, None]) * H + h) * D + dim[None, :]
    k = tl.load(Raw + raw_addr, dim[None, :] < D, other=0)
    v = tl.load(Value + val_addr, dim[None, :] < D, other=0)
    state_addr = (bh * M + slot[:, None]) * D + dim[None, :]
    valid = (slot[:, None] < M) & (dim[None, :] < D)
    sk = tl.load(SK + state_addr, valid, other=0)
    sv = tl.load(SV + state_addr, valid, other=0)
    sk = sk * tl.exp(last)[:, None]
    sv = sv * tl.exp(last)[:, None]
    sk += tl.dot(tl.trans(w), k)
    sv += tl.dot(tl.trans(w), v)
    tl.store(SK + state_addr, sk, valid)
    tl.store(SV + state_addr, sv, valid)
    snap = (((b * NB + tl.cast(BLOCK, tl.int64) + 1) * M + slot[:, None]) * H + h) * D + dim[None, :]
    tl.store(PK + snap, sk, valid)
    tl.store(PV + snap, sv, valid)


def finish_block(cache, block):
    """Update both FP32 anchors and write the next low-precision prefix snapshot."""
    grid = (cache.batch * cache.heads, triton.cdiv(cache.slots, 32), triton.cdiv(cache.width, 64))
    _boundary_update[grid](
        cache.raw_tail, cache.values, cache.log_tail, cache.write_tail,
        cache.state_k, cache.state_v, cache.pk, cache.pv, block,
        CAP=cache.capacity, NB=cache.nblocks, H=cache.heads, D=cache.width,
        M=cache.slots, C=cache.chunk_size, BM=32, BD=64, num_warps=4, num_stages=1,
    )


@triton.jit(do_not_specialize=["FIRST"])
def _centroid_cache(Keys, Centers, FIRST, CAP: tl.constexpr, NB: tl.constexpr,
                    H: tl.constexpr, D: tl.constexpr, C: tl.constexpr, BD: tl.constexpr):
    bh = tl.program_id(0).to(tl.int64)
    b, h = bh // H, bh % H
    block = tl.cast(FIRST, tl.int64) + tl.program_id(1).to(tl.int64)
    lane = tl.arange(0, 64).to(tl.int64)
    d = tl.arange(0, BD).to(tl.int64)
    total = tl.full((BD,), 0, tl.float32)
    for start in range(0, C, 64):
        row = block * C + start + lane
        address = ((b * CAP + row[:, None]) * H + h) * D + d[None, :]
        value = tl.load(Keys + address, (start + lane[:, None] < C) & (d[None, :] < D), other=0)
        total += tl.sum(value.to(tl.float32), axis=0)
    target = ((b * NB + block) * H + h) * D + d
    tl.store(Centers + target, total / C, d < D)


def cache_centroids(cache, first, count):
    if count:
        _centroid_cache[(cache.batch * cache.heads, count)](
            cache.keys, cache.centroids, first, CAP=cache.capacity, NB=cache.nblocks,
            H=cache.heads, D=cache.width, C=cache.chunk_size,
            BD=triton.next_power_of_2(cache.width), num_warps=4,
        )


@triton.jit(do_not_specialize=["VALID"])
def _routing_scores(Q, Centers, Scores, VALID, NB: tl.constexpr, H: tl.constexpr,
                    D: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    bh = tl.program_id(0).to(tl.int64)
    b, h = bh // H, bh % H
    n = tl.program_id(1).to(tl.int64) * BN + tl.arange(0, BN)
    d = tl.arange(0, BD).to(tl.int64)
    q = tl.load(Q + bh * D + d, d < D, other=0).to(tl.float32)
    addr = ((b * NB + n[:, None]) * H + h) * D + d[None, :]
    center = tl.load(Centers + addr, (n[:, None] < VALID) & (d[None, :] < D), other=0)
    center = center.to(Q.dtype.element_ty).to(tl.float32)
    score = tl.sum(center * q[None, :], axis=1)
    tl.store(Scores + bh * NB + n, tl.where(n < VALID, score, -float("inf")), n < NB)


@triton.jit(do_not_specialize=["VALID"])
def _routing_topk(Scores, Routes, VALID, NB: tl.constexpr, KH: tl.constexpr, BN: tl.constexpr):
    bh = tl.program_id(0).to(tl.int64)
    n = tl.arange(0, BN)
    scores = tl.load(Scores + bh * NB + n.to(tl.int64), (n < NB) & (n < VALID), other=-float("inf"))
    for k in tl.static_range(KH):
        best = tl.max(scores, axis=0)
        idx = tl.min(tl.where(scores == best, n, 2147483647), axis=0)
        valid = (idx < VALID) & (best > -float("inf"))
        tl.store(Routes + bh * KH + k, tl.where(valid, idx, -1))
        scores = tl.where(n == idx, -float("inf"), scores)


def route_one(cache, query, count):
    """Two small launches: tiled scores and stable exact top-K; no history pooling."""
    if not cache.topk - 1:
        return cache.route_buffer
    query = query.contiguous()
    _routing_scores[(cache.batch * cache.heads, triton.cdiv(cache.nblocks, 32))](
        query, cache.centroids, cache.route_scores, count, NB=cache.nblocks, H=cache.heads,
        D=cache.width, BN=32, BD=triton.next_power_of_2(cache.width), num_warps=4,
    )
    _routing_topk[(cache.batch * cache.heads,)](
        cache.route_scores, cache.route_buffer, count, NB=cache.nblocks, KH=cache.topk - 1,
        BN=triton.next_power_of_2(cache.nblocks), num_warps=4,
    )
    return cache.route_buffer


@triton.jit
def _accumulate(m, z, u, score, values):
    maximum = tl.maximum(m, tl.max(score, axis=0))
    maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    scale = tl.where(z > 0, tl.exp(m - maximum), 0.0)
    p = tl.exp(score - maximum)
    term = tl.sum(p.to(values.dtype).to(tl.float32)[:, None] * values.to(tl.float32), axis=0)
    return maximum, scale * z + tl.sum(p, axis=0), scale * u + term


@triton.jit(do_not_specialize=["POS"])
def _split_partial(QRaw, QRope, Keys, Values, PK, PV, Routes, Partials, POS,
                   CAP: tl.constexpr, NB: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                   M: tl.constexpr, C: tl.constexpr, K: tl.constexpr, BC: tl.constexpr, BD: tl.constexpr,
                   HISTORY_TOKENS: tl.constexpr, HISTORY_STATES: tl.constexpr):
    bh = tl.program_id(0).to(tl.int64)
    route = tl.program_id(1).to(tl.int64)
    batch, head = bh // H, bh % H
    if route == 0:
        block = tl.cast(POS, tl.int64) // C
    else:
        block = tl.load(Routes + bh * (K-1) + route - 1).to(tl.int64)
    d = tl.arange(0, BD).to(tl.int64)
    qr = tl.load(QRope + bh * D + d, d < D, other=0).to(tl.float32)
    qw = tl.load(QRaw + bh * D + d, d < D, other=0).to(tl.float32)
    m = tl.full((), -float("inf"), tl.float32)
    z = tl.full((), 0, tl.float32)
    u = tl.full((BD,), 0, tl.float32)
    lane = tl.arange(0, BC).to(tl.int64)
    safe = tl.maximum(block, 0)
    if (block >= 0) & ((route == 0) | HISTORY_TOKENS):
        for start in range(0, C, BC):
            token = safe * C + start + lane
            valid = (start + lane < C) & (token <= POS) & (token < CAP)
            addr = ((batch * CAP + token[:, None]) * H + head) * D + d[None, :]
            key = tl.load(Keys + addr, valid[:, None] & (d[None, :] < D), other=0)
            value = tl.load(Values + addr, valid[:, None] & (d[None, :] < D), other=0)
            scores = tl.sum(key.to(tl.float32) * qr[None, :], axis=1) * (D ** -0.5)
            scores = tl.where(valid, scores, -float("inf"))
            m, z, u = _accumulate(m, z, u, scores, value)
    if (block > 0) & ((route == 0) | HISTORY_STATES):
        for start in range(0, M, BC):
            slot = start + lane
            valid = slot < M
            addr = (((batch * NB + safe) * M + slot[:, None]) * H + head) * D + d[None, :]
            key = tl.load(PK + addr, valid[:, None] & (d[None, :] < D), other=0)
            value = tl.load(PV + addr, valid[:, None] & (d[None, :] < D), other=0)
            scores = tl.sum(key.to(tl.float32) * qw[None, :], axis=1) * (D ** -0.5)
            scores = tl.where(valid, scores, -float("inf"))
            m, z, u = _accumulate(m, z, u, scores, value)
    base = (bh * K + route) * (D+2)
    tl.store(Partials + base, m)
    tl.store(Partials + base + 1, z)
    tl.store(Partials + base + 2 + d, u, d < D)


@triton.jit
def _split_merge(Partials, Out, K: tl.constexpr, D: tl.constexpr, BK: tl.constexpr, BD: tl.constexpr):
    bh = tl.program_id(0).to(tl.int64)
    k = tl.arange(0, BK).to(tl.int64)
    d = tl.arange(0, BD).to(tl.int64)
    base = (bh * K + k) * (D+2)
    m = tl.load(Partials + base, k < K, other=-float("inf"))
    z = tl.load(Partials + base + 1, k < K, other=0)
    u = tl.load(Partials + base[:, None] + 2 + d[None, :], (k[:, None] < K) & (d[None, :] < D), other=0)
    maximum = tl.max(tl.where(z > 0, m, -float("inf")), axis=0)
    weight = tl.where(z > 0, tl.exp(m - maximum), 0.0)
    denominator = tl.sum(weight * z, axis=0)
    result = tl.sum(weight[:, None] * u, axis=0) / denominator
    tl.store(Out + bh * D + d, result, d < D)


def split_read(cache, q_raw, q_rope, position, routes):
    """Stream selected cache blocks directly; merge unnormalized softmax statistics."""
    out = torch.empty(cache.batch, cache.heads, cache.width, device=cache.device, dtype=cache.dtype)
    _split_partial[(cache.batch * cache.heads, cache.topk)](
        q_raw.contiguous(), q_rope.contiguous(), cache.keys, cache.values, cache.pk, cache.pv,
        routes, cache.partials, position, CAP=cache.capacity, NB=cache.nblocks,
        H=cache.heads, D=cache.width, M=cache.slots, C=cache.chunk_size, K=cache.topk,
        BC=32, BD=triton.next_power_of_2(cache.width), HISTORY_TOKENS=cache.mode != "routed_states_only",
        HISTORY_STATES=cache.mode != "current_state_only", num_warps=4, num_stages=1,
    )
    _split_merge[(cache.batch * cache.heads,)](
        cache.partials, out, K=cache.topk, D=cache.width,
        BK=triton.next_power_of_2(cache.topk), BD=triton.next_power_of_2(cache.width), num_warps=4,
    )
    return out
