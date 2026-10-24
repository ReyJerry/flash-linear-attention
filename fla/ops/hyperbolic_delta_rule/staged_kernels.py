# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Strict-precision staged WY kernels; equations are in docs/hyperbolic_staged.md."""

import triton
import triton.language as tl

from fla.ops.hyperbolic_delta_rule.tensor_core_kernels import _mm, _wy_weights


@triton.jit
def _chunk_metadata(Cu, slot, N: tl.constexpr, BT: tl.constexpr):
    low = tl.full((), 0, tl.int64)
    high = tl.full((), N, tl.int64)
    while high - low > 1:
        middle = (low + high) // 2
        base = tl.load(Cu + middle).to(tl.int64) // BT + middle
        right = base <= slot
        low = tl.where(right, middle, low)
        high = tl.where(right, high, middle)
    bos = tl.load(Cu + low).to(tl.int64)
    eos = tl.load(Cu + low + 1).to(tl.int64)
    start = bos + (slot - bos // BT - low) * BT
    return low, eos, start


@triton.jit
def hyperbolic_staged_prepare_fwd_kernel(
    Q, Key, Cu, Gram, QueryKey,
    H: tl.constexpr, K: tl.constexpr, N: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr,
):
    program = tl.program_id(0).to(tl.int64)
    slot, head = program // H, program % H
    _, eos, start = _chunk_metadata(Cu, slot, N, BT)
    if start < eos:
        rows = tl.arange(0, BT)
        cols = tl.arange(0, BT)
        ik = tl.arange(0, BK).to(tl.int64)
        token = start + rows
        offsets = (token[:, None] * H + head) * K + ik[None, :]
        mask = (token[:, None] < eos) & (ik[None, :] < K)
        key = tl.load(Key + offsets, mask, 0).to(tl.float32)
        query = tl.load(Q + offsets, mask, 0).to(tl.float32)
        gram = _mm(key, tl.trans(key), False, False, False)
        query_key = _mm(query, tl.trans(key), False, False, False)
        offsets = ((slot * H + head) * BT + rows[:, None]) * BT + cols[None, :]
        tl.store(Gram + offsets, gram)
        tl.store(QueryKey + offsets, tl.where(rows[:, None] >= cols[None, :], query_key, 0.0))


@triton.jit
def hyperbolic_staged_state_fwd_kernel(
    Key, Val, Beta, H0, Cu, Curvature, Gram, Weight, Checkpoints, HT,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
):
    nh = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    sequence, head = nh // H, nh % H
    bos = tl.load(Cu + sequence).to(tl.int64)
    eos = tl.load(Cu + sequence + 1).to(tl.int64)
    c = tl.load(Curvature).to(tl.float32)
    rows, cols = tl.arange(0, BT), tl.arange(0, BT)
    ik = tl.arange(0, BK).to(tl.int64)
    iv = tile * BV + tl.arange(0, BV)
    state_offsets = (nh * K + ik[:, None]) * V + iv[None, :]
    state_mask = (ik[:, None] < K) & (iv[None, :] < V)
    state = tl.load(H0 + state_offsets, state_mask, 0).to(tl.float32)
    for block_start in range(0, eos - bos, BT):
        slot = bos // BT + sequence + block_start // BT
        checkpoint = ((slot * H + head) * K + ik[:, None]) * V + iv[None, :]
        tl.store(Checkpoints + checkpoint, state, state_mask)
        token = bos + block_start + rows
        valid = token < eos
        key = tl.load(Key + (token[:, None] * H + head) * K + ik[None, :],
                      valid[:, None] & (ik[None, :] < K), 0).to(tl.float32)
        offsets = (token[:, None] * H + head) * V + iv[None, :]
        mask = valid[:, None] & (iv[None, :] < V)
        value = tl.load(Val + offsets, mask, 0).to(tl.float32)
        beta = tl.load(Beta + offsets, mask, 0).to(tl.float32)
        gram = tl.load(Gram + ((slot * H + head) * BT + rows[:, None]) * BT + cols[None, :])
        weight, _, _, _, _ = _wy_weights(gram, key, value, beta, state, c, valid, False, False, False, BT)
        tl.store(Weight + offsets, weight, mask)
        denominator = 1.0 + c * tl.cumsum(value * weight, axis=0)
        last = tl.minimum(BT, eos - bos - block_start) - 1
        final_denominator = tl.sum(tl.where(rows[:, None] == last, denominator, 0.0), axis=0)
        state = (state + _mm(tl.trans(key), weight, False, False, False)) / final_denominator[None, :]
    tl.store(HT + state_offsets, state, state_mask)


@triton.jit
def hyperbolic_staged_output_fwd_kernel(
    Q, Val, Cu, Curvature, QueryKey, Weight, Checkpoints, Out,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, N: tl.constexpr,
    SCALE: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
):
    program = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    slot, head = program // H, program % H
    _, eos, start = _chunk_metadata(Cu, slot, N, BT)
    if start < eos:
        rows, cols = tl.arange(0, BT), tl.arange(0, BT)
        ik = tl.arange(0, BK).to(tl.int64)
        iv = tile * BV + tl.arange(0, BV)
        token = start + rows
        valid = token < eos
        query = tl.load(Q + (token[:, None] * H + head) * K + ik[None, :],
                        valid[:, None] & (ik[None, :] < K), 0).to(tl.float32)
        state = tl.load(Checkpoints + ((slot * H + head) * K + ik[:, None]) * V + iv[None, :],
                        (ik[:, None] < K) & (iv[None, :] < V), 0).to(tl.float32)
        offsets = (token[:, None] * H + head) * V + iv[None, :]
        mask = valid[:, None] & (iv[None, :] < V)
        weight = tl.load(Weight + offsets, mask, 0)
        value = tl.load(Val + offsets, mask, 0).to(tl.float32)
        c = tl.load(Curvature).to(tl.float32)
        denominator = 1.0 + c * tl.cumsum(value * weight, axis=0)
        query_key = tl.load(QueryKey + ((slot * H + head) * BT + rows[:, None]) * BT + cols[None, :])
        numerator = _mm(query, state, False, False, False) + _mm(query_key, weight, False, False, False)
        tl.store(Out + offsets, SCALE * numerator / denominator, mask)


@triton.jit
def hyperbolic_staged_output_bwd_kernel(
    Q, Val, Cu, Curvature, QueryKey, Weight, Checkpoints, DOut, DNumerator, Z, Suffix,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, N: tl.constexpr,
    SCALE: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
):
    program = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    slot, head = program // H, program % H
    _, eos, start = _chunk_metadata(Cu, slot, N, BT)
    if start < eos:
        rows, cols = tl.arange(0, BT), tl.arange(0, BT)
        ik = tl.arange(0, BK).to(tl.int64)
        iv = tile * BV + tl.arange(0, BV)
        token = start + rows
        valid = token < eos
        query = tl.load(Q + (token[:, None] * H + head) * K + ik[None, :],
                        valid[:, None] & (ik[None, :] < K), 0).to(tl.float32)
        state = tl.load(Checkpoints + ((slot * H + head) * K + ik[:, None]) * V + iv[None, :],
                        (ik[:, None] < K) & (iv[None, :] < V), 0).to(tl.float32)
        offsets = (token[:, None] * H + head) * V + iv[None, :]
        mask = valid[:, None] & (iv[None, :] < V)
        weight = tl.load(Weight + offsets, mask, 0)
        value = tl.load(Val + offsets, mask, 0).to(tl.float32)
        do = tl.load(DOut + offsets, mask, 0).to(tl.float32)
        c = tl.load(Curvature).to(tl.float32)
        denominator = 1.0 + c * tl.cumsum(value * weight, axis=0)
        query_key = tl.load(QueryKey + ((slot * H + head) * BT + rows[:, None]) * BT + cols[None, :])
        numerator = _mm(query, state, False, False, False) + _mm(query_key, weight, False, False, False)
        output = SCALE * numerator / denominator
        d = SCALE * do / denominator
        dz = -do * output / denominator
        suffix = tl.sum(dz, axis=0)[None, :] - tl.cumsum(dz, axis=0) + dz
        tl.store(DNumerator + offsets, d, mask)
        tl.store(Z + offsets, _mm(tl.trans(query_key), d, False, False, False), mask)
        tl.store(Suffix + offsets, suffix, mask)


@triton.jit
def hyperbolic_staged_state_bwd_kernel(
    Q, Key, Val, Beta, Cu, Curvature, Gram, Weight, Checkpoints, HT, DHT,
    DNumerator, Z, U, Suffix, StateAdjoint, DH0,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
):
    nh = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    sequence, head = nh // H, nh % H
    bos = tl.load(Cu + sequence).to(tl.int64)
    eos = tl.load(Cu + sequence + 1).to(tl.int64)
    c = tl.load(Curvature).to(tl.float32)
    rows, cols = tl.arange(0, BT), tl.arange(0, BT)
    ik = tl.arange(0, BK).to(tl.int64)
    iv = tile * BV + tl.arange(0, BV)
    state_offsets = (nh * K + ik[:, None]) * V + iv[None, :]
    state_mask = (ik[:, None] < K) & (iv[None, :] < V)
    ds = tl.load(DHT + state_offsets, state_mask, 0).to(tl.float32)
    send = tl.load(HT + state_offsets, state_mask, 0).to(tl.float32)
    blocks = tl.cdiv(eos - bos, BT)
    for reverse in range(blocks):
        block = blocks - 1 - reverse
        slot = bos // BT + sequence + block
        token = bos + block * BT + rows
        valid = token < eos
        key_offsets = (token[:, None] * H + head) * K + ik[None, :]
        key_mask = valid[:, None] & (ik[None, :] < K)
        key = tl.load(Key + key_offsets, key_mask, 0).to(tl.float32)
        query = tl.load(Q + key_offsets, key_mask, 0).to(tl.float32)
        checkpoint = ((slot * H + head) * K + ik[:, None]) * V + iv[None, :]
        state = tl.load(Checkpoints + checkpoint, state_mask, 0)
        offsets = (token[:, None] * H + head) * V + iv[None, :]
        mask = valid[:, None] & (iv[None, :] < V)
        value = tl.load(Val + offsets, mask, 0).to(tl.float32)
        beta = tl.load(Beta + offsets, mask, 0).to(tl.float32)
        weight = tl.load(Weight + offsets, mask, 0)
        d = tl.load(DNumerator + offsets, mask, 0)
        dw_readout = tl.load(Z + offsets, mask, 0)
        suffix = tl.load(Suffix + offsets, mask, 0)
        gram = tl.load(Gram + ((slot * H + head) * BT + rows[:, None]) * BT + cols[None, :])
        denominator = 1.0 + c * tl.cumsum(value * weight, axis=0)
        last = tl.minimum(BT, eos - bos - block * BT) - 1
        final_denominator = tl.sum(tl.where(rows[:, None] == last, denominator, 0.0), axis=0)
        x = ds / final_denominator[None, :]
        suffix -= tl.sum(ds * send, axis=0)[None, :] / final_denominator[None, :]
        zz = dw_readout + _mm(key, x, False, False, False) + c * value * suffix
        a_raw = tl.sum(key * key, axis=1)[:, None] - c * value * value
        active = valid[:, None] & (a_raw > 0.0)
        alpha = tl.where(active, beta / tl.where(active, a_raw, 1.0), 0.0)
        for j in range(BT):
            index = BT - 1 - j
            zj = tl.sum(tl.where(rows[:, None] == index, zz, 0.0), axis=0)
            gj = tl.sum(tl.where(rows[:, None] == index, gram, 0.0), axis=0)
            aj = tl.sum(tl.where(rows[:, None] == index, alpha, 0.0), axis=0)
            vj = tl.sum(tl.where(rows[:, None] == index, value, 0.0), axis=0)
            correction = aj[None, :] * (gj[:, None] - c * vj[None, :] * value) * zj[None, :]
            zz -= tl.where(rows[:, None] < index, correction, 0.0)
        u = alpha * zz
        tl.store(Z + offsets, zz, mask)
        tl.store(U + offsets, u, mask)
        tl.store(Suffix + offsets, suffix, mask)
        tl.store(StateAdjoint + checkpoint, x, state_mask)
        ds = _mm(tl.trans(query), d, False, False, False) + x - _mm(tl.trans(key), u, False, False, False)
        send = state
    tl.store(DH0 + state_offsets, ds, state_mask)


@triton.jit
def hyperbolic_staged_values_bwd_kernel(
    Key, Val, Beta, Cu, Curvature, Gram, Weight, Checkpoints, Z, U, Suffix, DV, DBeta, DC,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, N: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, NEED_DC: tl.constexpr,
):
    program = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    slot, head = program // H, program % H
    _, eos, start = _chunk_metadata(Cu, slot, N, BT)
    dc = tl.full((), 0.0, tl.float32)
    if start < eos:
        rows, cols = tl.arange(0, BT), tl.arange(0, BT)
        token = start + rows
        valid = token < eos
        iv = tile * BV + tl.arange(0, BV).to(tl.int64)
        full_ik = tl.arange(0, BK).to(tl.int64)
        full_key = tl.load(Key + (token[:, None] * H + head) * K + full_ik[None, :],
                           valid[:, None] & (full_ik[None, :] < K), 0).to(tl.float32)
        # Preserve the full-Key feasibility reduction used by both state scans.
        norm = tl.sum(full_key * full_key, axis=1)
        projected = tl.full((BT, BV), 0.0, tl.float32)
        for key_tile in range(tl.cdiv(K, 32)):
            ik = key_tile * 32 + tl.arange(0, 32).to(tl.int64)
            key_offsets = (token[:, None] * H + head) * K + ik[None, :]
            key_mask = valid[:, None] & (ik[None, :] < K)
            key = tl.load(Key + key_offsets, key_mask, 0).to(tl.float32)
            checkpoint = ((slot * H + head) * K + ik[:, None]) * V + iv[None, :]
            state_mask = (ik[:, None] < K) & (iv[None, :] < V)
            state = tl.load(Checkpoints + checkpoint, state_mask, 0)
            projected += _mm(key, state, False, False, False)
        gram = tl.load(Gram + ((slot * H + head) * BT + rows[:, None]) * BT + cols[None, :])
        gram_lower = tl.where(rows[:, None] > cols[None, :], gram, 0.0)
        c = tl.load(Curvature).to(tl.float32)
        offsets = (token[:, None] * H + head) * V + iv[None, :]
        mask = valid[:, None] & (iv[None, :] < V)
        value = tl.load(Val + offsets, mask, 0).to(tl.float32)
        beta = tl.load(Beta + offsets, mask, 0).to(tl.float32)
        weight = tl.load(Weight + offsets, mask, 0)
        zz = tl.load(Z + offsets, mask, 0)
        u = tl.load(U + offsets, mask, 0)
        suffix = tl.load(Suffix + offsets, mask, 0)
        vw = value * weight
        prefix_vw = tl.cumsum(vw, axis=0) - vw
        uv = u * value
        suffix_uv = tl.sum(uv, axis=0)[None, :] - tl.cumsum(uv, axis=0)
        residual = value - projected
        gram_weight = _mm(gram_lower, weight, False, False, False) - c * value * prefix_vw
        d_alpha = zz * (residual - gram_weight)
        a_raw = norm[:, None] - c * value * value
        active = valid[:, None] & (a_raw > 0.0)
        a = tl.where(active, a_raw, 1.0)
        da = tl.where(active, -d_alpha * beta / (a * a), 0.0)
        dv = c * weight * suffix + c * (u * prefix_vw + weight * suffix_uv) + u - 2.0 * c * value * da
        db = tl.where(active, d_alpha / a, 0.0)
        tl.store(DV + offsets, dv, mask)
        tl.store(DBeta + offsets, db, mask)
        # The solve result has no consumer after d_alpha; reuse it for the next stage.
        tl.store(Z + offsets, da, mask)
        if NEED_DC:
            dc = tl.sum(tl.sum(value * weight * suffix + u * value * prefix_vw - da * value * value, axis=0), axis=0)
    if NEED_DC:
        tl.store(DC + program * tl.cdiv(V, BV) + tile, dc)


@triton.jit
def hyperbolic_staged_parameters_bwd_kernel(
    Q, Key, Cu, Weight, Checkpoints, DNumerator, DA, U, StateAdjoint, DQ, DK,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, N: tl.constexpr,
    BT: tl.constexpr, BV: tl.constexpr,
):
    program = tl.program_id(0).to(tl.int64)
    slot, head = program // H, program % H
    _, eos, start = _chunk_metadata(Cu, slot, N, BT)
    if start < eos:
        rows, cols = tl.arange(0, BT), tl.arange(0, BT)
        ik = tl.program_id(1).to(tl.int64) * 32 + tl.arange(0, 32).to(tl.int64)
        token = start + rows
        valid = token < eos
        key_offsets = (token[:, None] * H + head) * K + ik[None, :]
        key_mask = valid[:, None] & (ik[None, :] < K)
        key = tl.load(Key + key_offsets, key_mask, 0).to(tl.float32)
        query = tl.load(Q + key_offsets, key_mask, 0).to(tl.float32)
        dp = tl.full((BT, BT), 0.0, tl.float32)
        dg = tl.full((BT, BT), 0.0, tl.float32)
        da_sum = tl.full((BT,), 0.0, tl.float32)
        for tile in tl.range(tl.cdiv(V, BV), num_stages=1, loop_unroll_factor=1, disallow_acc_multi_buffer=True):
            iv = tile * BV + tl.arange(0, BV).to(tl.int64)
            offsets = (token[:, None] * H + head) * V + iv[None, :]
            mask = valid[:, None] & (iv[None, :] < V)
            weight = tl.load(Weight + offsets, mask, 0)
            d = tl.load(DNumerator + offsets, mask, 0)
            u = tl.load(U + offsets, mask, 0)
            da = tl.load(DA + offsets, mask, 0)
            dp += _mm(d, tl.trans(weight), False, False, False)
            dg -= _mm(u, tl.trans(weight), False, False, False)
            da_sum += tl.sum(da, axis=1)
        dp = tl.where(rows[:, None] >= cols[None, :], dp, 0.0)
        dg = tl.where(rows[:, None] > cols[None, :], dg, 0.0)
        dq = _mm(dp, key, False, False, False)
        dk = _mm(tl.trans(dp), query, False, False, False)
        dk += _mm(dg + tl.trans(dg), key, False, False, False) + 2.0 * key * da_sum[:, None]
        for tile in tl.range(tl.cdiv(V, BV), num_stages=1, loop_unroll_factor=1, disallow_acc_multi_buffer=True):
            iv = tile * BV + tl.arange(0, BV).to(tl.int64)
            checkpoint = ((slot * H + head) * K + ik[:, None]) * V + iv[None, :]
            state_mask = (ik[:, None] < K) & (iv[None, :] < V)
            state = tl.load(Checkpoints + checkpoint, state_mask, 0)
            x = tl.load(StateAdjoint + checkpoint, state_mask, 0)
            offsets = (token[:, None] * H + head) * V + iv[None, :]
            mask = valid[:, None] & (iv[None, :] < V)
            weight = tl.load(Weight + offsets, mask, 0)
            d = tl.load(DNumerator + offsets, mask, 0)
            u = tl.load(U + offsets, mask, 0)
            dq += _mm(d, tl.trans(state), False, False, False)
            dk += _mm(weight, tl.trans(x), False, False, False) - _mm(u, tl.trans(state), False, False, False)
        tl.store(DQ + key_offsets, dq, key_mask)
        tl.store(DK + key_offsets, dk, key_mask)
