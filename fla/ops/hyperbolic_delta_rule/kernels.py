# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton kernels for canonical-gauge hyperbolic KV updates.

The recurrent backward recomputes one checkpoint interval at a time. It never
inverts a token transition (beta=1 is singular), and never materializes a full
[T,K,V] state tape. Chunk forward has a separate triangular system per value.
"""

import triton
import triton.language as tl


@triton.jit
def hyperbolic_delta_rule_fwd_kernel(
    Q, Kp, Vp, Beta, H0, Cu, O, HT, CP, Curvature,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    SCALE: tl.constexpr, BT: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr, SAVE: tl.constexpr,
):
    nh, tile_v = tl.program_id(0), tl.program_id(1)
    C = tl.load(Curvature).to(tl.float32)
    n, h = nh // H, nh % H
    bos, eos = tl.load(Cu + n), tl.load(Cu + n + 1)
    ik = tl.arange(0, BK)
    iv = tile_v * BV + tl.arange(0, BV)
    hs = (nh * K + ik[:, None]) * V + iv[None, :]
    mask = (ik[:, None] < K) & (iv[None, :] < V)
    s = tl.load(H0 + hs, mask, 0).to(tl.float32)
    for t in range(0, eos - bos):
        if SAVE:
            if t % BT == 0:
                cp_id = bos // BT + n + t // BT
                offset = ((cp_id * H + h) * K + ik[:, None]) * V + iv[None, :]
                tl.store(CP + offset, s, mask)
        kt = tl.load(Kp + ((bos + t) * H + h) * K + ik, ik < K, 0).to(tl.float32)
        qt = tl.load(Q + ((bos + t) * H + h) * K + ik, ik < K, 0).to(tl.float32)
        vo = ((bos + t) * H + h) * V + iv
        vt = tl.load(Vp + vo, iv < V, 0).to(tl.float32)
        beta = tl.load(Beta + vo, iv < V, 0).to(tl.float32)
        a = tl.sum(kt * kt, 0) - C * vt * vt
        r = vt - tl.sum(kt[:, None] * s, 0)
        u = beta * r / a
        den = 1.0 + C * vt * u
        s = (s + kt[:, None] * u[None, :]) / den[None, :]
        out = SCALE * tl.sum(qt[:, None] * s, 0)
        tl.store(O + vo, out, iv < V)
    tl.store(HT + hs, s, mask)


@triton.jit
def hyperbolic_delta_rule_chunk_fwd_kernel(
    Q, Kp, Vp, Beta, H0, Cu, O, HT, CP, Curvature, ReadoutQ, Readout,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    SCALE: tl.constexpr, BT: tl.constexpr,
    BK: tl.constexpr, SAVE: tl.constexpr, READOUT: tl.constexpr,
):
    # One (sequence, head, value) per program. No cross-value atomics.
    nh, value = tl.program_id(0), tl.program_id(1)
    C = tl.load(Curvature).to(tl.float32)
    n, h = nh // H, nh % H
    bos, eos = tl.load(Cu + n), tl.load(Cu + n + 1)
    rows, ik = tl.arange(0, BT), tl.arange(0, BK)
    hs = (nh * K + ik) * V + value
    s = tl.load(H0 + hs, ik < K, 0).to(tl.float32)
    for start in range(0, eos - bos, BT):
        if SAVE:
            cp_id = bos // BT + n + start // BT
            tl.store(CP + ((cp_id * H + h) * K + ik) * V + value, s, ik < K)
        token = bos + start + rows
        valid = token < eos
        kk = tl.load(
            Kp + (token[:, None] * H + h) * K + ik[None, :],
            valid[:, None] & (ik[None, :] < K), 0,
        ).to(tl.float32)
        qq = tl.load(
            Q + (token[:, None] * H + h) * K + ik[None, :],
            valid[:, None] & (ik[None, :] < K), 0,
        ).to(tl.float32)
        vo = (token * H + h) * V + value
        vv = tl.load(Vp + vo, valid, 0).to(tl.float32)
        beta = tl.load(Beta + vo, valid, 0).to(tl.float32)
        a_raw = tl.sum(kk * kk, 1) - C * vv * vv
        update_valid = valid & (a_raw > 1.0e-5)
        a = tl.where(update_valid, a_raw, 1.0)
        alpha = tl.where(update_valid, beta / a, 0.0)
        gram = tl.dot(kk, tl.trans(kk), input_precision='ieee') - C * vv[:, None] * vv[None, :]
        lower = tl.where(rows[:, None] > rows[None, :], alpha[:, None] * gram, 0.0)
        b = alpha * (vv - tl.sum(kk * s[None, :], 1))
        # Forward substitution for (I + strictly_lower(alpha * gram)) b = rhs.
        for j in range(0, BT):
            bj = tl.sum(tl.where(rows == j, b, 0.0), 0)
            column = tl.sum(tl.where(rows[None, :] == j, lower, 0.0), 1)
            b = b - column * bj
        den = 1.0 + C * tl.cumsum(vv * b, 0)
        if READOUT:
            rkk = tl.load(
                ReadoutQ + (token[:, None] * H + h) * K + ik[None, :],
                valid[:, None] & (ik[None, :] < K), 0,
            ).to(tl.float32)
            state_rows = (s[None, :] + tl.cumsum(kk * b[:, None], 0)) / den[:, None]
            readout = tl.sum(rkk * state_rows, 1)
            tl.store(Readout + vo, readout, valid)
        qk = tl.dot(qq, tl.trans(kk), input_precision='ieee')
        causal_qk = tl.where(rows[:, None] >= rows[None, :], qk, 0.0)
        num = tl.sum(qq * s[None, :], 1) + tl.sum(causal_qk * b[None, :], 1)
        tl.store(O + vo, SCALE * num / den, valid)
        last = tl.minimum(BT, eos - bos - start) - 1
        zlast = tl.sum(tl.where(rows == last, den, 0.0), 0)
        s = (s + tl.sum(kk * b[:, None], 0)) / zlast
    tl.store(HT + hs, s, ik < K)


@triton.jit
def hyperbolic_delta_rule_bwd_kernel(
    Q, Kp, Vp, Beta, Cu, CP, DO, DHT, Scratch,
    DQ, DK, DV, DBeta, DH0, DC, Curvature, ReadoutQ, DReadout, DReadoutQ,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, TOTAL: tl.constexpr,
    SCALE: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    READOUT: tl.constexpr,
):
    nh, tile_v = tl.program_id(0), tl.program_id(1)
    n, h = nh // H, nh % H
    bos, eos = tl.load(Cu + n), tl.load(Cu + n + 1)
    ik = tl.arange(0, BK)
    iv = tile_v * BV + tl.arange(0, BV)
    hs = (nh * K + ik[:, None]) * V + iv[None, :]
    mask = (ik[:, None] < K) & (iv[None, :] < V)
    C = tl.load(Curvature).to(tl.float32)
    ds = tl.load(DHT + hs, mask, 0).to(tl.float32)
    num_chunks = tl.cdiv(eos - bos, BT)
    for block_rev in range(0, num_chunks):
        block = num_chunks - 1 - block_rev
        start = block * BT
        count = tl.minimum(BT, eos - bos - start)
        cp_id = bos // BT + n + block
        cp_offset = ((cp_id * H + h) * K + ik[:, None]) * V + iv[None, :]
        s = tl.load(CP + cp_offset, mask, 0).to(tl.float32)
        # Replay this interval only. Scratch is private to (sequence, head, value tile).
        for j in range(0, count):
            st = (((nh * BT + j) * K + ik[:, None]) * V + iv[None, :])
            tl.store(Scratch + st, s, mask)
            t = bos + start + j
            kt = tl.load(Kp + (t * H + h) * K + ik, ik < K, 0).to(tl.float32)
            vo = (t * H + h) * V + iv
            vt = tl.load(Vp + vo, iv < V, 0).to(tl.float32)
            beta = tl.load(Beta + vo, iv < V, 0).to(tl.float32)
            a_raw = tl.sum(kt * kt, 0) - C * vt * vt
            update_valid = a_raw > 1.0e-5
            a = tl.where(update_valid, a_raw, 1.0)
            r = vt - tl.sum(kt[:, None] * s, 0)
            u = tl.where(update_valid, beta * r / a, 0.0)
            den = 1.0 + C * vt * u
            s = (s + kt[:, None] * u[None, :]) / den[None, :]
        tl.debug_barrier()
        for j_rev in range(0, count):
            j = count - 1 - j_rev
            t = bos + start + j
            st = (((nh * BT + j) * K + ik[:, None]) * V + iv[None, :])
            prev = tl.load(Scratch + st, mask, 0).to(tl.float32)
            ko = (t * H + h) * K + ik
            vo = (t * H + h) * V + iv
            kt = tl.load(Kp + ko, ik < K, 0).to(tl.float32)
            qt = tl.load(Q + ko, ik < K, 0).to(tl.float32)
            vt = tl.load(Vp + vo, iv < V, 0).to(tl.float32)
            beta = tl.load(Beta + vo, iv < V, 0).to(tl.float32)
            dout = tl.load(DO + vo, iv < V, 0).to(tl.float32)
            if READOUT:
                dreadout = tl.load(DReadout + vo, iv < V, 0).to(tl.float32)
                kr = tl.load(ReadoutQ + ko, ik < K, 0).to(tl.float32)
            else:
                dreadout = tl.full((BV,), 0.0, tl.float32)
                kr = tl.full((BK,), 0.0, tl.float32)
            a_raw = tl.sum(kt * kt, 0) - C * vt * vt
            update_valid = a_raw > 1.0e-5
            a = tl.where(update_valid, a_raw, 1.0)
            r = vt - tl.sum(kt[:, None] * prev, 0)
            u = tl.where(update_valid, beta * r / a, 0.0)
            den = 1.0 + C * vt * u
            cur = (prev + kt[:, None] * u[None, :]) / den[None, :]
            dq = SCALE * tl.sum(cur * dout[None, :], 1)
            ds = ds + SCALE * qt[:, None] * dout[None, :] + kr[:, None] * dreadout[None, :]
            gnum = ds / den[None, :]
            gden = -tl.sum(ds * cur, 0) / den
            gu = tl.sum(gnum * kt[:, None], 0) + C * vt * gden
            gr = tl.where(update_valid, gu * beta / a, 0.0)
            ga = tl.where(update_valid, -gu * u / a, 0.0)
            dv = C * u * gden + gr - 2.0 * C * vt * ga
            db = tl.where(update_valid, gu * r / a, 0.0)
            dk = tl.sum(gnum * u[None, :] - prev * gr[None, :], 1) + 2.0 * kt * tl.sum(ga, 0)
            dc = tl.sum(vt * u * gden - vt * vt * ga, 0)
            ds = gnum - kt[:, None] * gr[None, :]
            # Sum shared q/k gradients deterministically after the kernel, not with atomics.
            partial = tile_v * TOTAL * H * K + ko
            tl.store(DQ + partial, dq, ik < K)
            tl.store(DK + partial, dk, ik < K)
            if READOUT:
                d_readout_q = tl.sum(cur * dreadout[None, :], 1)
                tl.store(DReadoutQ + partial, d_readout_q, ik < K)
            tl.store(DV + vo, dv, iv < V)
            tl.store(DBeta + vo, db, iv < V)
            tl.store(DC + (tile_v * TOTAL + t) * H + h, dc)
        tl.debug_barrier()
    tl.store(DH0 + hs, ds, mask)
