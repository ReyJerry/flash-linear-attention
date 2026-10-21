# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Triton forward and first-order backward for Lorentz-protected KV updates."""

import triton
import triton.language as tl


@triton.jit
def _solve_small(A, b, R: tl.constexpr):
    rows = tl.arange(0, R)
    for p in tl.static_range(0, R):
        mask = rows == p
        prow = tl.sum(tl.where(mask[:, None], A, 0.0), axis=0)
        prhs = tl.sum(tl.where(mask, b, 0.0), axis=0)
        diag = tl.sum(tl.where(rows == p, prow, 0.0), axis=0)
        diag = tl.where(tl.abs(diag) > 1.0e-7, diag, 1.0)
        prow, prhs = prow / diag, prhs / diag
        factors = tl.sum(tl.where(rows[None, :] == p, A, 0.0), axis=1)
        factors = tl.where(mask, 0.0, factors)
        A = tl.where(mask[:, None], prow[None, :], A - factors[:, None] * prow[None, :])
        b = tl.where(mask, prhs, b - factors * prhs)
    return b


@triton.jit
def _project(k, v, PK, PV, count, c, ik, rows,
             R: tl.constexpr, K: tl.constexpr, BR: tl.constexpr, BK: tl.constexpr):
    pk = tl.load(PK + rows[:, None] * K + ik[None, :],
                 (rows[:, None] < R) & (ik[None, :] < K), 0).to(tl.float32)
    pv = tl.load(PV + rows, rows < R, 0).to(tl.float32)
    active = rows < count
    pk, pv = tl.where(active[:, None], pk, 0.0), tl.where(active, pv, 0.0)
    gram = tl.dot(pk, tl.trans(pk), input_precision="ieee") - c * pv[:, None] * pv[None, :]
    gram = tl.where(active[:, None] & active[None, :], gram,
                    tl.where(rows[:, None] == rows[None, :], 1.0, 0.0))
    rhs = tl.sum(pk * k[None, :], axis=1) - c * pv * v
    alpha = _solve_small(gram, tl.where(active, rhs, 0.0), BR)
    return k - tl.sum(alpha[:, None] * pk, axis=0), v - tl.sum(alpha * pv, axis=0), alpha, pk, pv, gram


@triton.jit
def _append(k, v, PK, PV, PC, c, ik, rows,
            R: tl.constexpr, K: tl.constexpr, BR: tl.constexpr, BK: tl.constexpr):
    count = tl.load(PC).to(tl.int32)
    kb, vb, _, _, _, _ = _project(k, v, PK, PV, count, c, ik, rows, R, K, BR, BK)
    norm = tl.sum(kb * kb, axis=0) - c * vb * vb
    if (count < R) & (norm > 1.0e-5):
        tl.store(PK + count * K + ik, k, ik < K)
        tl.store(PV + count, v)
        tl.store(PC, count + 1)


@triton.jit
def protected_hyperbolic_fwd_kernel(
    Q, Kp, Vp, Beta, H0, PK0, PV0, PC0, O, HT, PK1, PV1, PC1,
    TapeS, TapePK, TapePV, TapePC, Curvature,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, R: tl.constexpr, T: tl.constexpr,
    SCALE: tl.constexpr, BK: tl.constexpr, BR: tl.constexpr, SAVE: tl.constexpr,
):
    p = tl.program_id(0)
    b, h, vi = p // (H * V), (p // V) % H, p % V
    c = tl.load(Curvature).to(tl.float32)
    ik, rows = tl.arange(0, BK), tl.arange(0, BR)
    hs = ((b * H + h) * K + ik) * V + vi
    s = tl.load(H0 + hs, ik < K, 0).to(tl.float32)
    bank = (b * H + h) * V + vi
    pk_in, pv_in, pc_in = PK0 + bank * R * K, PV0 + bank * R, PC0 + bank
    pk_out, pv_out, pc_out = PK1 + bank * R * K, PV1 + bank * R, PC1 + bank
    count = tl.load(pc_in).to(tl.int32)
    for r in tl.static_range(0, R):
        tl.store(pk_out + r * K + ik, tl.load(pk_in + r * K + ik, ik < K, 0), ik < K)
        tl.store(pv_out + r, tl.load(pv_in + r))
    tl.store(pc_out, count)
    for t in range(0, T):
        token = b * T + t
        tape = ((token * H + h) * V + vi)
        if SAVE:
            tl.store(TapeS + tape * K + ik, s, ik < K)
            tl.store(TapePC + tape, tl.load(pc_out))
            for r in tl.static_range(0, R):
                tl.store(TapePK + (tape * R + r) * K + ik,
                         tl.load(pk_out + r * K + ik, ik < K, 0), ik < K)
                tl.store(TapePV + tape * R + r, tl.load(pv_out + r))
        kt = tl.load(Kp + (token * H + h) * K + ik, ik < K, 0).to(tl.float32)
        qt = tl.load(Q + (token * H + h) * K + ik, ik < K, 0).to(tl.float32)
        vt = tl.load(Vp + (token * H + h) * V + vi).to(tl.float32)
        beta = tl.load(Beta + (token * H + h) * V + vi).to(tl.float32)
        count = tl.load(pc_out).to(tl.int32)
        kb, vb, _, _, _, _ = _project(kt, vt, pk_out, pv_out, count, c, ik, rows, R, K, BR, BK)
        a = tl.sum(kb * kb, axis=0) - c * vb * vb
        valid = a > 1.0e-8
        safe_a = tl.where(valid, a, 1.0)
        residual = vb - tl.sum(kb * s, axis=0)
        u = tl.where(valid, beta * residual / safe_a, 0.0)
        den = tl.where(valid, 1.0 + c * vb * u, 1.0)
        s = tl.where(valid, (s + kb * u) / den, s)
        tl.store(O + token * H * V + h * V + vi, SCALE * tl.sum(qt * s, axis=0))
        _append(kt, tl.sum(kt * s, axis=0), pk_out, pv_out, pc_out, c, ik, rows, R, K, BR, BK)
    tl.store(HT + hs, s, ik < K)


@triton.jit
def protected_hyperbolic_bwd_kernel(
    Q, Kp, Vp, Beta, TapeS, TapePK, TapePV, TapePC, Curvature,
    DO, DHT, DQ, DK, DV, DBeta, DH0, DC,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, R: tl.constexpr, T: tl.constexpr,
    SCALE: tl.constexpr, BK: tl.constexpr, BR: tl.constexpr,
):
    p = tl.program_id(0)
    b, h, vi = p // (H * V), (p // V) % H, p % V
    c = tl.load(Curvature).to(tl.float32)
    ik, rows = tl.arange(0, BK), tl.arange(0, BR)
    hs = ((b * H + h) * K + ik) * V + vi
    ds = tl.load(DHT + hs, ik < K, 0).to(tl.float32)
    for rev in range(0, T):
        t = T - 1 - rev
        token = b * T + t
        tape = ((token * H + h) * V + vi)
        prev = tl.load(TapeS + tape * K + ik, ik < K, 0).to(tl.float32)
        pc = tl.load(TapePC + tape).to(tl.int32)
        kt = tl.load(Kp + (token * H + h) * K + ik, ik < K, 0).to(tl.float32)
        qt = tl.load(Q + (token * H + h) * K + ik, ik < K, 0).to(tl.float32)
        vt = tl.load(Vp + (token * H + h) * V + vi).to(tl.float32)
        beta = tl.load(Beta + (token * H + h) * V + vi).to(tl.float32)
        dout = tl.load(DO + (token * H + h) * V + vi).to(tl.float32)
        kb, vb, alpha, pk, pv, gram = _project(
            kt, vt, TapePK + tape * R * K, TapePV + tape * R, pc, c, ik, rows, R, K, BR, BK,
        )
        a = tl.sum(kb * kb, axis=0) - c * vb * vb
        valid = a > 1.0e-8
        safe_a = tl.where(valid, a, 1.0)
        residual = vb - tl.sum(kb * prev, axis=0)
        u = tl.where(valid, beta * residual / safe_a, 0.0)
        den = tl.where(valid, 1.0 + c * vb * u, 1.0)
        cur = tl.where(valid, (prev + kb * u) / den, prev)
        dq = SCALE * cur * dout
        ds = ds + SCALE * qt * dout
        gnum = tl.where(valid, ds / den, 0.0)
        gden = tl.where(valid, -tl.sum(ds * cur, axis=0) / den, 0.0)
        gu = gnum * kb + c * vb * gden
        gr = tl.where(valid, gu * beta / safe_a, 0.0)
        ga = tl.where(valid, -gu * u / safe_a, 0.0)
        db = tl.where(valid, gu * residual / safe_a, 0.0)
        gkb = gnum * u - prev * gr + 2.0 * kb * ga
        gvb = c * u * gden + gr - 2.0 * c * vb * ga
        galpha = -(tl.sum(pk * gkb[None, :], axis=1) + pv * gvb)
        gh = _solve_small(gram, galpha, BR)
        dk = gkb + tl.sum(gh[:, None] * pk, axis=0)
        dv = gvb - c * tl.sum(pv * gh, axis=0)
        dc = vb * u * gden - vb * vb * ga + tl.sum(gh * pv * (tl.sum(pv * alpha, axis=0) - vt), axis=0)
        ds = tl.where(valid, gnum - kb * gr, ds)
        tl.store(DQ + ((token * H + h) * V + vi) * K + ik, dq, ik < K)
        tl.store(DK + ((token * H + h) * V + vi) * K + ik, dk, ik < K)
        tl.store(DV + token * H * V + h * V + vi, dv)
        tl.store(DBeta + token * H * V + h * V + vi, db)
        tl.store(DC + token * H * V + h * V + vi, dc)
    tl.store(DH0 + hs, ds, ik < K)
