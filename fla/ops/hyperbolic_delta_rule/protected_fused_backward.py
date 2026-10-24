# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Licensed under the MIT license in the repository root.

"""Single-launch per-(batch, head) fused backward for protected hyperbolic KV chunks.

The kernel replays the chunk's projection and normalized state recurrence,
reconstructs the continuous bank evolution, and differentiates all three
subsystems with one Triton CTA. This is the first-order VJP of the *smooth*
bank update; empty-slot insertion is a piecewise discrete occupancy transition.
"""

import triton
import triton.language as tl

from .protected_kernels import _invert_small


@triton.jit
def protected_fused_chunk_bwd_kernel(
    Q, Keys, Values, Beta, H0, BankK, BankV, BankScores, SelectScores, Curvature,
    DO, DHT, DNextK, DNextV, DNextScores,
    ScratchS, ScratchRealized, ScratchBankK, ScratchBankV, ScratchBankScores,
    ScratchDKeys, ScratchDRealized,
    DQ, DK, DV, DBeta, DH0, DInitK, DInitV, DInitScores, DSelectScores, DCurvature,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, R: tl.constexpr,
    SCALE: tl.constexpr, EPS: tl.constexpr, TEMP: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr, BR: tl.constexpr,
):
    p = tl.program_id(0)
    b, h = p // H, p % H
    c = tl.load(Curvature).to(tl.float32)
    ik, iv, ir = tl.arange(0, BK), tl.arange(0, BV), tl.arange(0, BR)
    mk, mv, mr = ik < K, iv < V, ir < R
    bank_base = (b * H + h) * R
    pk = tl.load(BankK + (bank_base + ir[:, None]) * K + ik[None, :], mr[:, None] & mk[None, :], 0).to(tl.float32)
    py = tl.load(BankV + (bank_base + ir[:, None]) * V + iv[None, :], mr[:, None] & mv[None, :], 0).to(tl.float32)
    ps = tl.load(BankScores + bank_base + ir, mr, -1.0e6).to(tl.float32)
    active = mr & (ps > -5.0e5)
    fixed_pk = tl.where(active[:, None], pk, 0.0)
    fixed_py = tl.where(active[:, None], py, 0.0)
    eye = tl.where(ir[:, None] == ir[None, :], 1.0, 0.0)
    gram = tl.dot(fixed_pk, tl.trans(fixed_pk), input_precision="ieee")
    gram = tl.where(active[:, None] & active[None, :], gram + EPS * eye, eye)
    ainv = _invert_small(gram, BR)
    u = tl.dot(ainv, fixed_py, input_precision="ieee")
    gden = 1.0 - c * tl.sum(fixed_py * u, 0)
    feasible = gden > EPS
    inv_gden = tl.where(feasible, 1.0 / gden, 0.0)
    state_base = ((b * H + h) * K + ik[:, None]) * V + iv[None, :]
    ms = mk[:, None] & mv[None, :]
    state = tl.load(H0 + state_base, ms, 0).to(tl.float32)
    # Recompute the chunk while saving only this CTA's temporary state history.
    for t in range(T):
        scratch_id = ((b * H + h) * T + t)
        tl.store(ScratchS + (scratch_id * K + ik[:, None]) * V + iv[None, :], state, ms)
        tok_k = ((b * T + t) * H + h) * K + ik
        tok_v = ((b * T + t) * H + h) * V + iv
        kt = tl.load(Keys + tok_k, mk, 0).to(tl.float32)
        vt = tl.load(Values + tok_v, mv, 0).to(tl.float32)
        beta = tl.load(Beta + tok_v, mv, 0).to(tl.float32)
        key_rhs = tl.sum(fixed_pk * kt[None, :], 1)
        rhs = key_rhs[:, None] - c * fixed_py * vt[None, :]
        inv_rhs = tl.dot(ainv, rhs, input_precision="ieee")
        alpha = inv_rhs + c * inv_gden[None, :] * u * tl.sum(u * rhs, 0)[None, :]
        kb = kt[:, None] - tl.dot(tl.trans(fixed_pk), alpha, input_precision="ieee")
        vb = vt - tl.sum(fixed_py * alpha, 0)
        norm = tl.sum(kb * kb, 0) - c * vb * vb
        projection_valid = feasible & (norm > EPS)
        kb = tl.where(projection_valid[None, :], kb, kt[:, None])
        vb = tl.where(projection_valid, vb, vt)
        a = tl.sum(kb * kb, 0) - c * vb * vb
        update_valid = (a > EPS) & mv
        inv_a = 1.0 / tl.where(update_valid, a, 1.0)
        residual = vb - tl.sum(kb * state, 0)
        weight = tl.where(update_valid, beta * residual * inv_a, 0.0)
        den = 1.0 + c * vb * weight
        state = (state + kb * weight[None, :]) / den[None, :]
        realized = tl.sum(kt[:, None] * state, 0)
        tl.store(ScratchRealized + scratch_id * V + iv, realized, mv)

    # Reconstruct fixed-bank snapshots prior to each smooth candidate insertion.
    current_pk, current_py, current_ps = pk, py, ps
    for t in range(T):
        sid = (b * H + h) * T + t
        tl.store(ScratchBankK + (sid * R + ir[:, None]) * K + ik[None, :], current_pk, mr[:, None] & mk[None, :])
        tl.store(ScratchBankV + (sid * R + ir[:, None]) * V + iv[None, :], current_py, mr[:, None] & mv[None, :])
        tl.store(ScratchBankScores + sid * R + ir, current_ps, mr)
        ck = tl.load(Keys + ((b * T + t) * H + h) * K + ik, mk, 0).to(tl.float32)
        cv = tl.load(ScratchRealized + sid * V + iv, mv, 0).to(tl.float32)
        cs = tl.load(SelectScores + (b * T + t) * H + h).to(tl.float32)
        count = tl.sum((mr & (current_ps > -5.0e5)).to(tl.int32), 0)
        if count < R:
            slot = ir == count
            current_pk = tl.where(slot[:, None], ck[None, :], current_pk)
            current_py = tl.where(slot[:, None], cv[None, :], current_py)
            current_ps = tl.where(slot, cs, current_ps)
        else:
            logits = tl.where(mr, -current_ps / TEMP, -1.0e9)
            pr = tl.softmax(logits, dim=0)
            sig = 1.0 / (1.0 + tl.exp(-(cs - current_ps) / TEMP))
            gate = pr * sig
            current_pk = (1.0 - gate[:, None]) * current_pk + gate[:, None] * ck[None, :]
            current_py = (1.0 - gate[:, None]) * current_py + gate[:, None] * cv[None, :]
            current_ps = (1.0 - gate) * current_ps + gate * cs
    tl.debug_barrier()

    dpk = tl.load(DNextK + (bank_base + ir[:, None]) * K + ik[None, :], mr[:, None] & mk[None, :], 0).to(tl.float32)
    dpy = tl.load(DNextV + (bank_base + ir[:, None]) * V + iv[None, :], mr[:, None] & mv[None, :], 0).to(tl.float32)
    dps = tl.load(DNextScores + bank_base + ir, mr, 0).to(tl.float32)
    for rev in range(T):
        t = T - 1 - rev
        sid = (b * H + h) * T + t
        old_pk = tl.load(ScratchBankK + (sid * R + ir[:, None]) * K + ik[None, :], mr[:, None] & mk[None, :], 0).to(tl.float32)
        old_py = tl.load(ScratchBankV + (sid * R + ir[:, None]) * V + iv[None, :], mr[:, None] & mv[None, :], 0).to(tl.float32)
        old_ps = tl.load(ScratchBankScores + sid * R + ir, mr, -1.0e6).to(tl.float32)
        ck = tl.load(Keys + ((b * T + t) * H + h) * K + ik, mk, 0).to(tl.float32)
        cv = tl.load(ScratchRealized + sid * V + iv, mv, 0).to(tl.float32)
        cs = tl.load(SelectScores + (b * T + t) * H + h).to(tl.float32)
        count = tl.sum((mr & (old_ps > -5.0e5)).to(tl.int32), 0)
        if count < R:
            slot = ir == count
            dk_bank = tl.sum(tl.where(slot[:, None], dpk, 0.0), 0)
            dv_realized = tl.sum(tl.where(slot[:, None], dpy, 0.0), 0)
            ds_select = tl.sum(tl.where(slot, dps, 0.0), 0)
            dpk = tl.where(slot[:, None], 0.0, dpk)
            dpy = tl.where(slot[:, None], 0.0, dpy)
            dps = tl.where(slot, 0.0, dps)
        else:
            logits = tl.where(mr, -old_ps / TEMP, -1.0e9)
            pr = tl.softmax(logits, dim=0)
            sig = 1.0 / (1.0 + tl.exp(-(cs - old_ps) / TEMP))
            gate = pr * sig
            dgate = tl.sum(dpk * (ck[None, :] - old_pk), 1)
            dgate += tl.sum(dpy * (cv[None, :] - old_py), 1)
            dgate += dps * (cs - old_ps)
            dk_bank = tl.sum(gate[:, None] * dpk, 0)
            dv_realized = tl.sum(gate[:, None] * dpy, 0)
            ds_select = tl.sum(gate * dps, 0) + tl.sum(dgate * gate * (1.0 - sig), 0) / TEMP
            dps = (1.0 - gate) * dps + pr / TEMP * tl.sum(dgate * gate, 0) - dgate * gate * (2.0 - sig) / TEMP
            dpk = (1.0 - gate[:, None]) * dpk
            dpy = (1.0 - gate[:, None]) * dpy
        tl.store(ScratchDKeys + sid * K + ik, dk_bank, mk)
        tl.store(ScratchDRealized + sid * V + iv, dv_realized, mv)
        tl.store(DSelectScores + (b * T + t) * H + h, ds_select)
    tl.debug_barrier()

    dbank_k, dbank_v = dpk, dpy
    dcurvature = tl.full((), 0.0, tl.float32)
    ds = tl.load(DHT + state_base, ms, 0).to(tl.float32)
    for rev in range(T):
        t = T - 1 - rev
        sid = (b * H + h) * T + t
        prev = tl.load(ScratchS + (sid * K + ik[:, None]) * V + iv[None, :], ms, 0).to(tl.float32)
        ko = ((b * T + t) * H + h) * K + ik
        vo = ((b * T + t) * H + h) * V + iv
        kt = tl.load(Keys + ko, mk, 0).to(tl.float32)
        qt = tl.load(Q + ko, mk, 0).to(tl.float32)
        vt = tl.load(Values + vo, mv, 0).to(tl.float32)
        beta = tl.load(Beta + vo, mv, 0).to(tl.float32)
        dout = tl.load(DO + vo, mv, 0).to(tl.float32)
        dk_bank = tl.load(ScratchDKeys + sid * K + ik, mk, 0).to(tl.float32)
        dy_real = tl.load(ScratchDRealized + sid * V + iv, mv, 0).to(tl.float32)
        key_rhs = tl.sum(fixed_pk * kt[None, :], 1)
        rhs = key_rhs[:, None] - c * fixed_py * vt[None, :]
        inv_rhs = tl.dot(ainv, rhs, input_precision="ieee")
        alpha = inv_rhs + c * inv_gden[None, :] * u * tl.sum(u * rhs, 0)[None, :]
        kb_proj = kt[:, None] - tl.dot(tl.trans(fixed_pk), alpha, input_precision="ieee")
        vb_proj = vt - tl.sum(fixed_py * alpha, 0)
        norm = tl.sum(kb_proj * kb_proj, 0) - c * vb_proj * vb_proj
        projection_valid = feasible & (norm > EPS) & mv
        alpha = tl.where(projection_valid[None, :], alpha, 0.0)
        kb = tl.where(projection_valid[None, :], kb_proj, kt[:, None])
        vb = tl.where(projection_valid, vb_proj, vt)
        a = tl.sum(kb * kb, 0) - c * vb * vb
        update_valid = (a > EPS) & mv
        inv_a = 1.0 / tl.where(update_valid, a, 1.0)
        residual = vb - tl.sum(kb * prev, 0)
        weight = tl.where(update_valid, beta * residual * inv_a, 0.0)
        den = 1.0 + c * vb * weight
        now = (prev + kb * weight[None, :]) / den[None, :]
        dq = SCALE * tl.sum(now * dout[None, :], 1)
        ds = ds + SCALE * qt[:, None] * dout[None, :] + kt[:, None] * dy_real[None, :]
        dk_real = tl.sum(now * dy_real[None, :], 1)
        gnum = ds / den[None, :]
        gden_state = -tl.sum(ds * now, 0) / den
        gu = tl.sum(gnum * kb, 0) + c * vb * gden_state
        gr = tl.where(update_valid, gu * beta * inv_a, 0.0)
        ga = tl.where(update_valid, -gu * weight * inv_a, 0.0)
        dbeta = tl.where(update_valid, gu * residual * inv_a, 0.0)
        gkb = tl.where(update_valid[None, :], gnum * weight[None, :] - prev * gr[None, :] + 2.0 * kb * ga[None, :], 0.0)
        gvb = tl.where(update_valid, c * weight * gden_state + gr - 2.0 * c * vb * ga, 0.0)
        dcurvature += tl.sum(vb * weight * gden_state - vb * vb * ga, 0)
        ds = gnum - kb * gr[None, :]
        galpha = -tl.dot(fixed_pk, gkb, input_precision="ieee") - fixed_py * gvb[None, :]
        z_raw = tl.dot(ainv, galpha, input_precision="ieee")
        z = z_raw + c * inv_gden[None, :] * u * tl.sum(u * galpha, 0)[None, :]
        z = tl.where(projection_valid[None, :], z, 0.0)
        kp_grad = tl.where(projection_valid[None, :], gkb + tl.dot(tl.trans(fixed_pk), z, input_precision="ieee"), gkb)
        dkey = dk_real + dk_bank + tl.sum(kp_grad, 1)
        dv = gvb - c * tl.sum(fixed_py * z, 0)
        gpk = -tl.dot(alpha, tl.trans(gkb), input_precision="ieee")
        gpk += tl.sum(z, 1)[:, None] * kt[None, :]
        za = tl.dot(z, tl.trans(alpha), input_precision="ieee")
        az = tl.dot(alpha, tl.trans(z), input_precision="ieee")
        gpk -= tl.dot(za + az, fixed_pk, input_precision="ieee")
        y_alpha = tl.sum(fixed_py * alpha, 0)
        y_z = tl.sum(fixed_py * z, 0)
        gpy = -alpha * gvb[None, :] - c * z * vt[None, :]
        gpy += c * (z * y_alpha[None, :] + alpha * y_z[None, :])
        dbank_k += tl.where(active[:, None], gpk, 0.0)
        dbank_v += tl.where(active[:, None], gpy, 0.0)
        dcurvature += tl.sum(y_z * (y_alpha - vt), 0)
        tl.store(DQ + ko, dq, mk)
        tl.store(DK + ko, dkey, mk)
        tl.store(DV + vo, dv, mv)
        tl.store(DBeta + vo, dbeta, mv)
    tl.store(DH0 + state_base, ds, ms)
    tl.store(DInitK + (bank_base + ir[:, None]) * K + ik[None, :], dbank_k, mr[:, None] & mk[None, :])
    tl.store(DInitV + (bank_base + ir[:, None]) * V + iv[None, :], dbank_v, mr[:, None] & mv[None, :])
    tl.store(DInitScores + bank_base + ir, dps, mr)
    tl.store(DCurvature + p, dcurvature)


__all__ = ["protected_fused_chunk_bwd_kernel"]
