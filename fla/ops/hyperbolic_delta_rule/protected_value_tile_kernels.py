# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# Licensed under the MIT license in the repository root.

"""Register-bounded, Value-parallel protected HyperbolicDeltaNet kernels.

One CTA per (batch, head, value-channel) instead of one CTA per head.
The protection Gram is frozen during each chunk. An optional shared inverse
avoids rebuilding the key-only Gram in each Value CTA and backward replay.
The forward uses a short fused recurrent tile (T<=32); it avoids materializing
T-by-T WY Grams and the enormous [K,V] per-head state in registers.
The backward performs one replay pass and one reverse pass in the same CTA.
Both inverse strategies retain V-fold CTA parallelism.
"""

import triton
import triton.language as tl

from .protected_kernels import _invert_small


@triton.jit
def _load_projector(BankK, BankV, BankScores, Curvature, AInv,
                    b, h, vi, H: tl.constexpr, K: tl.constexpr,
                    V: tl.constexpr, R: tl.constexpr,
                    BR: tl.constexpr, BK: tl.constexpr, EPS: tl.constexpr,
                    SHARED_INVERSE: tl.constexpr = False):
    ir = tl.arange(0, BR).to(tl.int64)
    ik = tl.arange(0, BK).to(tl.int64)
    base = (b * H + h) * R
    scores = tl.load(BankScores + base + ir, ir < R, -1e6).to(tl.float32)
    active = (ir < R) & (scores > -5e5)
    pk = tl.load(BankK + (base + ir[:, None]) * K + ik[None, :],
                 (ir[:, None] < R) & (ik[None, :] < K), 0).to(tl.float32)
    y = tl.load(BankV + (base + ir) * V + vi, ir < R, 0).to(tl.float32)
    pk = tl.where(active[:, None], pk, 0.0)
    y = tl.where(active, y, 0.0)
    eye = tl.where(ir[:, None] == ir[None, :], 1.0, 0.0)
    if SHARED_INVERSE:
        ainv = tl.load(AInv + (base + ir[:, None]) * R + ir[None, :],
                       (ir[:, None] < R) & (ir[None, :] < R), 0).to(tl.float32)
        ainv = tl.where((ir[:, None] < R) & (ir[None, :] < R), ainv, eye)
    else:
        # inactive padding has identity Gram to avoid singular pivots
        a = tl.dot(pk, tl.trans(pk), input_precision="ieee")
        a = tl.where(active[:, None] & active[None, :], a + EPS * eye, eye)
        ainv = _invert_small(a, BR)
    curvature = tl.load(Curvature).to(tl.float32)
    u = tl.sum(ainv * y[None, :], axis=1)
    denom = 1.0 - curvature * tl.sum(y * u, axis=0)
    feasible = denom > EPS
    ginv = ainv + (curvature / tl.where(feasible, denom, 1.0)) * u[:, None] * u[None, :]
    ginv = tl.where(feasible, ginv, ainv)
    return pk, y, ginv, active, feasible, curvature


@triton.jit
def _project_one(pk, y, ginv, feasible, curvature, key, value, EPS: tl.constexpr):
    rhs = tl.sum(pk * key[None, :], axis=1) - curvature * y * value
    alpha = tl.sum(ginv * rhs[None, :], axis=1)
    projected_k = key - tl.sum(alpha[:, None] * pk, axis=0)
    projected_v = value - tl.sum(alpha * y, axis=0)
    norm = tl.sum(projected_k * projected_k, axis=0) - curvature * projected_v * projected_v
    valid = feasible & (norm > EPS)
    return (tl.where(valid, projected_k, key),
            tl.where(valid, projected_v, value),
            tl.where(valid, alpha, 0.0), valid)


@triton.jit
def protected_value_tile_fwd_kernel(
    Q, Keys, Values, Beta, H0, BankK, BankV, BankScores, Curvature, AInv,
    O, Realized, HT,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, R: tl.constexpr,
    SCALE: tl.constexpr, EPS: tl.constexpr, BR: tl.constexpr, BK: tl.constexpr,
    SHARED_INVERSE: tl.constexpr = False,
):
    p = tl.program_id(0).to(tl.int64)
    vi = p % V
    h = (p // V) % H
    b = p // (H * V)
    ik = tl.arange(0, BK).to(tl.int64)
    pk, y, ginv, active, feasible, c = _load_projector(
        BankK, BankV, BankScores, Curvature, AInv, b, h, vi,
        H, K, V, R, BR, BK, EPS, SHARED_INVERSE,
    )
    state_offset = ((b * H + h) * K + ik) * V + vi
    state = tl.load(H0 + state_offset, ik < K, 0).to(tl.float32)
    for t in range(0, T):
        ko = ((b * T + t) * H + h) * K + ik
        vo = ((b * T + t) * H + h) * V + vi
        key = tl.load(Keys + ko, ik < K, 0).to(tl.float32)
        query = tl.load(Q + ko, ik < K, 0).to(tl.float32)
        value = tl.load(Values + vo).to(tl.float32)
        beta = tl.load(Beta + vo).to(tl.float32)
        kp, vp, alpha, project_valid = _project_one(pk, y, ginv, feasible, c, key, value, EPS)
        a = tl.sum(kp * kp, axis=0) - c * vp * vp
        valid = a > EPS
        a = tl.where(valid, a, 1.0)
        residual = vp - tl.sum(kp * state, axis=0)
        weight = tl.where(valid, beta * residual / a, 0.0)
        den = 1.0 + c * vp * weight
        state = (state + kp * weight) / den
        tl.store(O + vo, SCALE * tl.sum(query * state, axis=0))
        tl.store(Realized + vo, tl.sum(key * state, axis=0))
    tl.store(HT + state_offset, state, ik < K)


@triton.jit
def protected_value_tile_bwd_kernel(
    Q, Keys, Values, Beta, H0, BankK, BankV, BankScores, Curvature, AInv,
    DO, DHT, DRealized,
    ScratchState, ScratchProjK, ScratchProjV, ScratchAlpha, ScratchValid,
    DQPart, DKPart, DV, DBeta, DH0, DBankKPart, DBankV, DCurvPart,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, R: tl.constexpr,
    SCALE: tl.constexpr, EPS: tl.constexpr, BR: tl.constexpr, BK: tl.constexpr,
    SHARED_INVERSE: tl.constexpr = False,
):
    p = tl.program_id(0).to(tl.int64)
    vi = p % V
    h = (p // V) % H
    b = p // (H * V)
    ik, ir = tl.arange(0, BK).to(tl.int64), tl.arange(0, BR).to(tl.int64)
    pk, y, ginv, active, feasible, c = _load_projector(
        BankK, BankV, BankScores, Curvature, AInv, b, h, vi,
        H, K, V, R, BR, BK, EPS, SHARED_INVERSE,
    )
    state_offset = ((b * H + h) * K + ik) * V + vi
    state = tl.load(H0 + state_offset, ik < K, 0).to(tl.float32)
    # One replay of the forward state and projection, no O(T^2) recomputation.
    for t in range(0, T):
        ko = ((b * T + t) * H + h) * K + ik
        vo = ((b * T + t) * H + h) * V + vi
        tile_k = (((b * T + t) * H + h) * V + vi) * K + ik
        tile_a = (((b * T + t) * H + h) * V + vi) * R + ir
        tl.store(ScratchState + tile_k, state, ik < K)
        key = tl.load(Keys + ko, ik < K, 0).to(tl.float32)
        value = tl.load(Values + vo).to(tl.float32)
        beta = tl.load(Beta + vo).to(tl.float32)
        kp, vp, alpha, project_valid = _project_one(pk, y, ginv, feasible, c, key, value, EPS)
        tl.store(ScratchProjK + tile_k, kp, ik < K)
        tl.store(ScratchProjV + vo, vp)
        tl.store(ScratchAlpha + tile_a, alpha, ir < R)
        tl.store(ScratchValid + vo, project_valid.to(tl.uint8))
        a = tl.sum(kp * kp, axis=0) - c * vp * vp
        valid = a > EPS
        a = tl.where(valid, a, 1.0)
        residual = vp - tl.sum(kp * state, axis=0)
        weight = tl.where(valid, beta * residual / a, 0.0)
        den = 1.0 + c * vp * weight
        state = (state + kp * weight) / den

    ds = tl.load(DHT + state_offset, ik < K, 0).to(tl.float32)
    dpk = tl.full((BR, BK), 0., tl.float32)
    dy = tl.full((BR,), 0., tl.float32)
    dc = tl.full((), 0., tl.float32)
    for reverse in range(0, T):
        t = T - 1 - reverse
        ko = ((b * T + t) * H + h) * K + ik
        vo = ((b * T + t) * H + h) * V + vi
        tile_k = (((b * T + t) * H + h) * V + vi) * K + ik
        tile_a = (((b * T + t) * H + h) * V + vi) * R + ir
        prev = tl.load(ScratchState + tile_k, ik < K, 0).to(tl.float32)
        kp = tl.load(ScratchProjK + tile_k, ik < K, 0).to(tl.float32)
        vp = tl.load(ScratchProjV + vo).to(tl.float32)
        alpha = tl.load(ScratchAlpha + tile_a, ir < R, 0).to(tl.float32)
        project_valid = tl.load(ScratchValid + vo).to(tl.int1)
        key = tl.load(Keys + ko, ik < K, 0).to(tl.float32)
        query = tl.load(Q + ko, ik < K, 0).to(tl.float32)
        value = tl.load(Values + vo).to(tl.float32)
        beta = tl.load(Beta + vo).to(tl.float32)
        dout = tl.load(DO + vo).to(tl.float32)
        dread = tl.load(DRealized + vo).to(tl.float32)
        a = tl.sum(kp * kp, axis=0) - c * vp * vp
        valid = a > EPS
        a = tl.where(valid, a, 1.0)
        residual = vp - tl.sum(kp * prev, axis=0)
        weight = tl.where(valid, beta * residual / a, 0.0)
        den = 1.0 + c * vp * weight
        now = (prev + kp * weight) / den
        dq = SCALE * now * dout
        # Bank target uses the unscaled, original key readout.
        dkey_direct = now * dread
        ds = ds + SCALE * query * dout + key * dread
        gnum = ds / den
        gden = -tl.sum(ds * now, axis=0) / den
        gw = tl.sum(gnum * kp, axis=0) + c * vp * gden
        gr = tl.where(valid, gw * beta / a, 0.0)
        ga = tl.where(valid, -gw * weight / a, 0.0)
        gkp = gnum * weight - prev * gr + 2.0 * kp * ga
        gvp = c * weight * gden + gr - 2.0 * c * vp * ga
        dbeta = tl.where(valid, gw * residual / a, 0.0)
        dc += vp * weight * gden - vp * vp * ga
        ds = gnum - kp * gr

        galpha = -tl.sum(pk * gkp[None, :], axis=1) - y * gvp
        z = tl.sum(ginv * galpha[None, :], axis=1)
        z = tl.where(project_valid, z, 0.0)
        dkey = gkp + tl.sum(pk * z[:, None], axis=0) + dkey_direct
        dvalue = gvp - c * tl.sum(y * z, axis=0)
        # Full derivative of G^-1, not just the RHS. Each CTA accumulates one
        # value-channel contribution; the wrapper sums to the shared key bank.
        alpha_pk = tl.sum(alpha[:, None] * pk, axis=0)
        z_pk = tl.sum(z[:, None] * pk, axis=0)
        gpk = -alpha[:, None] * gkp[None, :]
        gpk += z[:, None] * (key - alpha_pk)[None, :]
        gpk -= alpha[:, None] * z_pk[None, :]
        y_alpha = tl.sum(y * alpha, axis=0)
        y_z = tl.sum(y * z, axis=0)
        gy = -alpha * gvp - c * z * value + c * z * y_alpha + c * alpha * y_z
        dpk += tl.where(active[:, None], gpk, 0.0)
        dy += tl.where(active, gy, 0.0)
        dc += y_z * (y_alpha - value)
        tl.store(DQPart + tile_k, dq, ik < K)
        tl.store(DKPart + tile_k, dkey, ik < K)
        tl.store(DV + vo, dvalue)
        tl.store(DBeta + vo, dbeta)

    tl.store(DH0 + state_offset, ds, ik < K)
    bank_k_base = (((b * H + h) * V + vi) * R + ir[:, None]) * K + ik[None, :]
    tl.store(DBankKPart + bank_k_base, dpk, (ir[:, None] < R) & (ik[None, :] < K))
    tl.store(DBankV + ((b * H + h) * R + ir) * V + vi, dy, ir < R)
    tl.store(DCurvPart + (b * H + h) * V + vi, dc)
