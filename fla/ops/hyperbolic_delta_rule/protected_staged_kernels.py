# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Shared key/score and independent value trajectories for protected HDN."""

import triton
import triton.language as tl

from fla.ops.hyperbolic_delta_rule.protected_value_tile_kernels import _project_one


@triton.jit
def _bank_gate(scores, candidate, rows, R: tl.constexpr, TEMP: tl.constexpr):
    count = tl.sum(((rows < R) & (scores > -5e5)).to(tl.int32), 0)
    probability = tl.softmax(tl.where(rows < R, -scores / TEMP, -1e9), 0)
    sigma = 1.0 / (1.0 + tl.exp(-(candidate - scores) / TEMP))
    gate = tl.where(count < R, (rows == count).to(tl.float32), probability * sigma)
    return tl.where(rows < R, gate, 0.0), probability, sigma, count < R


@triton.jit(do_not_specialize=['T'])
def protected_shared_trajectory_fwd_kernel(
    Keys, Selection, BankK, BankScores, KeyTrajectory, ScoreTrajectory, Gates, FinalK, FinalScores,
    T, H: tl.constexpr, K: tl.constexpr, R: tl.constexpr, NC: tl.constexpr, C: tl.constexpr,
    TEMP: tl.constexpr, BK: tl.constexpr, BR: tl.constexpr,
):
    p = tl.program_id(0).to(tl.int64)
    b, h = p // H, p % H
    ir, ik = tl.arange(0, BR).to(tl.int64), tl.arange(0, BK).to(tl.int64)
    mask = (ir[:, None] < R) & (ik[None, :] < K)
    bank = tl.load(BankK + (p * R + ir[:, None]) * K + ik[None, :], mask, 0).to(tl.float32)
    scores = tl.load(BankScores + p * R + ir, ir < R, -1e6).to(tl.float32)
    for chunk in range(NC):
        ic = tl.cast(chunk, tl.int64)
        base = ((b * NC + ic) * H + h) * R
        tl.store(KeyTrajectory + (base + ir[:, None]) * K + ik[None, :], bank, mask)
        tl.store(ScoreTrajectory + base + ir, scores, ir < R)
        for j in range(C):
            t = ic * C + j
            if t < T:
                pos = (b * T + t) * H + h
                key = tl.load(Keys + pos * K + ik, ik < K, 0).to(tl.float32)
                candidate = tl.load(Selection + pos).to(tl.float32)
                gate, probability, sigma, inserting = _bank_gate(scores, candidate, ir, R, TEMP)
                tl.store(Gates + pos * R + ir, gate, ir < R)
                bank = (1.0 - gate[:, None]) * bank + gate[:, None] * key[None, :]
                scores = (1.0 - gate) * scores + gate * candidate
    tl.store(FinalK + (p * R + ir[:, None]) * K + ik[None, :], bank, mask)
    tl.store(FinalScores + p * R + ir, scores, ir < R)


@triton.jit
def _trajectory_projector(
    KeyTrajectory, ScoreTrajectory, AInv, bank_values, c, b, h, chunk,
    H: tl.constexpr, K: tl.constexpr, R: tl.constexpr, NC: tl.constexpr,
    BK: tl.constexpr, BR: tl.constexpr, EPS: tl.constexpr,
):
    ir, ik = tl.arange(0, BR).to(tl.int64), tl.arange(0, BK).to(tl.int64)
    base = ((b * NC + chunk) * H + h) * R
    scores = tl.load(ScoreTrajectory + base + ir, ir < R, -1e6)
    active = (ir < R) & (scores > -5e5)
    pk = tl.load(KeyTrajectory + (base + ir[:, None]) * K + ik[None, :],
                 (ir[:, None] < R) & (ik[None, :] < K), 0).to(tl.float32)
    pk = tl.where(active[:, None], pk, 0.0)
    y = tl.where(active, bank_values, 0.0)
    ainv = tl.load(AInv + (base + ir[:, None]) * R + ir[None, :],
                   (ir[:, None] < R) & (ir[None, :] < R), 0).to(tl.float32)
    u = tl.sum(ainv * y[None, :], 1)
    denom = 1.0 - c * tl.sum(y * u, 0)
    feasible = denom > EPS
    ginv = ainv + (c / tl.where(feasible, denom, 1.0)) * u[:, None] * u[None, :]
    ginv = tl.where(feasible, ginv, ainv)
    return pk, y, ginv, active, feasible


@triton.jit(do_not_specialize=['T'])
def protected_value_trajectory_fwd_kernel(
    Q, Keys, Values, Beta, H0, BankV, Curvature, KeyTrajectory, ScoreTrajectory, AInv, Gates,
    O, HT, FinalV, StateCheckpoints, ValueCheckpoints,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, R: tl.constexpr, NC: tl.constexpr, C: tl.constexpr,
    SCALE: tl.constexpr, EPS: tl.constexpr, BK: tl.constexpr, BR: tl.constexpr,
):
    p = tl.program_id(0).to(tl.int64)
    b, h, vi = p // (H * V), (p // V) % H, p % V
    ir, ik = tl.arange(0, BR).to(tl.int64), tl.arange(0, BK).to(tl.int64)
    state_offset = ((b * H + h) * K + ik) * V + vi
    bank_offset = ((b * H + h) * R + ir) * V + vi
    state = tl.load(H0 + state_offset, ik < K, 0).to(tl.float32)
    bank_values = tl.load(BankV + bank_offset, ir < R, 0).to(tl.float32)
    c = tl.load(Curvature).to(tl.float32)
    for chunk in range(NC):
        ic = tl.cast(chunk, tl.int64)
        cp = ((b * NC + ic) * H + h)
        tl.store(StateCheckpoints + (cp * K + ik) * V + vi, state, ik < K)
        tl.store(ValueCheckpoints + (cp * R + ir) * V + vi, bank_values, ir < R)
        pk, y, ginv, active, feasible = _trajectory_projector(
            KeyTrajectory, ScoreTrajectory, AInv, bank_values, c, b, h, ic, H, K, R, NC, BK, BR, EPS,
        )
        for j in range(C):
            t = ic * C + j
            if t < T:
                pos = (b * T + t) * H + h
                key = tl.load(Keys + pos * K + ik, ik < K, 0).to(tl.float32)
                query = tl.load(Q + pos * K + ik, ik < K, 0).to(tl.float32)
                value = tl.load(Values + pos * V + vi).to(tl.float32)
                beta = tl.load(Beta + pos * V + vi).to(tl.float32)
                kp, vp, alpha, projected = _project_one(pk, y, ginv, feasible, c, key, value, EPS)
                a = tl.sum(kp * kp, 0) - c * vp * vp
                valid = a > EPS
                a = tl.where(valid, a, 1.0)
                residual = vp - tl.sum(kp * state, 0)
                weight = tl.where(valid, beta * residual / a, 0.0)
                den = 1.0 + c * vp * weight
                state = (state + kp * weight) / den
                tl.store(O + pos * V + vi, SCALE * tl.sum(query * state, 0))
                realized = tl.sum(key * state, 0)
                gate = tl.load(Gates + pos * R + ir, ir < R, 0).to(tl.float32)
                bank_values = (1.0 - gate) * bank_values + gate * realized
    tl.store(HT + state_offset, state, ik < K)
    tl.store(FinalV + bank_offset, bank_values, ir < R)


@triton.jit(do_not_specialize=['T', 'START', 'STOP'])
def protected_value_trajectory_bwd_kernel(
    Q, Keys, Values, Beta, Curvature, KeyTrajectory, ScoreTrajectory, AInv, Gates,
    StateCheckpoints, ValueCheckpoints, DO, CarryState, CarryValues, CarryCurvature,
    ScratchState, ScratchProjK, ScratchProjV, ScratchAlpha, ScratchValid, ScratchValues,
    DQPart, DKPart, DV, DBeta, DBankKPart, DGatePart,
    T, START, STOP, P: tl.constexpr,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, R: tl.constexpr, NC: tl.constexpr, C: tl.constexpr,
    SCALE: tl.constexpr, EPS: tl.constexpr, BK: tl.constexpr, BR: tl.constexpr,
):
    p = tl.program_id(0).to(tl.int64)
    b, h, vi = p // (H * V), (p // V) % H, p % V
    ir, ik = tl.arange(0, BR).to(tl.int64), tl.arange(0, BK).to(tl.int64)
    state_offset = ((b * H + h) * K + ik) * V + vi
    bank_offset = ((b * H + h) * R + ir) * V + vi
    ds = tl.load(CarryState + state_offset, ik < K, 0).to(tl.float32)
    dy = tl.load(CarryValues + bank_offset, ir < R, 0).to(tl.float32)
    dc = tl.load(CarryCurvature + p).to(tl.float32)
    c = tl.load(Curvature).to(tl.float32)
    start = START.to(tl.int64)
    for reverse in range(STOP - START):
        chunk = (STOP - reverse - 1).to(tl.int64)
        cp = ((b * NC + chunk) * H + h)
        state = tl.load(StateCheckpoints + (cp * K + ik) * V + vi, ik < K, 0).to(tl.float32)
        bank_values = tl.load(ValueCheckpoints + (cp * R + ir) * V + vi, ir < R, 0).to(tl.float32)
        pk, y, ginv, active, feasible = _trajectory_projector(
            KeyTrajectory, ScoreTrajectory, AInv, bank_values, c, b, h, chunk, H, K, R, NC, BK, BR, EPS,
        )
        for j in range(C):
            t = chunk * C + j
            if t < T:
                pos = (b * T + t) * H + h
                scratch = p * C + j
                tl.store(ScratchState + scratch * K + ik, state, ik < K)
                tl.store(ScratchValues + scratch * R + ir, bank_values, ir < R)
                key = tl.load(Keys + pos * K + ik, ik < K, 0).to(tl.float32)
                value = tl.load(Values + pos * V + vi).to(tl.float32)
                beta = tl.load(Beta + pos * V + vi).to(tl.float32)
                kp, vp, alpha, projected = _project_one(pk, y, ginv, feasible, c, key, value, EPS)
                tl.store(ScratchProjK + scratch * K + ik, kp, ik < K)
                tl.store(ScratchProjV + scratch, vp)
                tl.store(ScratchAlpha + scratch * R + ir, alpha, ir < R)
                tl.store(ScratchValid + scratch, projected.to(tl.uint8))
                a = tl.sum(kp * kp, 0) - c * vp * vp
                valid = a > EPS
                a = tl.where(valid, a, 1.0)
                residual = vp - tl.sum(kp * state, 0)
                weight = tl.where(valid, beta * residual / a, 0.0)
                state = (state + kp * weight) / (1.0 + c * vp * weight)
                realized = tl.sum(key * state, 0)
                gate = tl.load(Gates + pos * R + ir, ir < R, 0).to(tl.float32)
                bank_values = (1.0 - gate) * bank_values + gate * realized
        dpk = tl.full((BR, BK), 0.0, tl.float32)
        dyp = tl.full((BR,), 0.0, tl.float32)
        for rev in range(C):
            j = C - 1 - rev
            t = chunk * C + j
            if t < T:
                pos = (b * T + t) * H + h
                scratch = p * C + j
                prev = tl.load(ScratchState + scratch * K + ik, ik < K, 0).to(tl.float32)
                old_y = tl.load(ScratchValues + scratch * R + ir, ir < R, 0).to(tl.float32)
                kp = tl.load(ScratchProjK + scratch * K + ik, ik < K, 0).to(tl.float32)
                vp = tl.load(ScratchProjV + scratch).to(tl.float32)
                alpha = tl.load(ScratchAlpha + scratch * R + ir, ir < R, 0).to(tl.float32)
                projected = tl.load(ScratchValid + scratch).to(tl.int1)
                key = tl.load(Keys + pos * K + ik, ik < K, 0).to(tl.float32)
                query = tl.load(Q + pos * K + ik, ik < K, 0).to(tl.float32)
                value = tl.load(Values + pos * V + vi).to(tl.float32)
                beta = tl.load(Beta + pos * V + vi).to(tl.float32)
                dout = tl.load(DO + pos * V + vi).to(tl.float32)
                gate = tl.load(Gates + pos * R + ir, ir < R, 0).to(tl.float32)
                a = tl.sum(kp * kp, 0) - c * vp * vp
                valid = a > EPS
                a = tl.where(valid, a, 1.0)
                residual = vp - tl.sum(kp * prev, 0)
                weight = tl.where(valid, beta * residual / a, 0.0)
                den = 1.0 + c * vp * weight
                now = (prev + kp * weight) / den
                realized = tl.sum(key * now, 0)
                dread = tl.sum(gate * dy, 0)
                dg = dy * (realized - old_y)
                dy = (1.0 - gate) * dy
                dq = SCALE * now * dout
                direct = now * dread
                ds = ds + SCALE * query * dout + key * dread
                gnum = ds / den
                gden = -tl.sum(ds * now, 0) / den
                gw = tl.sum(gnum * kp, 0) + c * vp * gden
                gr = tl.where(valid, gw * beta / a, 0.0)
                ga = tl.where(valid, -gw * weight / a, 0.0)
                gkp = gnum * weight - prev * gr + 2.0 * kp * ga
                gvp = c * weight * gden + gr - 2.0 * c * vp * ga
                dbeta = tl.where(valid, gw * residual / a, 0.0)
                dc += vp * weight * gden - vp * vp * ga
                ds = gnum - kp * gr
                galpha = -tl.sum(pk * gkp[None, :], 1) - y * gvp
                z = tl.where(projected, tl.sum(ginv * galpha[None, :], 1), 0.0)
                dkey = gkp + tl.sum(pk * z[:, None], 0) + direct
                dvalue = gvp - c * tl.sum(y * z, 0)
                alpha_pk = tl.sum(alpha[:, None] * pk, 0)
                z_pk = tl.sum(z[:, None] * pk, 0)
                gpk = -alpha[:, None] * gkp[None, :]
                gpk += z[:, None] * (key - alpha_pk)[None, :]
                gpk -= alpha[:, None] * z_pk[None, :]
                ya, yz = tl.sum(y * alpha, 0), tl.sum(y * z, 0)
                gy = -alpha * gvp - c * z * value + c * z * ya + c * alpha * yz
                dpk += tl.where(active[:, None], gpk, 0.0)
                dyp += tl.where(active, gy, 0.0)
                dc += yz * (ya - value)
                partial = (t - start * C) * P + p
                tl.store(DQPart + partial * K + ik, dq, ik < K)
                tl.store(DKPart + partial * K + ik, dkey, ik < K)
                tl.store(DGatePart + partial * R + ir, dg, ir < R)
                tl.store(DV + pos * V + vi, dvalue)
                tl.store(DBeta + pos * V + vi, dbeta)
        dy += dyp
        bank_partial = ((chunk - start) * P + p) * R
        tl.store(DBankKPart + (bank_partial + ir[:, None]) * K + ik[None, :], dpk,
                 (ir[:, None] < R) & (ik[None, :] < K))
    tl.store(CarryState + state_offset, ds, ik < K)
    tl.store(CarryValues + bank_offset, dy, ir < R)
    tl.store(CarryCurvature + p, dc)


@triton.jit(do_not_specialize=['T', 'START'])
def protected_value_partials_bwd_kernel(
    DQPart, DKPart, DGatePart, DQ, DK, DGates,
    T, START, P: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, R: tl.constexpr,
    BV: tl.constexpr, BR: tl.constexpr, BK: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    p = tl.program_id(1).to(tl.int64)
    block = tl.program_id(2).to(tl.int64)
    b, h = p // H, p % H
    iv = tl.arange(0, BV).to(tl.int64)
    ik = block * BK + tl.arange(0, BK).to(tl.int64)
    base = t * P + p * V
    offsets = (base + iv[:, None]) * K + ik[None, :]
    mask = (iv[:, None] < V) & (ik[None, :] < K)
    dq = tl.sum(tl.load(DQPart + offsets, mask, 0), 0)
    dk = tl.sum(tl.load(DKPart + offsets, mask, 0), 0)
    output = (b * T + START + t) * H + h
    tl.store(DQ + output * K + ik, dq, ik < K)
    tl.store(DK + output * K + ik, dk, ik < K)
    if block == 0:
        ir = tl.arange(0, BR).to(tl.int64)
        dg = tl.sum(tl.load(DGatePart + (base + iv[:, None]) * R + ir[None, :],
                            (iv[:, None] < V) & (ir[None, :] < R), 0), 0)
        tl.store(DGates + output * R + ir, dg, ir < R)


@triton.jit(do_not_specialize=['START'])
def protected_bank_partials_bwd_kernel(
    DBankKPart, DKeyTrajectory, START, P: tl.constexpr,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, R: tl.constexpr, NC: tl.constexpr,
    BV: tl.constexpr, BLOCK: tl.constexpr,
):
    chunk = tl.program_id(0).to(tl.int64)
    p = tl.program_id(1).to(tl.int64)
    index = tl.program_id(2).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    iv = tl.arange(0, BV).to(tl.int64)
    b, h = p // H, p % H
    values = tl.load(DBankKPart + ((chunk * P + p * V + iv[:, None]) * R * K + index[None, :]),
                     (iv[:, None] < V) & (index[None, :] < R * K), 0)
    summed = tl.sum(values, 0)
    tl.store(DKeyTrajectory + ((b * NC + START + chunk) * H + h) * R * K + index, summed, index < R * K)


@triton.jit(do_not_specialize=['T'])
def protected_shared_trajectory_bwd_kernel(
    Keys, Selection, KeyTrajectory, ScoreTrajectory, Gates, DKeyTrajectory, DGates, DFinalK, DFinalScores,
    ScratchKeys, ScratchScores, DKeys, DSelection, DBankK, DBankScores,
    T, H: tl.constexpr, K: tl.constexpr, R: tl.constexpr, NC: tl.constexpr, C: tl.constexpr,
    TEMP: tl.constexpr, BK: tl.constexpr, BR: tl.constexpr,
):
    p = tl.program_id(0).to(tl.int64)
    b, h = p // H, p % H
    ir, ik = tl.arange(0, BR).to(tl.int64), tl.arange(0, BK).to(tl.int64)
    mask = (ir[:, None] < R) & (ik[None, :] < K)
    dkey = tl.load(DFinalK + (p * R + ir[:, None]) * K + ik[None, :], mask, 0).to(tl.float32)
    dscore = tl.load(DFinalScores + p * R + ir, ir < R, 0).to(tl.float32)
    for reverse in range(NC):
        chunk = tl.cast(NC - reverse - 1, tl.int64)
        base = ((b * NC + chunk) * H + h) * R
        bank = tl.load(KeyTrajectory + (base + ir[:, None]) * K + ik[None, :], mask, 0).to(tl.float32)
        scores = tl.load(ScoreTrajectory + base + ir, ir < R, -1e6).to(tl.float32)
        for j in range(C):
            t = chunk * C + j
            if t < T:
                scratch = (p * C + j) * R
                tl.store(ScratchKeys + (scratch + ir[:, None]) * K + ik[None, :], bank, mask)
                tl.store(ScratchScores + scratch + ir, scores, ir < R)
                pos = (b * T + t) * H + h
                key = tl.load(Keys + pos * K + ik, ik < K, 0).to(tl.float32)
                candidate = tl.load(Selection + pos).to(tl.float32)
                gate = tl.load(Gates + pos * R + ir, ir < R, 0).to(tl.float32)
                bank = (1.0 - gate[:, None]) * bank + gate[:, None] * key[None, :]
                scores = (1.0 - gate) * scores + gate * candidate
        for rev in range(C):
            j = C - 1 - rev
            t = chunk * C + j
            if t < T:
                pos = (b * T + t) * H + h
                scratch = (p * C + j) * R
                old_k = tl.load(ScratchKeys + (scratch + ir[:, None]) * K + ik[None, :], mask, 0).to(tl.float32)
                old_s = tl.load(ScratchScores + scratch + ir, ir < R, -1e6).to(tl.float32)
                key = tl.load(Keys + pos * K + ik, ik < K, 0).to(tl.float32)
                candidate = tl.load(Selection + pos).to(tl.float32)
                gate, probability, sigma, inserting = _bank_gate(old_s, candidate, ir, R, TEMP)
                dg = tl.load(DGates + pos * R + ir, ir < R, 0).to(tl.float32)
                dg += tl.sum(dkey * (key[None, :] - old_k), 1) + dscore * (candidate - old_s)
                dk = tl.sum(gate[:, None] * dkey, 0)
                dsel = tl.sum(gate * dscore, 0)
                dsigma = dg * probability * sigma * (1.0 - sigma) / TEMP
                softmax_term = probability * (dg * sigma - tl.sum(probability * dg * sigma, 0)) / TEMP
                dsel += tl.sum(tl.where(inserting, 0.0, dsigma), 0)
                dscore = (1.0 - gate) * dscore - tl.where(inserting, 0.0, softmax_term + dsigma)
                dkey = (1.0 - gate[:, None]) * dkey
                old_dk = tl.load(DKeys + pos * K + ik, ik < K, 0).to(tl.float32)
                # preserve the two input-dtype gradient edges of the original separate autograd functions
                old_dk = old_dk.to(Keys.dtype.element_ty).to(tl.float32)
                dk = dk.to(Keys.dtype.element_ty).to(tl.float32)
                tl.store(DKeys + pos * K + ik, old_dk + dk, ik < K)
                tl.store(DSelection + pos, dsel)
        dkey += tl.load(DKeyTrajectory + (base + ir[:, None]) * K + ik[None, :], mask, 0).to(tl.float32)
    tl.store(DBankK + (p * R + ir[:, None]) * K + ik[None, :], dkey, mask)
    tl.store(DBankScores + p * R + ir, dscore, ir < R)
