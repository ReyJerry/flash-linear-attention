# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Triton chunkwise projection and VJP kernels for shared-key protection."""

import triton
import triton.language as tl


@triton.jit
def _invert_small(A, R: tl.constexpr):
    rows = tl.arange(0, R)
    inv = tl.where(rows[:, None] == rows[None, :], 1.0, 0.0)
    for pivot in tl.static_range(0, R):
        mask = rows == pivot
        row = tl.sum(tl.where(mask[:, None], A, 0.0), axis=0)
        inv_row = tl.sum(tl.where(mask[:, None], inv, 0.0), axis=0)
        diagonal = tl.sum(tl.where(rows == pivot, row, 0.0), axis=0)
        diagonal = tl.where(tl.abs(diagonal) > 1.0e-8, diagonal, 1.0e-8)
        row, inv_row = row / diagonal, inv_row / diagonal
        factors = tl.sum(tl.where(rows[None, :] == pivot, A, 0.0), axis=1)
        factors = tl.where(mask, 0.0, factors)
        A = tl.where(mask[:, None], row[None, :], A - factors[:, None] * row[None, :])
        inv = tl.where(mask[:, None], inv_row[None, :], inv - factors[:, None] * inv_row[None, :])
    return inv


@triton.jit
def shared_key_gram_fwd_kernel(
    BankK, BankScores, AInv,
    H: tl.constexpr, K: tl.constexpr, R: tl.constexpr,
    EPS: tl.constexpr, BK: tl.constexpr, BR: tl.constexpr,
):
    """Compute the regularized key-only Gram inverse once per batch/head/chunk."""
    p = tl.program_id(0)
    b, h = p // H, p % H
    rows, cols, ik = tl.arange(0, BR), tl.arange(0, BR), tl.arange(0, BK)
    bank_base = (b * H + h) * R
    scores = tl.load(BankScores + bank_base + rows, rows < R, -1.0e6).to(tl.float32)
    active = (rows < R) & (scores > -5.0e5)
    pk = tl.load(
        BankK + (bank_base + rows[:, None]) * K + ik[None, :],
        (rows[:, None] < R) & (ik[None, :] < K), 0,
    ).to(tl.float32)
    pk = tl.where(active[:, None], pk, 0.0)
    gram = tl.dot(pk, tl.trans(pk), input_precision="ieee")
    eye = tl.where(rows[:, None] == cols[None, :], 1.0, 0.0)
    gram = tl.where(active[:, None] & active[None, :], gram + EPS * eye, eye)
    inv = _invert_small(gram, BR)
    tl.store(
        AInv + ((b * H + h) * R + rows[:, None]) * R + cols[None, :],
        inv, (rows[:, None] < R) & (cols[None, :] < R),
    )


@triton.jit
def protected_kv_project_fwd_kernel(
    Kp, Vp, BankK, BankV, BankScores, AInv, Curvature,
    KBar, VBar, Alpha, GInv, Valid,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, R: tl.constexpr,
    EPS: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BR: tl.constexpr,
):
    """Project a chunk; reuse A^-1 across every token and value channel."""
    p = tl.program_id(0)
    b, h, vi = p // (H * V), (p // V) % H, p % V
    c = tl.load(Curvature).to(tl.float32)
    rows, cols = tl.arange(0, BR), tl.arange(0, BR)
    it, ik = tl.arange(0, BT), tl.arange(0, BK)
    bank_base = (b * H + h) * R
    scores = tl.load(BankScores + bank_base + rows, rows < R, -1.0e6).to(tl.float32)
    active = (rows < R) & (scores > -5.0e5)
    pk = tl.load(
        BankK + (bank_base + rows[:, None]) * K + ik[None, :],
        (rows[:, None] < R) & (ik[None, :] < K), 0,
    ).to(tl.float32)
    pk = tl.where(active[:, None], pk, 0.0)
    y = tl.load(BankV + bank_base * V + rows * V + vi, rows < R, 0).to(tl.float32)
    y = tl.where(active, y, 0.0)
    a_base = ((b * H + h) * R + rows[:, None]) * R + cols[None, :]
    ainv = tl.load(AInv + a_base, (rows[:, None] < R) & (cols[None, :] < R), 0).to(tl.float32)
    u = tl.sum(ainv * y[None, :], axis=1)
    denom = 1.0 - c * tl.sum(y * u, axis=0)
    feasible = denom > EPS
    safe_denom = tl.where(feasible, denom, 1.0)
    ginv = ainv + (c / safe_denom) * u[:, None] * u[None, :]
    ginv = tl.where(feasible, ginv, ainv)

    k_base = (b * T + it[:, None]) * H * K + h * K
    kt = tl.load(Kp + k_base + ik[None, :],
                 (it[:, None] < T) & (ik[None, :] < K), 0).to(tl.float32)
    v_base = (b * T + it) * H * V + h * V + vi
    vt = tl.load(Vp + v_base, it < T, 0).to(tl.float32)
    rhs = tl.dot(pk, tl.trans(kt), input_precision="ieee") - c * y[:, None] * vt[None, :]
    alpha = tl.dot(ginv, rhs)
    kb = kt - tl.dot(tl.trans(alpha), pk, input_precision="ieee")
    vb = vt - tl.sum(y[:, None] * alpha, axis=0)
    norm = tl.sum(kb * kb, axis=1) - c * vb * vb
    valid = (it < T) & feasible & (norm > EPS)
    kb = tl.where(valid[:, None], kb, kt)
    vb = tl.where(valid, vb, vt)
    alpha = tl.where(valid[None, :], alpha, 0.0)

    kb_base = (((b * T + it[:, None]) * H + h) * V + vi) * K
    tl.store(KBar + kb_base + ik[None, :], kb, (it[:, None] < T) & (ik[None, :] < K))
    tl.store(VBar + v_base, vb, it < T)
    alpha_base = (((b * T + it[:, None]) * H + h) * V + vi) * R
    tl.store(Alpha + alpha_base + rows[None, :], tl.trans(alpha),
             (it[:, None] < T) & (rows[None, :] < R))
    ginv_base = ((b * H + h) * V + vi) * R * R
    tl.store(GInv + ginv_base + rows[:, None] * R + cols[None, :], ginv,
             (rows[:, None] < R) & (cols[None, :] < R))
    valid_base = (b * T + it) * H * V + h * V + vi
    tl.store(Valid + valid_base, valid.to(tl.uint8), it < T)


@triton.jit
def protected_kv_project_bwd_kernel(
    Kp, Vp, BankK, BankV, BankScores, Curvature, Alpha, GInv, Valid, DKBar, DVBar,
    DKByValue, DV, DBankKByValue, DBankV, DC,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, R: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BR: tl.constexpr,
):
    """Analytic projection VJP; aggregate shared-bank gradients over the chunk."""
    p = tl.program_id(0)
    b, h, vi = p // (H * V), (p // V) % H, p % V
    c = tl.load(Curvature).to(tl.float32)
    rows, cols = tl.arange(0, BR), tl.arange(0, BR)
    it, ik = tl.arange(0, BT), tl.arange(0, BK)
    bank_base = (b * H + h) * R
    scores = tl.load(BankScores + bank_base + rows, rows < R, -1.0e6).to(tl.float32)
    active = (rows < R) & (scores > -5.0e5)
    pk = tl.load(
        BankK + (bank_base + rows[:, None]) * K + ik[None, :],
        (rows[:, None] < R) & (ik[None, :] < K), 0,
    ).to(tl.float32)
    pk = tl.where(active[:, None], pk, 0.0)
    y = tl.load(BankV + bank_base * V + rows * V + vi, rows < R, 0).to(tl.float32)
    y = tl.where(active, y, 0.0)
    k_base = (b * T + it[:, None]) * H * K + h * K
    kt = tl.load(Kp + k_base + ik[None, :],
                 (it[:, None] < T) & (ik[None, :] < K), 0).to(tl.float32)
    v_base = (b * T + it) * H * V + h * V + vi
    vt = tl.load(Vp + v_base, it < T, 0).to(tl.float32)
    gkb_base = (((b * T + it[:, None]) * H + h) * V + vi) * K
    gkb = tl.load(DKBar + gkb_base + ik[None, :],
                  (it[:, None] < T) & (ik[None, :] < K), 0).to(tl.float32)
    gvb = tl.load(DVBar + v_base, it < T, 0).to(tl.float32)
    alpha_base = (((b * T + it[:, None]) * H + h) * V + vi) * R
    alpha = tl.trans(tl.load(Alpha + alpha_base + rows[None, :],
                             (it[:, None] < T) & (rows[None, :] < R), 0).to(tl.float32))
    valid_base = (b * T + it) * H * V + h * V + vi
    valid = tl.load(Valid + valid_base, it < T, 0).to(tl.int1)
    gkb = tl.where(valid[:, None], gkb, 0.0)
    gvb = tl.where(valid, gvb, 0.0)
    ginv_base = ((b * H + h) * V + vi) * R * R
    ginv = tl.load(GInv + ginv_base + rows[:, None] * R + cols[None, :],
                   (rows[:, None] < R) & (cols[None, :] < R), 0).to(tl.float32)

    # z = G^-T dL/dalpha; dKbank and dVbank are reduced over tokens here.
    galpha = -(tl.dot(pk, tl.trans(gkb), input_precision="ieee") + y[:, None] * gvb[None, :])
    z = tl.dot(ginv, galpha, input_precision="ieee")
    dk_projected = tl.trans(tl.dot(tl.trans(pk), z, input_precision="ieee")) + gkb
    dy_z = tl.sum(y[:, None] * z, axis=0)
    dv_projected = gvb - c * dy_z

    alpha_tk = tl.dot(tl.trans(alpha), pk, input_precision="ieee")
    z_tk = tl.dot(tl.trans(z), pk, input_precision="ieee")
    dbank_k = tl.dot(z, kt, input_precision="ieee")
    dbank_k -= tl.dot(z, alpha_tk, input_precision="ieee")
    dbank_k -= tl.dot(alpha, z_tk, input_precision="ieee")
    # Direct derivative of kbar = k - alpha^T K_bank.
    dbank_k -= tl.dot(alpha, gkb, input_precision="ieee")
    alpha_y = tl.sum(y[:, None] * alpha, axis=0)
    dbank_v = -tl.sum(alpha * gvb[None, :], axis=1)
    dbank_v -= c * tl.sum(z * vt[None, :], axis=1)
    dbank_v += c * tl.sum(z * alpha_y[None, :], axis=1)
    dbank_v += c * tl.sum(alpha * dy_z[None, :], axis=1)
    dc = tl.sum(dy_z * (alpha_y - vt), axis=0)

    dk = tl.where(valid[:, None], dk_projected, tl.load(DKBar + gkb_base + ik[None, :],
                    (it[:, None] < T) & (ik[None, :] < K), 0).to(tl.float32))
    dv = tl.where(valid, dv_projected, tl.load(DVBar + v_base, it < T, 0).to(tl.float32))
    dpk = tl.where(active[:, None], dbank_k, 0.0)
    dpv = tl.where(active, dbank_v, 0.0)
    dk_out_base = (((b * T + it[:, None]) * H + h) * V + vi) * K
    tl.store(DKByValue + dk_out_base + ik[None, :], dk, (it[:, None] < T) & (ik[None, :] < K))
    tl.store(DV + v_base, dv, it < T)
    dpk_base = (((b * H + h) * V + vi) * R + rows[:, None]) * K + ik[None, :]
    tl.store(DBankKByValue + dpk_base, dpk, (rows[:, None] < R) & (ik[None, :] < K))
    dpv_base = ((b * H + h) * R + rows) * V + vi
    tl.store(DBankV + dpv_base, dpv, rows < R)
    tl.store(DC + (b * H + h) * V + vi, dc)


__all__ = [
    "shared_key_gram_fwd_kernel",
    "protected_kv_project_fwd_kernel",
    "protected_kv_project_bwd_kernel",
]



@triton.jit
def soft_bank_update_fwd_kernel(
    BankK, BankV, BankScores, CandK, CandV, CandScores,
    NextK, NextV, NextScores, Count, TapeK, TapeV, TapeScores,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, R: tl.constexpr,
    TEMP: tl.constexpr, SENTINEL: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, BR: tl.constexpr,
):
    """Exact forward for smooth replacement; no discrete top-k or STE."""
    p = tl.program_id(0)
    b, h = p // H, p % H
    rows, ik, iv = tl.arange(0, BR), tl.arange(0, BK), tl.arange(0, BV)
    bank_base = (b * H + h) * R
    bank_k = tl.load(BankK + (bank_base + rows[:, None]) * K + ik[None, :],
                     (rows[:, None] < R) & (ik[None, :] < K), 0).to(tl.float32)
    bank_v = tl.load(BankV + (bank_base + rows[:, None]) * V + iv[None, :],
                     (rows[:, None] < R) & (iv[None, :] < V), 0).to(tl.float32)
    scores = tl.load(BankScores + bank_base + rows, rows < R, SENTINEL).to(tl.float32)
    for t in range(0, T):
        tape_base = (((b * T + t) * H + h) * R)
        tl.store(TapeK + (tape_base + rows[:, None]) * K + ik[None, :],
                 bank_k, (rows[:, None] < R) & (ik[None, :] < K))
        tl.store(TapeV + (tape_base + rows[:, None]) * V + iv[None, :],
                 bank_v, (rows[:, None] < R) & (iv[None, :] < V))
        tl.store(TapeScores + tape_base + rows, scores, rows < R)
        cand_k = tl.load(CandK + ((b * T + t) * H + h) * K + ik, ik < K, 0).to(tl.float32)
        cand_v = tl.load(CandV + ((b * T + t) * H + h) * V + iv, iv < V, 0).to(tl.float32)
        cand_s = tl.load(CandScores + (b * T + t) * H + h).to(tl.float32)
        active = (rows < R) & (scores > SENTINEL / 2.0)
        count = tl.sum(active.to(tl.int32), axis=0)
        if count < R:
            slot = rows == count
            bank_k = tl.where(slot[:, None], cand_k[None, :], bank_k)
            bank_v = tl.where(slot[:, None], cand_v[None, :], bank_v)
            scores = tl.where(slot, cand_s, scores)
        else:
            safe_scores = tl.where(rows < R, scores, 0.0)
            p_replace = tl.softmax(-safe_scores / TEMP, axis=0)
            sigma = 1.0 / (1.0 + tl.exp(-(cand_s - scores) / TEMP))
            gate = p_replace * sigma
            bank_k = (1.0 - gate[:, None]) * bank_k + gate[:, None] * cand_k[None, :]
            bank_v = (1.0 - gate[:, None]) * bank_v + gate[:, None] * cand_v[None, :]
            scores = (1.0 - gate) * scores + gate * cand_s
    tl.store(NextK + (bank_base + rows[:, None]) * K + ik[None, :],
             bank_k, (rows[:, None] < R) & (ik[None, :] < K))
    tl.store(NextV + (bank_base + rows[:, None]) * V + iv[None, :],
             bank_v, (rows[:, None] < R) & (iv[None, :] < V))
    tl.store(NextScores + bank_base + rows, scores, rows < R)
    count = tl.sum(((rows < R) & (scores > SENTINEL / 2.0)).to(tl.int32), axis=0)
    tl.store(Count + p, count)


@triton.jit
def soft_bank_update_bwd_kernel(
    BankK, BankV, BankScores, CandK, CandV, CandScores,
    TapeK, TapeV, TapeScores, DNextK, DNextV, DNextScores,
    DInitK, DInitV, DInitScores, DCandK, DCandV, DCandScores,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, R: tl.constexpr,
    TEMP: tl.constexpr, SENTINEL: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, BR: tl.constexpr,
):
    """Exact reverse-mode derivative of the smooth sequential bank update."""
    p = tl.program_id(0)
    b, h = p // H, p % H
    rows, ik, iv = tl.arange(0, BR), tl.arange(0, BK), tl.arange(0, BV)
    bank_base = (b * H + h) * R
    d_k = tl.load(DNextK + (bank_base + rows[:, None]) * K + ik[None, :],
                  (rows[:, None] < R) & (ik[None, :] < K), 0).to(tl.float32)
    d_v = tl.load(DNextV + (bank_base + rows[:, None]) * V + iv[None, :],
                  (rows[:, None] < R) & (iv[None, :] < V), 0).to(tl.float32)
    d_s = tl.load(DNextScores + bank_base + rows, rows < R, 0).to(tl.float32)
    for rev in range(0, T):
        t = T - 1 - rev
        tape_base = (((b * T + t) * H + h) * R)
        old_k = tl.load(TapeK + (tape_base + rows[:, None]) * K + ik[None, :],
                        (rows[:, None] < R) & (ik[None, :] < K), 0).to(tl.float32)
        old_v = tl.load(TapeV + (tape_base + rows[:, None]) * V + iv[None, :],
                        (rows[:, None] < R) & (iv[None, :] < V), 0).to(tl.float32)
        old_s = tl.load(TapeScores + tape_base + rows, rows < R, SENTINEL).to(tl.float32)
        cand_k = tl.load(CandK + ((b * T + t) * H + h) * K + ik, ik < K, 0).to(tl.float32)
        cand_v = tl.load(CandV + ((b * T + t) * H + h) * V + iv, iv < V, 0).to(tl.float32)
        cand_s = tl.load(CandScores + (b * T + t) * H + h).to(tl.float32)
        active = (rows < R) & (old_s > SENTINEL / 2.0)
        count = tl.sum(active.to(tl.int32), axis=0)
        if count < R:
            slot = rows == count
            d_cand_k = tl.sum(tl.where(slot[:, None], d_k, 0.0), axis=0)
            d_cand_v = tl.sum(tl.where(slot[:, None], d_v, 0.0), axis=0)
            d_cand_s = tl.sum(tl.where(slot, d_s, 0.0), axis=0)
            d_k = tl.where(slot[:, None], 0.0, d_k)
            d_v = tl.where(slot[:, None], 0.0, d_v)
            d_s = tl.where(slot, 0.0, d_s)
        else:
            p_replace = tl.softmax(-old_s / TEMP, axis=0)
            sigma = 1.0 / (1.0 + tl.exp(-(cand_s - old_s) / TEMP))
            gate = p_replace * sigma
            d_gate = tl.sum(d_k * (cand_k[None, :] - old_k), axis=1)
            d_gate += tl.sum(d_v * (cand_v[None, :] - old_v), axis=1)
            d_gate += d_s * (cand_s - old_s)
            d_cand_k = tl.sum(gate[:, None] * d_k, axis=0)
            d_cand_v = tl.sum(gate[:, None] * d_v, axis=0)
            d_cand_s = tl.sum(gate * d_s, axis=0)
            hgate = d_gate * p_replace * sigma / TEMP
            d_cand_s += tl.sum(hgate * (1.0 - sigma), axis=0)
            d_s = (1.0 - gate) * d_s + p_replace * tl.sum(hgate, axis=0) - hgate * (2.0 - sigma)
            d_k = (1.0 - gate[:, None]) * d_k
            d_v = (1.0 - gate[:, None]) * d_v
        tl.store(DCandK + ((b * T + t) * H + h) * K + ik, d_cand_k, ik < K)
        tl.store(DCandV + ((b * T + t) * H + h) * V + iv, d_cand_v, iv < V)
        tl.store(DCandScores + (b * T + t) * H + h, d_cand_s)
    tl.store(DInitK + (bank_base + rows[:, None]) * K + ik[None, :],
             d_k, (rows[:, None] < R) & (ik[None, :] < K))
    tl.store(DInitV + (bank_base + rows[:, None]) * V + iv[None, :],
             d_v, (rows[:, None] < R) & (iv[None, :] < V))
    tl.store(DInitScores + bank_base + rows, d_s, rows < R)
