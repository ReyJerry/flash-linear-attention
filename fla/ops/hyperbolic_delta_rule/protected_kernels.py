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
