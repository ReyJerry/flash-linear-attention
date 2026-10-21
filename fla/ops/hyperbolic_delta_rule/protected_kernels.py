# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Triton projection kernels for differentiable shared-key protection."""

import triton
import triton.language as tl


@triton.jit
def _invert_small(A, R: tl.constexpr):
    """Gauss-Jordan inverse for a small, regularized SPD matrix."""
    rows = tl.arange(0, R)
    inv = tl.where(rows[:, None] == rows[None, :], 1.0, 0.0)
    for pivot in tl.static_range(0, R):
        row_mask = rows == pivot
        pivot_row = tl.sum(tl.where(row_mask[:, None], A, 0.0), axis=0)
        inv_row = tl.sum(tl.where(row_mask[:, None], inv, 0.0), axis=0)
        diagonal = tl.sum(tl.where(rows == pivot, pivot_row, 0.0), axis=0)
        safe_diagonal = tl.where(tl.abs(diagonal) > 1.0e-8, diagonal, 1.0e-8)
        pivot_row = pivot_row / safe_diagonal
        inv_row = inv_row / safe_diagonal
        factors = tl.sum(tl.where(rows[None, :] == pivot, A, 0.0), axis=1)
        factors = tl.where(row_mask, 0.0, factors)
        A = tl.where(row_mask[:, None], pivot_row[None, :], A - factors[:, None] * pivot_row[None, :])
        inv = tl.where(row_mask[:, None], inv_row[None, :], inv - factors[:, None] * inv_row[None, :])
    return inv


@triton.jit
def protected_kv_project_fwd_kernel(
    Kp, Vp, PK, PV, Curvature, KBar, VBar, Alpha, GInv, Valid,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, R: tl.constexpr,
    EPS: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BR: tl.constexpr,
):
    """Project a whole chunk; the protection Gram inverse is computed once."""
    p = tl.program_id(0)
    b, h, vi = p // (H * V), (p // V) % H, p % V
    c = tl.load(Curvature).to(tl.float32)
    rows, cols = tl.arange(0, BR), tl.arange(0, BR)
    it, ik = tl.arange(0, BT), tl.arange(0, BK)
    pk_base = (b * H + h) * R * K
    pv_base = (b * H + h) * R * V
    pk = tl.load(PK + pk_base + rows[:, None] * K + ik[None, :],
                 (rows[:, None] < R) & (ik[None, :] < K), 0).to(tl.float32)
    y = tl.load(PV + pv_base + rows * V + vi, rows < R, 0).to(tl.float32)
    eye = tl.where(rows[:, None] == cols[None, :], 1.0, 0.0)
    gram = tl.dot(pk, tl.trans(pk), input_precision="ieee") - c * y[:, None] * y[None, :]
    gram = gram + EPS * eye
    ginv = _invert_small(gram, BR)
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
    valid = (it < T) & (norm > EPS)
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
    tl.store(Valid + valid_base, valid, it < T)


@triton.jit
def protected_kv_project_bwd_kernel(
    Kp, Vp, PK, PV, Curvature, Alpha, GInv, Valid, DKBar, DVBar,
    DK, DV, DPK, DPV, DC,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, R: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BR: tl.constexpr,
):
    """Analytic VJP through the regularized Lorentz projection."""
    p = tl.program_id(0)
    b, h, vi = p // (H * V), (p // V) % H, p % V
    c = tl.load(Curvature).to(tl.float32)
    rows, cols = tl.arange(0, BR), tl.arange(0, BR)
    it, ik = tl.arange(0, BT), tl.arange(0, BK)
    pk_base = (b * H + h) * R * K
    pv_base = (b * H + h) * R * V
    pk = tl.load(PK + pk_base + rows[:, None] * K + ik[None, :],
                 (rows[:, None] < R) & (ik[None, :] < K), 0).to(tl.float32)
    y = tl.load(PV + pv_base + rows * V + vi, rows < R, 0).to(tl.float32)
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
    valid = tl.load(Valid + valid_base, it < T, 0)
    ginv_base = ((b * H + h) * V + vi) * R * R
    ginv = tl.load(GInv + ginv_base + rows[:, None] * R + cols[None, :],
                   (rows[:, None] < R) & (cols[None, :] < R), 0).to(tl.float32)
    galpha = -(tl.dot(pk, tl.trans(gkb), input_precision="ieee") + y[:, None] * gvb[None, :])
    galpha = tl.where(valid[None, :], galpha, 0.0)
    hvec = tl.dot(ginv, galpha, input_precision="ieee")
    dk = gkb + tl.dot(tl.trans(hvec), pk, input_precision="ieee")
    y_h = tl.sum(y[:, None] * hvec, axis=0)
    dv = gvb - c * y_h
    hta = tl.dot(tl.trans(hvec), alpha, input_precision="ieee")
    ath = tl.dot(tl.trans(alpha), hvec, input_precision="ieee")
    dpk = tl.dot(hvec, kt, input_precision="ieee") - tl.dot(hta, pk, input_precision="ieee") - tl.dot(ath, pk, input_precision="ieee")
    y_alpha = tl.sum(y[:, None] * alpha, axis=0)
    dpv = -c * tl.sum(hvec * vt[None, :], axis=1)
    dpv = dpv + c * tl.sum(hvec * y_alpha[None, :], axis=1)
    dpv = dpv + c * tl.sum(alpha * y_h[None, :], axis=1)
    dc = -tl.sum(y_h * vt, axis=0) + tl.sum(y_h * y_alpha, axis=0)
    dk_base = (((b * T + it[:, None]) * H + h) * V + vi) * K
    tl.store(DK + dk_base + ik[None, :], dk, (it[:, None] < T) & (ik[None, :] < K))
    tl.store(DV + v_base, dv, it < T)
    dpk_base = (((b * H + h) * V + vi) * R + rows[:, None]) * K + ik[None, :]
    tl.store(DPK + dpk_base, dpk, (rows[:, None] < R) & (ik[None, :] < K))
    dpv_out_base = ((b * H + h) * R + rows) * V + vi
    tl.store(DPV + dpv_out_base, dpv, rows < R)
    tl.store(DC + (b * H + h) * V + vi, dc)
