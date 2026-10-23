# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tensor-Core-backed, Value-tiled WY forward and analytic WY backward.

A program owns (sequence, head, 16 Value channels). Shared key Gram and
query-key matrices are calculated once per Value tile instead of once per
channel. Each block uses dense GEMMs and a short triangular substitution;
the backward differentiates the *homogeneous WY system* with its transposed
triangular solve, rather than replaying a recurrent state for every token.

All dot products accumulate in FP32. Accurate mode uses TF32x3 (Tensor Cores,
Ampere and newer) for FP32 intermediate matrices. Fast mode uses native
FP16/BF16 Tensor Cores for low-precision inputs. Use the conservative
accurate mode for gradient validation. The packed/CPU and readout paths use
the original fused_recurrent.py kernels instead.
"""

import triton
import triton.language as tl


@triton.jit
def _mm(a, b, FAST: tl.constexpr, HALF: tl.constexpr, BF16: tl.constexpr):
    if FAST and HALF:
        return tl.dot(a.to(tl.float16), b.to(tl.float16))
    elif FAST and BF16:
        return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
    else:
        return tl.dot(a.to(tl.float32), b.to(tl.float32), input_precision="tf32x3")


@triton.jit
def _wy_weights(kg, key, val, beta, s, c, valid_t,
                FAST: tl.constexpr, HALF: tl.constexpr, BF16: tl.constexpr,
                BT: tl.constexpr):
    rows = tl.arange(0, BT)
    cols = tl.arange(0, BT)
    a_raw = tl.sum(key * key, axis=1)[:, None] - c * val * val
    active = valid_t[:, None] & (a_raw > 1.0e-5)
    a = tl.where(active, a_raw, 1.0)
    alpha = tl.where(active, beta / a, 0.0)
    residual = val - _mm(key, s, FAST, HALF, BF16)
    weight = alpha * residual
    # Forward substitution on L=(I+tril(alpha_i * (k_i k_j - c v_i v_j), -1)).
    # No BT-by-BT-by-BV tensor is formed, and the shared key Gram is reused.
    for j in range(BT):
        wj = tl.sum(tl.where(rows[:, None] == j, weight, 0.0), axis=0)
        kj = tl.sum(tl.where(cols[None, :] == j, kg, 0.0), axis=1)
        vj = tl.sum(tl.where(rows[:, None] == j, val, 0.0), axis=0)
        corr = alpha * (kj[:, None] - c * val * vj[None, :]) * wj[None, :]
        weight -= tl.where(rows[:, None] > j, corr, 0.0)
    return weight, alpha, a, residual, active


@triton.jit
def hyperbolic_delta_tc_chunk_fwd(
    Q, Key, Val, Beta, H0, Cu, Out, HT, Checkpoints, Curvature,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    SCALE: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr,
    BV: tl.constexpr, SAVE: tl.constexpr,
    FAST: tl.constexpr, HALF: tl.constexpr, BF16: tl.constexpr,
):
    nh, tile_v = tl.program_id(0), tl.program_id(1)
    n, h = nh // H, nh % H
    bos = tl.load(Cu + n)
    eos = tl.load(Cu + n + 1)
    c = tl.load(Curvature).to(tl.float32)
    rows, cols = tl.arange(0, BT), tl.arange(0, BT)
    ik = tl.arange(0, BK)
    iv = tile_v * BV + tl.arange(0, BV)
    ms = (ik[:, None] < K) & (iv[None, :] < V)
    hs = (nh * K + ik[:, None]) * V + iv[None, :]
    s = tl.load(H0 + hs, ms, 0).to(tl.float32)
    for start in range(0, eos - bos, BT):
        if SAVE:
            cp_id = bos // BT + n + start // BT
            tl.store(Checkpoints + ((cp_id * H + h) * K + ik[:, None]) * V + iv[None, :], s, ms)
        idx = bos + start + rows
        mt = idx < eos
        kk = tl.load(Key + ((idx[:, None] * H + h) * K + ik[None, :]),
                     mt[:, None] & (ik[None, :] < K), 0).to(tl.float32)
        qq = tl.load(Q + ((idx[:, None] * H + h) * K + ik[None, :]),
                     mt[:, None] & (ik[None, :] < K), 0).to(tl.float32)
        vptr = (idx[:, None] * H + h) * V + iv[None, :]
        mv = mt[:, None] & (iv[None, :] < V)
        vv = tl.load(Val + vptr, mv, 0).to(tl.float32)
        be = tl.load(Beta + vptr, mv, 0).to(tl.float32)
        kg = _mm(kk, tl.trans(kk), FAST, HALF, BF16)
        w, alpha, a, r, active = _wy_weights(
            kg, kk, vv, be, s, c, mt, FAST, HALF, BF16, BT
        )
        z = 1.0 + c * tl.cumsum(vv * w, axis=0)
        qk = _mm(qq, tl.trans(kk), FAST, HALF, BF16)
        causal_qk = tl.where(rows[:, None] >= cols[None, :], qk, 0.0)
        num = _mm(qq, s, FAST, HALF, BF16) + _mm(causal_qk, w, FAST, HALF, BF16)
        tl.store(Out + vptr, SCALE * num / z, mv)
        last = tl.minimum(BT, eos - bos - start) - 1
        z_last = tl.sum(tl.where(rows[:, None] == last, z, 0.0), axis=0)
        s = (s + _mm(tl.trans(kk), w, FAST, HALF, BF16)) / z_last[None, :]
    tl.store(HT + hs, s, ms)


@triton.jit
def hyperbolic_delta_tc_chunk_bwd(
    Q, Key, Val, Beta, Cu, Checkpoints, Out, DOut, DHT, Curvature,
    DQPart, DKPart, DV, DBeta, DH0, DCPart,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    TOTAL: tl.constexpr, SCALE: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    FAST: tl.constexpr, HALF: tl.constexpr, BF16: tl.constexpr,
):
    nh, tile_v = tl.program_id(0), tl.program_id(1)
    n, h = nh // H, nh % H
    bos, eos = tl.load(Cu + n), tl.load(Cu + n + 1)
    c = tl.load(Curvature).to(tl.float32)
    rows, cols = tl.arange(0, BT), tl.arange(0, BT)
    ik = tl.arange(0, BK)
    iv = tile_v * BV + tl.arange(0, BV)
    ms = (ik[:, None] < K) & (iv[None, :] < V)
    hs = (nh * K + ik[:, None]) * V + iv[None, :]
    ds = tl.load(DHT + hs, ms, 0).to(tl.float32)
    dc_accum = tl.full((), 0.0, tl.float32)
    num_chunks = tl.cdiv(eos - bos, BT)
    for rev in range(num_chunks):
        block = num_chunks - 1 - rev
        start = block * BT
        cp_id = bos // BT + n + block
        s = tl.load(Checkpoints + ((cp_id * H + h) * K + ik[:, None]) * V + iv[None, :], ms, 0).to(tl.float32)
        idx = bos + start + rows
        mt = idx < eos
        mv = mt[:, None] & (iv[None, :] < V)
        kk = tl.load(Key + ((idx[:, None] * H + h) * K + ik[None, :]),
                     mt[:, None] & (ik[None, :] < K), 0).to(tl.float32)
        qq = tl.load(Q + ((idx[:, None] * H + h) * K + ik[None, :]),
                     mt[:, None] & (ik[None, :] < K), 0).to(tl.float32)
        vptr = (idx[:, None] * H + h) * V + iv[None, :]
        vv = tl.load(Val + vptr, mv, 0).to(tl.float32)
        be = tl.load(Beta + vptr, mv, 0).to(tl.float32)
        o = tl.load(Out + vptr, mv, 0).to(tl.float32)
        do = tl.load(DOut + vptr, mv, 0).to(tl.float32)
        kg = _mm(kk, tl.trans(kk), FAST, HALF, BF16)
        w, alpha, a, residual, active = _wy_weights(
            kg, kk, vv, be, s, c, mt, FAST, HALF, BF16, BT
        )
        z = 1.0 + c * tl.cumsum(vv * w, axis=0)
        last = tl.minimum(BT, eos - bos - start) - 1
        zend = tl.sum(tl.where(rows[:, None] == last, z, 0.0), axis=0)
        send = (s + _mm(tl.trans(kk), w, FAST, HALF, BF16)) / zend[None, :]
        qk = _mm(qq, tl.trans(kk), FAST, HALF, BF16)
        cqk = tl.where(rows[:, None] >= cols[None, :], qk, 0.0)
        # Reverse-mode through the homogeneous state and its normalization.
        d = SCALE * do / z
        dwq = _mm(tl.trans(cqk), d, FAST, HALF, BF16)
        dq = _mm(d, tl.trans(s), FAST, HALF, BF16)
        dwcoeff = _mm(d, tl.trans(w), FAST, HALF, BF16)
        dq += _mm(tl.where(rows[:, None] >= cols[None, :], dwcoeff, 0.0),
                  kk, FAST, HALF, BF16)
        ds_from_num = _mm(tl.trans(qq), d, FAST, HALF, BF16) + ds / zend[None, :]
        # W_j contracts all output gradients at times i >= j.
        wdt = _mm(w, tl.trans(d), FAST, HALF, BF16)
        dk = _mm(tl.where(rows[:, None] <= cols[None, :], wdt, 0.0), qq, FAST, HALF, BF16)
        dk += _mm(w, tl.trans(ds / zend[None, :]), FAST, HALF, BF16)
        dz = -do * o / z
        dz_prefix = tl.cumsum(dz, axis=0)
        dz_suffix = tl.sum(dz, axis=0)[None, :] - dz_prefix + dz
        dz_suffix -= tl.sum(ds * send, axis=0)[None, :] / zend[None, :]
        dw = dwq + _mm(kk, ds / zend[None, :], FAST, HALF, BF16) + c * vv * dz_suffix
        dv = c * w * dz_suffix
        dc_accum += tl.sum(vv * w * dz_suffix, axis=(0, 1))
        # Backward of L W = alpha * (v - K S): solve L^T Z = dW.
        zz = dw
        for j in range(BT):
            token = BT - 1 - j
            zj = tl.sum(tl.where(rows[:, None] == token, zz, 0.0), axis=0)
            gj = tl.sum(tl.where(rows[:, None] == token, kg, 0.0), axis=0)
            aj = tl.sum(tl.where(rows[:, None] == token, alpha, 0.0), axis=0)
            vj = tl.sum(tl.where(rows[:, None] == token, vv, 0.0), axis=0)
            corr = aj[None, :] * (gj[:, None] - c * vj[None, :] * vv) * zj[None, :]
            zz -= tl.where(rows[:, None] < token, corr, 0.0)
        wz = alpha * zz
        # Aggregate d(K K^T) over Value channels before the Tensor Core matmul.
        gram_adj = -_mm(wz, tl.trans(w), FAST, HALF, BF16)
        gram_adj = tl.where(rows[:, None] > cols[None, :], gram_adj, 0.0)
        dk += _mm(gram_adj + tl.trans(gram_adj), kk, FAST, HALF, BF16)
        vw = vv * w
        vw_prefix = tl.cumsum(vw, axis=0) - vw
        suf_adj = wz * vv
        vw_suffix_adj = tl.sum(suf_adj, axis=0)[None, :] - tl.cumsum(suf_adj, axis=0)
        dv += c * (wz * vw_prefix + w * vw_suffix_adj)
        dc_accum += tl.sum(wz * vv * vw_prefix, axis=(0, 1))
        kg_lower = tl.where(rows[:, None] > cols[None, :], kg, 0.0)
        gram_w = _mm(kg_lower, w, FAST, HALF, BF16) - c * vv * vw_prefix
        d_alpha = zz * (residual - gram_w)
        da = tl.where(active, -d_alpha * be / (a * a), 0.0)
        ds = ds_from_num - _mm(tl.trans(kk), wz, FAST, HALF, BF16)
        dk -= _mm(wz, tl.trans(s), FAST, HALF, BF16)
        dk += 2.0 * kk * tl.sum(da, axis=1)[:, None]
        dv += wz - 2.0 * c * vv * da
        dc_accum -= tl.sum(da * vv * vv, axis=(0, 1))
        db = tl.where(active, d_alpha / a, 0.0)
        partial_k = (tile_v * TOTAL * H * K) + ((idx[:, None] * H + h) * K + ik[None, :])
        tl.store(DQPart + partial_k, dq, mt[:, None] & (ik[None, :] < K))
        tl.store(DKPart + partial_k, dk, mt[:, None] & (ik[None, :] < K))
        tl.store(DV + vptr, dv, mv)
        tl.store(DBeta + vptr, db, mv)
    tl.store(DH0 + hs, ds, ms)
    tl.store(DCPart + tile_v * (TOTAL * H) + nh, dc_accum)


__all__ = ["hyperbolic_delta_tc_chunk_fwd", "hyperbolic_delta_tc_chunk_bwd"]
