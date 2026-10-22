# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Experimental single-launch fused protected Chunk/WY and continuous bank update.

One CTA owns one (batch, head) chunk, so shared Key-bank updates never race.
The bank is fixed during projection/WY and updated after every token's realized
association has been computed, matching the chunkwise PyTorch reference.
"""

import triton
import triton.language as tl

from .protected_kernels import _invert_small


@triton.jit
def protected_fused_chunk_fwd_kernel(
    Q, Kp, Vp, Beta, H0, BankK, BankV, BankScores, SelectScores, Curvature,
    Out, HT, NextK, NextV, NextScores, Count,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, R: tl.constexpr,
    SCALE: tl.constexpr, EPS: tl.constexpr, TEMP: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, BR: tl.constexpr,
):
    p = tl.program_id(0)
    b, h = p // H, p % H
    c = tl.load(Curvature).to(tl.float32)
    it, ik = tl.arange(0, BT), tl.arange(0, BK)
    iv, ir = tl.arange(0, BV), tl.arange(0, BR)
    valid_t, valid_k, valid_v, valid_r = it < T, ik < K, iv < V, ir < R
    base_bank = (b * H + h) * R
    scores = tl.load(BankScores + base_bank + ir, valid_r, -1.0e6).to(tl.float32)
    active = valid_r & (scores > -5.0e5)
    pk = tl.load(BankK + (base_bank + ir[:, None]) * K + ik[None, :],
                 valid_r[:, None] & valid_k[None, :], 0).to(tl.float32)
    py = tl.load(BankV + (base_bank + ir[:, None]) * V + iv[None, :],
                 valid_r[:, None] & valid_v[None, :], 0).to(tl.float32)
    pk_mask = tl.where(active[:, None], pk, 0.0)
    py_mask = tl.where(active[:, None], py, 0.0)
    eye = tl.where(ir[:, None] == ir[None, :], 1.0, 0.0)
    gram_k = tl.dot(pk_mask, tl.trans(pk_mask), input_precision="ieee")
    gram_k = tl.where(active[:, None] & active[None, :], gram_k + EPS * eye, eye)
    a_inv = _invert_small(gram_k, BR)
    token_addr = (b * T + it[:, None]) * H * K + h * K
    raw_k = tl.load(Kp + token_addr + ik[None, :],
                    valid_t[:, None] & valid_k[None, :], 0).to(tl.float32)
    query = tl.load(Q + token_addr + ik[None, :],
                    valid_t[:, None] & valid_k[None, :], 0).to(tl.float32)
    key_rhs = tl.dot(pk_mask, tl.trans(raw_k), input_precision="ieee")
    realized_cache = tl.full((BT, BV), 0.0, tl.float32)
    for value in range(0, V):
        y = tl.sum(tl.where(iv[None, :] == value, py_mask, 0.0), 1)
        u = tl.sum(a_inv * y[None, :], 1)
        denom_g = 1.0 - c * tl.sum(y * u, 0)
        feasible = denom_g > EPS
        inv_g = a_inv + c / tl.where(feasible, denom_g, 1.0) * u[:, None] * u[None, :]
        v_addr = (b * T + it) * H * V + h * V + value
        raw_v = tl.load(Vp + v_addr, valid_t, 0).to(tl.float32)
        beta = tl.load(Beta + v_addr, valid_t, 0).to(tl.float32)
        rhs = key_rhs - c * y[:, None] * raw_v[None, :]
        alpha = tl.dot(inv_g, rhs)
        kp = raw_k - tl.dot(tl.trans(alpha), pk_mask, input_precision="ieee")
        vp = raw_v - tl.sum(y[:, None] * alpha, 0)
        norm = tl.sum(kp * kp, 1) - c * vp * vp
        valid_projection = feasible & (norm > EPS)
        kp = tl.where(valid_projection[:, None], kp, raw_k)
        vp = tl.where(valid_projection, vp, raw_v)
        state = tl.load(H0 + ((b * H + h) * K + ik) * V + value, valid_k, 0).to(tl.float32)
        a_raw = tl.sum(kp * kp, 1) - c * vp * vp
        update_valid = valid_t & (a_raw > EPS)
        a = tl.where(update_valid, a_raw, 1.0)
        coeff = tl.where(update_valid, beta / a, 0.0)
        gram = tl.dot(kp, tl.trans(kp), input_precision="ieee") - c * vp[:, None] * vp[None, :]
        lower = tl.where(it[:, None] > it[None, :], coeff[:, None] * gram, 0.0)
        recurrence = coeff * (vp - tl.sum(kp * state[None, :], 1))
        for j in range(0, BT):
            bj = tl.sum(tl.where(it == j, recurrence, 0.0), 0)
            column = tl.sum(tl.where(it[None, :] == j, lower, 0.0), 1)
            recurrence = recurrence - column * bj
        den = 1.0 + c * tl.cumsum(vp * recurrence, 0)
        states = (state[None, :] + tl.cumsum(kp * recurrence[:, None], 0)) / den[:, None]
        output = SCALE * tl.sum(query * states, 1)
        realized = tl.sum(raw_k * states, 1)
        tl.store(Out + v_addr, output, valid_t)
        realized_cache = tl.where(iv[None, :] == value, realized[:, None], realized_cache)
        final = tl.sum(tl.where(it[:, None] == T - 1, states, 0.0), 0)
        tl.store(HT + ((b * H + h) * K + ik) * V + value, final, valid_k)
    # A single program now owns all V channels and the shared bank. This permits
    # a sequential, differentiable bank update without atomics or a second launch.
    for t in range(0, T):
        cand_k = tl.sum(tl.where(it[:, None] == t, raw_k, 0.0), 0)
        cand_v = tl.sum(tl.where(it[:, None] == t, realized_cache, 0.0), 0)
        cand_s = tl.load(SelectScores + (b * T + t) * H + h).to(tl.float32)
        active = valid_r & (scores > -5.0e5)
        count = tl.sum(active.to(tl.int32), 0)
        if count < R:
            slot = ir == count
            pk = tl.where(slot[:, None], cand_k[None, :], pk)
            py = tl.where(slot[:, None], cand_v[None, :], py)
            scores = tl.where(slot, cand_s, scores)
        else:
            logits = tl.where(valid_r, -scores / TEMP, -1.0e9)
            weights = tl.softmax(logits, 0)
            sigma = 1.0 / (1.0 + tl.exp(-(cand_s - scores) / TEMP))
            gate = weights * sigma
            pk = (1.0 - gate[:, None]) * pk + gate[:, None] * cand_k[None, :]
            py = (1.0 - gate[:, None]) * py + gate[:, None] * cand_v[None, :]
            scores = (1.0 - gate) * scores + gate * cand_s
    tl.store(NextK + (base_bank + ir[:, None]) * K + ik[None, :], pk,
             valid_r[:, None] & valid_k[None, :])
    tl.store(NextV + (base_bank + ir[:, None]) * V + iv[None, :], py,
             valid_r[:, None] & valid_v[None, :])
    tl.store(NextScores + base_bank + ir, scores, valid_r)
    final_count = tl.sum((valid_r & (scores > -5.0e5)).to(tl.int32), 0)
    tl.store(Count + p, final_count)


__all__ = ["protected_fused_chunk_fwd_kernel"]
