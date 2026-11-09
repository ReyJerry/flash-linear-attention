# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Wide direct-residual state scan with independent mathematical value groups."""
import triton
import triton.language as tl

from fla.ops.key_gated_hyperbolic_delta_rule.bf16_grouped_kernels import sum_value_groups
from fla.ops.key_gated_hyperbolic_delta_rule.bf16_selective_kernels import (
    at,
    erase,
    exclusive_prefix,
    load_matrix,
    load_vector,
    mass_forward_scan,
    mass_reverse_scan,
    mass_system_sources,
    mass_system_transpose,
    mm,
    mm_backward,
    mm_precise,
    pointer_mm,
    pointer_mv,
    store_matrix,
    store_vector,
)


@triton.jit
def state_direct_grouped_fwd_kernel(V0, Curvature, S0, Final, Starts, Ends, SeqChunks, P, Gp,
                                    SavedS, SavedY, SavedU, SavedR, SavedT, SavedW, SavedZP,
                                    H: tl.constexpr, VD: tl.constexpr, GS: tl.constexpr, NG: tl.constexpr,
                                    C: tl.constexpr, K: tl.constexpr, BV: tl.constexpr, Fallback=None):
    tl.static_assert(C == 64 and K == 128 and GS == 32 and BV == 64 and VD % BV == 0)
    shv = tl.program_id(0).to(tl.int64)
    tile, head, seq = shv % (VD // BV), shv // (VD // BV) % H, shv // ((VD // BV) * H)
    columns = tile * BV + tl.arange(0, BV).to(tl.int64)
    groups, local = columns // GS, columns % GS
    a = tl.arange(0, K).to(tl.int64)
    i = tl.arange(0, C).to(tl.int64)
    state_off = ((seq * H + head) * K + a[:, None]) * VD + columns[None, :]
    state = tl.load(S0 + state_off).to(tl.float32)
    kap = tl.load(Curvature).to(tl.float32) / GS
    first, last = tl.load(SeqChunks + seq), tl.load(SeqChunks + seq + 1)
    for chunk in range(first, last):
        ch = chunk * H + head
        blocks = ch * NG + groups
        start, end = tl.load(Starts + chunk), tl.load(Ends + chunk)
        length = end - start
        saved_state_off = blocks[None, :] * K * GS + a[:, None] * GS + local[None, :]
        saved_token_off = blocks[None, :] * C * GS + i[:, None] * GS + local[None, :]
        scalar_off = blocks[None, :] * C + i[:, None]
        vo = ((start + i[:, None]) * H + head) * VD + columns[None, :]
        value = tl.load(V0 + vo, i[:, None] < length, 0.0).to(tl.float32)
        y = mm(load_matrix(P[5], ch, C, K), state, C, BV, K)
        tl.store(SavedS + saved_state_off, state)
        tl.store(SavedY + saved_token_off, y)
        beta = load_vector(P[21], ch, C)
        n = load_vector(P[18], ch, C)
        nu = tl.load(Gp[7] + scalar_off)
        w_all = tl.full((C, BV), 0.0, tl.float32)
        last_r = tl.full((BV,), 0.0, tl.float32)
        last_t = tl.full((BV,), 0.0, tl.float32)
        last_nu = tl.full((BV,), 0.0, tl.float32)
        for group_offset in tl.static_range(BV // GS):
            group = tile * (BV // GS) + group_offset
            block = ch * NG + group
            selected = groups == group
            h = tl.sum(tl.where(selected[None, :], value * y, 0.0), 1)
            nus = load_vector(Gp[7], block, C)
            af = load_vector(Gp[5], block, C)
            lam = load_vector(Gp[6], block, C)
            rhs = load_vector(Gp[8], block, C) * nus * (h + beta * exclusive_prefix(h, C))
            w = af - kap * pointer_mv(Gp[0], block, rhs, C)
            r, t, zp = mass_forward_scan(beta, lam, n, w, C)
            store_vector(SavedR, block, r, C)
            store_vector(SavedT, block, t, C)
            store_vector(SavedW, block, w, C)
            store_vector(SavedZP, block, zp, C)
            w_all = tl.where(selected[None, :], w[:, None], w_all)
            last_r = tl.where(selected, at(r, length - 1, C), last_r)
            last_t = tl.where(selected, at(t, length - 1, C), last_t)
            last_nu = tl.where(selected, at(nus, length - 1, C), last_nu)
        wave = w_all * value
        scaled_wave = (w_all / nu) * value
        aa = load_matrix(P[11], ch, C, C)
        strict_a = tl.where(i[:, None] > i[None, :], aa, 0.0)
        state_part = nu * y
        u = wave + nu * mm(strict_a, scaled_wave, C, BV, C) - state_part
        tl.store(SavedU + saved_token_off, u)
        residual_sizes = sum_value_groups(tl.abs(u), C, BV, GS)
        wave_sizes = sum_value_groups(tl.abs(wave), C, BV, GS)
        scaled_sizes = sum_value_groups(tl.abs(scaled_wave), C, BV, GS)
        state_sizes = sum_value_groups(tl.abs(state_part), C, BV, GS)
        group_ids = tile * (BV // GS) + tl.arange(0, BV // GS).to(tl.int64)
        bounds = tl.full((C, BV // GS), 0.0, tl.float32)
        for group_offset in tl.static_range(BV // GS):
            group = tile * (BV // GS) + group_offset
            selected = group_ids == group
            nus = load_vector(Gp[7], ch * NG + group, C)
            sizes = tl.sum(tl.where(selected[None, :], scaled_sizes, 0.0), 1)
            bound = nus * tl.sum(tl.abs(strict_a) * sizes[None, :], 1)
            bound += tl.sum(tl.where(selected[None, :], wave_sizes + state_sizes, 0.0), 1)
            bounds = tl.where(selected[None, :], bound[:, None], bounds)
        tl.debug_barrier()
        for group_offset in tl.static_range(BV // GS):
            group = tile * (BV // GS) + group_offset
            block = ch * NG + group
            selected = group_ids == group
            bound = tl.sum(tl.where(selected[None, :], bounds, 0.0), 1)
            residual = tl.sum(tl.where(selected[None, :], residual_sizes, 0.0), 1)
            fallback = tl.max(((4 * residual < bound) & (bound > 0) & (i < length)).to(tl.int32), 0) != 0
            tl.store(Fallback + block, fallback.to(tl.int32))
            if fallback:
                nus = load_vector(Gp[7], block, C)
                gg = tl.arange(0, GS).to(tl.int64)
                off = ((start + i[:, None]) * H + head) * VD + group * GS + gg[None, :]
                vg = tl.load(V0 + off, i[:, None] < length, 0.0).to(tl.float32)
                yg = load_matrix(SavedY, block, C, GS)
                wg = load_vector(SavedW, block, C)
                stable_u = wg[:, None] * vg - nus[:, None] * pointer_mm(P[10], ch, yg, C, GS, C)
                offset = ch * C * C + i[:, None] * C + i[None, :]
                stable_a = tl.load(P[11] + offset, volatile=True)
                stable_delta = nus[:, None] * mm_precise(stable_a, stable_u / nus[:, None], C, GS, C)
                store_matrix(SavedU, block, stable_delta, C, GS)
        tl.debug_barrier()
        u = tl.load(SavedU + saved_token_off, volatile=True)
        # the same bounded-gauge admission used by grouped output covers these ratios
        us = tl.where(i[:, None] < length - 1, (last_nu[None, :] / nu) * u, 0.0)
        kl = tl.load(P[1] + ch * C * K + (length - 1).to(tl.int64) * K + a)
        tau = tl.load(P[2] + ch * C * K + (length - 1).to(tl.int64) * K + a)
        gamma = tl.load(P[19] + ch * C + length - 1)
        rs = erase(kl, gamma, tau[:, None] * (last_nu[None, :] * state)
                   + mm(load_matrix(P[16], ch, K, C), us, K, BV, C))
        last_value = tl.load(V0 + ((end - 1) * H + head) * VD + columns).to(tl.float32)
        ts = kl[:, None] * last_value[None, :] / at(n, length - 1, C)
        z = last_r + last_t
        state = rs / z[None, :] + (last_t / z)[None, :] * ts
    tl.store(Final + state_off, state)


@triton.jit
def state_direct_grouped_bwd_kernel(V0, Curvature, DS, DS0, Starts, Ends, SeqChunks, P, Gp, SavedS,
                                    Local, Sh, OutputAdj, H: tl.constexpr, VD: tl.constexpr, GS: tl.constexpr,
                                    NG: tl.constexpr, C: tl.constexpr, K: tl.constexpr, BV: tl.constexpr):
    tl.static_assert(C == 64 and K == 128 and GS == 32 and BV == 64 and VD % BV == 0)
    shv = tl.program_id(0).to(tl.int64)
    tile, head, seq = shv % (VD // BV), shv // (VD // BV) % H, shv // ((VD // BV) * H)
    columns = tile * BV + tl.arange(0, BV).to(tl.int64)
    groups, local = columns // GS, columns % GS
    group_ids = tile * (BV // GS) + tl.arange(0, BV // GS).to(tl.int64)
    a = tl.arange(0, K).to(tl.int64)
    i = tl.arange(0, C).to(tl.int64)
    state_off = ((seq * H + head) * K + a[:, None]) * VD + columns[None, :]
    bs = tl.load(DS + state_off).to(tl.float32)
    kap = tl.load(Curvature).to(tl.float32) / GS
    first, last = tl.load(SeqChunks + seq), tl.load(SeqChunks + seq + 1)
    for step in range(last - first):
        chunk = last - 1 - step
        ch = chunk * H + head
        blocks = ch * NG + groups
        start, end = tl.load(Starts + chunk), tl.load(Ends + chunk)
        length = end - start
        saved_state_off = blocks[None, :] * K * GS + a[:, None] * GS + local[None, :]
        saved_token_off = blocks[None, :] * C * GS + i[:, None] * GS + local[None, :]
        scalar_off = blocks[None, :] * C + i[:, None]
        group_scalar_off = (ch * NG + group_ids[None, :]) * C + i[:, None]
        nu = tl.load(Gp[7] + scalar_off)
        last_scalar_off = blocks * C + length - 1
        nul = tl.load(Gp[7] + last_scalar_off)
        rl, tt = tl.load(Local[1] + last_scalar_off), tl.load(Local[2] + last_scalar_off)
        zz = rl + tt
        n = load_vector(P[18], ch, C)
        beta = load_vector(P[21], ch, C)
        kl = tl.load(P[1] + ch * C * K + (length - 1).to(tl.int64) * K + a)
        tau = tl.load(P[2] + ch * C * K + (length - 1).to(tl.int64) * K + a)
        gamma = tl.load(P[19] + ch * C + length - 1)
        ss = nul[None, :] * tl.load(SavedS + saved_state_off)
        last_value = tl.load(V0 + ((end - 1) * H + head) * VD + columns).to(tl.float32)
        ts = kl[:, None] * last_value[None, :] / at(n, length - 1, C)
        norm_s = tl.sum(sum_value_groups(bs * erase(kl, gamma, tau[:, None] * ss) / zz[None, :], K, BV, GS), 0)
        dot_s = tl.sum(sum_value_groups(bs * ts, K, BV, GS), 0)
        projected = mm_backward(tl.trans(load_matrix(P[16], ch, K, C)), erase(kl, gamma, bs), C, BV, K)
        rhol = nul[None, :] / nu
        u = tl.load(Local[0] + saved_token_off)
        us = tl.where(i[:, None] < length - 1, rhol * u, 0.0)
        norm_s += tl.sum(sum_value_groups(projected / zz[None, :] * us, C, BV, GS), 0)
        gu = tl.load(OutputAdj[1] + saved_token_off)
        gu += tl.where(i[:, None] < length - 1, rhol * projected / zz[None, :], 0.0)
        gr, gt = tl.load(OutputAdj[2] + group_scalar_off), tl.load(OutputAdj[3] + group_scalar_off)
        last_group_off = (ch * NG + group_ids) * C + length - 1
        rg, tg = tl.load(Local[1] + last_group_off), tl.load(Local[2] + last_group_off)
        zg = rg + tg
        gr += tl.where(i[:, None] == length - 1, (-(norm_s + tg / zg * dot_s) / zg)[None, :], 0.0)
        gt += tl.where(i[:, None] == length - 1, ((rg / zg * dot_s - norm_s) / zg)[None, :], 0.0)
        vo = ((start + i[:, None]) * H + head) * VD + columns[None, :]
        value = tl.load(V0 + vo, i[:, None] < length, 0.0).to(tl.float32)
        gu_old = mm_backward(tl.trans(load_matrix(P[11], ch, C, C)), nu * gu, C, BV, C) / nu
        gw_value = sum_value_groups(gu_old * value, C, BV, GS)
        b_all = tl.full((C, BV), 0.0, tl.float32)
        for group_offset in tl.static_range(BV // GS):
            group = tile * (BV // GS) + group_offset
            block = ch * NG + group
            selected = group_ids == group
            grs = tl.sum(tl.where(selected[None, :], gr, 0.0), 1)
            gts = tl.sum(tl.where(selected[None, :], gt, 0.0), 1)
            w, zp = load_vector(Local[4], block, C), load_vector(Local[5], block, C)
            lam = load_vector(Gp[6], block, C)
            gb, gn, gw = mass_reverse_scan(beta, lam, n, w, zp, grs, gts, C)
            gw += tl.sum(tl.where(selected[None, :], gw_value, 0.0), 1)
            f = load_vector(Gp[8], block, C) * pointer_mv(Gp[0], block, gw, C, True)
            b = mass_system_transpose(f, beta, lam, C)
            source_n, source_beta = mass_system_sources(b, beta, lam, w, zp, C)
            store_vector(OutputAdj[2], block, b, C)
            store_vector(Sh[8], block, gn + source_n, C)
            store_vector(Sh[12], block, gb + source_beta, C)
            b_all = tl.where((groups == group)[None, :], b[:, None], b_all)
        residual = gu + kap * b_all * value
        tl.store(OutputAdj[1] + saved_token_off, residual)
        # load the output state adjoint after the scalar feedback work
        gs = tl.load(OutputAdj[0] + saved_state_off)
        gs += nul[None, :] * tau[:, None] * erase(kl, gamma, bs / zz[None, :])
        tl.debug_barrier()
        tl.store(OutputAdj[0] + saved_state_off, bs)
        bs = gs - mm_backward(load_matrix(P[17], ch, K, C), nu * residual, K, BV, C)
    tl.store(DS0 + state_off, bs)
