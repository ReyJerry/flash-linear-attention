# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Experimental direct residual state coordinates with a stable cancellation branch."""
import triton
import triton.language as tl

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
    mm_precise,
    mm_state_backward,
    mm_state_three,
    mm_three,
    pointer_mm,
    pointer_mm_backward,
    pointer_mv,
    solve_lower_action,
    solve_upper_transpose_action,
    read_rho_row,
    row,
    store_matrix,
    store_vector,
)


@triton.jit
def common_direct_guard(P, C: tl.constexpr):
    tl.static_assert(C == 64)
    ch = tl.program_id(0).to(tl.int64)
    i = tl.arange(0, C).to(tl.int64)
    matrix = load_matrix(P[11], ch, C, C)
    strict = tl.where(i[:, None] > i[None, :], tl.abs(matrix), 0.0)
    bound = tl.sum(tl.reshape(strict, (C, C // 8, 8)), 2)
    # cover FP32 summation error while preserving a conservative cancellation bound
    bound *= 1.0 + 2.0 ** -16
    store_matrix(P[14], ch, bound, C, C // 8)


@triton.jit
def strict_inverse_product(P, ch, wave, C: tl.constexpr, G: tl.constexpr):
    i = tl.arange(0, C)
    matrix = load_matrix(P[11], ch, C, C)
    return mm_three(tl.where(i[:, None] > i[None, :], matrix, 0.0), wave, C, G, C)


@triton.jit
def local_direct_state_graph(P, Gp, ch, block, v, s, y, length, kap, C: tl.constexpr, K: tl.constexpr, G: tl.constexpr,
                            MIXED: tl.constexpr = False, RHO_MATERIALIZED: tl.constexpr = True,
                            FUSE_GROUP_RHS: tl.constexpr = False, ACTION_SOLVE: tl.constexpr = False):
    tl.static_assert(not RHO_MATERIALIZED)
    i = tl.arange(0, C)
    n = load_vector(P[18], ch, C)
    gamma = load_vector(P[19], ch, C)
    beta = load_vector(P[21], ch, C)
    lam = load_vector(Gp[6], block, C)
    nu = load_vector(Gp[7], block, C)
    h = tl.sum(v * y, 1)
    prefix_h = exclusive_prefix(h, C)
    diagonal_inverse = load_vector(Gp[8], block, C)
    if FUSE_GROUP_RHS:
        rhs = diagonal_inverse * nu * (beta - kap * (h + beta * prefix_h))
        w = solve_lower_action(Gp[1], block, rhs, C) if ACTION_SOLVE else pointer_mv(Gp[0], block, rhs, C)
    else:
        if ACTION_SOLVE:
            af = solve_lower_action(Gp[1], block, diagonal_inverse * nu * beta, C)
            auxiliary = solve_lower_action(Gp[1], block, diagonal_inverse * nu * (h + beta * prefix_h), C)
            w = af - kap * auxiliary
        else:
            af = load_vector(Gp[5], block, C)
            w = af - kap * pointer_mv(Gp[0], block, diagonal_inverse * nu * (h + beta * prefix_h), C)
    # keep the unit diagonal exact; bounded admission keeps the inverse scale finite
    scaled_wave = (w / nu)[:, None] * v
    target_part = w[:, None] * v + nu[:, None] * strict_inverse_product(P, ch, scaled_wave, C, G)
    state_part = nu[:, None] * y
    u = target_part - state_part
    wave_sizes = tl.sum(tl.abs(scaled_wave), 1)
    block_max = tl.max(tl.reshape(wave_sizes, (C // 8, 8)), 1)
    bound = load_matrix(P[14], ch, C, C // 8)
    offdiagonal_size = nu * tl.sum(bound * block_max[None, :], 1)
    residual_size = tl.sum(tl.abs(u), 1)
    target_size = tl.sum(tl.abs(w[:, None] * v), 1)
    operand_size = target_size + offdiagonal_size + tl.sum(tl.abs(state_part), 1)
    # include cancellation inside the contraction as well as the final subtraction
    fallback = tl.max(((4 * residual_size < operand_size) & (operand_size > 0)
                       & (i < length)).to(tl.int32), 0) != 0
    if fallback:
        stable_u = w[:, None] * v - nu[:, None] * pointer_mm(P[10], ch, y, C, G, C)
        # reload inside the branch so the full coefficient tile does not stay live
        offset = ch * C * C + i[:, None].to(tl.int64) * C + i[None, :].to(tl.int64)
        stable_a = tl.load(P[11] + offset, volatile=True)
        u = nu[:, None] * mm_precise(stable_a, stable_u / nu[:, None], C, G, C)
    if C >= 16:
        (r, t, zp) = mass_forward_scan(beta, lam, n, w, C)
    else:
        prev = tl.full((), 1, tl.float32)
        r = tl.full((C,), 0, tl.float32)
        t = tl.full((C,), 0, tl.float32)
        zp = tl.full((C,), 0, tl.float32)
        for j in range(C):
            rj = (1 - at(beta, j, C)) / at(lam, j, C) * prev
            tj = at(n, j, C) * at(w, j, C)
            zp = tl.where(i == j, prev, zp)
            (r, t) = (tl.where(i == j, rj, r), tl.where(i == j, tj, t))
            prev = rj + tj
    z = r + t
    a = tl.arange(0, K).to(tl.int64)
    last = (length - 1).to(tl.int64)
    kl = tl.load(P[1] + ch * C * K + last * K + a)
    tau = tl.load(P[2] + ch * C * K + last * K + a)
    gl = at(gamma, length - 1, C)
    ss = at(nu, length - 1, C) * s
    rhol = read_rho_row(Gp, block, last, i, C, RHO_MATERIALIZED)
    us = tl.where(i[:, None] < length - 1, rhol[:, None] * u, 0.0)
    rs = erase(kl, gl, tau[:, None] * ss + pointer_mm(P[16], ch, us, K, G, C))
    ts = kl[:, None] * row(v, length - 1, C, G)[None, :] / at(n, length - 1, C)
    (zl, tl_last) = (at(z, length - 1, C), at(t, length - 1, C))
    sn = rs / zl + tl_last / zl * ts
    return (sn, u, r, t, ss, us, w, zp, fallback)


@triton.jit
def state_direct_fwd_kernel(V0, Curvature, S0, Out, Final, Starts, Ends, SeqChunks, P, Gp, SavedS, SavedY, SavedU, SavedR, SavedT, SavedW, SavedZP, Carry, H: tl.constexpr, KD: tl.constexpr, VD: tl.constexpr, GS: tl.constexpr, NG: tl.constexpr, C: tl.constexpr, K: tl.constexpr, G: tl.constexpr, FINAL: tl.constexpr, CARRY: tl.constexpr, MIXED: tl.constexpr = False, BF16_FS6: tl.constexpr = False, RHO_MATERIALIZED: tl.constexpr = True, FUSE_GROUP_RHS: tl.constexpr = False, ACTION_SOLVE: tl.constexpr = False, Fallback=None):
    tl.static_assert(not CARRY and C == 64)
    kap = tl.load(Curvature).to(tl.float32) / GS
    shg = tl.program_id(0).to(tl.int64)
    (group, head, seq) = (shg % NG, shg // NG % H, shg // (NG * H))
    (a, g, i) = (tl.arange(0, K), tl.arange(0, G), tl.arange(0, C))
    so = ((seq * H + head) * KD + a[:, None].to(tl.int64)) * VD + group * GS + g[None, :].to(tl.int64)
    smask = (a[:, None] < KD) & (g[None, :] < GS)
    s = tl.load(S0 + so, smask, 0).to(tl.float32)
    (first, last) = (tl.load(SeqChunks + seq), tl.load(SeqChunks + seq + 1))
    yn = tl.full((C, G), 0, tl.float32)
    for chunk in range(first, last):
        ch = chunk * H + head
        block = ch * NG + group
        (start, end) = (tl.load(Starts + chunk), tl.load(Ends + chunk))
        vo = ((start + i.to(tl.int64))[:, None] * H + head) * VD + group * GS + g[None, :].to(tl.int64)
        vmask = (i[:, None] < end - start) & (g[None, :] < GS)
        if CARRY and chunk > first:
            y = yn
        elif BF16_FS6:
            tl.static_assert(not MIXED and (not CARRY))
            y = pointer_mm(P[5], ch, s, C, G, K)
        else:
            y = pointer_mm(P[5], ch, s, C, G, K)
        store_matrix(SavedS, block, s, K, G)
        store_matrix(SavedY, block, y, C, G)
        v = tl.load(V0 + vo, vmask, 0).to(tl.float32)
        (s, u, r, t, ss, us, w, zp, fallback) = local_direct_state_graph(P, Gp, ch, block,
                                                                         v, s, y, end - start, kap, C, K, G, MIXED, RHO_MATERIALIZED,
                                                                         FUSE_GROUP_RHS, ACTION_SOLVE)
        if CARRY and chunk + 1 < last:
            e = load_matrix(Carry[0], ch, C, K)
            hc = load_matrix(Carry[1], ch, C, C)
            fn = load_matrix(P[5], ch + H, C, K)
            kl = row(load_matrix(P[1], ch, C, K), end - start - 1, C, K)
            nl = at(load_vector(P[18], ch, C), end - start - 1, C)
            ry = mm_state_three(e, ss, C, G, K, MIXED) + mm_state_three(hc, us, C, G, C, MIXED)
            ty = (tl.sum(fn * kl[None, :], 1) / nl)[:, None] * row(v, end - start - 1, C, G)[None, :]
            (zz, tt) = (at(r + t, end - start - 1, C), at(t, end - start - 1, C))
            yn = ry / zz + tt / zz * ty
        tl.store(Fallback + block, fallback.to(tl.int32))
        store_matrix(SavedU, block, u, C, G)
        store_vector(SavedR, block, r, C)
        store_vector(SavedT, block, t, C)
        store_vector(SavedW, block, w, C)
        store_vector(SavedZP, block, zp, C)
    if FINAL:
        tl.store(Final + so, s, smask)


@triton.jit
def state_direct_bwd_kernel(V0, Curvature, DS, DV, DS0, Starts, Ends, SeqChunks, P, Gp, SavedS, Local, Sh, OutputAdj, Carry, EdgeY, H: tl.constexpr, KD: tl.constexpr, VD: tl.constexpr, GS: tl.constexpr, NG: tl.constexpr, C: tl.constexpr, K: tl.constexpr, G: tl.constexpr, CARRY: tl.constexpr, RHO_MATERIALIZED: tl.constexpr = True, COUPLED_STATE: tl.constexpr = False, ACTION_SOLVE: tl.constexpr = False):
    tl.static_assert(COUPLED_STATE and not CARRY and C == 64)
    tl.static_assert(not RHO_MATERIALIZED)
    shg = tl.program_id(0).to(tl.int64)
    (group, head, seq) = (shg % NG, shg // NG % H, shg // (NG * H))
    (a, g, i) = (tl.arange(0, K), tl.arange(0, G), tl.arange(0, C))
    so = ((seq * H + head) * KD + a[:, None].to(tl.int64)) * VD + group * GS + g[None, :].to(tl.int64)
    smask = (a[:, None] < KD) & (g[None, :] < GS)
    bs = tl.load(DS + so, smask, 0).to(tl.float32)
    kap = tl.load(Curvature).to(tl.float32) / GS
    (first, last) = (tl.load(SeqChunks + seq), tl.load(SeqChunks + seq + 1))
    by = tl.full((C, G), 0, tl.float32)
    for step in range(last - first):
        chunk = last - 1 - step
        (ch, block) = (chunk * H + head, (chunk * H + head) * NG + group)
        (start, end) = (tl.load(Starts + chunk), tl.load(Ends + chunk))
        length = end - start
        vo = ((start + i.to(tl.int64))[:, None] * H + head) * VD + group * GS + g[None, :].to(tl.int64)
        mask = (i[:, None] < length) & (g[None, :] < GS)
        (r, t) = (load_vector(Local[1], block, C), load_vector(Local[2], block, C))
        n = load_vector(P[18], ch, C)
        nu = load_vector(Gp[7], block, C)
        z = r + t
        (rl, tt, zz) = (at(r, length - 1, C), at(t, length - 1, C), at(z, length - 1, C))
        kl = tl.load(P[1] + ch * C * K + (length - 1).to(tl.int64) * K + a)
        tau = tl.load(P[2] + ch * C * K + (length - 1).to(tl.int64) * K + a)
        gl = tl.load(P[19] + ch * C + (length - 1).to(tl.int64))
        ss = at(nu, length - 1, C) * load_matrix(SavedS, block, K, G)
        last_v = tl.load(V0 + ((end - 1) * H + head) * VD + group * GS + g.to(tl.int64), g < GS, 0.0).to(tl.float32)
        ts = kl[:, None] * last_v[None, :] / at(n, length - 1, C)
        adj = bs
        if CARRY:
            store_matrix(EdgeY, block, by, C, G)
        if CARRY and chunk + 1 < last:
            fn = load_matrix(P[5], ch + H, C, K)
            adj += mm_state_backward(tl.trans(fn), by, K, G, C, False)
        gx_norm = erase(kl, gl, adj / zz)
        norm_s = tl.sum(tl.sum(gx_norm * (tau[:, None] * ss), 0), 0)
        dot_s = tl.sum(tl.sum(adj * ts, 0), 0)
        gs = load_matrix(OutputAdj[0], block, K, G)
        gs += at(nu, length - 1, C) * tau[:, None] * gx_norm
        tl.debug_barrier()
        store_matrix(OutputAdj[0], block, adj, K, G)
        projected = pointer_mm_backward(P[16], ch, erase(kl, gl, bs), C, G, K, True)
        if CARRY and chunk + 1 < last:
            hc = load_matrix(Carry[1], ch, C, C)
            projected += mm_state_backward(tl.trans(hc), by, C, G, C, False)
        rhol = read_rho_row(Gp, block, length - 1, i, C, RHO_MATERIALIZED)
        u = load_matrix(Local[0], block, C, G)
        us = tl.where(i[:, None] < length - 1, rhol[:, None] * u, 0.0)
        norm_s += tl.sum(tl.sum(projected / zz * us, 0), 0)
        gu = load_matrix(OutputAdj[1], block, C, G)
        (gr, gt) = (load_vector(OutputAdj[2], block, C), load_vector(OutputAdj[3], block, C))
        gr += tl.where(i == length - 1, -(norm_s + tt / zz * dot_s) / zz, 0.0)
        gt += tl.where(i == length - 1, (rl / zz * dot_s - norm_s) / zz, 0.0)
        gu += tl.where(i[:, None] < length - 1, rhol[:, None] * projected / zz, 0.0)
        (w, zp) = (load_vector(Local[4], block, C), load_vector(Local[5], block, C))
        beta = load_vector(P[21], ch, C)
        lam = load_vector(Gp[6], block, C)
        if C >= 16:
            (gb, gn, gw) = mass_reverse_scan(beta, lam, n, w, zp, gr, gt, C)
        else:
            (gb, gn, gw) = (tl.full((C,), 0, tl.float32), tl.full((C,), 0, tl.float32), tl.full((C,), 0, tl.float32))
            gz = tl.full((), 0, tl.float32)
            for jj in range(C):
                j = C - 1 - jj
                (gri, gti) = (at(gr, j, C) + gz, at(gt, j, C) + gz)
                gb += tl.where(i == j, -gri * at(zp, j, C) / at(lam, j, C), 0.0)
                gz = gri * (1 - at(beta, j, C)) / at(lam, j, C)
                gn += tl.where(i == j, gti * at(w, j, C), 0.0)
                gw += tl.where(i == j, gti * at(n, j, C), 0.0)
        gu_old = mm_precise(tl.trans(load_matrix(P[11], ch, C, C)), nu[:, None] * gu, C, G, C) / nu[:, None]
        gw += tl.sum(gu_old * tl.load(V0 + vo, mask, 0, volatile=True).to(tl.float32), 1)
        f = load_vector(Gp[8], block, C) * (solve_upper_transpose_action(Gp[1], block, gw, C) if ACTION_SOLVE else pointer_mv(Gp[0], block, gw, C, True))
        b = mass_system_transpose(f, beta, lam, C)
        source_n, source_beta = mass_system_sources(b, beta, lam, w, zp, C)
        gn += source_n
        gb += source_beta
        if COUPLED_STATE:
            tl.static_assert(not CARRY and C == 64)
            v = tl.load(V0 + vo, mask, 0).to(tl.float32)
            pi = gu + kap * b[:, None] * v
            gs -= pointer_mm_backward(P[17], ch, nu[:, None] * pi, K, G, C)
            tl.debug_barrier()
            store_matrix(OutputAdj[1], block, pi, C, G)
        else:
            rhs = nu * b
            gh = -kap * rhs
            gy = -pointer_mm_backward(P[10], ch, nu[:, None] * gu, C, G, C, True)
            gy += gh[:, None] * tl.load(V0 + vo, mask, 0).to(tl.float32)
            tl.debug_barrier()
            store_matrix(OutputAdj[1], block, gu, C, G)
            if not CARRY or chunk == first:
                gs += pointer_mm_backward(P[5], ch, gy, K, G, C, True)
            tl.store(DV + vo, gy, mask)
            by = gy
        store_vector(OutputAdj[2], block, b, C)
        store_vector(Sh[8], block, gn, C)
        store_vector(Sh[12], block, gb, C)
        bs = gs
    tl.store(DS0 + so, bs, smask)


@triton.jit
def direct_residual_bwd(P, Gp, R, Pi, NG: tl.constexpr, C: tl.constexpr, G: tl.constexpr,
                        RHO_MATERIALIZED: tl.constexpr):
    tl.static_assert(not RHO_MATERIALIZED)
    block = tl.program_id(0).to(tl.int64)
    a_matrix = load_matrix(P[11], block // NG, C, C)
    nu = load_vector(Gp[7], block, C)
    residual = load_matrix(R, block, C, G)
    value = mm_precise(tl.trans(a_matrix), nu[:, None] * residual, C, G, C) / nu[:, None]
    store_matrix(Pi, block, value, C, G)


@triton.jit
def common_direct_transpose(P, C: tl.constexpr, K: tl.constexpr):
    tl.static_assert(C % 32 == 0 and K % 64 == 0)
    ch = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1).to(tl.int64) * 32 + tl.arange(0, 32).to(tl.int64)
    a = tl.program_id(2).to(tl.int64) * 64 + tl.arange(0, 64).to(tl.int64)
    source = ch * C * K + i[:, None] * K + a[None, :]
    target = ch * K * C + a[None, :] * C + i[:, None]
    tl.store(P[16] + target, tl.load(P[8] + source))
    tl.store(P[17] + target, tl.load(P[5] + source))
