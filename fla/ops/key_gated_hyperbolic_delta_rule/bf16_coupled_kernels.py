# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Direct source pullbacks for the C64 coupled residual equations."""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from fla.ops.key_gated_hyperbolic_delta_rule.bf16_selective_kernels import (
    centered_diagonal_scale,
    erase,
    interval_ell,
    interval_tile_load,
    load_matrix,
    load_vector,
    mm_left_two,
    mm_precise,
    read_rho_matrix,
    read_rho_row,
    store_matrix,
)


@triton.jit
def coupled_residual_bwd(V0, Starts, Ends, P, Gp, Curvature, SavedU, OutputAdj, Coupled, H: tl.constexpr,
                         VD: tl.constexpr, NG: tl.constexpr, GS: tl.constexpr, C: tl.constexpr, G: tl.constexpr,
                         RHO_MATERIALIZED: tl.constexpr, PI_READY: tl.constexpr = False):
    block = tl.program_id(0).to(tl.int64)
    ch = block // NG
    i = tl.arange(0, C)
    a = load_matrix(P[11], ch, C, C)
    rho = read_rho_matrix(Gp, block, i, i, C, RHO_MATERIALIZED)
    ahat = a * rho
    u = load_matrix(SavedU, block, C, G)
    delta = mm_precise(ahat, u, C, G, C)
    store_matrix(Coupled[0], block, delta, C, G)
    if not PI_READY:
        g = tl.arange(0, G).to(tl.int64)
        start, end = tl.load(Starts + ch // H), tl.load(Ends + ch // H)
        off = ((start + i[:, None].to(tl.int64)) * H + ch % H) * VD + (block % NG) * GS + g[None, :]
        v = tl.load(V0 + off, (i[:, None] < end - start) & (g[None, :] < GS), 0.0).to(tl.float32)
        bv = load_vector(OutputAdj[2], block, C)[:, None] * v
        kap = tl.load(Curvature).to(tl.float32) / GS
        pi = load_matrix(OutputAdj[1], block, C, G) + kap * mm_precise(tl.trans(ahat), bv, C, G, C)
        store_matrix(Coupled[1], block, pi, C, G)


@triton.jit
def coupled_sources_bwd(DO, Starts, Ends, P, Gp, SavedS, SavedR, SavedT, OutputAdj, Coupled, Sh, Work,
                        H: tl.constexpr, VD: tl.constexpr, GS: tl.constexpr, NG: tl.constexpr,
                        C: tl.constexpr, K: tl.constexpr, G: tl.constexpr, MODE: tl.constexpr,
                        RHO_MATERIALIZED: tl.constexpr):
    M: tl.constexpr = 32
    N: tl.constexpr = 64
    ch = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1).to(tl.int64) * M + tl.arange(0, M).to(tl.int64)
    j = tl.program_id(2).to(tl.int64) * N + tl.arange(0, N).to(tl.int64)
    g = tl.arange(0, G).to(tl.int64)
    start, end = tl.load(Starts + ch // H), tl.load(Ends + ch // H)
    last = end - start - 1
    total = tl.full((M, N), 0.0, tl.float32)
    for group in range(NG):
        block = ch * NG + group
        if MODE == 3:
            a = tl.arange(0, K).to(tl.int64)
            z = tl.load(SavedR + block * C + last) + tl.load(SavedT + block * C + last)
            grs = load_matrix(OutputAdj[0], block, K, G) / z
            kl = tl.load(P[1] + ch * C * K + last * K + a)
            gamma = tl.load(P[19] + ch * C + last)
            grs = erase(kl, gamma, grs)
            left = tl.gather(grs, tl.broadcast_to(i[:, None].to(tl.int32), (M, G)), 0)
            right = tl.load(Coupled[0] + block * C * G + j[:, None] * G + g[None, :])
            rho = read_rho_row(Gp, block, last, j, C, RHO_MATERIALIZED)
            right = tl.where(j[:, None] < last, rho[:, None] * right, 0.0)
        else:
            if MODE == 2:
                off = ((start + i[:, None]) * H + ch % H) * VD + group * GS + g[None, :]
                z = tl.load(SavedR + block * C + i) + tl.load(SavedT + block * C + i)
                left = tl.load(DO + off, (i[:, None] < end - start) & (g[None, :] < GS), 0.0).to(tl.float32) / z[:, None]
            else:
                left = tl.load(Coupled[1] + block * C * G + i[:, None] * G + g[None, :])
                if MODE == 1:
                    left *= -tl.load(Gp[7] + block * C + i)[:, None]
                else:
                    left = -left
            if MODE == 1:
                right = tl.load(SavedS + block * K * G + j[:, None] * G + g[None, :])
            else:
                right = tl.load(Coupled[0] + block * C * G + j[:, None] * G + g[None, :])
        product = mm_precise(left, tl.trans(right), M, N, G)
        if MODE == 0 or MODE == 2:
            rho = read_rho_matrix(Gp, block, i, j, C, RHO_MATERIALIZED)
            product = tl.where(i[:, None] > j[None, :], rho * product, 0.0)
        total += product
    if MODE == 0:
        tl.store(Sh[0] + ch * C * C + i[:, None] * C + j[None, :], total)
    elif MODE == 1:
        tl.store(Work[3] + ch * C * K + i[:, None] * K + j[None, :], total)
    elif MODE == 2:
        tl.store(Sh[1] + ch * C * C + i[:, None] * C + j[None, :], total)
    else:
        tl.store(Work[0] + ch * K * C + i[:, None] * C + j[None, :], total)


@triton.jit
def coupled_diagonal_sources(Ell, Starts, Ends, P, Sh, Flags, ch, tile: tl.constexpr, a,
                             H: tl.constexpr, KD: tl.constexpr, C: tl.constexpr, K: tl.constexpr):
    i = tl.arange(0, 32).to(tl.int64)
    start, end = tl.load(Starts + ch // H), tl.load(Ends + ch // H)
    k = interval_tile_load(P[1], ch, tile, a, C, K)
    qr = interval_tile_load(P[6], ch, tile, a, C, K)
    off = ch * C * C + (tile * 32 + i[:, None]) * C + tile * 32 + i[None, :]
    if tl.load(Flags + ch * 2 + tile):
        decay = centered_diagonal_scale(Ell, start, end - start, ch % H, tile, a, H, KD, C)
        gg, gc = tl.load(Sh[0] + off), tl.load(Sh[1] + off)
        inverse = k / decay
        rowk = mm_left_two(gg, inverse, 32, 32, 32) * decay
        rowq = mm_left_two(gc, inverse, 32, 32, 32) * decay
        colk = (mm_left_two(tl.trans(gg), k * decay, 32, 32, 32)
                + mm_left_two(tl.trans(gc), qr * decay, 32, 32, 32)) / decay
        dk, dq = rowk + colk, rowq
        incidence = k * rowk + qr * rowq - k * colk
    else:
        dk = tl.full((32, 32), 0.0, tl.float32)
        dq = tl.full((32, 32), 0.0, tl.float32)
        incidence = tl.full((32, 32), 0.0, tl.float32)
        ell = interval_ell(Ell, start, end - start, ch % H, tile, a, H, KD, C)
        for j in range(32):
            interval = libdevice.exp(tl.cumsum(tl.where(i[:, None] > j, ell, 0.0), 0))
            kj = tl.sum(tl.where(i[:, None] == j, k, 0.0), 0)
            goff = ch * C * C + (tile * 32 + i) * C + tile * 32 + j
            gg, gc = tl.load(Sh[0] + goff), tl.load(Sh[1] + goff)
            left = gg[:, None] * k + gc[:, None] * qr
            dk += gg[:, None] * kj[None, :] * interval
            dq += gc[:, None] * kj[None, :] * interval
            col = tl.sum(left * interval, 0)
            dk += tl.where(i[:, None] == j, col[None, :], 0.0)
            weight = left * kj[None, :] * interval
            incidence += weight - tl.where(i[:, None] == j, tl.sum(weight, 0)[None, :], 0.0)
    return dk, dq, incidence


@triton.jit
def coupled_interval_sources_bwd(Ell, DQ, DK, DE, Starts, Ends, P, Sh, Work, Flags,
                                 H: tl.constexpr, KD: tl.constexpr, NG: tl.constexpr,
                                 C: tl.constexpr, K: tl.constexpr):
    tl.static_assert(C == 64 and K == 128)
    ch = tl.program_id(0).to(tl.int64)
    owner = tl.program_id(1).to(tl.int64)
    ktile = tl.program_id(2).to(tl.int64)
    a = ktile * 32 + tl.arange(0, 32).to(tl.int64)
    i = tl.arange(0, 32).to(tl.int64)
    rows = owner * 32 + i
    start, end = tl.load(Starts + ch // H), tl.load(Ends + ch // H)
    k = interval_tile_load(P[1], ch, owner, a, C, K)
    qr = interval_tile_load(P[6], ch, owner, a, C, K)
    goff = ch * C * C + (32 + i[:, None]) * C + i[None, :]
    gg, gc = tl.load(Sh[0] + goff), tl.load(Sh[1] + goff)
    if owner == 0:
        prefix = interval_tile_load(P[3], ch, 1, a, C, K)
        kr = interval_tile_load(P[1], ch, 1, a, C, K) * prefix
        qrr = interval_tile_load(P[6], ch, 1, a, C, K) * prefix
        crossk = (mm_left_two(tl.trans(gg), kr, 32, 32, 32)
                  + mm_left_two(tl.trans(gc), qrr, 32, 32, 32)) * interval_tile_load(P[4], ch, 0, a, C, K)
        crossq = tl.full((32, 32), 0.0, tl.float32)
        incidence = -k * crossk
        dk, dq, diag_incidence = coupled_diagonal_sources(Ell, Starts, Ends, P, Sh, Flags, ch, 0, a, H, KD, C, K)
    else:
        kc = interval_tile_load(P[1], ch, 0, a, C, K) * interval_tile_load(P[4], ch, 0, a, C, K)
        prefix = interval_tile_load(P[3], ch, 1, a, C, K)
        crossk = mm_left_two(gg, kc, 32, 32, 32) * prefix
        crossq = mm_left_two(gc, kc, 32, 32, 32) * prefix
        incidence = k * crossk + qr * crossq
        dk, dq, diag_incidence = coupled_diagonal_sources(Ell, Starts, Ends, P, Sh, Flags, ch, 1, a, H, KD, C, K)
    dk += crossk
    dq += crossq
    incidence += diag_incidence
    pos = ch * C * K + rows[:, None] * K + a[None, :]
    decay = tl.load(P[2] + pos)
    gbmat = tl.load(Work[3] + pos)
    gamma = tl.load(P[19] + ch * C + rows)
    gke = gamma[:, None] * gbmat
    gqer = tl.load(Sh[5] + pos)
    tail = interval_tile_load(P[4], ch, owner, a, C, K)
    if owner == 0:
        tail *= tl.load(P[3] + ch * C * K + 63 * K + a)[None, :]
    dk += tl.load(Sh[7] + pos) * tail + decay * gke
    dk += tl.where(rows[:, None] == end - start - 1, tl.load(Work[2] + ch * K + a)[None, :], 0.0)
    dq += gqer * decay
    direct_tail = tl.full((32,), 0.0, tl.float32)
    for group in range(NG):
        direct_tail += tl.load(Sh[13] + (ch * NG + group) * K + a)
    gd = tl.where(rows[:, None] == end - start - 1, direct_tail[None, :], 0.0)
    gd += gqer * qr + k * gke
    incidence += gd * decay
    off = ((start + rows[:, None]) * H + ch % H) * KD + a[None, :]
    valid = (rows[:, None] < end - start) & (a[None, :] < KD)
    tl.store(DQ + off, dq, valid)
    tl.store(DK + off, dk, valid)
    tl.store(DE + off, incidence, valid)
    tl.store(Work[0] + (ch * 4 + ktile) * C + rows, tl.sum(gbmat * (decay * k), 1))


@triton.jit
def coupled_interval_gate_bwd(DE, Starts, Ends, P, Sh, H: tl.constexpr, KD: tl.constexpr,
                              C: tl.constexpr, K: tl.constexpr):
    ch = tl.program_id(0).to(tl.int64)
    a = tl.program_id(1).to(tl.int64) * 32 + tl.arange(0, 32).to(tl.int64)
    i = tl.arange(0, C).to(tl.int64)
    start, end = tl.load(Starts + ch // H), tl.load(Ends + ch // H)
    pos = ch * C * K + i[:, None] * K + a[None, :]
    tail = tl.load(P[4] + pos)
    extra = tl.load(P[3] + ch * C * K + 63 * K + a)
    tail *= tl.where(i[:, None] < 32, extra[None, :], 1.0)
    tailgrad = tl.load(Sh[7] + pos) * tail * tl.load(P[1] + pos)
    off = ((start + i[:, None]) * H + ch % H) * KD + a[None, :]
    valid = (i[:, None] < end - start) & (a[None, :] < KD)
    value = tl.cumsum(tl.load(DE + off, valid, 0.0), 0, reverse=True)
    inclusive = tl.cumsum(tailgrad, 0)
    previous = tl.gather(inclusive, tl.broadcast_to(tl.maximum(i - 1, 0)[:, None].to(tl.int32), (C, 32)), 0)
    tl.store(DE + off, value + tl.where(i[:, None] > 0, previous, 0.0), valid)


@triton.jit
def coupled_gamma_sources_bwd(Sh, Work, DB, Starts, Ends, H: tl.constexpr, NG: tl.constexpr, C: tl.constexpr):
    ch = tl.program_id(0).to(tl.int64)
    i = tl.arange(0, C).to(tl.int64)
    start, end = tl.load(Starts + ch // H), tl.load(Ends + ch // H)
    value = tl.full((C,), 0.0, tl.float32)
    for group in range(NG):
        value += tl.load(Sh[9] + (ch * NG + group) * C + i)
    for tile in range(4):
        value += tl.load(Work[0] + (ch * 4 + tile) * C + i)
    off = (start + i) * H + ch % H
    value += tl.load(DB + off, i < end - start, 0.0)
    tl.store(DB + off, value, i < end - start)
