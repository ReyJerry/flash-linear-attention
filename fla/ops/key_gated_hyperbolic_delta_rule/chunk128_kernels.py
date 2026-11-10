# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""BF16 contractions with FP32 state and first-order source VJPs."""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def mm(a, b, M: tl.constexpr, N: tl.constexpr, R: tl.constexpr):
    if M >= 16 and N >= 16 and R >= 16:
        return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16), out_dtype=tl.float32)
    acc = tl.full((M, N), 0, tl.float32)
    for r in range(R):
        x = tl.gather(a, tl.full((M, 1), r, tl.int32), 1)
        y = tl.gather(b, tl.full((1, N), r, tl.int32), 0)
        acc = acc + x * y
    return acc


@triton.jit
def round_bf16_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; cvt.f32.bf16 $0, h; }",
        constraints="=f,f", args=[x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def add_rn_f32(a, b):
    return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", constraints="=f,f,f",
                                     args=[a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def mm_precise(a, b, M: tl.constexpr, N: tl.constexpr, R: tl.constexpr):
    if M >= 16 and N >= 16 and R >= 16:
        ah32, bh32 = round_bf16_f32(a), round_bf16_f32(b)
        ah, bh = ah32.to(tl.bfloat16), bh32.to(tl.bfloat16)
        al, bl = (a - ah32).to(tl.bfloat16), (b - bh32).to(tl.bfloat16)
        correction = tl.dot(al, bh, out_dtype=tl.float32)
        correction += tl.dot(ah, bl, out_dtype=tl.float32)
        high = tl.dot(ah, bh, out_dtype=tl.float32)
        return add_rn_f32(high, correction)
    return mm(a, b, M, N, R)


@triton.jit
def mm_three(a, b, M: tl.constexpr, N: tl.constexpr, R: tl.constexpr):
    return mm(a, b, M, N, R)


@triton.jit
def mm_state_three(a, b, M: tl.constexpr, N: tl.constexpr, R: tl.constexpr, MIXED: tl.constexpr):
    return mm_three(a, b, M, N, R)


@triton.jit
def row(a, i, M: tl.constexpr, N: tl.constexpr):
    return tl.reshape(tl.gather(a, tl.full((1, N), i, tl.int32), 0), (N,))


@triton.jit
def at(a, i, N: tl.constexpr):
    return tl.sum(tl.where(tl.arange(0, N) == i, a, 0), 0)


@triton.jit
def load_matrix(p, block, M: tl.constexpr, N: tl.constexpr):
    x = tl.arange(0, M).to(tl.int64)
    y = tl.arange(0, N).to(tl.int64)
    return tl.load(p + block * M * N + x[:, None] * N + y[None, :]).to(tl.float32)


@triton.jit
def store_matrix(p, block, a, M: tl.constexpr, N: tl.constexpr):
    x = tl.arange(0, M).to(tl.int64)
    y = tl.arange(0, N).to(tl.int64)
    tl.store(p + block * M * N + x[:, None] * N + y[None, :], a)


@triton.jit
def load_vector(p, block, N: tl.constexpr):
    return tl.load(p + block * N + tl.arange(0, N).to(tl.int64)).to(tl.float32)


@triton.jit
def store_vector(p, block, a, N: tl.constexpr):
    tl.store(p + block * N + tl.arange(0, N).to(tl.int64), a)


@triton.jit
def inverse_lower(a, C: tl.constexpr):
    i = tl.arange(0, C)
    result = tl.full((C, C), 0, tl.float32)
    for r in range(C):
        ar = row(a, r, C, C)
        rhs = (i == r).to(tl.float32) - tl.sum(ar[:, None] * result, 0)
        value = rhs / at(ar, r, C)
        result = tl.where(i[:, None] == r, value[None, :], result)
    return result


@triton.jit
def erase(k, gamma, x):
    return x - gamma * k[:, None] * tl.sum((k[:, None] * x).to(tl.float32), 0)[None, :]


@triton.jit
def erase_vjp(k, gamma, x, grad):
    p = tl.sum((k[:, None] * x).to(tl.float32), 0)
    h = tl.sum((k[:, None] * grad).to(tl.float32), 0)
    gx = grad - gamma * k[:, None] * h[None, :]
    gk = -gamma * tl.sum(grad * p[None, :] + x * h[None, :], 1)
    gg = -tl.sum(p * h, 0)
    return gx, gk, gg


@triton.jit
def common_decay_kernel(Ell, Starts, Ends, P, H: tl.constexpr, KD: tl.constexpr,
                        C: tl.constexpr, K: tl.constexpr):
    ch = tl.program_id(0).to(tl.int64)
    a = tl.program_id(1).to(tl.int64) * 32 + tl.arange(0, 32).to(tl.int64)
    i = tl.arange(0, C).to(tl.int64)
    start, end = tl.load(Starts + ch // H), tl.load(Ends + ch // H)
    off = ((start + i[:, None]) * H + ch % H) * KD + a[None, :]
    ell = tl.load(Ell + off, (i[:, None] < end-start) & (a[None, :] < KD), 0.).to(tl.float32)
    decay = libdevice.exp(tl.cumsum(ell, 0))
    tl.store(P[2] + ch * C * K + i[:, None] * K + a[None, :], decay)


@triton.jit
def common_input_kernel(Q, K0, Beta, Starts, Ends, P,
                        H: tl.constexpr, KD: tl.constexpr, C: tl.constexpr, K: tl.constexpr,
                        SCALE: tl.constexpr):
    ch = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1).to(tl.int64) * 16 + tl.arange(0, 16).to(tl.int64)
    a = tl.arange(0, K).to(tl.int64)
    start, end = tl.load(Starts + ch // H), tl.load(Ends + ch // H)
    off = ((start + i[:, None]) * H + ch % H) * KD + a[None, :]
    valid = (i[:, None] < end-start) & (a[None, :] < KD)
    q = tl.load(Q + off, valid, 0.).to(tl.float32) * SCALE
    k = tl.load(K0 + off, valid, 0.).to(tl.float32)
    beta = tl.load(Beta + (start + i) * H + ch % H, i < end-start, 0.).to(tl.float32)
    n = tl.where(i < end-start, tl.sum(k * k, 1), 1.)
    gamma = beta / n
    aq = tl.sum(q * k, 1) / n
    qr = q - (beta * aq)[:, None] * k
    out = ch * C * K + i[:, None] * K + a[None, :]
    decay = tl.load(P[2] + out)
    tl.store(P[0] + out, q)
    tl.store(P[1] + out, k)
    tl.store(P[6] + out, qr)
    tl.store(P[7] + out, qr * decay)
    tl.store(P[18] + ch * C + i, n)
    tl.store(P[19] + ch * C + i, gamma)
    tl.store(P[20] + ch * C + i, aq)
    tl.store(P[21] + ch * C + i, beta)


@triton.jit
def product_combine(x, y):
    return x*y


@triton.jit
def common_products_kernel(P, C: tl.constexpr):
    tl.static_assert(C == 128)
    ch = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    rt, ct = tile // (C//16), tile % (C//16)
    i = tl.arange(0, 16).to(tl.int64)
    all_i = tl.arange(0, C).to(tl.int64)
    rows, cols = rt*16+i, ct*16+i
    br = tl.load(P[21]+ch*C+rows)
    nc = tl.load(P[18]+ch*C+cols)
    if rt == ct:
        factor = tl.where(i[:, None] > i[None, :], 1.-br[:, None], 1.)
        inclusive = tl.cumprod(factor, axis=0)
        previous = tl.gather(inclusive, tl.broadcast_to(tl.maximum(i-1, 0)[:, None].to(tl.int32), (16, 16)), 0)
        prod = tl.where(i[:, None] > i[None, :], previous, 0.)
    elif rt > ct:
        bc = tl.load(P[21]+ch*C+cols)
        rp = tl.cumprod(1.-br, 0)
        rp = tl.where(i > 0, tl.gather(rp, tl.maximum(i-1, 0).to(tl.int32), 0), 1.)
        cp = tl.cumprod(1.-bc, 0, reverse=True)
        cp = tl.where(i < 15, tl.gather(cp, tl.minimum(i+1, 15).to(tl.int32), 0), 1.)
        all_beta = load_vector(P[21], ch, C)
        middle = tl.reduce(tl.where((all_i >= (ct+1)*16) & (all_i < rt*16), 1.-all_beta, 1.), 0, product_combine)
        prod = rp[:, None]*cp[None, :]*middle
    else:
        prod = tl.full((16, 16), 0., tl.float32)
    offset = ch*C*C+rows[:, None]*C+cols[None, :]
    qs = tl.where(rows[:, None] > cols[None, :], -br[:, None]*nc[None, :]*prod,
                  tl.where(rows[:, None] == cols[None, :], nc[None, :], 0.))
    tl.store(P[15]+offset, prod)
    tl.store(P[14]+offset, qs)
    if ct == 0:
        all_beta = load_vector(P[21], ch, C)
        prior = tl.reduce(tl.where(all_i < rt*16, 1.-all_beta, 1.), 0, product_combine)
        prefix = tl.cumprod(1.-br, 0)
        prefix = tl.where(i > 0, tl.gather(prefix, tl.maximum(i-1, 0).to(tl.int32), 0), 1.)
        tl.store(P[22]+ch*C+rows, br*prior*prefix)


@triton.jit
def common_lower_kernel(P, C: tl.constexpr):
    ch = tl.program_id(0).to(tl.int64)
    i = tl.arange(0, C).to(tl.int64)
    off = ch*C*C+i[:, None]*C+i[None, :]
    upper = i[:, None]//32 < i[None, :]//32
    gram = tl.load(P[9]+off, ~upper, 0.).to(tl.float32)
    tl.store(P[9]+off, 0., upper)
    tl.store(P[12]+off, 0., upper)
    gamma = load_vector(P[19], ch, C)
    store_matrix(P[10], ch, (i[:, None] == i[None, :]).to(tl.float32) + gamma[:, None]*gram, C, C)


@triton.jit
def group_gram_kernel(V0, Starts, Ends, Gp, H: tl.constexpr, VD: tl.constexpr, GS: tl.constexpr,
                      NG: tl.constexpr, C: tl.constexpr, G: tl.constexpr):
    block = tl.program_id(0).to(tl.int64)
    ch, group = block // NG, block % NG
    start, end = tl.load(Starts + ch // H), tl.load(Ends + ch // H)
    i, g = tl.arange(0, C), tl.arange(0, G)
    off = ((start + i.to(tl.int64))[:, None] * H + ch % H) * VD + group * GS + g[None, :].to(tl.int64)
    v = tl.load(V0 + off, (i[:, None] < end-start) & (g[None, :] < GS), 0).to(tl.float32)
    store_matrix(Gp[3], block, mm_three(v, tl.trans(v), C, C, G), C, C)


@triton.jit
def group_gauge_kernel(Curvature, P, Gp, GS: tl.constexpr, NG: tl.constexpr, C: tl.constexpr):
    block = tl.program_id(0).to(tl.int64)
    ch = block // NG
    i = tl.arange(0, C)
    kap = tl.load(Curvature).to(tl.float32) / GS
    n, beta = load_vector(P[18], ch, C), load_vector(P[21], ch, C)
    radius2 = kap * tl.load(Gp[3] + block * C * C + i.to(tl.int64) * C + i.to(tl.int64)) / n
    # A conservative interior domain keeps the normalized mass bounds within FP32.
    bounded = (tl.max(radius2, 0) <= 0.24) & (tl.min(n, 0) >= 0.5) & (tl.max(n, 0) <= 2.)
    if bounded:
        lam = libdevice.sqrt(1. + beta * (2. - beta) * radius2 / (1. - radius2))
    else:
        history, lam = tl.full((C,), 0, tl.float32), tl.full((C,), 1, tl.float32)
        for r in range(C):
            offset = r.to(tl.int64) * C + i.to(tl.int64)
            rr = tl.load(P[11] + ch * C * C + offset).to(tl.float32) * tl.load(Gp[3] + block * C * C + offset).to(tl.float32)
            nr, br = at(n, r, C), at(beta, r, C)
            u0 = (br + kap * tl.sum(rr * history, 0)) / (nr - kap * at(rr, r, C))
            ratio = 1 - br + nr * u0
            history = tl.where(i == r, u0 / ratio, history / ratio)
            lam = tl.where(i == r, ratio, lam)
    store_vector(Gp[6], block, lam, C)
    store_vector(Gp[7], block, tl.cumprod(1 / lam, 0), C)


@triton.jit
def mass_affine_compose(a_left, b_left, a_right, b_right):
    return a_right * a_left, b_right + a_right * b_left


@triton.jit
def mass_forward_scan(beta, lam, n, w, C: tl.constexpr):
    i = tl.arange(0, C)
    alpha = (1 - beta) / lam
    t = n * w
    aa, bb = tl.associative_scan((alpha, t), axis=0, combine_fn=mass_affine_compose)
    z = aa + bb
    zp = tl.where(i == 0, 1., tl.gather(z, tl.maximum(i - 1, 0), axis=0))
    return alpha * zp, t, zp


@triton.jit
def mass_reverse_scan(beta, lam, n, w, zp, gr, gt, C: tl.constexpr):
    i = tl.arange(0, C)
    reverse = C - 1 - i
    alpha = (1 - beta) / lam
    ar = tl.gather(alpha, reverse, axis=0)
    br = tl.gather(alpha * gr, reverse, axis=0)
    _, b = tl.associative_scan((ar, br), axis=0, combine_fn=mass_affine_compose)
    inclusive = tl.gather(b, reverse, axis=0)
    future = tl.where(i < C - 1, tl.gather(inclusive, tl.minimum(i + 1, C - 1), axis=0), 0.)
    gri, gti = gr + future, gt + future
    return -gri * zp / lam, gti * w, gti * n


@triton.jit
def local_state_graph(P, Gp, ch, block, v, s, y, length, kap,
                      C: tl.constexpr, K: tl.constexpr, G: tl.constexpr, MIXED: tl.constexpr = False, RHO_MATERIALIZED: tl.constexpr = True):
    i = tl.arange(0, C)
    n = load_vector(P[18], ch, C)
    gamma = load_vector(P[19], ch, C)
    beta = load_vector(P[21], ch, C)
    af = load_vector(Gp[5], block, C)
    lam = load_vector(Gp[6], block, C)
    nu = load_vector(Gp[7], block, C)
    h = tl.sum(v * y, 1)
    w = af - kap * pointer_mv(Gp[0], block, nu * h, C)
    u = (w[:, None] * v - nu[:, None] * pointer_mm(P[10], ch, y, C, G, C)).to(tl.bfloat16)
    if C >= 16:
        r, t, zp = mass_forward_scan(beta, lam, n, w, C)
    else:
        prev = tl.full((), 1, tl.float32)
        r = tl.full((C,), 0, tl.float32)
        t = tl.full((C,), 0, tl.float32)
        zp = tl.full((C,), 0, tl.float32)
        for j in range(C):
            rj = (1 - at(beta, j, C)) / at(lam, j, C) * prev
            tj = at(n, j, C) * at(w, j, C)
            zp = tl.where(i == j, prev, zp)
            r, t = tl.where(i == j, rj, r), tl.where(i == j, tj, t)
            prev = rj + tj
    z = r + t
    a = tl.arange(0, K).to(tl.int64)
    last = (length - 1).to(tl.int64)
    kl = tl.load(P[1] + ch * C * K + last * K + a)
    tau = tl.load(P[2] + ch * C * K + last * K + a)
    gl = at(gamma, length - 1, C)
    ss = at(nu, length - 1, C) * s
    rhol = read_rho_row(Gp, block, last, i, C, RHO_MATERIALIZED)
    us = tl.where(i[:, None] < length - 1, rhol[:, None] * u, 0.)
    rs = erase(kl, gl, tau[:, None] * ss) + pointer_mm(P[17], ch, us, K, G, C)
    ts = kl[:, None] * row(v, length - 1, C, G)[None, :] / at(n, length - 1, C)
    zl, tl_last = at(z, length - 1, C), at(t, length - 1, C)
    sn = rs / zl + (tl_last / zl) * ts
    return sn, u, r, t, ss, us, w, zp


@triton.jit
def state_fwd_kernel(V0, Curvature, S0, Out, Final, Starts, Ends, SeqChunks, P, Gp, SavedS, SavedY, SavedU, SavedR, SavedT, SavedW, SavedZP, Carry,
                     H: tl.constexpr, KD: tl.constexpr, VD: tl.constexpr, GS: tl.constexpr, NG: tl.constexpr,
                     C: tl.constexpr, K: tl.constexpr, G: tl.constexpr, FINAL: tl.constexpr, CARRY: tl.constexpr,
                     MIXED: tl.constexpr = False, BF16_FS6: tl.constexpr = False, RHO_MATERIALIZED: tl.constexpr = True):
    kap = tl.load(Curvature).to(tl.float32) / GS
    shg = tl.program_id(0).to(tl.int64)
    group, head, seq = shg % NG, (shg // NG) % H, shg // (NG * H)
    a, g, i = tl.arange(0, K), tl.arange(0, G), tl.arange(0, C)
    so = ((seq * H + head) * KD + a[:, None].to(tl.int64)) * VD + group * GS + g[None, :].to(tl.int64)
    smask = (a[:, None] < KD) & (g[None, :] < GS)
    s = tl.load(S0 + so, smask, 0).to(tl.float32)
    first, last = tl.load(SeqChunks + seq), tl.load(SeqChunks + seq + 1)
    yn = tl.full((C, G), 0, tl.float32)
    for chunk in range(first, last):
        ch = chunk * H + head
        block = ch * NG + group
        start, end = tl.load(Starts + chunk), tl.load(Ends + chunk)
        vo = ((start + i.to(tl.int64))[:, None] * H + head) * VD + group * GS + g[None, :].to(tl.int64)
        vmask = (i[:, None] < end - start) & (g[None, :] < GS)
        if CARRY and chunk > first:
            y = yn
        else:
            if BF16_FS6:
                tl.static_assert(not MIXED and not CARRY)
                y = pointer_mm(P[5], ch, s, C, G, K).to(tl.bfloat16)
            else:
                y = pointer_mm(P[5], ch, s, C, G, K).to(tl.bfloat16)
        store_matrix(SavedS, block, s, K, G)
        store_matrix(SavedY, block, y, C, G)
        v = tl.load(V0 + vo, vmask, 0).to(tl.float32)
        s, u, r, t, ss, us, w, zp = local_state_graph(
            P, Gp, ch, block, v, s, y, end - start, kap, C, K, G, MIXED, RHO_MATERIALIZED)
        if CARRY and chunk + 1 < last:
            e = load_matrix(Carry[0], ch, C, K)
            hc = load_matrix(Carry[1], ch, C, C)
            fn = load_matrix(P[5], ch + H, C, K)
            kl = row(load_matrix(P[1], ch, C, K), end - start - 1, C, K)
            nl = at(load_vector(P[18], ch, C), end - start - 1, C)
            ry = mm_state_three(e, ss, C, G, K, MIXED) + mm_state_three(hc, us, C, G, C, MIXED)
            ty = (tl.sum(fn * kl[None, :], 1) / nl)[:, None] * row(v, end - start - 1, C, G)[None, :]
            zz, tt = at(r + t, end - start - 1, C), at(t, end - start - 1, C)
            yn = ry / zz + (tt / zz) * ty
        store_matrix(SavedU, block, u, C, G)
        store_vector(SavedR, block, r, C)
        store_vector(SavedT, block, t, C)
        store_vector(SavedW, block, w, C)
        store_vector(SavedZP, block, zp, C)
    if FINAL:
        tl.store(Final + so, s, smask)


@triton.jit
def output_fwd_kernel(V0, Out, Starts, Ends, P, Gp, SavedS, SavedU, SavedR, SavedT, SavedRO,
                      H: tl.constexpr, VD: tl.constexpr, GS: tl.constexpr, NG: tl.constexpr,
                      C: tl.constexpr, K: tl.constexpr, G: tl.constexpr, RHO_MATERIALIZED: tl.constexpr = True):
    block = tl.program_id(0).to(tl.int64)
    group, ch = block % NG, block // NG
    head, chunk = ch % H, ch // H
    i, g = tl.arange(0, C), tl.arange(0, G)
    start, end = tl.load(Starts + chunk), tl.load(Ends + chunk)
    off = ((start + i.to(tl.int64))[:, None] * H + head) * VD + group * GS + g[None, :].to(tl.int64)
    valid = (i[:, None] < end - start) & (g[None, :] < GS)
    v = tl.load(V0 + off, valid, 0).to(tl.float32)
    s = load_matrix(SavedS, block, K, G)
    u = load_matrix(SavedU, block, C, G)
    r, t = load_vector(SavedR, block, C), load_vector(SavedT, block, C)
    qer = load_matrix(P[7], ch, C, K)
    to = load_matrix(P[13], ch, C, C)
    aq = load_vector(P[20], ch, C)
    rho = read_rho_matrix(Gp, block, i, i, C, RHO_MATERIALIZED)
    nu = load_vector(Gp[7], block, C)
    ro = nu[:, None] * mm_three(qer, s, C, G, K) + mm_three(to * rho, u, C, G, C)
    target = aq[:, None] * v
    z = r + t
    output = ro / z[:, None] + (t / z)[:, None] * target
    store_matrix(SavedRO, block, ro, C, G)
    tl.store(Out + off, output, valid)


@triton.jit
def output_bwd_kernel(V0, DO, DV, Starts, Ends, P, Gp, SavedS, SavedU, SavedR, SavedT, SavedRO, Sh, OutputAdj,
                      H: tl.constexpr, KD: tl.constexpr, VD: tl.constexpr, GS: tl.constexpr, NG: tl.constexpr,
                      C: tl.constexpr, K: tl.constexpr, G: tl.constexpr, RHO_MATERIALIZED: tl.constexpr = True):
    block = tl.program_id(0).to(tl.int64)
    group, ch = block % NG, block // NG
    head, chunk = ch % H, ch // H
    i, g = tl.arange(0, C), tl.arange(0, G)
    start, end = tl.load(Starts + chunk), tl.load(Ends + chunk)
    length = end - start
    vo = ((start + i.to(tl.int64))[:, None] * H + head) * VD + group * GS + g[None, :].to(tl.int64)
    mask = (i[:, None] < length) & (g[None, :] < GS)
    v = tl.load(V0 + vo, mask, 0).to(tl.float32)
    go = tl.load(DO + vo, mask, 0).to(tl.float32)
    aq = load_vector(P[20], ch, C)
    r, t = load_vector(SavedR, block, C), load_vector(SavedT, block, C)
    ro = load_matrix(SavedRO, block, C, G)
    target = aq[:, None] * v
    z = r + t
    norm = tl.sum(go * (ro / z[:, None]), 1)
    dot = tl.sum(go * target, 1)
    gro, gto = go / z[:, None], (t / z)[:, None] * go
    gr, gt = -(norm + (t / z) * dot) / z, ((r / z) * dot - norm) / z
    store_vector(OutputAdj[2], block, gr, C)
    store_vector(OutputAdj[3], block, gt, C)
    gaq = tl.sum(gto * v, 1)
    store_vector(Sh[10], block, gaq, C)
    qer = load_matrix(P[7], ch, C, K)
    nu = load_vector(Gp[7], block, C)
    gs = mm_three(tl.trans(qer), nu[:, None] * gro, K, G, C)
    store_matrix(OutputAdj[0], block, gs, K, G)
    to = load_matrix(P[13], ch, C, C)
    rho = read_rho_matrix(Gp, block, i, i, C, RHO_MATERIALIZED)
    gu = mm_three(tl.trans(to * rho), gro, C, G, C)
    store_matrix(OutputAdj[1], block, gu, C, G)


@triton.jit
def saved_state_tail(P, Gp, ch, block, v, s, u, length,
                     C: tl.constexpr, K: tl.constexpr, G: tl.constexpr, RHO_MATERIALIZED: tl.constexpr = True):
    i, a = tl.arange(0, C), tl.arange(0, K)
    last = (length - 1).to(tl.int64)
    kl = tl.load(P[1] + ch * C * K + last * K + a)
    tau = tl.load(P[2] + ch * C * K + last * K + a)
    gl = tl.load(P[19] + ch * C + last)
    nl = tl.load(P[18] + ch * C + last)
    nul = tl.load(Gp[7] + block * C + last)
    rhol = read_rho_row(Gp, block, last, i, C, RHO_MATERIALIZED)
    ss = nul * s
    us = tl.where(i[:, None] < length - 1, rhol[:, None] * u, 0.)
    past = load_matrix(P[17], ch, K, C)
    rs = erase(kl, gl, tau[:, None] * ss) + mm_state_three(past, us, K, G, C, False)
    ts = kl[:, None] * row(v, length - 1, C, G)[None, :] / nl
    return ss, us, rs, ts


@triton.jit
def state_bwd_kernel(V0, Curvature, DS, DV, DS0, Starts, Ends, SeqChunks, P, Gp, SavedS, Local, Sh, OutputAdj, Carry, EdgeY,
                     H: tl.constexpr, KD: tl.constexpr, VD: tl.constexpr, GS: tl.constexpr, NG: tl.constexpr,
                     C: tl.constexpr, K: tl.constexpr, G: tl.constexpr, CARRY: tl.constexpr, RHO_MATERIALIZED: tl.constexpr = True):
    shg = tl.program_id(0).to(tl.int64)
    group, head, seq = shg % NG, (shg // NG) % H, shg // (NG * H)
    a, g, i = tl.arange(0, K), tl.arange(0, G), tl.arange(0, C)
    so = ((seq * H + head) * KD + a[:, None].to(tl.int64)) * VD + group * GS + g[None, :].to(tl.int64)
    smask = (a[:, None] < KD) & (g[None, :] < GS)
    bs = tl.load(DS + so, smask, 0).to(tl.float32)
    kap = tl.load(Curvature).to(tl.float32) / GS
    first, last = tl.load(SeqChunks + seq), tl.load(SeqChunks + seq + 1)
    by = tl.full((C, G), 0, tl.float32)
    for step in range(last - first):
        chunk = last - 1 - step
        ch, block = chunk * H + head, (chunk * H + head) * NG + group
        start, end = tl.load(Starts + chunk), tl.load(Ends + chunk)
        length = end - start
        vo = ((start + i.to(tl.int64))[:, None] * H + head) * VD + group * GS + g[None, :].to(tl.int64)
        mask = (i[:, None] < length) & (g[None, :] < GS)
        r, t = load_vector(Local[1], block, C), load_vector(Local[2], block, C)
        n = load_vector(P[18], ch, C)
        nu = load_vector(Gp[7], block, C)
        z = r + t
        rl, tt, zz = at(r, length - 1, C), at(t, length - 1, C), at(z, length - 1, C)
        kl = tl.load(P[1] + ch * C * K + (length - 1).to(tl.int64) * K + a)
        tau = tl.load(P[2] + ch * C * K + (length - 1).to(tl.int64) * K + a)
        gl = tl.load(P[19] + ch * C + (length - 1).to(tl.int64))
        ss = at(nu, length - 1, C) * load_matrix(SavedS, block, K, G)
        last_v = tl.load(V0+((end-1)*H+head)*VD+group*GS+g.to(tl.int64), g < GS, 0.).to(tl.float32)
        ts = kl[:, None] * last_v[None, :] / at(n, length - 1, C)
        adj = bs
        if CARRY:
            store_matrix(EdgeY, block, by, C, G)
        if CARRY and chunk + 1 < last:
            fn = load_matrix(P[5], ch + H, C, K)
            adj += mm_state_three(tl.trans(fn), by, K, G, C, False)
        # contraction duality avoids rebuilding the state and carry residual matrices
        gx_norm = erase(kl, gl, adj / zz)
        norm_s = tl.sum(tl.sum(gx_norm * (tau[:, None] * ss), 0), 0)
        dot_s = tl.sum(tl.sum(adj * ts, 0), 0)
        gs = load_matrix(OutputAdj[0], block, K, G)
        gs += at(nu, length - 1, C) * tau[:, None] * gx_norm
        # consume the output-state partial before reusing its storage for the tail adjoint
        tl.debug_barrier()
        store_matrix(OutputAdj[0], block, adj, K, G)
        projected = pointer_mm(P[17], ch, bs, C, G, K, True)
        if CARRY and chunk + 1 < last:
            hc = load_matrix(Carry[1], ch, C, C)
            projected += mm_state_three(tl.trans(hc), by, C, G, C, False)
        rhol = read_rho_row(Gp, block, length-1, i, C, RHO_MATERIALIZED)
        u = load_matrix(Local[0], block, C, G)
        us = tl.where(i[:, None] < length - 1, rhol[:, None] * u, 0.)
        norm_s += tl.sum(tl.sum((projected / zz) * us, 0), 0)
        gu = load_matrix(OutputAdj[1], block, C, G)
        gr, gt = load_vector(OutputAdj[2], block, C), load_vector(OutputAdj[3], block, C)
        gr += tl.where(i == length - 1, -(norm_s + (tt / zz) * dot_s) / zz, 0.)
        gt += tl.where(i == length - 1, ((rl / zz) * dot_s - norm_s) / zz, 0.)
        gu += tl.where(i[:, None] < length - 1,
                       rhol[:, None] * projected / zz, 0.)
        w, zp = load_vector(Local[4], block, C), load_vector(Local[5], block, C)
        beta = load_vector(P[21], ch, C)
        lam = load_vector(Gp[6], block, C)
        if C >= 16:
            gb, gn, gw = mass_reverse_scan(beta, lam, n, w, zp, gr, gt, C)
        else:
            gb, gn, gw = tl.full((C,), 0, tl.float32), tl.full((C,), 0, tl.float32), tl.full((C,), 0, tl.float32)
            gz = tl.full((), 0, tl.float32)
            for jj in range(C):
                j = C - 1 - jj
                gri, gti = at(gr, j, C) + gz, at(gt, j, C) + gz
                gb += tl.where(i == j, -gri * at(zp, j, C) / at(lam, j, C), 0.)
                gz = gri * (1 - at(beta, j, C)) / at(lam, j, C)
                gn += tl.where(i == j, gti * at(w, j, C), 0.)
                gw += tl.where(i == j, gti * at(n, j, C), 0.)
        # Reload V after the transpose contraction instead of keeping it live across that GEMM.
        gw += tl.sum(gu * tl.load(V0 + vo, mask, 0, volatile=True).to(tl.float32), 1)
        b = pointer_mv(Gp[0], block, gw, C, True)
        rhs = nu * b
        gh = -kap * rhs
        gy = -pointer_mm(P[10], ch, nu[:, None] * gu, C, G, C, True)
        gy += gh[:, None] * tl.load(V0 + vo, mask, 0).to(tl.float32)
        # the U and mass partials have been consumed before their storage is reused
        tl.debug_barrier()
        store_matrix(OutputAdj[1], block, gu, C, G)
        store_vector(OutputAdj[2], block, b, C)
        if not CARRY or chunk == first:
            gs += pointer_mm(P[5], ch, gy, K, G, C, True)
        tl.store(DV + vo, gy, mask)
        store_vector(Sh[8], block, gn, C)
        store_vector(Sh[12], block, gb, C)
        bs, by = gs, gy
    tl.store(DS0 + so, bs, smask)


@triton.jit
def parameter_bwd_kernel(V0, Curvature, DO, DV, DC, Starts, Ends, Prev, Next, P, Gp, SavedS, SavedY, Local, Sh, OutputAdj, EdgeY,
                         H: tl.constexpr, KD: tl.constexpr, VD: tl.constexpr, GS: tl.constexpr, NG: tl.constexpr,
                         C: tl.constexpr, K: tl.constexpr, G: tl.constexpr, CARRY: tl.constexpr, NEED_DC: tl.constexpr = True, RHO_MATERIALIZED: tl.constexpr = True):
    block = tl.program_id(0).to(tl.int64)
    group, ch = block % NG, block // NG
    head, chunk = ch % H, ch // H
    i, g, a = tl.arange(0, C), tl.arange(0, G), tl.arange(0, K)
    start, end = tl.load(Starts + chunk), tl.load(Ends + chunk)
    length = end - start
    vo = ((start + i.to(tl.int64))[:, None] * H + head) * VD + group * GS + g[None, :].to(tl.int64)
    mask = (i[:, None] < length) & (g[None, :] < GS)
    v = tl.load(V0 + vo, mask, 0).to(tl.float32)
    go = tl.load(DO + vo, mask, 0).to(tl.float32)
    gy = tl.load(DV + vo, mask, 0).to(tl.float32)  # noqa: F841
    gn_mass = load_vector(Sh[8], block, C)
    adj = load_matrix(OutputAdj[0], block, K, G)
    gu = load_matrix(OutputAdj[1], block, C, G)
    b = load_vector(OutputAdj[2], block, C)
    # DV and Sh[8] contain scan intermediates until every lane has read them
    tl.debug_barrier()
    s = load_matrix(SavedS, block, K, G)
    u = load_matrix(Local[0], block, C, G)
    r, t = load_vector(Local[1], block, C), load_vector(Local[2], block, C)
    w = load_vector(Local[4], block, C)
    z = r + t
    tt, zz = at(t, length - 1, C), at(z, length - 1, C)
    ss, us, rs, ts = saved_state_tail(P, Gp, ch, block, v, s, u, length, C, K, G, RHO_MATERIALIZED)
    residual_grad = adj / zz
    kl = tl.load(P[1] + ch * C * K + (length - 1).to(tl.int64) * K + a)
    tau = tl.load(P[2] + ch * C * K + (length - 1).to(tl.int64) * K + a)
    gl = tl.load(P[19] + ch * C + (length - 1).to(tl.int64))
    gx, gkl, ggl = erase_vjp(kl, gl, tau[:, None] * ss, residual_grad)
    gd = tl.sum(gx * ss, 1)
    gk = gkl
    gtarget = (tt / zz) * adj
    nl = at(load_vector(P[18], ch, C), length - 1, C)
    vl = row(v, length - 1, C, G)
    gk += tl.sum(gtarget * vl[None, :], 1) / nl
    store_vector(Sh[11], block, gk, K)
    store_vector(Sh[13], block, gd, K)
    store_vector(Sh[9], block, tl.where(i == length - 1, ggl, 0.), C)
    gn = tl.where(i == length - 1, -tl.sum(tl.sum(gtarget * ts, 0), 0) / nl, 0.) + gn_mass
    store_vector(Sh[8], block, gn, C)
    previous, following = tl.load(Prev + chunk), tl.load(Next + chunk)  # noqa: F841
    if CARRY:
        gfnext = tl.full((C, K), 0, tl.float32)
        if following >= 0:
            by = load_matrix(EdgeY, block, C, G)
            sn = rs / zz + (tt / zz) * ts
            gfnext = mm_state_three(by, tl.trans(sn), C, K, G, False)
        store_matrix(Sh[14], block, gfnext, C, K)
    aq = load_vector(P[20], ch, C)
    gto = (t / z)[:, None] * go
    gv = aq[:, None] * gto
    gv += tl.where(i[:, None] == length - 1, (tl.sum(gtarget * kl[:, None], 0) / nl)[None, :], 0.)
    gv += w[:, None] * gu
    nu = load_vector(Gp[7], block, C)
    y = load_matrix(SavedY, block, C, G)
    rho = read_rho_matrix(Gp, block, i, i, C, RHO_MATERIALIZED)
    gzmatrix = -rho * b[:, None] * w[None, :]
    rhs = nu * b
    store_vector(Sh[4], block, rhs, C)
    kap = tl.load(Curvature).to(tl.float32) / GS
    gh = -kap * rhs
    grmat = -kap * gzmatrix
    if NEED_DC:
        rmat = load_matrix(Gp[4], block, C, C)
        h = tl.sum(v * y, 1)
        gc = -tl.sum(rhs * h, 0) - tl.sum(tl.sum(gzmatrix * rmat, 0), 0)
        tl.store(DC + block, gc / GS)
    gv += gh[:, None] * y
    inv = load_matrix(P[11], ch, C, C)
    gram_grad = grmat * inv
    gv += mm_state_three(gram_grad + tl.trans(gram_grad), v, C, G, C, False)
    tl.store(DV + vo, gv, mask)


@triton.jit
def sum_vector(p, ch, N: tl.constexpr, NG: tl.constexpr):
    result = tl.full((N,), 0, tl.float32)
    for group in range(NG):
        result += load_vector(p, ch * NG + group, N)
    return result


@triton.jit
def carry_fwd_kernel(Ends, Starts, Next, P, Carry, H: tl.constexpr, C: tl.constexpr, K: tl.constexpr):
    ch = tl.program_id(0).to(tl.int64)
    chunk, head = ch // H, ch % H
    next_chunk = tl.load(Next + chunk)
    e = tl.full((C, K), 0, tl.float32)
    hc = tl.full((C, C), 0, tl.float32)
    if next_chunk >= 0:
        length = tl.load(Ends + chunk) - tl.load(Starts + chunk)
        fn = load_matrix(P[5], next_chunk * H + head, C, K)
        k = row(load_matrix(P[1], ch, C, K), length - 1, C, K)
        tau = row(load_matrix(P[2], ch, C, K), length - 1, C, K)
        gamma = at(load_vector(P[19], ch, C), length - 1, C)
        e = (fn - gamma * tl.sum(fn * k[None, :], 1)[:, None] * k[None, :]) * tau[None, :]
        hc = mm_three(fn, load_matrix(P[17], ch, K, C), C, C, K)
    store_matrix(Carry[0], ch, e, C, K)
    store_matrix(Carry[1], ch, hc, C, C)


@triton.jit
def diagonal_inverse_kernel(A, Inv, C: tl.constexpr, B: tl.constexpr):
    matrix = tl.program_id(0).to(tl.int64)
    row_block = tl.program_id(1).to(tl.int64)
    i = tl.arange(0, B).to(tl.int64)
    row = row_block*B+i
    for column_block in range(row_block+1, C//B):
        upper = matrix*C*C+row[:, None]*C+column_block*B+i[None, :]
        tl.store(Inv+upper, tl.full((B, B), 0., tl.float32))
    offsets = matrix*C*C+row[:, None]*C+row_block*B+i[None, :]
    value = inverse_lower(tl.load(A+offsets).to(tl.float32), B)
    tl.store(Inv+offsets, value)


@triton.jit
def inverse_offdiagonal_kernel(A, Inv, C: tl.constexpr, N: tl.constexpr, SINGLE: tl.constexpr):
    matrix = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1).to(tl.int64) * (2 * N)
    i = tl.arange(0, N).to(tl.int64)
    base = matrix * C * C
    left = tl.load(Inv + base + (block + i[:, None]) * C + block + i[None, :]).to(tl.float32)
    right = tl.load(Inv + base + (block + N + i[:, None]) * C + block + N + i[None, :]).to(tl.float32)
    coupling = tl.load(A + base + (block + N + i[:, None]) * C + block + i[None, :]).to(tl.float32)
    if SINGLE:
        tmp = tl.dot(right.to(tl.bfloat16), coupling.to(tl.bfloat16), out_dtype=tl.float32)
        value = -tl.dot(tmp.to(tl.bfloat16), left.to(tl.bfloat16), out_dtype=tl.float32)
    else:
        tmp = mm_precise(right, coupling, N, N, N)
        value = -mm_precise(tmp, left, N, N, N)
    tl.store(Inv + base + (block + N + i[:, None]) * C + block + i[None, :], value)


@triton.jit
def pointer_mm(p, block, b, M: tl.constexpr, N: tl.constexpr,
               R: tl.constexpr, TRANS: tl.constexpr = False):
    tl.static_assert(M == 128 and R == 128 and N == 32)
    m = tl.arange(0, M).to(tl.int64)
    k = tl.arange(0, 64).to(tl.int64)
    b0, b1 = tl.split(tl.permute(tl.reshape(b, (2, 64, N)), (1, 2, 0)))
    if TRANS:
        off = block*R*M + k[None, :]*M + m[:, None]
        stride = 64*M
    else:
        off = block*M*R + m[:, None]*R + k[None, :]
        stride = 64
    a0 = tl.load(p+off).to(tl.bfloat16)
    result = tl.dot(a0, b0.to(tl.bfloat16), out_dtype=tl.float32)
    a1 = tl.load(p+off+stride).to(tl.bfloat16)
    return tl.dot(a1, b1.to(tl.bfloat16), acc=result, out_dtype=tl.float32)


@triton.jit
def pointer_mv(p, block, v, C: tl.constexpr, TRANS: tl.constexpr = False):
    i = tl.arange(0, C).to(tl.int64)
    j = tl.arange(0, 32).to(tl.int64)
    result = tl.full((C,), 0, tl.float32)
    for tile in range(C // 32):
        jj = tile * 32 + j
        if TRANS:
            off = jj[None, :] * C + i[:, None]
        else:
            off = i[:, None] * C + jj[None, :]
        a = tl.load(p + block * C * C + off)
        x = tl.gather(v, jj.to(tl.int32), axis=0)
        result += tl.sum(a * x[None, :], 1)
    return result


@triton.jit
def interval_ell(Ell, start, length, head, tile, a, H: tl.constexpr, KD: tl.constexpr):
    i = tile * 32 + tl.arange(0, 32).to(tl.int64)
    off = ((start + i[:, None]) * H + head) * KD + a[None, :]
    return tl.load(Ell + off, (i[:, None] < length) & (a[None, :] < KD), 0.).to(tl.float32)


@triton.jit
def interval_factors(Ell, start, length, head, rt, ct, a,
                     H: tl.constexpr, KD: tl.constexpr, BK: tl.constexpr,
                     P, ch, C: tl.constexpr, K: tl.constexpr):
    i = tl.arange(0, 32).to(tl.int64)
    prefix = tl.load(P[3]+ch*C*K+(rt*32+i[:, None])*K+a[None, :])
    suffix = tl.load(P[4]+ch*C*K+(ct*32+i[:, None])*K+a[None, :])
    middle = tl.full((BK,), 1., tl.float32)
    for tile in range(ct+1, rt):
        middle *= tl.load(P[3]+ch*C*K+(tile*32+31)*K+a)
    return prefix*middle[None, :], suffix


@triton.jit
def interval_tile_load(ptr, ch, tile, a, C: tl.constexpr, K: tl.constexpr):
    i = tile * 32 + tl.arange(0, 32).to(tl.int64)
    return tl.load(ptr + ch * C * K + i[:, None] * K + a[None, :])


@triton.jit
def interval_fwd_cross(Ell, Starts, Ends, P, H: tl.constexpr, KD: tl.constexpr,
                       C: tl.constexpr, K: tl.constexpr):
    tl.static_assert(C == 128 and K == 128)
    ch = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    ct = tl.where(tile < 2, 0, tile-1)
    rt = tl.where(tile == 0, 1, tl.where(tile == 2, 2, 3))
    i = tl.arange(0, 64).to(tl.int64)
    j = tl.arange(0, 32).to(tl.int64)
    a = tl.arange(0, K).to(tl.int64)
    rows = rt*32+i
    roff = ch*C*K+rows[:, None]*K+a[None, :]
    coff = ch*C*K+(ct*32+j[:, None])*K+a[None, :]
    prefix = tl.load(P[3]+roff, rows[:, None] < C, 0.).to(tl.float32)
    middle = tl.full((K,), 1., tl.float32)
    for block in range(ct+1, rt):
        middle *= tl.load(P[3]+ch*C*K+(block*32+31)*K+a)
    next_middle = middle*tl.load(P[3]+ch*C*K+(rt*32+31)*K+a)
    prefix *= tl.where(i[:, None] < 32, middle[None, :], next_middle[None, :])
    kc = tl.load(P[1]+coff).to(tl.float32)*tl.load(P[4]+coff)
    kr = tl.load(P[1]+roff, rows[:, None] < C, 0.).to(tl.float32)
    off = ch*C*C+rows[:, None]*C+ct*32+j[None, :]
    gram = mm_three(kr*prefix, tl.trans(kc), 64, 32, K)
    tl.store(P[9]+off, gram, rows[:, None] < C)
    qr = tl.load(P[6]+roff, rows[:, None] < C, 0.).to(tl.float32)
    readout = mm_three(qr*prefix, tl.trans(kc), 64, 32, K)
    tl.store(P[12]+off, readout, rows[:, None] < C)


@triton.jit
def interval_fwd_diagonal(Ell, Starts, Ends, P, Flags, H: tl.constexpr, KD: tl.constexpr,
                          C: tl.constexpr, K: tl.constexpr):
    ch = tl.program_id(0).to(tl.int64)
    pair = tl.program_id(1).to(tl.int64)
    rt, ct = pair, pair
    start, end = tl.load(Starts + ch//H), tl.load(Ends + ch//H)
    i = tl.arange(0, 32).to(tl.int64)
    a = tl.arange(0, K).to(tl.int64)
    off = ch*C*C + (rt*32+i[:, None])*C + ct*32+i[None, :]  # noqa: F841
    if not tl.load(Flags+ch*(C//32)+rt):
        ell = interval_ell(Ell, start, end-start, ch % H, rt, a, H, KD)
        k = interval_tile_load(P[1], ch, rt, a, C, K)
        qr = interval_tile_load(P[6], ch, rt, a, C, K)
        for j in range(32):
            interval = libdevice.exp(tl.cumsum(tl.where(i[:, None] > j, ell, 0.), 0))
            kj = row(k, j, 32, K)
            gj = tl.sum(k*interval*kj[None, :], 1)
            cj = tl.sum(qr*interval*kj[None, :], 1)
            col = ch*C*C+(rt*32+i)*C+ct*32+j
            tl.store(P[9]+col, tl.where(i > j, gj, 0.))
            tl.store(P[12]+col, tl.where(i > j, cj, 0.))


@triton.jit
def interval_tail_factor(Ell, start, length, head, tile, a,
                         H: tl.constexpr, KD: tl.constexpr, C: tl.constexpr, P, ch, K: tl.constexpr):
    i = tl.arange(0, 32).to(tl.int64)
    suffix = tl.load(P[4]+ch*C*K+(tile*32+i[:, None])*K+a[None, :])
    future = tl.full((32,), 1., tl.float32)
    for other in range(tile+1, C//32):
        future *= tl.load(P[3]+ch*C*K+(other*32+31)*K+a)
    return suffix*future[None, :]


@triton.jit
def interval_tail_kernel(Ell, Starts, Ends, P, H: tl.constexpr, KD: tl.constexpr,
                         C: tl.constexpr, K: tl.constexpr):
    ch = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    a = tl.program_id(2).to(tl.int64)*32 + tl.arange(0, 32).to(tl.int64)
    start, end = tl.load(Starts+ch//H), tl.load(Ends+ch//H)
    tail = interval_tail_factor(Ell, start, end-start, ch % H, tile, a, H, KD, C, P, ch, K)
    k = interval_tile_load(P[1], ch, tile, a, C, K)
    i = tile*32+tl.arange(0, 32).to(tl.int64)
    tl.store(P[8]+ch*C*K+i[:, None]*K+a[None, :], k*tail)


@triton.jit
def interval_bwd_cross(Ell, DQ, DK, Starts, Ends, P, Sh, Scratch, H: tl.constexpr, KD: tl.constexpr,
                       NG: tl.constexpr, C: tl.constexpr, K: tl.constexpr):
    tl.static_assert(C == 128 and K == 128)
    ch = tl.program_id(0).to(tl.int64)
    base = tl.program_id(1).to(tl.int64)*2
    a = tl.program_id(2).to(tl.int64)*64+tl.arange(0, 64).to(tl.int64)
    i = tl.arange(0, 64).to(tl.int64)
    j = tl.arange(0, 32).to(tl.int64)
    rows = base*32+i
    start, end = tl.load(Starts+ch//H), tl.load(Ends+ch//H)
    pos = ch*C*K+rows[:, None]*K+a[None, :]
    k = tl.load(P[1]+pos).to(tl.float32)
    qr = tl.load(P[6]+pos).to(tl.float32)
    own_prefix = tl.load(P[3]+pos).to(tl.float32)
    own_suffix = tl.load(P[4]+pos).to(tl.float32)
    off = ((start+rows[:, None])*H+ch % H)*KD+a[None, :]
    valid = (rows[:, None] < end-start) & (a[None, :] < KD)
    gk, gq = tl.load(DK+off, valid, 0.), tl.load(DQ+off, valid, 0.)
    incidence = tl.full((64, 64), 0., tl.float32)
    for other in range(C//32):
        other_pos = ch*C*K+(other*32+j[:, None])*K+a[None, :]
        if other < base+1:
            middle = tl.full((64,), 1., tl.float32)
            for block in range(other+1, base):
                middle *= tl.load(P[3]+ch*C*K+(block*32+31)*K+a)
            extra = tl.load(P[3]+ch*C*K+(base*32+31)*K+a)
            upper_middle = tl.where(other < base, middle*extra, middle)
            prefix = own_prefix*tl.where(i[:, None] < 32, middle[None, :], upper_middle[None, :])
            kc = tl.load(P[1]+other_pos).to(tl.float32)*tl.load(P[4]+other_pos)
            goff = ch*C*C+rows[:, None]*C+other*32+j[None, :]
            mask = rows[:, None]//32 > other
            gg, gc = tl.load(Sh[0]+goff, mask, 0.), tl.load(Sh[1]+goff, mask, 0.)
            dk = mm_three(gg, kc, 64, 64, 32)*prefix
            dq = mm_three(gc, kc, 64, 64, 32)*prefix
            gk += dk
            gq += dq
            incidence += k*dk+qr*dq
        if other > base:
            middle = tl.full((64,), 1., tl.float32)
            for block in range(base+2, other):
                middle *= tl.load(P[3]+ch*C*K+(block*32+31)*K+a)
            prefix = tl.load(P[3]+other_pos).to(tl.float32)*middle[None, :]
            extra = tl.load(P[3]+ch*C*K+((base+1)*32+31)*K+a)
            factor = tl.where((i[:, None] < 32) & (other > base+1), extra[None, :], 1.)
            suffix = own_suffix*factor
            kr = tl.load(P[1]+other_pos).to(tl.float32)*prefix
            qro = tl.load(P[6]+other_pos).to(tl.float32)*prefix
            goff = ch*C*C+(other*32+j[None, :])*C+rows[:, None]
            mask = rows[:, None]//32 < other
            gg, gc = tl.load(Sh[0]+goff, mask, 0.), tl.load(Sh[1]+goff, mask, 0.)
            dk = (mm_three(gg, kr, 64, 64, 32)+mm_three(gc, qro, 64, 64, 32))*suffix
            gk += dk
            incidence -= k*dk
    future = tl.full((64,), 1., tl.float32)
    for block in range(base+2, C//32):
        future *= tl.load(P[3]+ch*C*K+(block*32+31)*K+a)
    extra = tl.load(P[3]+ch*C*K+((base+1)*32+31)*K+a)
    tail = own_suffix*tl.where(i[:, None] < 32, (extra*future)[None, :], future[None, :])
    tailgrad = tl.load(Sh[7]+pos).to(tl.float32)*tail
    gk += tailgrad
    tl.store(DK+off, gk, valid)
    tl.store(DQ+off, gq, valid)
    tl.store(Scratch[0]+pos, incidence)
    tl.store(Scratch[1]+pos, tailgrad*k)


@triton.jit
def interval_bwd_diagonal(Ell, DQ, DK, Starts, Ends, P, Sh, Scratch, Flags, H: tl.constexpr, KD: tl.constexpr,
                          NG: tl.constexpr, C: tl.constexpr, K: tl.constexpr):
    ch = tl.program_id(0).to(tl.int64)
    owner = tl.program_id(1).to(tl.int64)
    a = tl.program_id(2).to(tl.int64)*32+tl.arange(0, 32).to(tl.int64)
    i = tl.arange(0, 32).to(tl.int64)
    start, end = tl.load(Starts+ch//H), tl.load(Ends+ch//H)
    if not tl.load(Flags+ch*(C//32)+owner):
        k = interval_tile_load(P[1], ch, owner, a, C, K)
        qr = interval_tile_load(P[6], ch, owner, a, C, K)
        off = ((start+owner*32+i[:, None])*H+ch % H)*KD+a[None, :]
        valid = (owner*32+i[:, None] < end-start) & (a[None, :] < KD)
        gk = tl.load(DK+off, valid, 0.)
        gq = tl.load(DQ+off, valid, 0.)
        scratch = ch*C*K+(owner*32+i[:, None])*K+a[None, :]
        incidence = tl.load(Scratch[0]+scratch).to(tl.float32)
        ell = interval_ell(Ell, start, end-start, ch % H, owner, a, H, KD)
        for j in range(32):
            interval = libdevice.exp(tl.cumsum(tl.where(i[:, None] > j, ell, 0.), 0))
            kj = row(k, j, 32, 32)
            goff = ch*C*C+(owner*32+i)*C+owner*32+j
            gg, gc = tl.load(Sh[0]+goff), tl.load(Sh[1]+goff)
            left = gg[:, None]*k+gc[:, None]*qr
            gk += gg[:, None]*kj[None, :]*interval
            gq += gc[:, None]*kj[None, :]*interval
            col = tl.sum(left*interval, 0)
            gk += tl.where(i[:, None] == j, col[None, :], 0.)
            weight = left*kj[None, :]*interval
            incidence += weight-tl.where(i[:, None] == j, tl.sum(weight, 0)[None, :], 0.)
        tl.store(DK+off, gk, valid)
        tl.store(DQ+off, gq, valid)
        tl.store(Scratch[0]+scratch, incidence)


@triton.jit
def interval_gate_bwd(DE, Starts, Ends, P, Scratch, H: tl.constexpr, KD: tl.constexpr,
                      C: tl.constexpr, K: tl.constexpr):
    ch = tl.program_id(0).to(tl.int64)
    a = tl.program_id(1).to(tl.int64)*32+tl.arange(0, 32).to(tl.int64)
    i = tl.arange(0, C).to(tl.int64)
    start, end = tl.load(Starts+ch//H), tl.load(Ends+ch//H)
    scratch = ch*C*K+i[:, None]*K+a[None, :]
    incidence, tail = tl.load(Scratch[0]+scratch).to(tl.float32), tl.load(Scratch[1]+scratch).to(tl.float32)
    inclusive = tl.cumsum(tail, 0)
    exclusive = tl.gather(inclusive, tl.broadcast_to(tl.maximum(i-1, 0)[:, None].to(tl.int32), (C, 32)), 0)
    off = ((start+i[:, None])*H+ch % H)*KD+a[None, :]
    valid = (i[:, None] < end-start) & (a[None, :] < KD)
    value = tl.cumsum(tl.load(DE+off, valid, 0.) + incidence, 0, reverse=True)
    value += tl.where(i[:, None] > 0, exclusive, 0.)
    tl.store(DE+off, value, valid)


@triton.jit
def common_input_bwd(DQ, DK, DB, Starts, Ends, P, Sh, Gauge, H: tl.constexpr, KD: tl.constexpr,
                     NG: tl.constexpr, C: tl.constexpr, K: tl.constexpr, SCALE: tl.constexpr):
    ch = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    i = tile*32+tl.arange(0, 32).to(tl.int64)
    a = tl.arange(0, K).to(tl.int64)
    start, end = tl.load(Starts+ch//H), tl.load(Ends+ch//H)
    off = ((start+i[:, None])*H+ch % H)*KD+a[None, :]
    valid = (i[:, None] < end-start) & (a[None, :] < KD)
    gk, gqr = tl.load(DK+off, valid, 0.), tl.load(DQ+off, valid, 0.)
    q, k = interval_tile_load(P[0], ch, tile, a, C, K), interval_tile_load(P[1], ch, tile, a, C, K)
    n, gamma = tl.load(P[18]+ch*C+i), tl.load(P[19]+ch*C+i)
    aq, beta = tl.load(P[20]+ch*C+i), tl.load(P[21]+ch*C+i)
    gaq = tl.full((32,), 0., tl.float32)
    for group in range(NG):
        gaq += tl.load(Sh[10]+(ch*NG+group)*C+i)
    gg = tl.load(DB+(start+i)*H+ch % H, i < end-start, 0.)
    gn, gb = tl.load(Gauge[0]+ch*C+i), tl.load(Gauge[1]+ch*C+i)
    dot = tl.sum(gqr*k, 1)
    gaq -= beta*dot
    gq = gqr+(gaq/n)[:, None]*k
    gb -= aq*dot
    gk -= (beta*aq)[:, None]*gqr
    gk += (gaq/n)[:, None]*q
    gn -= gaq*aq/n
    gb += gg/n
    gn -= gg*gamma/n
    gk += 2*gn[:, None]*k
    tl.store(DQ+off, gq*SCALE, valid)
    tl.store(DK+off, gk, valid)
    tl.store(DB+(start+i)*H+ch % H, gb, i < end-start)


@triton.jit
def common_tail_bwd(Starts, Ends, P, Sh, Work, DK, DB,
                    H: tl.constexpr, KD: tl.constexpr, NG: tl.constexpr,
                    C: tl.constexpr, K: tl.constexpr):
    ch = tl.program_id(0).to(tl.int64)
    i, a = tl.arange(0, C).to(tl.int64), tl.arange(0, K).to(tl.int64)
    start, end = tl.load(Starts+ch//H), tl.load(Ends+ch//H)
    last = end-start-1
    kl = tl.load(P[1]+ch*C*K+last*K+a)
    gamma = tl.load(P[19]+ch*C+last)
    gp = load_matrix(Sh[7], ch, K, C)
    p0 = load_matrix(P[16], ch, K, C)
    gp0, gkl, ggl = erase_vjp(kl, gamma, p0, gp)
    store_matrix(Work[0], ch, gp0, K, C)
    store_matrix(Work[2], ch, load_matrix(Sh[1], ch, C, C), C, C)
    gk = gkl+sum_vector(Sh[11], ch, K, NG)
    off = ((start+i[:, None])*H+ch % H)*KD+a[None, :]
    tl.store(DK+off, tl.where(i[:, None] == last, gk[None, :], 0.),
             (i[:, None] < end-start) & (a[None, :] < KD))
    tl.store(DB+(start+i)*H+ch % H, tl.where(i == last, ggl, 0.), i < end-start)


@triton.jit
def common_ga_bwd(Starts, Ends, P, Sh, Work, H: tl.constexpr, NG: tl.constexpr,
                  C: tl.constexpr, K: tl.constexpr):
    ch = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1).to(tl.int64)*64+tl.arange(0, 64).to(tl.int64)
    j = tl.program_id(2).to(tl.int64)*64+tl.arange(0, 64).to(tl.int64)
    ar = tl.arange(0, 32).to(tl.int64)
    last = tl.load(Ends+ch//H)-tl.load(Starts+ch//H)-1
    tail_acc = tl.full((64, 64), 0., tl.float32)
    total = tl.load(Sh[0]+ch*C*C+i[:, None]*C+j[None, :]).to(tl.float32)
    for tile in range(K//32):
        r = tile*32+ar
        tail = tl.load(P[8]+ch*C*K+i[:, None]*K+r[None, :])
        gp0 = tl.load(Work[0]+ch*K*C+r[:, None]*C+j[None, :])
        tail_acc += mm_three(tail, gp0, 64, 64, 32)
        gf = tl.load(Sh[2]+ch*C*K+i[:, None]*K+r[None, :])
        source = ch*C*K+j[:, None]*K+r[None, :]
        bmat = tl.load(P[19]+ch*C+j)[:, None]*tl.load(P[2]+source)*tl.load(P[1]+source)
        total += mm_three(gf, tl.trans(bmat), 64, 64, 32)
    total += tl.where((i[:, None] < last) & (j[None, :] < last), tail_acc, 0.)
    for tile in range(C//32):
        r = tile*32+ar
        cr = tl.load(P[12]+ch*C*C+r[:, None]*C+i[None, :])
        gto = tl.load(Sh[6]+ch*C*C+r[:, None]*C+j[None, :])
        total += mm_three(tl.trans(cr), gto, 64, 64, 32)
    tl.store(Work[1]+ch*C*C+i[:, None]*C+j[None, :], total)


@triton.jit
def common_products_bwd(Starts, Ends, P, Sh, Work, H: tl.constexpr, NG: tl.constexpr,
                        C: tl.constexpr, K: tl.constexpr, MODE: tl.constexpr):
    ch = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1).to(tl.int64)*64+tl.arange(0, 64).to(tl.int64)
    j = tl.program_id(2).to(tl.int64)*64+tl.arange(0, 64).to(tl.int64)
    ar = tl.arange(0, 32).to(tl.int64)
    last = tl.load(Ends+ch//H)-tl.load(Starts+ch//H)-1
    total = tl.full((64, 64), 0., tl.float32)
    for tile in range(C//32):
        r = tile*32+ar
        if MODE == 0:
            left = tl.load(P[11]+ch*C*C+i[:, None]*C+r[None, :])
            left = tl.where((i[:, None] < last) & (r[None, :] < last), left, 0.)
            right = tl.load(Work[0]+ch*K*C+j[None, :]*C+r[:, None])
        elif MODE == 1:
            left = tl.load(Sh[6]+ch*C*C+i[:, None]*C+r[None, :])
            right = tl.load(P[11]+ch*C*C+j[None, :]*C+r[:, None])
        else:
            left = tl.load(P[11]+ch*C*C+r[None, :]*C+i[:, None])
            right = tl.load(Sh[2]+ch*C*K+r[:, None]*K+j[None, :])
        total += mm_three(left, right, 64, 64, 32)
    if MODE == 0:
        tl.store(Sh[7]+ch*C*K+i[:, None]*K+j[None, :], total)
    elif MODE == 1:
        tl.store(Sh[1]+ch*C*C+i[:, None]*C+j[None, :], tl.where(i[:, None] > j[None, :], total, 0.))
    else:
        tl.store(Work[3]+ch*C*K+i[:, None]*K+j[None, :], total)


@triton.jit
def common_inverse_left_bwd(P, Work, C: tl.constexpr):
    ch = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1).to(tl.int64)*64+tl.arange(0, 64).to(tl.int64)
    j = tl.program_id(2).to(tl.int64)*64+tl.arange(0, 64).to(tl.int64)
    ar = tl.arange(0, 32).to(tl.int64)
    total = tl.full((64, 64), 0., tl.float32)
    for tile in range(C//32):
        r = tile*32+ar
        left = tl.load(P[11]+ch*C*C+r[None, :]*C+i[:, None])
        right = tl.load(Work[1]+ch*C*C+r[:, None]*C+j[None, :])
        total += mm_three(left, right, 64, 64, 32)
    tl.store(Work[0]+ch*C*C+i[:, None]*C+j[None, :], total)


@triton.jit
def common_lower_bwd(P, Sh, Work, DB, Starts, Ends, H: tl.constexpr, NG: tl.constexpr, C: tl.constexpr):
    ch = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1).to(tl.int64)*16+tl.arange(0, 16).to(tl.int64)
    j = tl.arange(0, C).to(tl.int64)
    ar = tl.arange(0, 32).to(tl.int64)
    total = tl.full((16, C), 0., tl.float32)
    for tile in range(C//32):
        r = tile*32+ar
        left = tl.load(Work[0]+ch*C*C+i[:, None]*C+r[None, :])
        right = tl.load(P[11]+ch*C*C+j[None, :]*C+r[:, None])
        total += mm_three(left, right, 16, C, 32)
    offset = ch*C*C+i[:, None]*C+j[None, :]
    gl = tl.where(i[:, None] > j[None, :], tl.load(Work[2]+offset)-total, 0.)
    gamma = tl.load(P[19]+ch*C+i)
    tl.store(Sh[0]+ch*C*C+i[:, None]*C+j[None, :], gamma[:, None]*gl)
    gram = tl.load(P[9]+offset)
    start, end = tl.load(Starts+ch//H), tl.load(Ends+ch//H)
    goff = (start+i)*H+ch % H
    gg = tl.load(DB+goff, i < end-start, 0.)+tl.sum(gl*gram, 1)
    tl.store(DB+goff, gg, i < end-start)


@triton.jit
def common_base_bwd(P, Sh, Work, DQ, DK, DB, DE, Starts, Ends,
                    H: tl.constexpr, KD: tl.constexpr, NG: tl.constexpr, C: tl.constexpr, K: tl.constexpr):
    ch = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1).to(tl.int64)*16+tl.arange(0, 16).to(tl.int64)
    a = tl.arange(0, K).to(tl.int64)
    off = ch*C*K+i[:, None]*K+a[None, :]
    decay, k = tl.load(P[2]+off), tl.load(P[1]+off)
    gbmat = tl.load(Work[3]+off)
    gamma = tl.load(P[19]+ch*C+i)
    gke = gamma[:, None]*gbmat
    gqer = tl.load(Sh[5]+ch*C*K+i[:, None]*K+a[None, :])
    start, end = tl.load(Starts+ch//H), tl.load(Ends+ch//H)
    token = ((start+i[:, None])*H+ch % H)*KD+a[None, :]
    valid = (i[:, None] < end-start) & (a[None, :] < KD)
    tl.store(DK+token, tl.load(DK+token, valid, 0.)+decay*gke, valid)
    tl.store(DQ+token, gqer*decay, valid)
    gd = tl.where(i[:, None] == end-start-1, sum_vector(Sh[13], ch, K, NG)[None, :], 0.)
    gd += gqer*tl.load(P[6]+off)+k*gke
    tl.store(DE+token, gd*decay, valid)
    gg = tl.full((16,), 0., tl.float32)
    for group in range(NG):
        gg += tl.load(Sh[9]+(ch*NG+group)*C+i)
    gg += tl.sum(gbmat*(decay*k), 1)
    boff = (start+i)*H+ch % H
    tl.store(DB+boff, tl.load(DB+boff, i < end-start, 0.)+gg, i < end-start)


@triton.jit
def group_system_tiles(Curvature, P, Gp, GS: tl.constexpr, NG: tl.constexpr,
                       C: tl.constexpr, NEED_DC: tl.constexpr, RHO_MATERIALIZED: tl.constexpr = True):
    block = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1).to(tl.int64)*32+tl.arange(0, 32).to(tl.int64)
    i = tl.arange(0, C).to(tl.int64)
    ch = block//NG
    kap = tl.load(Curvature).to(tl.float32)/GS
    if RHO_MATERIALIZED:
        lam = load_vector(Gp[6], block, C)
        rho = tl.cumprod(tl.where(i[:, None] > j[None, :], 1./lam[:, None], 1.), 0)
        rho = tl.where(i[:, None] >= j[None, :], rho, 0.)
    else:
        rho = read_rho_matrix(Gp, block, i, j, C, False)
    off = i[:, None]*C+j[None, :]
    rmat = tl.load(P[11]+ch*C*C+off)*tl.load(Gp[3]+block*C*C+off)
    qs = tl.load(P[14]+ch*C*C+off)
    if RHO_MATERIALIZED:
        tl.store(Gp[2]+block*C*C+off, rho)
    tl.store(Gp[1]+block*C*C+off, (qs-kap*rmat)*rho)
    if NEED_DC:
        tl.store(Gp[4]+block*C*C+off, rmat)


@triton.jit
def group_feedback_tiles(Curvature, P, Gp, GS: tl.constexpr, NG: tl.constexpr, C: tl.constexpr):
    block = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1).to(tl.int64)*32+tl.arange(0, 32).to(tl.int64)
    j = tl.arange(0, C).to(tl.int64)
    kap = tl.load(Curvature).to(tl.float32)/GS  # noqa: F841
    zi = tl.load(Gp[0]+block*C*C+i[:, None]*C+j[None, :])
    nu = load_vector(Gp[7], block, C)
    eta = load_vector(P[22], block//NG, C)
    tl.store(Gp[5]+block*C+i, tl.sum(zi*(nu*eta)[None, :], 1))


@triton.jit
def common_finish_tiles(P, Starts, Ends, H: tl.constexpr, C: tl.constexpr, K: tl.constexpr, MODE: tl.constexpr):
    ch = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1).to(tl.int64)*64+tl.arange(0, 64).to(tl.int64)
    j = tl.program_id(2).to(tl.int64)*64+tl.arange(0, 64).to(tl.int64)
    ar = tl.arange(0, 32).to(tl.int64)
    last = tl.load(Ends+ch//H)-tl.load(Starts+ch//H)-1
    total = tl.full((64, 64), 0., tl.float32)
    for tile in range(C//32):
        r = tile*32+ar
        if MODE == 0:
            left = tl.load(P[11]+ch*C*C+i[:, None]*C+r[None, :])
            source = ch*C*K+r[:, None]*K+j[None, :]
            right = tl.load(P[19]+ch*C+r)[:, None]*(tl.load(P[2]+source)*tl.load(P[1]+source))
        elif MODE == 1:
            left = tl.load(P[12]+ch*C*C+i[:, None]*C+r[None, :])
            right = tl.load(P[11]+ch*C*C+r[:, None]*C+j[None, :])
        else:
            left = tl.load(P[8]+ch*C*K+r[None, :]*K+i[:, None])
            right = tl.load(P[11]+ch*C*C+r[:, None]*C+j[None, :])
            right = tl.where((r[:, None] < last) & (j[None, :] < last), right, 0.)
        total += mm_three(left, right, 64, 64, 32)
    if MODE == 0:
        tl.store(P[5]+ch*C*K+i[:, None]*K+j[None, :], total)
    elif MODE == 1:
        tl.store(P[13]+ch*C*C+i[:, None]*C+j[None, :], total)
    else:
        tl.store(P[16]+ch*K*C+i[:, None]*C+j[None, :], total)


@triton.jit
def common_tail_finish(P, Starts, Ends, H: tl.constexpr, C: tl.constexpr, K: tl.constexpr):
    ch = tl.program_id(0).to(tl.int64)
    a = tl.arange(0, K).to(tl.int64)
    j = tl.program_id(1).to(tl.int64)*32+tl.arange(0, 32).to(tl.int64)
    last = tl.load(Ends+ch//H)-tl.load(Starts+ch//H)-1
    k = tl.load(P[1]+ch*C*K+last*K+a)
    gamma = tl.load(P[19]+ch*C+last)
    off = ch*K*C+a[:, None]*C+j[None, :]
    p0 = tl.load(P[16]+off)
    tl.store(P[17]+off, erase(k, gamma, p0))


@triton.jit
def interval_cache_kernel(Ell, Starts, Ends, P, H: tl.constexpr, KD: tl.constexpr,
                          C: tl.constexpr, K: tl.constexpr):
    ch = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    a = tl.program_id(2).to(tl.int64)*32+tl.arange(0, 32).to(tl.int64)
    i = tl.arange(0, 32).to(tl.int64)
    start, end = tl.load(Starts+ch//H), tl.load(Ends+ch//H)
    ell = interval_ell(Ell, start, end-start, ch % H, tile, a, H, KD)
    prefix = libdevice.exp(tl.cumsum(ell, 0))
    rev = tl.cumsum(ell, 0, reverse=True)
    suffix = tl.gather(rev, tl.broadcast_to(tl.minimum(i+1, 31)[:, None].to(tl.int32), (32, 32)), 0)
    suffix = libdevice.exp(tl.where(i[:, None] < 31, suffix, 0.))
    off = ch*C*K+(tile*32+i[:, None])*K+a[None, :]
    tl.store(P[3]+off, prefix)
    tl.store(P[4]+off, suffix)


@triton.jit
def shared_coefficients_bwd(DO, GY, Starts, Ends, Gp, SavedS, SavedU, SavedR, SavedT,
                            OutputAdj, Sh, H: tl.constexpr, VD: tl.constexpr, GS: tl.constexpr,
                            NG: tl.constexpr, C: tl.constexpr, K: tl.constexpr, G: tl.constexpr,
                            MODE: tl.constexpr, RHO_MATERIALIZED: tl.constexpr = True):
    ch = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1).to(tl.int64)*64+tl.arange(0, 64).to(tl.int64)
    j = tl.program_id(2).to(tl.int64)*64+tl.arange(0, 64).to(tl.int64)
    g = tl.arange(0, G).to(tl.int64)
    start, end = tl.load(Starts+ch//H), tl.load(Ends+ch//H)
    last = end-start-1
    total = tl.full((64, 64), 0., tl.float32)
    for group in range(NG):
        block = ch*NG+group
        if MODE == 3:
            z = tl.load(SavedR+block*C+last)+tl.load(SavedT+block*C+last)
            left = tl.load(OutputAdj[0]+block*K*G+i[:, None]*G+g[None, :])/z
            right = tl.load(SavedU+block*C*G+j[:, None]*G+g[None, :])
            rho = read_rho_row(Gp, block, last, j, C, RHO_MATERIALIZED)
            right = tl.where(j[:, None] < last, rho[:, None]*right, 0.)
        else:
            off = ((start+i[:, None])*H+ch % H)*VD+group*GS+g[None, :]
            valid = (i[:, None] < end-start) & (g[None, :] < GS)
            if MODE == 2:
                left = tl.load(GY+off, valid, 0.).to(tl.float32)
            else:
                z = tl.load(SavedR+block*C+i)+tl.load(SavedT+block*C+i)
                left = tl.load(DO+off, valid, 0.).to(tl.float32)/z[:, None]
                if MODE == 0:
                    nu = tl.load(Gp[7]+block*C+i)
                    left = nu[:, None]*left
            if MODE == 1:
                right = tl.load(SavedU+block*C*G+j[:, None]*G+g[None, :])
            else:
                right = tl.load(SavedS+block*K*G+j[:, None]*G+g[None, :])
        product = mm_three(left, tl.trans(right), 64, 64, G)
        if MODE == 1:
            rho = read_rho_matrix(Gp, block, i, j, C, RHO_MATERIALIZED)
            product = tl.where(i[:, None] > j[None, :], rho*product, 0.)
        total += product
    if MODE == 0:
        tl.store(Sh[5]+ch*C*K+i[:, None]*K+j[None, :], total)
    elif MODE == 1:
        tl.store(Sh[6]+ch*C*C+i[:, None]*C+j[None, :], total)
    elif MODE == 2:
        tl.store(Sh[2]+ch*C*K+i[:, None]*K+j[None, :], total)
    else:
        tl.store(Sh[7]+ch*K*C+i[:, None]*C+j[None, :], total)


@triton.jit
def gauge_collect(Sh, Work, NG: tl.constexpr, C: tl.constexpr):
    ch = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1).to(tl.int64)*32+tl.arange(0, 32).to(tl.int64)
    j = tl.program_id(2).to(tl.int64)*64+tl.arange(0, 64).to(tl.int64)
    gqs = tl.load(Sh[3]+ch*C*C+i[:, None]*C+j[None, :]).to(tl.float32)
    tl.store(Work[0]+ch*C*C+i[:, None]*C+j[None, :], gqs)


@triton.jit
def gauge_product_bwd(P, Work, C: tl.constexpr):
    ch = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1).to(tl.int64)*64+tl.arange(0, 64).to(tl.int64)
    j = tl.program_id(2).to(tl.int64)*64+tl.arange(0, 64).to(tl.int64)
    ar = tl.arange(0, 32).to(tl.int64)
    value = tl.full((64, 64), 0., tl.float32)
    for tile in range(C//32):
        k = tile*32+ar
        gqs = tl.load(Work[0]+ch*C*C+i[:, None]*C+k[None, :])
        n = tl.load(P[18]+ch*C+k)
        products = tl.load(P[15]+ch*C*C+j[None, :]*C+k[:, None])
        value += mm_three(gqs*n[None, :], products, 64, 64, 32)
    tl.store(Work[1]+ch*C*C+i[:, None]*C+j[None, :], value)


@triton.jit
def gauge_finish_bwd(P, Sh, Work, Gauge, NG: tl.constexpr, C: tl.constexpr):
    ch = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1).to(tl.int64)*16+tl.arange(0, 16).to(tl.int64)
    i = tl.arange(0, C).to(tl.int64)
    beta, n = load_vector(P[21], ch, C), load_vector(P[18], ch, C)
    ge = sum_vector(Sh[4], ch, C, NG)
    inclusive = tl.cumprod(1.-beta, 0)
    prefix = tl.where(i > 0, tl.gather(inclusive, tl.maximum(i-1, 0).to(tl.int32), 0), 1.)
    prefj = tl.gather(prefix, j.to(tl.int32), 0)
    gej = tl.gather(ge, j.to(tl.int32), 0)
    own_p = tl.load(P[15]+ch*C*C+j[:, None]*C+i[None, :])
    own_g = tl.load(Work[0]+ch*C*C+j[:, None]*C+i[None, :]).to(tl.float32)
    gb = gej*prefj-tl.sum(own_g*own_p*n[None, :], 1)
    gn = tl.sum(tl.where(j[:, None] == i[None, :], own_g, 0.), 1)
    for group in range(NG):
        gn += tl.load(Sh[8]+(ch*NG+group)*C+j)
        gb += tl.load(Sh[12]+(ch*NG+group)*C+j)
    pc = tl.load(P[15]+ch*C*C+i[:, None]*C+j[None, :])
    gc = tl.load(Work[0]+ch*C*C+i[:, None]*C+j[None, :])
    cross = tl.load(Work[1]+ch*C*C+i[:, None]*C+j[None, :])
    weight = beta[:, None]*pc
    gn -= tl.sum(weight*gc, 0)
    gb += tl.sum(weight*(cross-ge[:, None]*prefj[None, :]), 0)
    tl.store(Gauge[0]+ch*C+j, gn)
    tl.store(Gauge[1]+ch*C+j, gb)


@triton.jit
def interval_diagonal_tc_fwd(Ell, Starts, Ends, P, Flags, H: tl.constexpr, KD: tl.constexpr, C: tl.constexpr, K: tl.constexpr):
    ch = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    i = tl.arange(0, 32).to(tl.int64)
    start, end = tl.load(Starts+ch//H), tl.load(Ends+ch//H)
    if tl.load(Flags+ch*(C//32)+tile):
        gram, readout = tl.full((32, 32), 0., tl.float32), tl.full((32, 32), 0., tl.float32)
        for ktile in range(K//32):
            a = ktile*32+i
            decay = centered_diagonal_scale(Ell, start, end-start, ch % H, tile, a, H, KD)
            k = interval_tile_load(P[1], ch, tile, a, C, K)
            qr = interval_tile_load(P[6], ch, tile, a, C, K)
            right = k/decay
            gram += mm_three(k*decay, tl.trans(right), 32, 32, 32)
            readout += mm_three(qr*decay, tl.trans(right), 32, 32, 32)
        off = ch*C*C+(tile*32+i[:, None])*C+tile*32+i[None, :]
        tl.store(P[9]+off, tl.where(i[:, None] > i[None, :], gram, 0.))
        tl.store(P[12]+off, tl.where(i[:, None] > i[None, :], readout, 0.))


@triton.jit
def interval_diagonal_tc_bwd(Ell, DQ, DK, Starts, Ends, P, Sh, Scratch, Flags,
                             H: tl.constexpr, KD: tl.constexpr, NG: tl.constexpr,
                             C: tl.constexpr, K: tl.constexpr):
    ch = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    a = tl.program_id(2).to(tl.int64)*32+tl.arange(0, 32).to(tl.int64)
    i = tl.arange(0, 32).to(tl.int64)
    if tl.load(Flags+ch*(C//32)+tile):
        start, end = tl.load(Starts+ch//H), tl.load(Ends+ch//H)
        decay = centered_diagonal_scale(Ell, start, end-start, ch % H, tile, a, H, KD)
        k = interval_tile_load(P[1], ch, tile, a, C, K)
        qr = interval_tile_load(P[6], ch, tile, a, C, K)
        goff = ch*C*C+(tile*32+i[:, None])*C+tile*32+i[None, :]
        gg, gc = tl.load(Sh[0]+goff).to(tl.float32), tl.load(Sh[1]+goff).to(tl.float32)
        inverse = k/decay
        rowk = mm_three(gg, inverse, 32, 32, 32)*decay
        rowq = mm_three(gc, inverse, 32, 32, 32)*decay
        colk = (mm_three(tl.trans(gg), k*decay, 32, 32, 32) +
                mm_three(tl.trans(gc), qr*decay, 32, 32, 32))/decay
        off = ((start+tile*32+i[:, None])*H+ch % H)*KD+a[None, :]
        valid = (tile*32+i[:, None] < end-start) & (a[None, :] < KD)
        tl.store(DK+off, tl.load(DK+off, valid, 0.)+rowk+colk, valid)
        tl.store(DQ+off, tl.load(DQ+off, valid, 0.)+rowq, valid)
        scratch = ch*C*K+(tile*32+i[:, None])*K+a[None, :]
        incidence = tl.load(Scratch[0]+scratch).to(tl.float32).to(tl.float32)+k*rowk+qr*rowq-k*colk
        tl.store(Scratch[0]+scratch, incidence)


@triton.jit
def diagonal_eligibility(Ell, Starts, Ends, P, Flags, H: tl.constexpr, KD: tl.constexpr,
                         C: tl.constexpr, K: tl.constexpr):
    ch = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    i, a = tl.arange(0, 32).to(tl.int64), tl.arange(0, K).to(tl.int64)
    start, end = tl.load(Starts+ch//H), tl.load(Ends+ch//H)
    ell = interval_ell(Ell, start, end-start, ch % H, tile, a, H, KD)
    relative = tl.cumsum(tl.where(i[:, None] > 0, ell, 0.), 0)
    n = tl.load(P[18]+ch*C+tile*32+i)
    qr = interval_tile_load(P[6], ch, tile, a, C, K).to(tl.float32)
    safe = (tl.min(relative) >= -64.) & (tl.min(n) >= 0.5) & (tl.max(n) <= 2.) & (tl.max(tl.abs(qr)) <= 2.)
    tl.store(Flags+ch*(C//32)+tile, safe)


@triton.jit
def centered_diagonal_scale(Ell, start, length, head, tile, a, H: tl.constexpr, KD: tl.constexpr):
    i = tl.arange(0, 32)
    ell = interval_ell(Ell, start, length, head, tile, a, H, KD)
    relative = tl.cumsum(tl.where(i[:, None] > 0, ell, 0.), 0)
    center = 0.5 * tl.min(relative, 0)
    return libdevice.exp(relative-center[None, :])


@triton.jit
def finite_check_kernel(Sources, Sizes: tl.constexpr, Flags, BLOCK: tl.constexpr):
    task = tl.program_id(0).to(tl.int64)
    x = task*BLOCK+tl.arange(0, BLOCK).to(tl.int64)
    bad = tl.full((BLOCK,), False, tl.int1)
    offset = 0
    for index in tl.static_range(len(Sizes)):
        mask = (x >= offset) & (x < offset+Sizes[index])
        value = tl.load(Sources[index]+x-offset, mask, 0.).to(tl.float32)
        bad |= mask & ((value != value) | (tl.abs(value) == float('inf')))
        offset += Sizes[index]
    tl.store(Flags+task, tl.sum(bad.to(tl.int32), 0))


@triton.jit
def nonfinite(x):
    return (x != x) | (tl.abs(x) == float('inf'))


@triton.jit
def input_domain_flags(Q, K0, V, Beta, Ell, Curve, Flags, N: tl.constexpr,
                       K: tl.constexpr, VD: tl.constexpr, GS: tl.constexpr):
    task = tl.program_id(0).to(tl.int64)
    row = task*16+tl.arange(0, 16).to(tl.int64)
    a = tl.arange(0, K).to(tl.int64)
    vcol = tl.arange(0, VD).to(tl.int64)
    q = tl.load(Q+row[:, None]*K+a[None, :], row[:, None] < N, 0.).to(tl.float32)
    k = tl.load(K0+row[:, None]*K+a[None, :], row[:, None] < N, 0.).to(tl.float32)
    ell = tl.load(Ell+row[:, None]*K+a[None, :], row[:, None] < N, 0.).to(tl.float32)
    v = tl.load(V+row[:, None]*VD+vcol[None, :], row[:, None] < N, 0.).to(tl.float32)
    beta = tl.load(Beta+row, row < N, 0.).to(tl.float32)
    c = tl.load(Curve).to(tl.float32)
    source_bad = tl.sum((nonfinite(q) | nonfinite(k) | nonfinite(ell) | (ell > 0)).to(tl.int32)) > 0
    source_bad |= tl.sum(nonfinite(v).to(tl.int32)) > 0
    source_bad |= tl.sum((nonfinite(beta) | (beta < 0) | (beta > 1)).to(tl.int32)) > 0
    source_bad |= nonfinite(c)
    n = tl.sum(k*k, 1)
    vv = tl.sum(tl.reshape(v*v, (16, VD//GS, GS)), 2)
    mass = (c/GS)*vv
    margin = n[:, None]-mass
    domain_bad = tl.sum(((row < N) & (nonfinite(n) | (n <= 0))).to(tl.int32)) > 0
    domain_bad |= tl.sum(((row[:, None] < N) & (~(margin > 0))).to(tl.int32)) > 0
    near = tl.sum(((row[:, None] < N) & (tl.abs(margin) <= 0.0009765625 *
                  tl.maximum(n[:, None], tl.abs(mass)))).to(tl.int32)) > 0
    near |= tl.sum(((row < N) & ((n < 7.888609052210118e-31) | (n > 1.2676506002282294e30))).to(tl.int32)) > 0
    near |= tl.sum(((row[:, None] < N) & (vv > 1.2676506002282294e30)).to(tl.int32)) > 0
    near |= ((c > 0) & (c < 7.888609052210118e-31)) | (c > 1.2676506002282294e30)
    force_rho = tl.sum(((row < N) & ((n < 0.51) | (n > 1.99))).to(tl.int32)) > 0
    force_rho |= tl.sum(((row[:, None] < N) & (mass/n[:, None] > 0.23)).to(tl.int32)) > 0
    code = source_bad.to(tl.int32) | (domain_bad.to(tl.int32) << 1) | (near.to(tl.int32) << 2) | (force_rho.to(tl.int32) << 3)
    tl.store(Flags+task, code)


@triton.jit
def state_domain_flags(S, Curve, Flags, Offset: tl.constexpr, K: tl.constexpr,
                       VD: tl.constexpr, GS: tl.constexpr):
    task = tl.program_id(0).to(tl.int64)
    ng = VD//GS
    base, group = task//ng, task % ng
    a, g = tl.arange(0, K).to(tl.int64), tl.arange(0, GS).to(tl.int64)
    s = tl.load(S+base*K*VD+a[:, None]*VD+group*GS+g[None, :]).to(tl.float32)
    c = tl.load(Curve).to(tl.float32)
    source_bad = (tl.sum(nonfinite(s).to(tl.int32)) > 0) | nonfinite(c)
    radius = tl.sum(s*s)
    mass = (c/GS)*radius
    domain_bad = ~(mass < 1.)
    near = (tl.abs(mass-1.) <= 0.0009765625*tl.maximum(tl.abs(mass), 1.)) | (radius > 1.2676506002282294e30)
    near |= ((c > 0) & (c < 7.888609052210118e-31)) | (c > 1.2676506002282294e30)
    code = source_bad.to(tl.int32) | (domain_bad.to(tl.int32) << 1) | (near.to(tl.int32) << 2)
    tl.store(Flags+Offset+task, code)


@triton.jit
def merge_flag_codes(a, b):
    return a | b


@triton.jit
def reduce_domain_flags(Flags, Reduced, N: tl.constexpr, BLOCK: tl.constexpr):
    task = tl.program_id(0).to(tl.int64)
    i = task*BLOCK+tl.arange(0, BLOCK).to(tl.int64)
    values = tl.load(Flags+i, i < N, 0)
    value = tl.reduce(values, 0, merge_flag_codes)
    tl.store(Reduced+task, value)


@triton.jit
def shared_feedback_bwd(Curvature, Gp, SavedY, SavedW, OutputAdj, Sh,
                        NG: tl.constexpr, GS: tl.constexpr, C: tl.constexpr, G: tl.constexpr, RHO_MATERIALIZED: tl.constexpr = True):
    ch = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1).to(tl.int64)*32+tl.arange(0, 32).to(tl.int64)
    j = tl.program_id(2).to(tl.int64)*64+tl.arange(0, 64).to(tl.int64)
    g = tl.arange(0, G).to(tl.int64)
    kap = tl.load(Curvature).to(tl.float32)/GS
    ga = tl.full((32, 64), 0., tl.float32)
    gl = tl.full((32, 64), 0., tl.float32)
    gqs = tl.full((32, 64), 0., tl.float32)
    for group in range(NG):
        block = ch*NG+group
        nu = tl.load(Gp[7]+block*C+i)
        gu = tl.load(OutputAdj[1]+block*C*G+i[:, None]*G+g[None, :])
        y = tl.load(SavedY+block*C*G+j[:, None]*G+g[None, :])
        gl -= mm_three(nu[:, None]*gu, tl.trans(y), 32, 64, G)
        b = tl.load(OutputAdj[2]+block*C+i)
        w = tl.load(SavedW+block*C+j)
        off = block*C*C+i[:, None]*C+j[None, :]
        rho = read_rho_matrix(Gp, block, i, j, C, RHO_MATERIALIZED)
        gz = -rho*b[:, None]*w[None, :]
        gqs += gz
        vv = tl.load(Gp[3]+off)
        ga += (-kap*gz)*vv
    off = ch*C*C+i[:, None]*C+j[None, :]
    tl.store(Sh[0]+off, ga)
    tl.store(Sh[1]+off, gl)
    tl.store(Sh[3]+off, gqs)


@triton.jit
def read_rho_matrix(Gp, block, rows, cols, C: tl.constexpr, MATERIALIZED: tl.constexpr):
    rr, cc = rows.to(tl.int64), cols.to(tl.int64)
    if MATERIALIZED:
        return tl.load(Gp[2]+block*C*C+rr[:, None]*C+cc[None, :]).to(tl.float32)
    nr = tl.load(Gp[7]+block*C+rr)
    inverse = 1./tl.load(Gp[7]+block*C+cc)
    return tl.where(rr[:, None] > cc[None, :], nr[:, None]*inverse[None, :], (rr[:, None] == cc[None, :]).to(tl.float32))


@triton.jit
def read_rho_row(Gp, block, row_id, cols, C: tl.constexpr, MATERIALIZED: tl.constexpr):
    cc = cols.to(tl.int64)
    if MATERIALIZED:
        return tl.load(Gp[2]+block*C*C+row_id*C+cc).to(tl.float32)
    nr = tl.load(Gp[7]+block*C+row_id)
    inverse = 1./tl.load(Gp[7]+block*C+cc)
    return tl.where(cc < row_id, nr*inverse, (cc == row_id).to(tl.float32))


@triton.jit
def shared_feedback_gl_bwd(Gp, SavedY, OutputAdj, Sh, NG: tl.constexpr, C: tl.constexpr, G: tl.constexpr):
    ch = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1).to(tl.int64)*64+tl.arange(0, 64).to(tl.int64)
    j = tl.program_id(2).to(tl.int64)*64+tl.arange(0, 64).to(tl.int64)
    r = tl.arange(0, NG*G).to(tl.int64)
    blocks = ch*NG+r//G
    nu = tl.load(Gp[7]+blocks[None, :]*C+i[:, None])
    gu = tl.load(OutputAdj[1]+blocks[None, :]*C*G+i[:, None]*G+(r % G)[None, :])
    y = tl.load(SavedY+blocks[None, :]*C*G+j[:, None]*G+(r % G)[None, :])
    value = -tl.dot((nu*gu).to(tl.bfloat16), tl.trans(y).to(tl.bfloat16), out_dtype=tl.float32)
    tl.store(Sh[1]+ch*C*C+i[:, None]*C+j[None, :], value)


@triton.jit
def shared_feedback_system_bwd(Curvature, Gp, SavedY, SavedW, OutputAdj, Sh,
                               NG: tl.constexpr, GS: tl.constexpr, C: tl.constexpr, G: tl.constexpr, RHO_MATERIALIZED: tl.constexpr = True):
    ch = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1).to(tl.int64)*64+tl.arange(0, 64).to(tl.int64)
    j = tl.program_id(2).to(tl.int64)*64+tl.arange(0, 64).to(tl.int64)
    kap = tl.load(Curvature).to(tl.float32)/GS
    ga = tl.full((64, 64), 0., tl.float32)
    gqs = tl.full((64, 64), 0., tl.float32)
    for group in range(NG):
        block = ch*NG+group
        b = tl.load(OutputAdj[2]+block*C+i)
        w = tl.load(SavedW+block*C+j)
        off = block*C*C+i[:, None]*C+j[None, :]
        rho = read_rho_matrix(Gp, block, i, j, C, RHO_MATERIALIZED)
        gz = -rho*b[:, None]*w[None, :]
        gqs += gz
        vv = tl.load(Gp[3]+off)
        ga += (-kap*gz)*vv
    off = ch*C*C+i[:, None]*C+j[None, :]
    tl.store(Sh[0]+off, ga)
    tl.store(Sh[3]+off, gqs)
