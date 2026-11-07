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
def mm_three(a, b, M: tl.constexpr, N: tl.constexpr, R: tl.constexpr):
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
def mm_state_three(a, b, M: tl.constexpr, N: tl.constexpr, R: tl.constexpr, MIXED: tl.constexpr):
    return mm_three(a, b, M, N, R)


@triton.jit
def mm_state_product(a, b, M: tl.constexpr, N: tl.constexpr, R: tl.constexpr, MIXED: tl.constexpr):
    return mm(a, b, M, N, R)


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
    return tl.load(p + block * M * N + x[:, None] * N + y[None, :])


@triton.jit
def store_matrix(p, block, a, M: tl.constexpr, N: tl.constexpr):
    x = tl.arange(0, M).to(tl.int64)
    y = tl.arange(0, N).to(tl.int64)
    tl.store(p + block * M * N + x[:, None] * N + y[None, :], a)


@triton.jit
def load_vector(p, block, N: tl.constexpr):
    return tl.load(p + block * N + tl.arange(0, N).to(tl.int64))


@triton.jit
def store_vector(p, block, a, N: tl.constexpr):
    tl.store(p + block * N + tl.arange(0, N).to(tl.int64), a)


@triton.jit
def reduced_row(a, N: tl.constexpr):
    # preserve each FP32 value while changing the reduction axis to a row axis
    i = tl.arange(0, N)
    result = tl.full((1, N), 0, tl.float32)
    for j in range(N):
        result = tl.where(i[None, :] == j, at(a, j, N), result)
    return result


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
    return x - gamma * k[:, None] * tl.sum(k[:, None] * x, 0)[None, :]


@triton.jit
def erase_vjp(k, gamma, x, grad):
    p = tl.sum(k[:, None] * x, 0)
    h = tl.sum(k[:, None] * grad, 0)
    gx = grad - gamma * k[:, None] * h[None, :]
    gk = -gamma * tl.sum(grad * p[None, :] + x * h[None, :], 1)
    gg = -tl.sum(p * h, 0)
    return gx, gk, gg


@triton.jit
def common_base_fwd_kernel(Q, K0, Beta, Ell, Starts, Ends, P,
                           H: tl.constexpr, KD: tl.constexpr, C: tl.constexpr, K: tl.constexpr, SCALE: tl.constexpr):
    block = tl.program_id(0).to(tl.int64)
    head = block % H
    start, end = tl.load(Starts + block // H), tl.load(Ends + block // H)
    length = end - start
    i, a = tl.arange(0, C), tl.arange(0, K)
    pos = start + i.to(tl.int64)
    off = (pos[:, None] * H + head) * KD + a[None, :].to(tl.int64)
    valid = (i[:, None] < length) & (a[None, :] < KD)
    q = tl.load(Q + off, valid, 0).to(tl.float32) * SCALE
    k = tl.load(K0 + off, valid, 0).to(tl.float32)
    ell = tl.load(Ell + off, valid, 0).to(tl.float32)
    beta = tl.load(Beta + pos * H + head, i < length, 0).to(tl.float32)
    n = tl.where(i < length, tl.sum(k * k, 1), 1.)
    gamma = beta / n
    decay = libdevice.exp(tl.cumsum(ell, 0))
    aq = tl.sum(q * k, 1) / n
    if C == 1:
        qr = q - tl.reshape(beta * aq, ()) * k
    else:
        qr = q - (beta * aq)[:, None] * k
    ke = decay * k
    bmat = gamma[:, None] * ke
    store_matrix(P[0], block, q, C, K)
    store_matrix(P[1], block, k, C, K)
    store_matrix(P[2], block, decay, C, K)
    store_matrix(P[3], block, ke, C, K)
    store_matrix(P[4], block, bmat, C, K)
    store_matrix(P[6], block, qr, C, K)
    store_matrix(P[7], block, qr * decay, C, K)
    store_vector(P[18], block, n, C)
    store_vector(P[19], block, gamma, C)
    store_vector(P[20], block, aq, C)
    store_vector(P[21], block, beta, C)
    products = tl.full((C, C), 0, tl.float32)
    qs = tl.full((C, C), 0, tl.float32)
    product = tl.full((C,), 1, tl.float32)
    # rows are independent; retain the original descending-j multiplication order per row.
    for jj in range(C):
        j = C - 1 - jj
        active = i > j
        mask = active[:, None] & (i[None, :] == j)
        products = tl.where(mask, product[:, None], products)
        qs = tl.where(mask, -beta[:, None] * at(n, j, C) * product[:, None], qs)
        product = tl.where(active, product * (1 - at(beta, j, C)), product)
    eta = beta * product
    qs += tl.where(i[:, None] == i[None, :], n[:, None], 0.)
    store_matrix(P[14], block, qs, C, C)
    store_matrix(P[15], block, products, C, C)
    store_vector(P[22], block, eta, C)


@triton.jit
def common_interval_fwd_kernel(Ell, Starts, Ends, P,
                               H: tl.constexpr, KD: tl.constexpr, C: tl.constexpr, K: tl.constexpr):
    task = tl.program_id(0).to(tl.int64)
    ch, j = task // C, (task % C).to(tl.int32)
    head = ch % H
    start, end = tl.load(Starts + ch // H), tl.load(Ends + ch // H)
    length = end - start
    i, a = tl.arange(0, C), tl.arange(0, K)
    pos = start + i.to(tl.int64)
    off = (pos[:, None] * H + head) * KD + a[None, :].to(tl.int64)
    valid = (i[:, None] < length) & (a[None, :] < KD)
    ell = tl.load(Ell + off, valid, 0).to(tl.float32)
    k = load_matrix(P[1], ch, C, K)
    qr = load_matrix(P[6], ch, C, K)
    # preserve direct local intervals, including a small decay following a huge negative gate.
    interval = libdevice.exp(tl.cumsum(tl.where(i[:, None] > j, ell, 0.), 0))
    kj = tl.load(P[1] + ch * C * K + j.to(tl.int64) * K + a)
    gj = tl.sum(k * interval * kj[None, :], 1)
    cj = tl.sum(qr * interval * kj[None, :], 1)
    offset = ch * C * C + i.to(tl.int64) * C + j
    active = (i > j) & (i < length)
    tl.store(P[9] + offset, tl.where(active, gj, 0.))
    tl.store(P[12] + offset, tl.where(active, cj, 0.))
    last = row(interval, length - 1, C, K) * kj
    tl.store(P[8] + ch * C * K + j.to(tl.int64) * K + a, last)


@triton.jit
def common_system_fwd_kernel(Starts, Ends, P, H: tl.constexpr, C: tl.constexpr, K: tl.constexpr):
    ch = tl.program_id(0).to(tl.int64)
    start, end = tl.load(Starts + ch // H), tl.load(Ends + ch // H)
    length = end - start
    i, a = tl.arange(0, C), tl.arange(0, K)
    gamma = load_vector(P[19], ch, C)
    gram = load_matrix(P[9], ch, C, C)
    lower = (i[:, None] == i[None, :]).to(tl.float32) + gamma[:, None] * gram
    store_matrix(P[10], ch, lower, C, C)
    inv = inverse_lower(lower, C)
    store_matrix(P[11], ch, inv, C, C)
    bmat = load_matrix(P[4], ch, C, K)
    f = mm_three(inv, bmat, C, K, C)
    store_matrix(P[5], ch, f, C, K)
    cr = load_matrix(P[12], ch, C, C)
    to = mm_three(cr, inv, C, C, C)
    store_matrix(P[13], ch, to, C, C)
    tail = load_matrix(P[8], ch, C, K)
    pastmask = (i[:, None] < length - 1) & (i[None, :] < length - 1)
    p0 = mm_three(tl.trans(tail), tl.where(pastmask, inv, 0.), K, C, C)
    store_matrix(P[16], ch, p0, K, C)
    kl = tl.load(P[1] + ch * C * K + (length - 1) * K + a)
    past = erase(kl, at(gamma, length - 1, C), p0)
    store_matrix(P[17], ch, past, K, C)


@triton.jit
def group_fwd_kernel(V0, Curvature, Starts, Ends, P, Gp,
                     H: tl.constexpr, VD: tl.constexpr, GS: tl.constexpr, NG: tl.constexpr,
                     C: tl.constexpr, G: tl.constexpr):
    block = tl.program_id(0).to(tl.int64)
    ch, group = block // NG, block % NG
    head = ch % H
    start, end = tl.load(Starts + ch // H), tl.load(Ends + ch // H)
    i, g = tl.arange(0, C), tl.arange(0, G)
    off = ((start + i.to(tl.int64))[:, None] * H + head) * VD + group * GS + g[None, :].to(tl.int64)
    v = tl.load(V0 + off, (i[:, None] < end - start) & (g[None, :] < GS), 0).to(tl.float32)
    kap = tl.load(Curvature).to(tl.float32) / GS
    inv = load_matrix(P[11], ch, C, C)
    qs = load_matrix(P[14], ch, C, C)
    n = load_vector(P[18], ch, C)
    beta = load_vector(P[21], ch, C)
    vv = mm_three(v, tl.trans(v), C, C, G)
    rmat = inv * vv
    history = tl.full((C,), 0, tl.float32)
    lam = tl.full((C,), 1, tl.float32)
    for r in range(C):
        rr = row(rmat, r, C, C)
        nr, br = at(n, r, C), at(beta, r, C)
        u0 = (br + kap * tl.sum(rr * history, 0)) / (nr - kap * at(rr, r, C))
        ratio = 1 - br + nr * u0
        history = history / ratio
        history = tl.where(i == r, u0 / ratio, history)
        lam = tl.where(i == r, ratio, lam)
    nu = tl.cumprod(1 / lam, 0)
    rho = tl.full((C, C), 0, tl.float32)
    for j in range(C):
        value = tl.cumprod(tl.where(i > j, 1 / lam, 1.), 0)
        rho = tl.where((i[:, None] >= j) & (i[None, :] == j), value[:, None], rho)
    zi = inverse_lower((qs - kap * rmat) * rho, C)
    jf = -kap * zi * nu[None, :]
    store_matrix(Gp[0], block, zi, C, C)
    store_matrix(Gp[1], block, jf, C, C)
    store_matrix(Gp[2], block, rho, C, C)
    store_matrix(Gp[3], block, vv, C, C)
    store_matrix(Gp[4], block, rmat, C, C)
    store_vector(Gp[6], block, lam, C)
    store_vector(Gp[7], block, nu, C)


@triton.jit
def group_af_fwd_kernel(P, Gp, NG: tl.constexpr, C: tl.constexpr):
    # A scalar output per program avoids the replicated reduction-to-vector shared store.
    block = tl.program_id(0).to(tl.int64)
    r = tl.program_id(1).to(tl.int64)
    j = tl.arange(0, C).to(tl.int64)
    zi = tl.load(Gp[0] + block * C * C + r * C + j)
    nu = load_vector(Gp[7], block, C)
    eta = load_vector(P[22], block // NG, C)
    af = tl.sum(zi * (nu * eta), 0)
    tl.store(Gp[5] + block * C + r, af)


@triton.jit
def local_graph(P, Gp, ch, block, v, s, y, length,
                C: tl.constexpr, K: tl.constexpr, G: tl.constexpr, MIXED: tl.constexpr = False):
    i = tl.arange(0, C)
    lower = load_matrix(P[10], ch, C, C)
    qer = load_matrix(P[7], ch, C, K)
    to = load_matrix(P[13], ch, C, C)
    past = load_matrix(P[17], ch, K, C)
    k = load_matrix(P[1], ch, C, K)
    decay = load_matrix(P[2], ch, C, K)
    n = load_vector(P[18], ch, C)
    gamma = load_vector(P[19], ch, C)
    aq = load_vector(P[20], ch, C)
    beta = load_vector(P[21], ch, C)
    jf = load_matrix(Gp[1], block, C, C)
    rho = load_matrix(Gp[2], block, C, C)
    af = load_vector(Gp[5], block, C)
    lam = load_vector(Gp[6], block, C)
    nu = load_vector(Gp[7], block, C)
    h = tl.sum(v * y, 1)
    w = af + tl.sum(jf * h[None, :], 1)
    u = w[:, None] * v - nu[:, None] * mm_state_three(lower, y, C, G, C, MIXED)
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
    ro = nu[:, None] * mm_three(qer, s, C, G, K) + mm_three(to * rho, u, C, G, C)
    target = aq[:, None] * v
    z = r + t
    output = ro / z[:, None] + (t / z)[:, None] * target
    kl = row(k, length - 1, C, K)
    tau = row(decay, length - 1, C, K)
    gl = at(gamma, length - 1, C)
    ss = at(nu, length - 1, C) * s
    us = tl.where(i[:, None] < length - 1, row(rho, length - 1, C, C)[:, None] * u, 0.)
    rs = erase(kl, gl, tau[:, None] * ss) + mm_state_three(past, us, K, G, C, MIXED)
    ts = kl[:, None] * row(v, length - 1, C, G)[None, :] / at(n, length - 1, C)
    zl, tl_last = at(z, length - 1, C), at(t, length - 1, C)
    sn = rs / zl + (tl_last / zl) * ts
    return output, sn, h, w, u, r, t, zp, ro, target, ss, us, rs, ts


@triton.jit
def local_state_graph(P, Gp, ch, block, v, s, y, length,
                      C: tl.constexpr, K: tl.constexpr, G: tl.constexpr, MIXED: tl.constexpr = False):
    i = tl.arange(0, C)
    lower = load_matrix(P[10], ch, C, C)
    past = load_matrix(P[17], ch, K, C)
    k = load_matrix(P[1], ch, C, K)
    decay = load_matrix(P[2], ch, C, K)
    n = load_vector(P[18], ch, C)
    gamma = load_vector(P[19], ch, C)
    beta = load_vector(P[21], ch, C)
    jf = load_matrix(Gp[1], block, C, C)
    rho = load_matrix(Gp[2], block, C, C)
    af = load_vector(Gp[5], block, C)
    lam = load_vector(Gp[6], block, C)
    nu = load_vector(Gp[7], block, C)
    h = tl.sum(v * y, 1)
    w = af + tl.sum(jf * h[None, :], 1)
    u = w[:, None] * v - nu[:, None] * mm_state_three(lower, y, C, G, C, MIXED)
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
    kl = row(k, length - 1, C, K)
    tau = row(decay, length - 1, C, K)
    gl = at(gamma, length - 1, C)
    ss = at(nu, length - 1, C) * s
    us = tl.where(i[:, None] < length - 1, row(rho, length - 1, C, C)[:, None] * u, 0.)
    rs = erase(kl, gl, tau[:, None] * ss) + mm_state_three(past, us, K, G, C, MIXED)
    ts = kl[:, None] * row(v, length - 1, C, G)[None, :] / at(n, length - 1, C)
    zl, tl_last = at(z, length - 1, C), at(t, length - 1, C)
    sn = rs / zl + (tl_last / zl) * ts
    return sn, u, r, t, ss, us, w, zp


@triton.jit
def state_fwd_kernel(V0, S0, Out, Final, Starts, Ends, SeqChunks, P, Gp, SavedS, SavedY, SavedU, SavedR, SavedT, SavedW, SavedZP, Carry,
                     H: tl.constexpr, KD: tl.constexpr, VD: tl.constexpr, GS: tl.constexpr, NG: tl.constexpr,
                     C: tl.constexpr, K: tl.constexpr, G: tl.constexpr, FINAL: tl.constexpr, CARRY: tl.constexpr,
                     MIXED: tl.constexpr = False, BF16_FS6: tl.constexpr = False):
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
        v = tl.load(V0 + vo, vmask, 0).to(tl.float32)
        f = load_matrix(P[5], ch, C, K)
        if CARRY and chunk > first:
            y = yn
        else:
            if BF16_FS6:
                tl.static_assert(not MIXED and not CARRY)
                y = mm_three(f, s, C, G, K)
            else:
                y = mm_state_three(f, s, C, G, K, MIXED)
        store_matrix(SavedS, block, s, K, G)
        store_matrix(SavedY, block, y, C, G)
        s, u, r, t, ss, us, w, zp = local_state_graph(
            P, Gp, ch, block, v, s, y, end - start, C, K, G, MIXED)
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
                      C: tl.constexpr, K: tl.constexpr, G: tl.constexpr):
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
    rho = load_matrix(Gp[2], block, C, C)
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
                      C: tl.constexpr, K: tl.constexpr, G: tl.constexpr):
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
    s = load_matrix(SavedS, block, K, G)
    if C <= 16:
        gqer = mm(nu[:, None] * gro, tl.trans(s), C, K, G)
    else:
        gqer = mm_three(nu[:, None] * gro, tl.trans(s), C, K, G)
    store_matrix(Sh[5], block, gqer, C, K)
    to = load_matrix(P[13], ch, C, C)
    rho = load_matrix(Gp[2], block, C, C)
    gu = mm_three(tl.trans(to * rho), gro, C, G, C)
    store_matrix(OutputAdj[1], block, gu, C, G)
    u = load_matrix(SavedU, block, C, G)
    if C <= 16:
        g_to = tl.where(i[:, None] > i[None, :], rho * mm(gro, tl.trans(u), C, C, G), 0.)
    else:
        g_to = tl.where(i[:, None] > i[None, :], rho * mm_three(gro, tl.trans(u), C, C, G), 0.)
    store_matrix(Sh[6], block, g_to, C, C)


@triton.jit
def saved_state_tail(P, Gp, ch, block, v, s, u, length,
                     C: tl.constexpr, K: tl.constexpr, G: tl.constexpr):
    i, a = tl.arange(0, C), tl.arange(0, K)
    last = (length - 1).to(tl.int64)
    kl = tl.load(P[1] + ch * C * K + last * K + a)
    tau = tl.load(P[2] + ch * C * K + last * K + a)
    gl = tl.load(P[19] + ch * C + last)
    nl = tl.load(P[18] + ch * C + last)
    nul = tl.load(Gp[7] + block * C + last)
    rhol = tl.load(Gp[2] + block * C * C + last * C + i)
    ss = nul * s
    us = tl.where(i[:, None] < length - 1, rhol[:, None] * u, 0.)
    past = load_matrix(P[17], ch, K, C)
    rs = erase(kl, gl, tau[:, None] * ss) + mm_state_three(past, us, K, G, C, False)
    ts = kl[:, None] * row(v, length - 1, C, G)[None, :] / nl
    return ss, us, rs, ts


@triton.jit
def state_bwd_kernel(V0, Curvature, DS, DV, DS0, Starts, Ends, SeqChunks, P, Gp, SavedS, Local, Sh, OutputAdj, Carry, EdgeY,
                     H: tl.constexpr, KD: tl.constexpr, VD: tl.constexpr, GS: tl.constexpr, NG: tl.constexpr,
                     C: tl.constexpr, K: tl.constexpr, G: tl.constexpr, CARRY: tl.constexpr):
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
        v = tl.load(V0 + vo, mask, 0).to(tl.float32)
        s = load_matrix(SavedS, block, K, G)
        u = load_matrix(Local[0], block, C, G)
        r, t = load_vector(Local[1], block, C), load_vector(Local[2], block, C)
        w, zp = load_vector(Local[4], block, C), load_vector(Local[5], block, C)
        ss, us, rs, ts = saved_state_tail(P, Gp, ch, block, v, s, u, length, C, K, G)
        n = load_vector(P[18], ch, C)
        beta = load_vector(P[21], ch, C)
        rho = load_matrix(Gp[2], block, C, C)
        nu = load_vector(Gp[7], block, C)
        lam = load_vector(Gp[6], block, C)
        z = r + t
        gu = load_matrix(OutputAdj[1], block, C, G)
        gr, gt = load_vector(OutputAdj[2], block, C), load_vector(OutputAdj[3], block, C)
        rl, tt, zz = at(r, length - 1, C), at(t, length - 1, C), at(z, length - 1, C)
        adj = bs
        carry_gu = tl.full((C, G), 0, tl.float32)
        norm_y, dot_y = tl.full((), 0, tl.float32), tl.full((), 0, tl.float32)
        if CARRY:
            store_matrix(EdgeY, block, by, C, G)
        if CARRY and chunk + 1 < last:
            e = load_matrix(Carry[0], ch, C, K)
            hc = load_matrix(Carry[1], ch, C, C)
            fn = load_matrix(P[5], ch + H, C, K)
            klc = tl.load(P[1] + ch * C * K + (length - 1).to(tl.int64) * K + a)
            nlc = at(n, length - 1, C)
            ry = mm_state_three(e, ss, C, G, K, False) + mm_state_three(hc, us, C, G, C, False)
            ty = (tl.sum(fn * klc[None, :], 1) / nlc)[:, None] * row(v, length - 1, C, G)[None, :]
            adj += mm_state_three(tl.trans(fn), by, K, G, C, False)
            norm_y = tl.sum(tl.sum(by * (ry / zz), 0), 0)
            dot_y = tl.sum(tl.sum(by * ty, 0), 0)
            carry_gu = mm_state_three(tl.trans(hc), by, C, G, C, False)
        norm_s = norm_y + tl.sum(tl.sum(bs * (rs / zz), 0), 0)
        dot_s = dot_y + tl.sum(tl.sum(bs * ts, 0), 0)
        gr += tl.where(i == length - 1, -(norm_s + (tt / zz) * dot_s) / zz, 0.)
        gt += tl.where(i == length - 1, ((rl / zz) * dot_s - norm_s) / zz, 0.)
        past = load_matrix(P[17], ch, K, C)
        gu += tl.where(i[:, None] < length - 1,
                       row(rho, length - 1, C, C)[:, None] * (
                           mm_state_three(tl.trans(past), bs, C, G, K, False) + carry_gu) / zz, 0.)
        gb, gn, gw = tl.full((C,), 0, tl.float32), tl.full((C,), 0, tl.float32), tl.full((C,), 0, tl.float32)
        gz = tl.full((), 0, tl.float32)
        for jj in range(C):
            j = C - 1 - jj
            gri, gti = at(gr, j, C) + gz, at(gt, j, C) + gz
            gb += tl.where(i == j, -gri * at(zp, j, C) / at(lam, j, C), 0.)
            gz = gri * (1 - at(beta, j, C)) / at(lam, j, C)
            gn += tl.where(i == j, gti * at(w, j, C), 0.)
            gw += tl.where(i == j, gti * at(n, j, C), 0.)
        gw += tl.sum(gu * v, 1)
        zi = load_matrix(Gp[0], block, C, C)
        b = tl.sum(tl.trans(zi) * gw[None, :], 1)
        rhs = nu * b
        gh = -kap * rhs
        lower = load_matrix(P[10], ch, C, C)
        gy = -mm_state_three(tl.trans(lower), nu[:, None] * gu, C, G, C, False)
        gy += gh[:, None] * v
        gs = load_matrix(OutputAdj[0], block, K, G)
        # all aliased output partials are consumed before their storage is reused
        tl.debug_barrier()
        store_matrix(OutputAdj[0], block, adj, K, G)
        store_matrix(OutputAdj[1], block, gu, C, G)
        store_vector(OutputAdj[2], block, b, C)
        kl = tl.load(P[1] + ch * C * K + (length - 1).to(tl.int64) * K + a)
        tau = tl.load(P[2] + ch * C * K + (length - 1).to(tl.int64) * K + a)
        gl = tl.load(P[19] + ch * C + (length - 1).to(tl.int64))
        gx = erase(kl, gl, adj / zz)
        gs += at(nu, length - 1, C) * tau[:, None] * gx
        if not CARRY or chunk == first:
            f = load_matrix(P[5], ch, C, K)
            gs += mm_state_three(tl.trans(f), gy, K, G, C, False)
        tl.store(DV + vo, gy, mask)
        store_vector(Sh[8], block, gn, C)
        store_vector(Sh[12], block, gb, C)
        bs, by = gs, gy
    tl.store(DS0 + so, bs, smask)


@triton.jit
def parameter_bwd_kernel(V0, Curvature, DO, DV, DC, Starts, Ends, Prev, Next, P, Gp, SavedS, SavedY, Local, Sh, OutputAdj, EdgeY,
                         H: tl.constexpr, KD: tl.constexpr, VD: tl.constexpr, GS: tl.constexpr, NG: tl.constexpr,
                         C: tl.constexpr, K: tl.constexpr, G: tl.constexpr, CARRY: tl.constexpr):
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
    gy = tl.load(DV + vo, mask, 0).to(tl.float32)
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
    ss, us, rs, ts = saved_state_tail(P, Gp, ch, block, v, s, u, length, C, K, G)
    residual_grad = adj / zz
    gp = mm_state_three(residual_grad, tl.trans(us), K, C, G, False)
    store_matrix(Sh[7], block, gp, K, C)
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
    gf = tl.full((C, K), 0, tl.float32)
    previous, following = tl.load(Prev + chunk), tl.load(Next + chunk)
    if not CARRY or previous < 0:
        gf = mm_state_three(gy, tl.trans(s), C, K, G, False)
    store_matrix(Sh[2], block, gf, C, K)
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
    glower = -mm_state_three(nu[:, None] * gu, tl.trans(y), C, C, G, False)
    store_matrix(Sh[1], block, glower, C, C)
    rho = load_matrix(Gp[2], block, C, C)
    gzmatrix = -rho * b[:, None] * w[None, :]
    rhs = nu * b
    store_matrix(Sh[3], block, gzmatrix, C, C)
    store_vector(Sh[4], block, rhs, C)
    kap = tl.load(Curvature).to(tl.float32) / GS
    gh = -kap * rhs
    grmat = -kap * gzmatrix
    rmat = load_matrix(Gp[4], block, C, C)
    h = tl.sum(v * y, 1)
    gc = -tl.sum(rhs * h, 0) - tl.sum(tl.sum(gzmatrix * rmat, 0), 0)
    tl.store(DC + block, gc / GS)
    gv += gh[:, None] * y
    vv = load_matrix(Gp[3], block, C, C)
    ga = grmat * vv
    store_matrix(Sh[0], block, ga, C, C)
    inv = load_matrix(P[11], ch, C, C)
    gram_grad = grmat * inv
    gv += mm_state_three(gram_grad + tl.trans(gram_grad), v, C, G, C, False)
    tl.store(DV + vo, gv, mask)


@triton.jit
def sum_matrix(p, ch, M: tl.constexpr, N: tl.constexpr, NG: tl.constexpr):
    result = tl.full((M, N), 0, tl.float32)
    for group in range(NG):
        result += load_matrix(p, ch * NG + group, M, N)
    return result


@triton.jit
def sum_vector(p, ch, N: tl.constexpr, NG: tl.constexpr):
    result = tl.full((N,), 0, tl.float32)
    for group in range(NG):
        result += load_vector(p, ch * NG + group, N)
    return result


@triton.jit
def gauge_bwd_kernel(P, Sh, Gauge, NG: tl.constexpr, C: tl.constexpr):
    ch = tl.program_id(0).to(tl.int64)
    i = tl.arange(0, C)
    n = load_vector(P[18], ch, C)
    beta = load_vector(P[21], ch, C)
    products = load_matrix(P[15], ch, C, C)
    gqs = sum_matrix(Sh[3], ch, C, C, NG)
    ge = sum_vector(Sh[4], ch, C, NG)
    gn = sum_vector(Sh[8], ch, C, NG)
    gb = sum_vector(Sh[12], ch, C, NG)
    prefixes = tl.cumprod(tl.where(i[None, :] < i[:, None], 1 - beta[None, :], 1.), 1)
    final_product = tl.sum(tl.where(i[None, :] == C - 1, prefixes, 0.), 1)
    gn += tl.sum(tl.where(i[:, None] == i[None, :], gqs, 0.), 1)
    gb += final_product * ge
    prod_grad = beta * ge
    n_cross = tl.full((C, C), 0, tl.float32)
    b_cross = tl.full((C, C), 0, tl.float32)
    for j in range(C):
        active = i > j
        pr = tl.sum(tl.where(i[None, :] == j, products, 0.), 1)
        qj = tl.sum(tl.where(i[None, :] == j, gqs, 0.), 1)
        nj = at(n, j, C)
        column = active[:, None] & (i[None, :] == j)
        n_cross = tl.where(column, (-beta * pr * qj)[:, None], n_cross)
        gb += tl.where(active, -nj * pr * qj, 0.)
        b_cross = tl.where(column, (-pr * prod_grad)[:, None], b_cross)
        prod_grad = tl.where(active, (1 - at(beta, j, C)) * prod_grad - beta * nj * qj, prod_grad)
    # each own-row term precedes later-row cross terms, in the original ascending-r order.
    for r in range(C):
        gn += row(n_cross, r, C, C)
        gb += row(b_cross, r, C, C)
    store_vector(Gauge[0], ch, gn, C)
    store_vector(Gauge[1], ch, gb, C)


@triton.jit
def common_matrix_bwd_kernel(DQ, DK, DB, DE, Starts, Ends, Prev, P, Sh,
                             H: tl.constexpr, KD: tl.constexpr, NG: tl.constexpr,
                             C: tl.constexpr, K: tl.constexpr, CARRY: tl.constexpr):
    ch = tl.program_id(0).to(tl.int64)
    head, chunk = ch % H, ch // H
    start, end = tl.load(Starts + chunk), tl.load(Ends + chunk)
    length = end - start
    i, a = tl.arange(0, C), tl.arange(0, K)
    off = ((start + i.to(tl.int64))[:, None] * H + head) * KD + a[None, :].to(tl.int64)
    valid = (i[:, None] < length) & (a[None, :] < KD)
    gamma = load_vector(P[19], ch, C)
    kl = tl.load(P[1] + ch * C * K + (length - 1) * K + a)
    gp = sum_matrix(Sh[7], ch, K, C, NG)
    p0 = load_matrix(P[16], ch, K, C)
    gp0, gkl, ggl = erase_vjp(kl, at(gamma, length - 1, C), p0, gp)
    gk = tl.where(i[:, None] == length - 1, sum_vector(Sh[11], ch, K, NG)[None, :], 0.)
    gk += tl.where(i[:, None] == length - 1, gkl[None, :], 0.)
    tl.store(DK + off, gk, valid)
    inv = load_matrix(P[11], ch, C, C)
    pastmask = (i[:, None] < length - 1) & (i[None, :] < length - 1)
    gtail = mm_three(tl.where(pastmask, inv, 0.), tl.trans(gp0), C, K, C)
    tl.debug_barrier()
    store_matrix(Sh[7], ch * NG, gtail, C, K)
    ga = sum_matrix(Sh[0], ch, C, C, NG)
    tail = load_matrix(P[8], ch, C, K)
    ga += tl.where(pastmask, mm_three(tail, gp0, C, C, K), 0.)
    gl = sum_matrix(Sh[1], ch, C, C, NG)
    gto = sum_matrix(Sh[6], ch, C, C, NG)
    gcr = tl.where(i[:, None] > i[None, :], mm_three(gto, tl.trans(inv), C, C, C), 0.)
    tl.debug_barrier()
    store_matrix(Sh[1], ch * NG, gcr, C, C)
    cr = load_matrix(P[12], ch, C, C)
    ga += mm_three(tl.trans(cr), gto, C, C, C)
    gf = sum_matrix(Sh[2], ch, C, K, NG)
    if CARRY:
        prev = tl.load(Prev + chunk)
        if prev >= 0:
            gf += sum_matrix(Sh[14], prev * H + head, C, K, NG)
    bmat = load_matrix(P[4], ch, C, K)
    ga += mm_three(gf, tl.trans(bmat), C, C, K)
    g_bmat = mm_three(tl.trans(inv), gf, C, K, C)
    g_ke = gamma[:, None] * g_bmat
    gg = sum_vector(Sh[9], ch, C, NG)
    gg += tl.where(i == length - 1, ggl, 0.)
    ke = load_matrix(P[3], ch, C, K)
    gg += tl.sum(g_bmat * ke, 1)
    gl = tl.where(i[:, None] > i[None, :],
                  gl - mm_three(mm_three(tl.trans(inv), ga, C, C, C), tl.trans(inv), C, C, C), 0.)
    gram = load_matrix(P[9], ch, C, C)
    gg += tl.sum(gl * gram, 1)
    ggram = gamma[:, None] * gl
    tl.debug_barrier()
    store_matrix(Sh[0], ch * NG, ggram, C, C)
    tl.store(DB + (start + i.to(tl.int64)) * H + head, gg, i < length)
    decay = load_matrix(P[2], ch, C, K)
    gk = tl.load(DK + off, valid, 0.)
    gk += decay * g_ke
    tl.store(DK + off, gk, valid)
    gqer = sum_matrix(Sh[5], ch, C, K, NG)
    gqr = gqer * decay
    tl.store(DQ + off, gqr, valid)
    gd = tl.where(i[:, None] == length - 1, sum_vector(Sh[13], ch, K, NG)[None, :], 0.)
    qr = load_matrix(P[6], ch, C, K)
    gd += gqer * qr
    k = load_matrix(P[1], ch, C, K)
    gd += k * g_ke
    gate = tl.cumsum(gd * decay, 0, reverse=True)
    tl.store(DE + off, gate, valid)


@triton.jit
def common_interval_bwd_kernel(Ell, DQ, DK, DB, DE, Starts, Ends, P, Sh, Gauge,
                               H: tl.constexpr, KD: tl.constexpr, NG: tl.constexpr,
                               C: tl.constexpr, K: tl.constexpr, SCALE: tl.constexpr):
    ch = tl.program_id(0).to(tl.int64)
    head, chunk = ch % H, ch // H
    start, end = tl.load(Starts + chunk), tl.load(Ends + chunk)
    length = end - start
    i, a = tl.arange(0, C), tl.arange(0, K)
    off = ((start + i.to(tl.int64))[:, None] * H + head) * KD + a[None, :].to(tl.int64)
    valid = (i[:, None] < length) & (a[None, :] < KD)
    ell = tl.load(Ell + off, valid, 0).to(tl.float32)
    k = load_matrix(P[1], ch, C, K)
    qr = load_matrix(P[6], ch, C, K)
    gk = tl.load(DK + off, valid, 0.)
    gqr = tl.load(DQ + off, valid, 0.)
    gate = tl.load(DE + off, valid, 0.)
    for j in range(C):
        interval = libdevice.exp(tl.cumsum(tl.where(i[:, None] > j, ell, 0.), 0))
        kj = tl.load(P[1] + ch * C * K + j * K + a)
        gj = tl.load(Sh[0] + ch * NG * C * C + i.to(tl.int64) * C + j)
        cj = tl.load(Sh[1] + ch * NG * C * C + i.to(tl.int64) * C + j)
        left = gj[:, None] * k + cj[:, None] * qr
        gk += gj[:, None] * kj[None, :] * interval
        gqr += cj[:, None] * kj[None, :] * interval
        gk += tl.where(i[:, None] == j, tl.sum(left * interval, 0)[None, :], 0.)
        eg = left * kj[None, :] * interval
        gate += tl.where(i[:, None] > j, tl.cumsum(eg, 0, reverse=True), 0.)
        tailgrad = tl.load(Sh[7] + ch * NG * C * K + j * K + a) * row(interval, length - 1, C, K)
        gk += tl.where(i[:, None] == j, tailgrad[None, :], 0.)
        gate += tl.where(i[:, None] > j, (tailgrad * kj)[None, :], 0.)
    q = load_matrix(P[0], ch, C, K)
    n, gamma = load_vector(P[18], ch, C), load_vector(P[19], ch, C)
    aq, beta = load_vector(P[20], ch, C), load_vector(P[21], ch, C)
    gaq = sum_vector(Sh[10], ch, C, NG)
    gg = tl.load(DB + (start + i.to(tl.int64)) * H + head, i < length, 0.)
    gn, gb = load_vector(Gauge[0], ch, C), load_vector(Gauge[1], ch, C)
    dot = tl.sum(gqr * k, 1)
    gaq -= beta * dot
    gq = gqr + (gaq / n)[:, None] * k
    gb -= aq * dot
    gk -= (beta * aq)[:, None] * gqr
    gk += (gaq / n)[:, None] * q
    gn -= gaq * aq / n
    gb += gg / n
    gn -= gg * gamma / n
    gk += 2 * gn[:, None] * k
    tl.debug_barrier()
    tl.store(DQ + off, gq * SCALE, valid)
    tl.store(DK + off, gk, valid)
    tl.store(DE + off, gate, valid)
    tl.store(DB + (start + i.to(tl.int64)) * H + head, gb, i < length)


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
