# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Experimental strict readout and state/control with seventeen TF32x3 state-product sites."""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def mm(a, b, M: tl.constexpr, N: tl.constexpr, R: tl.constexpr):
    acc = tl.full((M, N), 0, tl.float64)
    for r in range(R):
        x = tl.gather(a, tl.full((M, 1), r, tl.int32), 1)
        y = tl.gather(b, tl.full((1, N), r, tl.int32), 0)
        acc = acc + x * y
    return acc


@triton.jit
def mm_state_product(a, b, M: tl.constexpr, N: tl.constexpr, R: tl.constexpr, MIXED: tl.constexpr):
    if MIXED and M >= 16 and N >= 16 and R >= 16:
        return tl.dot(a.to(tl.float32), b.to(tl.float32), out_dtype=tl.float32, input_precision="tf32x3").to(tl.float64)
    else:
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
    # Preserve each FP64 value while changing the reduction axis to a row axis.
    i = tl.arange(0, N)
    result = tl.full((1, N), 0, tl.float64)
    for j in range(N):
        result = tl.where(i[None, :] == j, at(a, j, N), result)
    return result


@triton.jit
def inverse_lower(a, C: tl.constexpr):
    i = tl.arange(0, C)
    result = tl.full((C, C), 0, tl.float64)
    for r in range(C):
        ar = row(a, r, C, C)
        rhs = (i == r).to(tl.float64) - tl.sum(ar[:, None] * result, 0)
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
def common_fwd_kernel(Q, K0, Beta, Ell, Starts, Ends, P,
                      H: tl.constexpr, KD: tl.constexpr, C: tl.constexpr, K: tl.constexpr, SCALE: tl.constexpr):
    block = tl.program_id(0).to(tl.int64)
    head = block % H
    start, end = tl.load(Starts + block // H), tl.load(Ends + block // H)
    length = end - start
    i, a = tl.arange(0, C), tl.arange(0, K)
    pos = start + i.to(tl.int64)
    off = (pos[:, None] * H + head) * KD + a[None, :].to(tl.int64)
    valid = (i[:, None] < length) & (a[None, :] < KD)
    q = tl.load(Q + off, valid, 0).to(tl.float64) * SCALE
    k = tl.load(K0 + off, valid, 0).to(tl.float64)
    ell = tl.load(Ell + off, valid, 0).to(tl.float64)
    beta = tl.load(Beta + pos * H + head, i < length, 0).to(tl.float64)
    n = tl.where(i < length, tl.sum(k * k, 1), 1.)
    gamma = beta / n
    decay = libdevice.exp(tl.cumsum(ell, 0))
    aq = tl.sum(q * k, 1) / n
    if C == 1:
        qr = q - tl.reshape(beta * aq, ()) * k
    else:
        qr = q - (beta * aq)[:, None] * k
    gram = tl.full((C, C), 0, tl.float64)
    cr = tl.full((C, C), 0, tl.float64)
    tail = tl.full((C, K), 0, tl.float64)
    # one left endpoint at a time retains only C*K interval values in registers
    for j in range(C):
        interval = libdevice.exp(tl.cumsum(tl.where(i[:, None] > j, ell, 0.), 0))
        kj = row(k, j, C, K)
        gj = tl.sum(k * interval * kj[None, :], 1)
        cj = tl.sum(qr * interval * kj[None, :], 1)
        mask = (i[:, None] > j) & (i[None, :] == j) & (i[:, None] < length)
        gram = tl.where(mask, gj[:, None], gram)
        cr = tl.where(mask, cj[:, None], cr)
        last = row(interval, length - 1, C, K) * kj
        tail = tl.where(i[:, None] == j, last[None, :], tail)
    lower = (i[:, None] == i[None, :]).to(tl.float64) + gamma[:, None] * gram
    inv = inverse_lower(lower, C)
    ke = decay * k
    bmat = gamma[:, None] * ke
    f = mm(inv, bmat, C, K, C)
    to = mm(cr, inv, C, C, C)
    pastmask = (i[:, None] < length - 1) & (i[None, :] < length - 1)
    p0 = mm(tl.trans(tail), tl.where(pastmask, inv, 0.), K, C, C)
    kl = row(k, length - 1, C, K)
    past = erase(kl, at(gamma, length - 1, C), p0)
    products = tl.full((C, C), 0, tl.float64)
    qs = tl.full((C, C), 0, tl.float64)
    eta = tl.full((C,), 0, tl.float64)
    for r in range(C):
        product = tl.full((), 1, tl.float64)
        br = at(beta, r, C)
        for jj in range(C):
            j = C - 1 - jj
            active = j < r
            products = tl.where((i[:, None] == r) & (i[None, :] == j) & active, product, products)
            qs = tl.where((i[:, None] == r) & (i[None, :] == j) & active,
                          -br * at(n, j, C) * product, qs)
            product = tl.where(active, product * (1 - at(beta, j, C)), product)
        eta = tl.where(i == r, br * product, eta)
    qs += tl.where(i[:, None] == i[None, :], n[:, None], 0.)
    store_matrix(P[0], block, q, C, K)
    store_matrix(P[1], block, k, C, K)
    store_matrix(P[2], block, decay, C, K)
    store_matrix(P[3], block, ke, C, K)
    store_matrix(P[4], block, bmat, C, K)
    store_matrix(P[5], block, f, C, K)
    store_matrix(P[6], block, qr, C, K)
    store_matrix(P[7], block, qr * decay, C, K)
    store_matrix(P[8], block, tail, C, K)
    store_matrix(P[9], block, gram, C, C)
    store_matrix(P[10], block, lower, C, C)
    store_matrix(P[11], block, inv, C, C)
    store_matrix(P[12], block, cr, C, C)
    store_matrix(P[13], block, to, C, C)
    store_matrix(P[14], block, qs, C, C)
    store_matrix(P[15], block, products, C, C)
    store_matrix(P[16], block, p0, K, C)
    store_matrix(P[17], block, past, K, C)
    store_vector(P[18], block, n, C)
    store_vector(P[19], block, gamma, C)
    store_vector(P[20], block, aq, C)
    store_vector(P[21], block, beta, C)
    store_vector(P[22], block, eta, C)


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
    v = tl.load(V0 + off, (i[:, None] < end - start) & (g[None, :] < GS), 0).to(tl.float64)
    kap = tl.load(Curvature).to(tl.float64) / GS
    inv = load_matrix(P[11], ch, C, C)
    qs = load_matrix(P[14], ch, C, C)
    n = load_vector(P[18], ch, C)
    beta = load_vector(P[21], ch, C)
    vv = mm(v, tl.trans(v), C, C, G)
    rmat = inv * vv
    history = tl.full((C,), 0, tl.float64)
    lam = tl.full((C,), 1, tl.float64)
    for r in range(C):
        rr = row(rmat, r, C, C)
        nr, br = at(n, r, C), at(beta, r, C)
        u0 = (br + kap * tl.sum(rr * history, 0)) / (nr - kap * at(rr, r, C))
        ratio = 1 - br + nr * u0
        history = history / ratio
        history = tl.where(i == r, u0 / ratio, history)
        lam = tl.where(i == r, ratio, lam)
    nu = tl.cumprod(1 / lam, 0)
    rho = tl.full((C, C), 0, tl.float64)
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
    u = w[:, None] * v - nu[:, None] * mm_state_product(lower, y, C, G, C, MIXED)
    prev = tl.full((), 1, tl.float64)
    r = tl.full((C,), 0, tl.float64)
    t = tl.full((C,), 0, tl.float64)
    zp = tl.full((C,), 0, tl.float64)
    for j in range(C):
        rj = (1 - at(beta, j, C)) / at(lam, j, C) * prev
        tj = at(n, j, C) * at(w, j, C)
        zp = tl.where(i == j, prev, zp)
        r, t = tl.where(i == j, rj, r), tl.where(i == j, tj, t)
        prev = rj + tj
    ro = nu[:, None] * mm(qer, s, C, G, K) + mm(to * rho, u, C, G, C)
    target = aq[:, None] * v
    z = r + t
    output = ro / z[:, None] + (t / z)[:, None] * target
    kl = row(k, length - 1, C, K)
    tau = row(decay, length - 1, C, K)
    gl = at(gamma, length - 1, C)
    ss = at(nu, length - 1, C) * s
    us = tl.where(i[:, None] < length - 1, row(rho, length - 1, C, C)[:, None] * u, 0.)
    rs = erase(kl, gl, tau[:, None] * ss) + mm_state_product(past, us, K, G, C, MIXED)
    ts = kl[:, None] * row(v, length - 1, C, G)[None, :] / at(n, length - 1, C)
    zl, tl_last = at(z, length - 1, C), at(t, length - 1, C)
    sn = rs / zl + (tl_last / zl) * ts
    return output, sn, h, w, u, r, t, zp, ro, target, ss, us, rs, ts


@triton.jit
def state_fwd_kernel(V0, S0, Out, Final, Starts, Ends, SeqChunks, P, Gp, SavedS, SavedY, Carry,
                     H: tl.constexpr, KD: tl.constexpr, VD: tl.constexpr, GS: tl.constexpr, NG: tl.constexpr,
                     C: tl.constexpr, K: tl.constexpr, G: tl.constexpr, FINAL: tl.constexpr, CARRY: tl.constexpr,
                     MIXED: tl.constexpr = False):
    shg = tl.program_id(0).to(tl.int64)
    group, head, seq = shg % NG, (shg // NG) % H, shg // (NG * H)
    a, g, i = tl.arange(0, K), tl.arange(0, G), tl.arange(0, C)
    so = ((seq * H + head) * KD + a[:, None].to(tl.int64)) * VD + group * GS + g[None, :].to(tl.int64)
    smask = (a[:, None] < KD) & (g[None, :] < GS)
    s = tl.load(S0 + so, smask, 0).to(tl.float64)
    first, last = tl.load(SeqChunks + seq), tl.load(SeqChunks + seq + 1)
    yn = tl.full((C, G), 0, tl.float64)
    for chunk in range(first, last):
        ch = chunk * H + head
        block = ch * NG + group
        start, end = tl.load(Starts + chunk), tl.load(Ends + chunk)
        vo = ((start + i.to(tl.int64))[:, None] * H + head) * VD + group * GS + g[None, :].to(tl.int64)
        vmask = (i[:, None] < end - start) & (g[None, :] < GS)
        v = tl.load(V0 + vo, vmask, 0).to(tl.float64)
        f = load_matrix(P[5], ch, C, K)
        if CARRY and chunk > first:
            y = yn
        else:
            y = mm_state_product(f, s, C, G, K, MIXED)
        store_matrix(SavedS, block, s, K, G)
        store_matrix(SavedY, block, y, C, G)
        output, s, h, w, u, r, t, zp, ro, target, ss, us, rs, ts = local_graph(
            P, Gp, ch, block, v, s, y, end - start, C, K, G, MIXED)
        if CARRY and chunk + 1 < last:
            e = load_matrix(Carry[0], ch, C, K)
            hc = load_matrix(Carry[1], ch, C, C)
            fn = load_matrix(P[5], ch + H, C, K)
            kl = row(load_matrix(P[1], ch, C, K), end - start - 1, C, K)
            nl = at(load_vector(P[18], ch, C), end - start - 1, C)
            ry = mm_state_product(e, ss, C, G, K, MIXED) + mm_state_product(hc, us, C, G, C, MIXED)
            ty = (tl.sum(fn * kl[None, :], 1) / nl)[:, None] * row(v, end - start - 1, C, G)[None, :]
            zz, tt = at(r + t, end - start - 1, C), at(t, end - start - 1, C)
            yn = ry / zz + (tt / zz) * ty
        tl.store(Out + vo, output, vmask)
    if FINAL:
        tl.store(Final + so, s, smask)


@triton.jit
def state_bwd_kernel(V0, Curvature, DO, DS, DV, DS0, DC, Starts, Ends, SeqChunks, P, Gp, SavedS, SavedY, Sh, Carry,
                     H: tl.constexpr, KD: tl.constexpr, VD: tl.constexpr, GS: tl.constexpr, NG: tl.constexpr,
                     C: tl.constexpr, K: tl.constexpr, G: tl.constexpr, CARRY: tl.constexpr, MIXED: tl.constexpr = False):
    shg = tl.program_id(0).to(tl.int64)
    group, head, seq = shg % NG, (shg // NG) % H, shg // (NG * H)
    a, g, i = tl.arange(0, K), tl.arange(0, G), tl.arange(0, C)
    so = ((seq * H + head) * KD + a[:, None].to(tl.int64)) * VD + group * GS + g[None, :].to(tl.int64)
    smask = (a[:, None] < KD) & (g[None, :] < GS)
    bs = tl.load(DS + so, smask, 0).to(tl.float64)
    kap = tl.load(Curvature).to(tl.float64) / GS
    first, last = tl.load(SeqChunks + seq), tl.load(SeqChunks + seq + 1)
    by = tl.full((C, G), 0, tl.float64)
    for step in range(last - first):
        chunk = last - 1 - step
        ch, block = chunk * H + head, (chunk * H + head) * NG + group
        start, end = tl.load(Starts + chunk), tl.load(Ends + chunk)
        length = end - start
        vo = ((start + i.to(tl.int64))[:, None] * H + head) * VD + group * GS + g[None, :].to(tl.int64)
        mask = (i[:, None] < length) & (g[None, :] < GS)
        v = tl.load(V0 + vo, mask, 0).to(tl.float64)
        go = tl.load(DO + vo, mask, 0).to(tl.float64)
        s = load_matrix(SavedS, block, K, G)
        y = load_matrix(SavedY, block, C, G)
        output, sn, h, w, u, r, t, zp, ro, target, ss, us, rs, ts = local_graph(
            P, Gp, ch, block, v, s, y, length, C, K, G, MIXED)
        qer = load_matrix(P[7], ch, C, K)
        to = load_matrix(P[13], ch, C, C)
        past = load_matrix(P[17], ch, K, C)
        k = load_matrix(P[1], ch, C, K)
        decay = load_matrix(P[2], ch, C, K)
        lower = load_matrix(P[10], ch, C, C)
        inv = load_matrix(P[11], ch, C, C)
        f = load_matrix(P[5], ch, C, K)
        n = load_vector(P[18], ch, C)
        gamma = load_vector(P[19], ch, C)
        aq = load_vector(P[20], ch, C)
        beta = load_vector(P[21], ch, C)
        rho = load_matrix(Gp[2], block, C, C)
        nu = load_vector(Gp[7], block, C)
        lam = load_vector(Gp[6], block, C)
        z = r + t
        norm = tl.sum(go * (ro / z[:, None]), 1)
        dot = tl.sum(go * target, 1)
        gro, gto = go / z[:, None], (t / z)[:, None] * go
        gr, gt = -(norm + (t / z) * dot) / z, ((r / z) * dot - norm) / z
        gs = mm(tl.trans(qer), nu[:, None] * gro, K, G, C)
        gqer = mm(nu[:, None] * gro, tl.trans(s), C, K, G)
        gu = mm(tl.trans(to * rho), gro, C, G, C)
        g_to = tl.where(i[:, None] > i[None, :], rho * mm(gro, tl.trans(u), C, C, G), 0.)
        gaq = tl.sum(gto * v, 1)
        gv = aq[:, None] * gto
        rl, tt, zz = at(r, length - 1, C), at(t, length - 1, C), at(z, length - 1, C)
        adj = bs
        carry_gu = tl.full((C, G), 0, tl.float64)
        norm_y, dot_y = tl.full((), 0, tl.float64), tl.full((), 0, tl.float64)
        gfnext = tl.full((C, K), 0, tl.float64)
        if CARRY and chunk + 1 < last:
            e = load_matrix(Carry[0], ch, C, K)
            hc = load_matrix(Carry[1], ch, C, C)
            fn = load_matrix(P[5], ch + H, C, K)
            klc = row(k, length - 1, C, K)
            nlc = at(n, length - 1, C)
            ry = mm_state_product(e, ss, C, G, K, MIXED) + mm_state_product(hc, us, C, G, C, MIXED)
            ty = (tl.sum(fn * klc[None, :], 1) / nlc)[:, None] * row(v, length - 1, C, G)[None, :]
            adj += mm_state_product(tl.trans(fn), by, K, G, C, MIXED)
            norm_y = tl.sum(tl.sum(by * (ry / zz), 0), 0)
            dot_y = tl.sum(tl.sum(by * ty, 0), 0)
            carry_gu = mm_state_product(tl.trans(hc), by, C, G, C, MIXED)
            gfnext = mm_state_product(by, tl.trans(sn), C, K, G, MIXED)
        norm_s = norm_y + tl.sum(tl.sum(bs * (rs / zz), 0), 0)
        dot_s = dot_y + tl.sum(tl.sum(bs * ts, 0), 0)
        gr += tl.where(i == length - 1, -(norm_s + (tt / zz) * dot_s) / zz, 0.)
        gt += tl.where(i == length - 1, ((rl / zz) * dot_s - norm_s) / zz, 0.)
        gu += tl.where(i[:, None] < length - 1,
                       row(rho, length - 1, C, C)[:, None] * (
                           mm_state_product(tl.trans(past), bs, C, G, K, MIXED) + carry_gu) / zz, 0.)
        residual_grad = adj / zz
        gp = mm_state_product(residual_grad, tl.trans(us), K, C, G, MIXED)
        kl = row(k, length - 1, C, K)
        tau = row(decay, length - 1, C, K)
        gl = at(gamma, length - 1, C)
        gx, gkl, ggl = erase_vjp(kl, gl, tau[:, None] * ss, residual_grad)
        gs += at(nu, length - 1, C) * tau[:, None] * gx
        if C == 1:
            gd = reduced_row(tl.sum(gx * ss, 1), K)
            gk = reduced_row(gkl, K)
        else:
            gd = tl.where(i[:, None] == length - 1, tl.sum(gx * ss, 1)[None, :], 0.)
            gk = tl.where(i[:, None] == length - 1, gkl[None, :], 0.)
        gg = tl.where(i == length - 1, ggl, 0.)
        gtarget = (tt / zz) * adj
        nl, vl = at(n, length - 1, C), row(v, length - 1, C, G)
        if C == 1:
            gk += reduced_row(tl.sum(gtarget * vl[None, :], 1) / nl, K)
        else:
            gk += tl.where(i[:, None] == length - 1, (tl.sum(gtarget * vl[None, :], 1) / nl)[None, :], 0.)
        gv += tl.where(i[:, None] == length - 1, (tl.sum(gtarget * kl[:, None], 0) / nl)[None, :], 0.)
        gn = tl.where(i == length - 1, -tl.sum(tl.sum(gtarget * ts, 0), 0) / nl, 0.)
        gb, gw = tl.full((C,), 0, tl.float64), tl.full((C,), 0, tl.float64)
        gz = tl.full((), 0, tl.float64)
        for jj in range(C):
            j = C - 1 - jj
            gri, gti = at(gr, j, C) + gz, at(gt, j, C) + gz
            gb += tl.where(i == j, -gri * at(zp, j, C) / at(lam, j, C), 0.)
            gz = gri * (1 - at(beta, j, C)) / at(lam, j, C)
            gn += tl.where(i == j, gti * at(w, j, C), 0.)
            gw += tl.where(i == j, gti * at(n, j, C), 0.)
        gw += tl.sum(gu * v, 1)
        gv += w[:, None] * gu
        scaled = nu[:, None] * gu
        glower = -mm_state_product(scaled, tl.trans(y), C, C, G, MIXED)
        gy = -mm_state_product(tl.trans(lower), scaled, C, G, C, MIXED)
        zi = load_matrix(Gp[0], block, C, C)
        b = tl.sum(tl.trans(zi) * gw[None, :], 1)
        gzmatrix = -rho * b[:, None] * w[None, :]
        rhs = nu * b
        gh = -kap * rhs
        grmat = -kap * gzmatrix
        rmat = load_matrix(Gp[4], block, C, C)
        gc = -tl.sum(rhs * h, 0) - tl.sum(tl.sum(gzmatrix * rmat, 0), 0)
        gv += gh[:, None] * y
        gy += gh[:, None] * v
        vv = load_matrix(Gp[3], block, C, C)
        ga = grmat * vv
        gram_grad = grmat * inv
        gv += mm_state_product(gram_grad + tl.trans(gram_grad), v, C, G, C, MIXED)
        gf = tl.full((C, K), 0, tl.float64)
        if not CARRY or chunk == first:
            gf = mm_state_product(gy, tl.trans(s), C, K, G, MIXED)
            gs += mm_state_product(tl.trans(f), gy, K, G, C, MIXED)
        by = gy
        store_matrix(Sh[14], block, gfnext, C, K)
        store_matrix(Sh[0], block, ga, C, C)
        store_matrix(Sh[1], block, glower, C, C)
        store_matrix(Sh[2], block, gf, C, K)
        store_matrix(Sh[3], block, gzmatrix, C, C)
        store_vector(Sh[4], block, rhs, C)
        store_matrix(Sh[5], block, gqer, C, K)
        store_matrix(Sh[6], block, g_to, C, C)
        store_matrix(Sh[7], block, gp, K, C)
        store_vector(Sh[8], block, gn, C)
        store_vector(Sh[9], block, gg, C)
        store_vector(Sh[10], block, gaq, C)
        store_matrix(Sh[11], block, gk, C, K)
        store_vector(Sh[12], block, gb, C)
        store_matrix(Sh[13], block, gd, C, K)
        tl.store(DV + vo, gv, mask)
        tl.store(DC + block, gc / GS)
        bs = gs
    tl.store(DS0 + so, bs, smask)


@triton.jit
def sum_matrix(p, ch, M: tl.constexpr, N: tl.constexpr, NG: tl.constexpr):
    result = tl.full((M, N), 0, tl.float64)
    for group in range(NG):
        result += load_matrix(p, ch * NG + group, M, N)
    return result


@triton.jit
def sum_vector(p, ch, N: tl.constexpr, NG: tl.constexpr):
    result = tl.full((N,), 0, tl.float64)
    for group in range(NG):
        result += load_vector(p, ch * NG + group, N)
    return result


@triton.jit
def common_bwd_kernel(Ell, DQ, DK, DB, DE, Starts, Ends, Prev, P, Sh,
                      H: tl.constexpr, KD: tl.constexpr, NG: tl.constexpr,
                      C: tl.constexpr, K: tl.constexpr, SCALE: tl.constexpr):
    ch = tl.program_id(0).to(tl.int64)
    head, chunk = ch % H, ch // H
    start, end = tl.load(Starts + chunk), tl.load(Ends + chunk)
    length = end - start
    i, a = tl.arange(0, C), tl.arange(0, K)
    off = ((start + i.to(tl.int64))[:, None] * H + head) * KD + a[None, :].to(tl.int64)
    valid = (i[:, None] < length) & (a[None, :] < KD)
    ell = tl.load(Ell + off, valid, 0).to(tl.float64)
    q, k = load_matrix(P[0], ch, C, K), load_matrix(P[1], ch, C, K)
    decay, ke = load_matrix(P[2], ch, C, K), load_matrix(P[3], ch, C, K)
    bmat, qr = load_matrix(P[4], ch, C, K), load_matrix(P[6], ch, C, K)
    tail = load_matrix(P[8], ch, C, K)
    gram, inv = load_matrix(P[9], ch, C, C), load_matrix(P[11], ch, C, C)
    cr = load_matrix(P[12], ch, C, C)
    products = load_matrix(P[15], ch, C, C)
    p0 = load_matrix(P[16], ch, K, C)
    n, gamma = load_vector(P[18], ch, C), load_vector(P[19], ch, C)
    aq, beta = load_vector(P[20], ch, C), load_vector(P[21], ch, C)
    ga = sum_matrix(Sh[0], ch, C, C, NG)
    gl = sum_matrix(Sh[1], ch, C, C, NG)
    gf = sum_matrix(Sh[2], ch, C, K, NG)
    prev = tl.load(Prev + chunk)
    if prev >= 0:
        gf += sum_matrix(Sh[14], prev * H + head, C, K, NG)
    gqs, ge = sum_matrix(Sh[3], ch, C, C, NG), sum_vector(Sh[4], ch, C, NG)
    gqer = sum_matrix(Sh[5], ch, C, K, NG)
    gto = sum_matrix(Sh[6], ch, C, C, NG)
    gp = sum_matrix(Sh[7], ch, K, C, NG)
    gn, gg = sum_vector(Sh[8], ch, C, NG), sum_vector(Sh[9], ch, C, NG)
    gaq = sum_vector(Sh[10], ch, C, NG)
    gk = sum_matrix(Sh[11], ch, C, K, NG)
    gb, gd = sum_vector(Sh[12], ch, C, NG), sum_matrix(Sh[13], ch, C, K, NG)
    gp0, gkl, ggl = erase_vjp(row(k, length - 1, C, K), at(gamma, length - 1, C), p0, gp)
    gk += tl.where(i[:, None] == length - 1, gkl[None, :], 0.)
    gg += tl.where(i == length - 1, ggl, 0.)
    pastmask = (i[:, None] < length - 1) & (i[None, :] < length - 1)
    gtail = mm(tl.where(pastmask, inv, 0.), tl.trans(gp0), C, K, C)
    ga += tl.where(pastmask, mm(tail, gp0, C, C, K), 0.)
    gcr = tl.where(i[:, None] > i[None, :], mm(gto, tl.trans(inv), C, C, C), 0.)
    ga += mm(tl.trans(cr), gto, C, C, C)
    gqr = gqer * decay
    gd += gqer * qr
    ga += mm(gf, tl.trans(bmat), C, C, K)
    g_bmat = mm(tl.trans(inv), gf, C, K, C)
    g_ke = gamma[:, None] * g_bmat
    gg += tl.sum(g_bmat * ke, 1)
    gk += decay * g_ke
    gd += k * g_ke
    gl = tl.where(i[:, None] > i[None, :],
                  gl - mm(mm(tl.trans(inv), ga, C, C, C), tl.trans(inv), C, C, C), 0.)
    gg += tl.sum(gl * gram, 1)
    ggram = gamma[:, None] * gl
    # reverse products without division by beta or 1-beta
    for r in range(C):
        br, ger = at(beta, r, C), at(ge, r, C)
        qrow, prow = row(gqs, r, C, C), row(products, r, C, C)
        final_product = at(tl.cumprod(tl.where(i < r, 1 - beta, 1.), 0), C - 1, C)
        gn += tl.where(i == r, at(qrow, r, C), 0.)
        gb += tl.where(i == r, final_product * ger, 0.)
        prod_grad = br * ger
        for j in range(C):
            active = j < r
            pr, qrj, nj = at(prow, j, C), at(qrow, j, C), at(n, j, C)
            gn += tl.where((i == j) & active, -br * pr * qrj, 0.)
            gb += tl.where((i == r) & active, -nj * pr * qrj, 0.)
            gb += tl.where((i == j) & active, -pr * prod_grad, 0.)
            prod_grad = tl.where(active, (1 - at(beta, j, C)) * prod_grad - br * nj * qrj, prod_grad)
    gate = tl.cumsum(gd * decay, 0, reverse=True)
    for j in range(C):
        interval = libdevice.exp(tl.cumsum(tl.where(i[:, None] > j, ell, 0.), 0))
        kj = row(k, j, C, K)
        gj = row(tl.trans(ggram), j, C, C)
        cj = row(tl.trans(gcr), j, C, C)
        left = gj[:, None] * k + cj[:, None] * qr
        gk += gj[:, None] * kj[None, :] * interval
        gqr += cj[:, None] * kj[None, :] * interval
        gk += tl.where(i[:, None] == j, tl.sum(left * interval, 0)[None, :], 0.)
        eg = left * kj[None, :] * interval
        gate += tl.where(i[:, None] > j, tl.cumsum(eg, 0, reverse=True), 0.)
        tailgrad = row(gtail, j, C, K) * row(interval, length - 1, C, K)
        gk += tl.where(i[:, None] == j, tailgrad[None, :], 0.)
        gate += tl.where(i[:, None] > j, (tailgrad * kj)[None, :], 0.)
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
    tl.store(DQ + off, gq * SCALE, valid)
    tl.store(DK + off, gk, valid)
    tl.store(DE + off, gate, valid)
    tl.store(DB + (start + i.to(tl.int64)) * H + head, gb, i < length)


@triton.jit
def carry_fwd_kernel(Ends, Starts, Next, P, Carry, H: tl.constexpr, C: tl.constexpr, K: tl.constexpr):
    ch = tl.program_id(0).to(tl.int64)
    chunk, head = ch // H, ch % H
    next_chunk = tl.load(Next + chunk)
    e = tl.full((C, K), 0, tl.float64)
    hc = tl.full((C, C), 0, tl.float64)
    if next_chunk >= 0:
        length = tl.load(Ends + chunk) - tl.load(Starts + chunk)
        fn = load_matrix(P[5], next_chunk * H + head, C, K)
        k = row(load_matrix(P[1], ch, C, K), length - 1, C, K)
        tau = row(load_matrix(P[2], ch, C, K), length - 1, C, K)
        gamma = at(load_vector(P[19], ch, C), length - 1, C)
        e = (fn - gamma * tl.sum(fn * k[None, :], 1)[:, None] * k[None, :]) * tau[None, :]
        hc = mm(fn, load_matrix(P[17], ch, K, C), C, C, K)
    store_matrix(Carry[0], ch, e, C, K)
    store_matrix(Carry[1], ch, hc, C, C)
