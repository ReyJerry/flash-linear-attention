"""Batched token preprocessing with FP32 parameter partials."""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def preprocessing_fwd(Q0, K0, V0, B0, G0, A, DT, Curve, Q, K, V, Beta, Ell, Norms,
                      N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, MARGIN: tl.constexpr, BT: tl.constexpr):
    block = tl.program_id(1).to(tl.int64)
    head = tl.program_id(0).to(tl.int64)
    t = block*BT+tl.arange(0, BT).to(tl.int64)
    i = tl.arange(0, D).to(tl.int64)
    row = t*H+head
    off = row[:, None]*D+i[None, :]
    valid = t[:, None] < N
    q = tl.load(Q0+off, valid, 0.).to(tl.float32)
    k = tl.load(K0+off, valid, 0.).to(tl.float32)
    v = tl.load(V0+off, valid, 0.).to(tl.float32)
    nq = libdevice.sqrt(tl.sum(q*q, 1))
    nk = libdevice.sqrt(tl.sum(k*k, 1))
    kn = tl.where(nk[:, None] > 1.e-6, k/tl.maximum(nk[:, None], 1.e-6), (i[None, :] == 0).to(tl.float32))
    unit = libdevice.sqrt(tl.sum(kn*kn, 1))
    c = tl.load(Curve).to(tl.float32)
    radius = MARGIN*libdevice.rsqrt(tl.maximum(c, 1.))
    tl.store(Q+off, q/tl.maximum(nq[:, None], 1.e-6), valid)
    tl.store(K+off, kn, valid)
    tl.store(V+off, libdevice.tanh(v)*radius*unit[:, None], valid)
    tl.store(Norms+row*3, nq, t < N)
    tl.store(Norms+row*3+1, nk, t < N)
    tl.store(Norms+row*3+2, unit, t < N)
    br = tl.load(B0+row, t < N, 0.).to(tl.float32)
    beta = 1./(1.+libdevice.exp(-br))
    tl.store(Beta+row, beta, t < N)
    raw = tl.load(G0+off, valid, 0.).to(tl.float32)+tl.load(DT+head*D+i)[None, :].to(tl.float32)
    sp = tl.where(raw > 20., raw, libdevice.log1p(libdevice.exp(raw)))
    scale = libdevice.exp(tl.load(A+head).to(tl.float32))
    tl.store(Ell+off, -scale*sp, valid)


@triton.jit
def preprocessing_bwd(Q0, K0, V0, G0, A, DT, Curve, Norms, Beta, Ell, GQ, GK, GV, GB, GE, DQ, DK, DV, DB, DG, DA, DDT, DC,
                      N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, MARGIN: tl.constexpr, NEED_C: tl.constexpr, BT: tl.constexpr):
    block = tl.program_id(1).to(tl.int64)
    head = tl.program_id(0).to(tl.int64)
    t = block*BT+tl.arange(0, BT).to(tl.int64)
    i = tl.arange(0, D).to(tl.int64)
    row = t*H+head
    off = row[:, None]*D+i[None, :]
    valid = t[:, None] < N
    q = tl.load(Q0+off, valid, 0.).to(tl.float32)
    k = tl.load(K0+off, valid, 0.).to(tl.float32)
    v = tl.load(V0+off, valid, 0.).to(tl.float32)
    nq = tl.load(Norms+row*3, t < N, 1.)
    nk = tl.load(Norms+row*3+1, t < N, 1.)
    unit = tl.load(Norms+row*3+2, t < N, 1.)
    qn = q/tl.maximum(nq[:, None], 1.e-6)
    kn = tl.where(nk[:, None] > 1.e-6, k/tl.maximum(nk[:, None], 1.e-6), (i[None, :] == 0).to(tl.float32))
    gq = tl.load(GQ+off, valid, 0.).to(tl.float32)
    gk = tl.load(GK+off, valid, 0.).to(tl.float32)
    gv = tl.load(GV+off, valid, 0.).to(tl.float32)
    dq = (gq-tl.where(nq[:, None] >= 1.e-6, qn*tl.sum(gq*qn, 1)[:, None], 0.))/tl.maximum(nq[:, None], 1.e-6)
    c = tl.load(Curve).to(tl.float32)
    cap = tl.maximum(c, 1.)
    radius = MARGIN*libdevice.rsqrt(cap)
    tv = libdevice.tanh(v)
    gu = tl.sum(gv*tv*radius, 1)
    gk += (gu/unit)[:, None]*kn
    dk = tl.where(nk[:, None] > 1.e-6, (gk-kn*tl.sum(gk*kn, 1)[:, None])/tl.maximum(nk[:, None], 1.e-6), 0.)
    dv = gv*radius*unit[:, None]*(1.-tv*tv)
    tl.store(DQ+off, dq, valid)
    tl.store(DK+off, dk, valid)
    tl.store(DV+off, dv, valid)
    beta = tl.load(Beta+row, t < N, 0.)
    tl.store(DB+row, tl.load(GB+row, t < N, 0.).to(tl.float32)*beta*(1.-beta), t < N)
    raw = tl.load(G0+off, valid, 0.).to(tl.float32)+tl.load(DT+head*D+i)[None, :].to(tl.float32)
    derivative = tl.where(raw > 20., 1., 1./(1.+libdevice.exp(-raw)))
    ge = tl.load(GE+off, valid, 0.).to(tl.float32)
    scale = libdevice.exp(tl.load(A+head).to(tl.float32))
    dg = -ge*scale*derivative
    tl.store(DG+off, dg, valid)
    partial = (block*H+head)*D+i
    tl.store(DDT+partial, tl.sum(dg, 0))
    da = tl.sum(ge*tl.load(Ell+off, valid, 0.).to(tl.float32), 1)
    tl.store(DA+block*H+head, tl.sum(da, 0))
    if NEED_C:
        slope = tl.where(c > 1., 1., tl.where(c == 1., .5, 0.))
        dc = tl.sum(gv*tv*unit[:, None], 1)*(-.5*radius/cap)*slope
        tl.store(DC+block*H+head, tl.sum(dc, 0))
