# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Layer Q/K normalization, value geometry, beta and key gates with their VJP."""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def preprocessing_fwd(Q0, K0, V0, B0, G0, A, DT, Curve, Q, K, V, Beta, Ell, Norms,
                      H: tl.constexpr, D: tl.constexpr, MARGIN: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    i = tl.arange(0, D).to(tl.int64)
    q, k, v = tl.load(Q0+row*D+i).to(tl.float32), tl.load(K0+row*D+i).to(tl.float32), tl.load(V0+row*D+i).to(tl.float32)
    nq = libdevice.sqrt(tl.sum(q*q, 0))
    nk = libdevice.sqrt(tl.sum(k*k, 0))
    kn = tl.where(nk > 1.e-6, k/tl.maximum(nk, 1.e-6), (i == 0).to(tl.float32))
    unit = libdevice.sqrt(tl.sum(kn*kn, 0))
    c = tl.load(Curve).to(tl.float32)
    radius = MARGIN*libdevice.rsqrt(tl.maximum(c, 1.))
    tl.store(Q+row*D+i, q/tl.maximum(nq, 1.e-6))
    tl.store(K+row*D+i, kn)
    tl.store(V+row*D+i, libdevice.tanh(v)*radius*unit)
    tl.store(Norms+row*3, nq)
    tl.store(Norms+row*3+1, nk)
    tl.store(Norms+row*3+2, unit)
    br = tl.load(B0+row).to(tl.float32)
    beta = 1./(1.+libdevice.exp(-br))
    tl.store(Beta+row, beta)
    head = row % H
    raw = tl.load(G0+row*D+i).to(tl.float32)+tl.load(DT+head*D+i).to(tl.float32)
    sp = tl.where(raw > 20., raw, libdevice.log1p(libdevice.exp(raw)))
    scale = libdevice.exp(tl.load(A+head).to(tl.float32))
    tl.store(Ell+row*D+i, -scale*sp)


@triton.jit
def preprocessing_bwd(Q0, K0, V0, G0, A, DT, Curve, Norms, Beta, Ell,
                      GQ, GK, GV, GB, GE, DQ, DK, DV, DB, DG, DA, DC,
                      H: tl.constexpr, D: tl.constexpr, MARGIN: tl.constexpr, NEED_C: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    i = tl.arange(0, D).to(tl.int64)
    q, k, v = tl.load(Q0+row*D+i).to(tl.float32), tl.load(K0+row*D+i).to(tl.float32), tl.load(V0+row*D+i).to(tl.float32)
    nq, nk, unit = tl.load(Norms+row*3), tl.load(Norms+row*3+1), tl.load(Norms+row*3+2)
    qn = q/tl.maximum(nq, 1.e-6)
    kn = tl.where(nk > 1.e-6, k/tl.maximum(nk, 1.e-6), (i == 0).to(tl.float32))
    gq, gk, gv = tl.load(GQ+row*D+i).to(tl.float32), tl.load(GK+row*D+i).to(tl.float32), tl.load(GV+row*D+i).to(tl.float32)
    dq = (gq-tl.where(nq >= 1.e-6, qn*tl.sum(gq*qn, 0), 0.))/tl.maximum(nq, 1.e-6)
    c = tl.load(Curve).to(tl.float32)
    cap = tl.maximum(c, 1.)
    radius = MARGIN*libdevice.rsqrt(cap)
    tv = libdevice.tanh(v)
    gu = tl.sum(gv*tv*radius, 0)
    gk += (gu/unit)*kn
    dk = tl.where(nk > 1.e-6, (gk-kn*tl.sum(gk*kn, 0))/tl.maximum(nk, 1.e-6), 0.)
    dv = gv*radius*unit*(1.-tv*tv)
    tl.store(DQ+row*D+i, dq)
    tl.store(DK+row*D+i, dk)
    tl.store(DV+row*D+i, dv)
    beta = tl.load(Beta+row)
    tl.store(DB+row, tl.load(GB+row).to(tl.float32)*beta*(1.-beta))
    head = row % H
    raw = tl.load(G0+row*D+i).to(tl.float32)+tl.load(DT+head*D+i).to(tl.float32)
    derivative = tl.where(raw > 20., 1., 1./(1.+libdevice.exp(-raw)))
    ge = tl.load(GE+row*D+i).to(tl.float32)
    scale = libdevice.exp(tl.load(A+head).to(tl.float32))
    tl.store(DG+row*D+i, -ge*scale*derivative)
    tl.store(DA+row, tl.sum(ge*tl.load(Ell+row*D+i), 0))
    if NEED_C:
        slope = tl.where(c > 1., 1., tl.where(c == 1., 0.5, 0.))
        dc = tl.sum(gv*tv*unit, 0)*(-0.5*radius/cap)*slope
        tl.store(DC+row, dc)
