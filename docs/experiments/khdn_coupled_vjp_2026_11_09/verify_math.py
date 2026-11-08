# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Untimed CPU FP64 proof of the coupled source VJP, independent of GPU code."""
import json
from pathlib import Path

import torch

torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)


def expanded(l, bmat, qs, eta, v, s, kap, nu):
    a = torch.linalg.inv(l)
    rho = nu[:, None] / nu[None, :]
    y = a @ bmat @ s
    z = (qs - kap * a * (v @ v.T)) * rho
    w = torch.linalg.solve(z, nu * (eta - kap * (v * y).sum(-1)))
    u = w[:, None] * v - nu[:, None] * (l @ y)
    return (a * rho) @ u, w


class Coupled(torch.autograd.Function):
    @staticmethod
    def forward(ctx, l, bmat, qs, eta, v, s, kap, nu):
        delta, w = expanded(l, bmat, qs, eta, v, s, kap, nu)
        a = torch.linalg.inv(l)
        rho = nu[:, None] / nu[None, :]
        z = (qs - kap * a * (v @ v.T)) * rho
        ctx.save_for_backward(l, bmat, v, s, kap, nu, a * rho, z, delta, w)
        return delta, w

    @staticmethod
    def backward(ctx, gd, gw):
        l, bmat, v, s, kap, nu, ahat, z, delta, w = ctx.saved_tensors
        gu = ahat.T @ gd
        b = torch.linalg.solve(z.T, gw + (gu * v).sum(-1))
        pi = gu + kap * (ahat.T @ (b[:, None] * v))
        rho = nu[:, None] / nu[None, :]
        return (
            -rho * (pi @ delta.T),
            -(nu[:, None] * pi) @ s.T,
            -rho * (b[:, None] * w[None, :]),
            nu * b,
            w[:, None] * pi + kap * b[:, None] * delta,
            -bmat.T @ (nu[:, None] * pi),
            (b[:, None] * v * delta).sum(),
            None,
        )


def preparation(q, k, v, beta, ell, s, c):
    t, d = k.shape
    n = k.square().sum(-1)
    gamma = beta / n
    prefix = ell.cumsum(0).exp()
    gram, readout, qs, eta = [], [], [], []
    for i in range(t):
        gr, cr, qr = [], [], []
        for j in range(t):
            factor = ell[j + 1:i + 1].sum(0).exp()
            gr.append((k[i] * k[j] * factor).sum() if j < i else k.new_zeros(()))
            cr.append((q[i] * k[j] * factor).sum() if j <= i else k.new_zeros(()))
            qr.append(n[i] if i == j else -beta[i] * n[j] * (1-beta[j+1:i]).prod()
                      if j < i else k.new_zeros(()))
        gram.append(torch.stack(gr))
        readout.append(torch.stack(cr))
        qs.append(torch.stack(qr))
        eta.append(beta[i] * (1-beta[:i]).prod())
    l = torch.eye(t) + gamma[:, None] * torch.stack(gram)
    bmat = gamma[:, None] * k * prefix
    return l, bmat, torch.stack(qs), torch.stack(eta), torch.stack(readout), prefix, n


def block(q, k, v, beta, ell, s, c, manual):
    l, bmat, qs, eta, cr, prefix, n = preparation(q, k, v, beta, ell, s, c)
    nu = torch.ones_like(beta)
    fn = Coupled.apply if manual else expanded
    delta, w = fn(l, bmat, qs, eta, v, s, c / v.shape[-1], nu)
    mass, states = k.new_ones(()), []
    for i in range(len(beta)):
        mass = (1-beta[i]) * mass + n[i] * w[i]
        numer = prefix[i, :, None] * s
        for j in range(i+1):
            numer = numer + (k[j] * ell[j+1:i+1].sum(0).exp())[:, None] * delta[j]
        states.append(numer / mass)
    states = torch.stack(states)
    return (q[:, :, None] * states).sum(1), states[-1]


def canonical(q, k, v, beta, ell, s, c):
    out = []
    kap = c / v.shape[-1]
    for qi, ki, vi, bi, ei in zip(q, k, v, beta, ell):
        x = ei.exp()[:, None] * s
        matrix = ki.square().sum() * torch.eye(len(vi)) - kap * vi[:, None] * vi[None, :]
        u = bi * torch.linalg.solve(matrix, vi - x.T @ ki)
        s = (x + ki[:, None] * u[None, :]) / (1 + kap * (vi * u).sum())
        out.append(qi @ s)
    return torch.stack(out), s


def run():
    torch.manual_seed(42)
    rows = []
    for t, d, g in [(1, 3, 2), (4, 5, 3), (7, 8, 4), (17, 128, 32)]:
        for zero in [False, True]:
            q = torch.randn(t, d) * .1
            k = torch.nn.functional.normalize(torch.randn(t, d), dim=-1)
            v = torch.randn(t, g) * .05
            beta = torch.rand(t) * .8
            if t > 1:
                beta[0], beta[-1] = 0, 1
            source = [q, k, v, beta, -torch.rand(t, d) * .08,
                      torch.randn(d, g) * .005, torch.tensor(0. if zero else .7)]
            source = [x.requires_grad_() for x in source]
            cot = (torch.randn(t, g), torch.randn(d, g))
            reference = canonical(*source)
            refgrad = torch.autograd.grad(reference, source, cot)
            for manual in [False, True]:
                actual = block(*source, manual)
                grad = torch.autograd.grad(actual, source, cot)
                error = max(float((a-b).abs().max()) for a, b in zip((*actual, *grad), (*reference, *refgrad)))
                assert error < 1e-10, (t, d, g, zero, manual, error)
                rows.append(dict(t=t, k=d, g=g, c0=zero, manual=manual, max_abs=error))
    # Gauge is constant in this local source comparison; its derivative cancels in the full graph.
    for t in [4, 16, 64]:
        l = torch.eye(t) + torch.randn(t, t).tril(-1) * .015
        source = [l, torch.randn(t, 8) * .03, torch.eye(t) + torch.randn(t, t).tril(-1) * .02,
                  torch.rand(t) * .2, torch.randn(t, 4) * .05, torch.randn(8, 4) * .005, torch.tensor(.2)]
        source = [x.requires_grad_() for x in source]
        nu = (-torch.arange(t) * .01).exp()
        cot = (torch.randn(t, 4), torch.randn(t))
        actual, reference = Coupled.apply(*source, nu), expanded(*source, nu)
        gradients = torch.autograd.grad(actual, source, cot)
        targets = torch.autograd.grad(reference, source, cot)
        # L and Qs contain structural constants; compare their active triangles only.
        errors = [(a-b).tril(-1).abs().max() if i == 0 else
                  (a-b).tril().abs().max() if i == 2 else (a-b).abs().max()
                  for i, (a, b) in enumerate(zip(gradients, targets))]
        error = max(map(float, errors))
        assert error < 1e-10, (t, error)
        rows.append(dict(local_gauge_t=t, max_abs=error))
    result = dict(passed=True, cases=len(rows), max_abs=max(r['max_abs'] for r in rows), rows=rows,
                  scope='Untimed independent CPU FP64 algebra; no BF16/GPU qualification.')
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}))


if __name__ == '__main__':
    run()
