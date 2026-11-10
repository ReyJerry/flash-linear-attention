# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Untimed FP64 verification of fused C64 preparation-source ownership."""
import json
from pathlib import Path

import torch

torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)


def scanback(x):
    return x.flip(0).cumsum(0).flip(0)


def diag(k, q, e, gg, gc, safe):
    if safe:
        rel = e.clone()
        rel[0] = 0
        rel = rel.cumsum(0)
        d = (rel - 0.5 * rel[-1]).exp()
        inv = k / d
        rk = gg @ inv * d
        rq = gc @ inv * d
        ck = (gg.T @ (k * d) + gc.T @ (q * d)) / d
        return (rk + ck, rq, k * rk + q * rq - k * ck)
    (dk, dq, inc) = (torch.zeros_like(k), torch.zeros_like(q), torch.zeros_like(k))
    for j in range(32):
        it = (e * (torch.arange(32) > j)[:, None]).cumsum(0).exp()
        left = gg[:, j, None] * k + gc[:, j, None] * q
        dk = dk + gg[:, j, None] * k[j] * it
        dq = dq + gc[:, j, None] * k[j] * it
        sel = (torch.arange(32) == j)[:, None]
        dk = dk + sel * (left * it).sum(0)
        w = left * k[j] * it
        inc = inc + w - sel * w.sum(0)
    return (dk, dq, inc)


rows = []
torch.manual_seed(42)
for (length, strong) in [(2, False), (31, False), (32, False), (33, False), (63, False), (64, False), (2, True)]:
    (C, K) = (64, 128)
    mask = (torch.arange(C) < length)[:, None]
    k = (torch.nn.functional.normalize(torch.randn(C, K), dim=-1) * mask).requires_grad_()
    q = (0.1 * torch.randn(C, K) * mask).requires_grad_()
    e = (-0.02 * torch.rand(C, K) * mask).requires_grad_()
    gamma = torch.rand(C, requires_grad=True)
    if strong:
        with torch.no_grad():
            e[0] = -100000000.0
            e[1] = -1
    gL = torch.randn(C, C).tril(-1)
    gL[length:] = 0
    gC = torch.randn(C, C).tril(-1)
    gC[length:] = 0
    gB = torch.randn(C, K) * mask
    gQ = torch.randn(C, K) * mask
    gP = torch.randn(C, K) * (torch.arange(C) < length - 1)[:, None]
    gkl = torch.randn(K)
    gtau = torch.randn(K)
    gram = []
    cr = []
    for i in range(C):
        gram.append(torch.stack([(k[i] * k[j] * e[j + 1:i + 1].sum(0).exp()).sum()
                    if j < i else k.new_zeros(()) for j in range(C)]))
        cr.append(torch.stack([(q[i] * k[j] * e[j + 1:i + 1].sum(0).exp()).sum()
                  if j < i else k.new_zeros(()) for j in range(C)]))
    decay = e.cumsum(0).exp()
    tail = torch.stack([e[i + 1:].sum(0).exp() for i in range(C)])
    loss = (gL * torch.stack(gram) + gC * torch.stack(cr)).sum() + (gB * gamma[:, None] * k * decay).sum() + (
        gQ * q * decay).sum() + (gP * k * tail).sum() + (gkl * k[length - 1]).sum() + (gtau * decay[length - 1]).sum()
    truth = torch.autograd.grad(loss, (k, q, e, gamma))
    with torch.no_grad():
        es = e.reshape(2, 32, K)
        pref = es.cumsum(1).exp()
        suf = torch.cat([es[:, 1:].flip(1).cumsum(1).flip(1).exp(), torch.ones(2, 1, K)], 1)
        (gg, gc) = (gL[32:, :32], gC[32:, :32])
        rk = gg @ (k[:32] * suf[0]) * pref[1]
        rq = gc @ (k[:32] * suf[0]) * pref[1]
        ck = (gg.T @ (k[32:] * pref[1]) + gc.T @ (q[32:] * pref[1])) * suf[0]
        pieces = []
        for j in range(2):
            rel = es[j].clone()
            rel[0] = 0
            safe = bool(rel.cumsum(0).min() >= -64)
            pieces.append(diag(k[j * 32:(j + 1) * 32], q[j * 32:(j + 1) * 32], es[j], gL[j * 32:(j + 1)
                          * 32, j * 32:(j + 1) * 32], gC[j * 32:(j + 1) * 32, j * 32:(j + 1) * 32], safe))
        dk = torch.cat([pieces[0][0] + ck, pieces[1][0] + rk])
        dq = torch.cat([pieces[0][1], pieces[1][1] + rq])
        inc = torch.cat([pieces[0][2] - k[:32] * ck, pieces[1][2] + k[32:] * rk + q[32:] * rq])
        tailfactor = torch.cat([suf[0] * pref[1, -1], suf[1]])
        tailgrad = gP * tailfactor
        dk = dk + tailgrad + decay * (gamma[:, None] * gB)
        dk[length - 1] += gkl
        dq = dq + gQ * decay
        gd = gQ * q + k * (gamma[:, None] * gB)
        gd[length - 1] += gtau
        de = scanback(inc + gd * decay) + torch.cat([torch.zeros(1, K), (tailgrad * k).cumsum(0)[:-1]])
        dg = (gB * (decay * k)).sum(-1)
        actual = (dk, dq, de, dg)
        error = max(float((a - b).abs().max()) for (a, b) in zip(actual, truth))
        assert error < 1e-10, (length, strong, error)
        rows.append(dict(length=length, strong=strong, max_abs=error))
result = dict(passed=True, cases=len(rows), max_abs=max(
    r['max_abs'] for r in rows), rows=rows, scope='Untimed CPU FP64 layout proof, not GPU qualification.')
Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2))
print({k: v for (k, v) in result.items() if k != 'rows'})
