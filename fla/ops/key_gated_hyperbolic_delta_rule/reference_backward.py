# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Complete first-order source VJP of the stable CPU KHDN chunk graph.

No autograd differentiation is used inside backward. Shared preparation is reversed once
after group reduction; carried S/Y edges and next-F ownership remain distinct.
"""

import math

import torch

from fla.ops.key_gated_hyperbolic_delta_rule.reference_arithmetic import (
    MPArray,
    arithmetic,
    extended,
    like,
    precision_context,
    tensor,
)
from fla.ops.key_gated_hyperbolic_delta_rule.reference_chunk import ReferenceRangeError, _arguments, _execute


def _double_tree(value, context=None):
    if isinstance(value, (torch.Tensor, MPArray)):
        return extended(value, context) if context is not None else value.double() if isinstance(value, torch.Tensor) else value
    if isinstance(value, dict):
        return {key: _double_tree(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [_double_tree(item, context) for item in value]
    if isinstance(value, tuple):
        return tuple(_double_tree(item, context) for item in value)
    return value


def _mass_vjp(residual, target, r, t, cotangent):
    """Normalize before contracting, retaining both endpoint mass derivatives."""
    z = r + t
    if z.ndim:
        rz = residual / z[:, None]
        norm = (cotangent * rz).sum(-1)
        dot = (cotangent * target).sum(-1)
        gr = cotangent / z[:, None]
        gt = (t / z)[:, None] * cotangent
    else:
        norm = (cotangent * (residual / z)).sum()
        dot = (cotangent * target).sum()
        gr, gt = cotangent / z, (t / z) * cotangent
    return gr, gt, -(norm + (t / z) * dot) / z, ((r / z) * dot - norm) / z


def _erase_vjp(k, gamma, x, cotangent):
    p, h = k @ x, k @ cotangent
    gx = cotangent - gamma * k[:, None] * h[None]
    gk = -gamma * (cotangent @ p + x @ h)
    return gx, gk, -(p * h).sum()


def _shared_zeros(p):
    ops = arithmetic(p['k'])
    keys = ('A', 'L', 'F', 'Qs', 'eta', 'Qer', 'To', 'P', 'n', 'gamma', 'aq', 'k', 'beta', 'decay')
    return {key: ops.zeros_like(p[key]) for key in keys}


def _reverse_group(p, a, go, bs, by, fn, hcarry, shared):
    ops = arithmetic(p['k'])
    v, s, y, w, u = (a[key] for key in ('v', 'S', 'Y', 'w', 'U'))
    nu, rho, lam = (a[key] for key in ('nu', 'rho', 'lam'))
    gr_o, gt_o, gr, gt = _mass_vjp(a['RO'], a['TO'], a['r'], a['t'], go)
    gs = p['Qer'].T @ (nu[:, None] * gr_o)
    shared['Qer'] += (nu[:, None] * gr_o) @ s.T
    gu = (p['To'] * rho).T @ gr_o
    shared['To'] += ops.tril(rho * (gr_o @ u.T), diagonal=-1)
    shared['aq'] += (gt_o * v).sum(-1)
    gv = p['aq'][:, None] * gt_o

    r, t = a['r'][-1], a['t'][-1]
    z = r + t
    adj = bs + fn.T @ by
    normalized = (bs * (a['RS'] / z)).sum() + (by * (a['RY'] / z)).sum()
    target_dot = (bs * a['TS']).sum() + (by * a['TY']).sum()
    gr[-1] += -(normalized + (t / z) * target_dot) / z
    gt[-1] += ((r / z) * target_dot - normalized) / z
    gu[:-1] += rho[-1, :-1, None] * ((p['P'].T @ bs + hcarry.T @ by) / z)
    residual_grad = adj / z
    shared['P'] += residual_grad @ a['Us'].T
    k, gamma, tau = p['k'][-1], p['gamma'][-1], p['decay'][-1]
    gx, gk, ggamma = _erase_vjp(k, gamma, tau[:, None] * a['Ss'], residual_grad)
    gs += nu[-1] * tau[:, None] * gx
    shared['decay'][-1] += (gx * a['Ss']).sum(-1)
    shared['k'][-1] += gk
    shared['gamma'][-1] += ggamma
    gtarget = (t / z) * adj
    shared['k'][-1] += (gtarget @ v[-1]) / p['n'][-1]
    gv[-1] += (gtarget.T @ k) / p['n'][-1]
    shared['n'][-1] -= (gtarget * a['TS']).sum() / p['n'][-1]
    gfnext = by @ a['Sout'].T

    gw = ops.zeros_like(w)
    gz = w.new_zeros(())
    for i in reversed(range(len(w))):
        gri, gti = gr[i] + gz, gt[i] + gz
        shared['beta'][i] -= gri * a['zprev'][i] / lam[i]
        gz = gri * (1 - p['beta'][i]) / lam[i]
        shared['n'][i] += gti * w[i]
        gw[i] += gti * p['n'][i]
    gw += (gu * v).sum(-1)
    gv += w[:, None] * gu
    scaled_gu = nu[:, None] * gu
    shared['L'] -= scaled_gu @ y.T
    gy = -p['L'].T @ scaled_gu

    b = ops.linalg.solve_triangular(a['Z'].T, gw[:, None], upper=True)[:, 0]
    gzmat = -rho * ops.tril(ops.outer(b, w))
    rhs_grad = nu * b
    shared['eta'] += rhs_grad
    shared['Qs'] += gzmat
    gh = -a['kappa'] * rhs_grad
    grmat = -a['kappa'] * gzmat
    gkappa = -(rhs_grad * a['h']).sum() - (gzmat * a['R']).sum()
    gv += gh[:, None] * y
    gy += gh[:, None] * v
    shared['A'] += grmat * a['VV']
    gram_grad = grmat * p['A']
    gv += (gram_grad + gram_grad.T) @ v
    return gs, gy, gv, gkappa, gfnext


def _coefficients_vjp(beta, n, products, gq, ge):
    ops = arithmetic(n)
    gb, gn = ops.zeros_like(beta), ops.zeros_like(n)
    for i, (row, final) in enumerate(products):
        gn[i] += gq[i, i]
        gb[i] += final * ge[i]
        gp = beta[i] * ge[i]
        for j in range(i):
            gn[j] -= beta[i] * row[j] * gq[i, j]
            gb[i] -= n[j] * row[j] * gq[i, j]
            gb[j] -= row[j] * gp
            gp = (1 - beta[j]) * gp - beta[i] * n[j] * gq[i, j]
    return gb, gn


def _reverse_common(p, shared):
    ops = arithmetic(p['k'])
    k, q, beta, n = (p[key] for key in ('k', 'q', 'beta', 'n'))
    gk, gb, gn, ggamma = (shared[key] for key in ('k', 'beta', 'n', 'gamma'))
    gdecay = shared['decay']
    ga = shared['A']
    gp0, gk_last, ggamma_last = _erase_vjp(k[-1], p['gamma'][-1], p['P0'], shared['P'])
    gk[-1] += gk_last
    ggamma[-1] += ggamma_last
    gtail = ops.zeros_like(p['tail'])
    gtail[:-1] += p['A'][:-1, :-1] @ gp0.T
    ga[:-1, :-1] += p['tail'][:-1] @ gp0
    gcr = ops.tril(shared['To'] @ p['A'].T, diagonal=-1)
    ga += p['Cr'].T @ shared['To']
    gqr = shared['Qer'] * p['decay']
    gdecay += shared['Qer'] * p['qr']

    ga += shared['F'] @ p['B'].T
    g_bmat = p['A'].T @ shared['F']
    g_ke = p['gamma'][:, None] * g_bmat
    ggamma += (g_bmat * p['Ke']).sum(-1)
    gk += p['decay'] * g_ke
    gdecay += k * g_ke
    # one explicit inverse VJP per head/chunk, after all Value groups and F owners have contributed
    gl = ops.tril(shared['L'] - p['A'].T @ ga @ p['A'].T, diagonal=-1)
    ggamma += (gl * p['GK']).sum(-1)
    ggram = p['gamma'][:, None] * gl
    beta_q, n_q = _coefficients_vjp(beta, n, p['products'], shared['Qs'], shared['eta'])
    gb += beta_q
    gn += n_q

    gate = ops.zeros_like(p['ell'])
    for i in range(len(k)):
        for j in range(i):
            decay = p['rel'][i, j]
            left = ggram[i, j] * k[i] + gcr[i, j] * p['qr'][i]
            gk[i] += ggram[i, j] * k[j] * decay
            gqr[i] += gcr[i, j] * k[j] * decay
            gk[j] += left * decay
            gate[j + 1:i + 1] += left * k[j] * decay
        gk[i] += gtail[i] * p['rel'][-1, i]
        gate[i + 1:] += gtail[i] * k[i] * p['rel'][-1, i]
        gate[:i + 1] += gdecay[i] * p['decay'][i]

    gaq = shared['aq'] - beta * (gqr * k).sum(-1)
    gq = gqr + (gaq / n)[:, None] * k
    gb -= p['aq'] * (gqr * k).sum(-1)
    gk -= (beta * p['aq'])[:, None] * gqr
    gk += (gaq / n)[:, None] * q
    gn -= gaq * p['aq'] / n
    gb += ggamma / n
    gn -= ggamma * p['gamma'] / n
    gk += 2 * gn[:, None] * k
    return gq, gk, gb, gate


def _backward(tapes, sources, options, go, gs, telemetry, needs_input_grad=None):
    B, T, H, K = sources[0].shape
    V, g = sources[2].shape[-1], options['group_size']
    # flattened accumulators must be views even when the source has noncontiguous strides
    grads = [torch.zeros_like(x, dtype=torch.float64, memory_format=torch.contiguous_format) for x in sources]
    contexts = [b['p']['k'].ctx for t in tapes for b in t['blocks'] if isinstance(b['p']['k'], MPArray)]
    go = go.double().reshape(B * T, H, V)
    gs = gs.double()
    if not torch.isfinite(go).all().item() or not torch.isfinite(gs).all().item():
        raise ReferenceRangeError('Nonfinite incoming cotangent.')
    max_cot = max(go.abs().max().item() if go.numel() else 0, gs.abs().max().item())
    shift = max(0, math.ceil(math.log2(max_cot)) - 32) if max_cot else 0
    cot_scale = math.ldexp(1.0, shift)
    # scaling must not erase a small component before it meets a large saved coefficient
    scaled_subnormal = any(((x != 0) & ((x / cot_scale).abs() < torch.finfo(torch.float64).tiny)).any().item()
                           for x in (go, gs))
    mp_context = None
    if contexts or scaled_subnormal:
        mp_context = precision_context((*sources, go, gs))
        mp_context.prec = max(mp_context.prec, *(ctx.prec for ctx in contexts)) if contexts else mp_context.prec
        grads = [extended(x, mp_context) for x in grads]
        go, gs = extended(go, mp_context), extended(gs, mp_context)
    flat_grads = [x.reshape(B * T, *x.shape[2:]) for x in grads[:5]]
    go, gs = go / cot_scale, gs / cot_scale
    direct_count = shared_count = 0
    owners = []
    for original in tapes:
        seq, head, bos = (original[key] for key in ('sequence', 'head', 'bos'))
        blocks = _double_tree(original['blocks'], mp_context)
        shared = [_shared_zeros(block['p']) for block in blocks]
        bs = gs[seq, head]
        by = None
        for index in reversed(range(len(blocks))):
            block = blocks[index]
            p, parts = block['p'], block['groups']
            gin = shared[index]
            bs = like(bs, p['k'])
            state_grads, y_grads = [], []
            for j, a in enumerate(parts):
                sl = slice(j*g, (j+1)*g)
                gout = go[bos + block['start']:bos + block['stop'], head, sl]
                gout = like(gout, p['k'])
                yout = arithmetic(p['k']).zeros_like(a['RY']) if by is None else like(by[j], p['k'])
                gstate, gy, gv, gc, gfnext = _reverse_group(p, a, gout, bs[:, sl], yout,
                                                            block['Fnext'], block['H'], gin)
                if block['outgoing']:
                    shared[index + 1]['F'] += gfnext
                if block['direct']:
                    gin['F'] += gy @ a['S'].T
                    gstate += p['F'].T @ gy
                state_grads.append(gstate)
                y_grads.append(gy)
                flat_grads[2][bos + block['start']:bos + block['stop'], head, sl] += like(gv, grads[2])
                grads[6] += like(gc / g, grads[6])
            direct_count += int(block['direct'])
            if block['outgoing']:
                owners.append((seq, head, index, index + 1))
            bs = arithmetic(p['k']).cat(state_grads, -1)
            by = None if block['direct'] else y_grads
        grads[5][seq, head] += like(bs, grads[5])
        for block, gin in zip(blocks, shared):
            qg, kg, bg, eg = _reverse_common(block['p'], gin)
            sl = slice(bos + block['start'], bos + block['stop'])
            for dest, value in zip((0, 1, 3, 4), (qg * options['scale'], kg, bg, eg)):
                flat_grads[dest][sl, head] += like(value, grads[dest])
            shared_count += 1
    results = []
    for index, (source, grad) in enumerate(zip(sources, grads)):
        if needs_input_grad is not None and not needs_input_grad[index]:
            results.append(None)
            continue
        grad = tensor(grad * cot_scale, source.dtype)
        if not torch.isfinite(grad).all().item():
            raise ReferenceRangeError('A source VJP is not representable in its public source dtype.')
        results.append(grad)
    if telemetry is not None:
        telemetry.update(backward_precision='mixed_float64_mpmath' if mp_context is not None else 'torch.float64',
                         backward_direct_FS=direct_count,
                         shared_A_vjp=shared_count, carry_Fnext_owners=owners, cotangent_scale_log2=shift,
                         backward='complete_manual_chunk_source_vjp',
                         requested_source_gradients=list(needs_input_grad) if needs_input_grad is not None else [True]*7,
                         backward_extended_subblocks=sum(len(t['blocks']) for t in tapes) if mp_context is not None else 0,
                         backward_cotangent_promotion=scaled_subnormal,
                         backward_extended_precision_bits=None if mp_context is None else mp_context.prec)
    return tuple(results)


class _ManualChunk(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, beta, ell, state, c, options, telemetry, execution_plan):
        sources = (q, k, v, beta, ell, state, c)
        with torch.no_grad(), torch.autocast(device_type='cpu', enabled=False):
            output, final, tapes = _execute(sources, options, telemetry, execution_plan)
        ctx.save_for_backward(*sources)
        ctx.tapes, ctx.options, ctx.telemetry = tapes, options, telemetry
        ctx.set_materialize_grads(False)
        return output, final

    @staticmethod
    def backward(ctx, output_grad, final_grad):
        if torch.is_grad_enabled():
            raise RuntimeError('manual_chunk_key_gated_hyperbolic_delta_rule supports first-order gradients only; '
                               'double backward is not supported.')
        sources = ctx.saved_tensors
        if output_grad is None:
            output_grad = torch.zeros_like(sources[2])
        if final_grad is None:
            final_grad = torch.zeros_like(sources[5])
        with torch.no_grad(), torch.autocast(device_type='cpu', enabled=False):
            grads = _backward(ctx.tapes, sources, ctx.options, output_grad,
                              final_grad, ctx.telemetry, ctx.needs_input_grad[:7])
        return (*grads, None, None, None)


def manual_chunk_key_gated_hyperbolic_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    log_decay: torch.Tensor,
    curvature: float | torch.Tensor = 1.0,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    group_size: int = 32,
    check_inputs: bool = True,
    chunk_size: int = 32,
    carry_mode: str = 'no_carry',
    gauge: str = 'zero_detached',
    gauge_values: torch.Tensor | None = None,
    feedback_precomposition: bool = True,
    reanchor_interval: int = 0,
    telemetry: dict | None = None,
    execution_plan: tuple | None = None,
):
    """Run the CPU chunk graph with a complete handwritten first-order source VJP.

    All options match naive_chunk_key_gated_hyperbolic_delta_rule. The forward runs without AD;
    backward uses explicit FP64 or arbitrary-precision source algebra on the saved partition.
    Gauge selection is fixed only by complete canonical gauge invariance. Requesting create_graph=True
    raises an explicit double-backward error; only the ordinary non-extended AD route supports higher derivatives.
    """
    sources, options = _arguments(q, k, v, beta, log_decay, curvature, initial_state, scale, cu_seqlens, group_size,
                                  check_inputs, chunk_size, carry_mode, gauge, gauge_values,
                                  feedback_precomposition, reanchor_interval)
    options['output_final_state'] = output_final_state
    output, final = _ManualChunk.apply(*sources, options, telemetry, execution_plan)
    return output, final
