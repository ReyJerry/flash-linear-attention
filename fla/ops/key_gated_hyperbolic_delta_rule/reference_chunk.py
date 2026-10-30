# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""CPU strict-past chunk graph with ordinary AD and explicit execution telemetry.

This is a mathematical reference, not a GPU dispatch or a performance implementation.
Local decay intervals deliberately use O(C²K) reference scratch. See docs/key_gated_hyperbolic_math.md.
"""

import math

import torch

from fla.ops.key_gated_hyperbolic_delta_rule.naive import _validate
from fla.ops.key_gated_hyperbolic_delta_rule.reference_arithmetic import (
    MPArray,
    arithmetic,
    exact_domain,
    extended,
    like,
    precision_context,
    source_roundoff_bits,
    source_roundoff_reason,
    tensor,
)


class ReferenceRangeError(RuntimeError):
    """The CPU reference cannot certify representable arithmetic for this execution."""


def _log_norm(x, dim):
    x = x.detach().double().abs()
    largest = x.amax(dim=dim, keepdim=True)
    safe = torch.where(largest > 0, largest, torch.ones_like(largest))
    return safe.squeeze(dim).log2() + 0.5 * (x / safe).square().sum(dim=dim).log2()


def _domain(q, k, v, beta, ell, c, state, g):
    if any(not torch.isfinite(x).all().item() for x in (q, k, v, beta, ell, state)):
        raise ValueError('All inputs must be finite.')
    if not ((beta >= 0) & (beta <= 1)).all().item() or not (ell <= 0).all().item():
        raise ValueError('Require beta in [0,1] and nonpositive log_decay.')
    if k.numel():
        logk = _log_norm(k, -1)
        if not torch.isfinite(logk).all().item():
            raise ValueError('Every valid token must have a nonzero Key.')
        if c.item() > 0:
            logv = _log_norm(v.reshape(*v.shape[:-1], -1, g), -1)
            logdelta = math.log2(c.item()) - math.log2(g) + 2 * (logv - logk[..., None])
            if not (logdelta < -32 * torch.finfo(torch.float64).eps).all().item() and not exact_domain(k, v, c, state, g)[0]:
                raise ValueError('Require ||k||^2 - (curvature/group_size) * ||v_group||^2 > 0.')
    if c.item() > 0:
        grouped = state.reshape(*state.shape[:2], state.shape[2], -1, g).transpose(-3, -2).flatten(-2)
        log_radius = math.log2(c.item()) - math.log2(g) + 2 * _log_norm(grouped, -1)
        if not (log_radius < -32 * torch.finfo(torch.float64).eps).all().item() and not exact_domain(k, v, c, state, g)[1]:
            raise ValueError('Each initial-state group must lie strictly inside its Frobenius ball.')


def _logadd2(values):
    largest = max(values)
    if largest == -math.inf:
        return largest
    return largest + math.log2(sum(2.0 ** (v - largest) for v in values))


def _input_bound(k, v, beta, c, g, dtype, fixed, state_radius=0.0):
    """Bound amplification and screen coefficient ranges before inverses or gauges."""
    logn = 2 * _log_norm(k, -1)
    logv = _log_norm(v.reshape(len(v), -1, g), -1)
    if c.item() > 0:
        logdelta = math.log2(c.item()) - math.log2(g) + 2 * logv - logn[:, None]
        delta = torch.exp2(logdelta)
    else:
        delta = torch.zeros_like(logv)
    if not (delta < 1).all().item():
        return dict(amplitude_bits=None, inverse_bound_bits=None, min_input_margin=0.0,
                    source_roundoff_screen=None, reasons=('unresolved_input_margin',))
    radius = delta.sqrt()
    b = beta.detach().double()[:, None]
    lower = 1 - b + b / (1 + radius)
    upper = 1 - b + b * (1 + radius) / (1 - delta)
    upper_bits = upper.log2().sum(0)
    inverse_bits = []
    for j in range(delta.shape[1]):
        rows = []
        for i in range(len(k)):
            terms = [0.0]
            if b[i, 0].item() > 0:
                terms += [math.log2(b[i, 0].item() * (1 + math.sqrt(delta[i, j].item() * delta[t, j].item())))
                          + rows[t] for t in range(i)]
            rows.append(_logadd2(terms) - math.log2(1 - delta[i, j].item()))
        # Z^-1=diag(1/n)H^-1; all lower gauge ratios are bounded by products of 1/L.
        inverse_bits.append(max(x - n for x, n in zip(rows, logn.tolist())) - lower[:, j].log2().sum().item())
    reasons = []
    amplitude = upper_bits.max().item()
    # the strict inequality leaves a rounding guard around the fixed 32-bit boundary
    if amplitude >= 32 - 1e-10:
        reasons.append('amplitude_budget_32')
    if max(inverse_bits) >= (80 if dtype == torch.float32 else 400):
        reasons.append('triangular_inverse_majorant')
    if logn.abs().max().item() > (40 if dtype == torch.float32 else 200):
        reasons.append('raw_key_norm_range')
    # the two subtractive margins condition coefficient preparation and state feedback
    margin = (1 - delta).min().item()
    sensitivity = (2 - margin) / margin * (1 + state_radius) / max(1 - state_radius, torch.finfo(torch.float64).tiny)
    source_roundoff = torch.finfo(dtype).eps * sensitivity
    if source_roundoff > (2e-5 if dtype == torch.float32 else 1e-9) / 8:
        reasons.append('source_sensitivity')
    if fixed is not None:
        logs = fixed.detach().double().log2()
        if logs.abs().max().item() > 32 or (logs.amax(0) - logs.amin(0)).max().item() > 32:
            reasons.append('fixed_gauge_range')
    return dict(amplitude_bits=amplitude, inverse_bound_bits=max(inverse_bits),
                min_input_margin=margin, source_roundoff_screen=source_roundoff if math.isfinite(source_roundoff) else None,
                reasons=tuple(reasons))


def _condition(matrix, dtype, name):
    if isinstance(matrix, MPArray):
        return None
    value = torch.linalg.cond(matrix.detach().double()).item()
    if not math.isfinite(value) or value > (2 ** 10 if dtype == torch.float32 else 2 ** 32):
        raise ReferenceRangeError(f'{name}_condition={value:.9g}')
    return value


def _erase(k, gamma, x):
    return x - gamma * k[:, None] * (k @ x)[None, :]


def _positive(residual, target, r, t):
    z = r + t
    if z.ndim:
        return residual / z[:, None] + (t / z)[:, None] * target
    return residual / z + (t / z) * target


def _coefficients(beta, n):
    ops = arithmetic(n)
    rows, eta, products = [], [], []
    for i in range(len(n)):
        product = n.new_ones(())
        values = [n.new_zeros(()) for _ in range(len(n))]
        values[i] = n[i]
        row_products = [None] * i
        for j in reversed(range(i)):
            row_products[j] = product
            values[j] = -beta[i] * n[j] * product
            product = product * (1 - beta[j])
        rows.append(ops.stack(values))
        eta.append(beta[i] * product)
        products.append((row_products, product))
    return ops.stack(rows), ops.stack(eta), products


def _prepare_common(q, k, beta, ell):
    ops = arithmetic(k)
    count, keydim = k.shape
    eye = ops.eye(count, dtype=k.dtype, device=k.device)
    # each interval starts at its own left endpoint; no subtraction of cumulative logs
    rel = ops.stack([ops.stack([
        ell[j + 1:i + 1].sum(0).exp() if j <= i else ell.new_zeros(keydim)
        for j in range(count)]) for i in range(count)])
    decay = ops.stack([ell[:i + 1].sum(0).exp() for i in range(count)])
    n = k.square().sum(-1)
    gamma = beta / n
    gram = ops.tril((k[:, None] * rel * k[None]).sum(-1), diagonal=-1)
    lower = eye + gamma[:, None] * gram
    cond_l = _condition(lower, k.dtype, 'L')
    inverse = ops.linalg.solve_triangular(lower, eye, upper=False)
    ke = decay * k
    bmat = gamma[:, None] * ke
    fmat = inverse @ bmat
    aq = (q * k).sum(-1) / n
    qr = q - (beta * aq)[:, None] * k
    cr = ops.tril((qr[:, None] * rel * k[None]).sum(-1), diagonal=-1)
    to = cr @ inverse
    tail = k * rel[-1]
    p0 = tail[:-1].T @ inverse[:-1, :-1]
    past = _erase(k[-1], gamma[-1], p0)
    qs, eta, products = _coefficients(beta, n)
    return dict(q=q, k=k, beta=beta, ell=ell, n=n, gamma=gamma, rel=rel, decay=decay, GK=gram,
                L=lower, A=inverse, Ke=ke, B=bmat, F=fmat, aq=aq, qr=qr, Qer=qr * decay, Cr=cr, To=to,
                tail=tail, P0=p0, P=past, Qs=qs, eta=eta, products=products, cond_L=cond_l)


def _prepare_group(p, v, kappa, gauge, fixed, precompose):
    ops = arithmetic(v)
    gram = v @ v.T
    rmat = p['A'] * gram
    count = len(v)
    if gauge == 'none':
        lam = v.new_ones(count)
    elif gauge == 'fixed':
        dt = torch.float64 if isinstance(v, MPArray) else v.dtype
        s = fixed if fixed is not None else like((0.7 * torch.sin(torch.arange(count, dtype=dt))).exp(), v)
        lam = s / ops.cat((s.new_ones(1), s[:-1]))
    else:
        history = v.new_empty(0)
        ratios = []
        for i in range(count):
            u0 = (p['beta'][i] + kappa * (rmat[i, :i] @ history)) / (p['n'][i] - kappa * rmat[i, i])
            ratio = 1 - p['beta'][i] + p['n'][i] * u0
            history = ops.cat((history / ratio, (u0 / ratio).reshape(1)))
            ratios.append(ratio)
        lam = ops.stack(ratios)
        if gauge == 'zero_detached':
            lam = lam.detach()
    if not (ops.isfinite(lam) & (lam > 0)).all().item():
        raise ReferenceRangeError('Invalid adjacent gauge ratio.')
    nu = ops.stack([lam[:i + 1].reciprocal().prod() for i in range(count)])
    rho = ops.stack([ops.stack([
        lam[j + 1:i + 1].reciprocal().prod() if j <= i else lam.new_zeros(())
        for j in range(count)]) for i in range(count)])
    if not (ops.isfinite(nu) & (nu > 0)).all().item() or not ops.isfinite(rho).all().item():
        raise ReferenceRangeError('Gauge factor is not representable.')
    mask = torch.ones(rho.shape, dtype=torch.bool).tril()
    np_mask = mask.numpy() if isinstance(rho, MPArray) else mask
    if not (rho[np_mask] > 0).all().item():
        raise ReferenceRangeError('A causal gauge ratio underflowed.')
    zmat = (p['Qs'] - kappa * rmat) * rho
    cond_z = _condition(zmat, v.dtype, 'Zhat')
    if precompose:
        af = ops.linalg.solve_triangular(zmat, (nu * p['eta'])[:, None], upper=False)[:, 0]
        jf = -kappa * ops.linalg.solve_triangular(zmat, ops.diag(nu), upper=False)
    else:
        af = jf = None
    return dict(v=v, kappa=kappa, VV=gram, R=rmat, lam=lam, nu=nu, rho=rho, Z=zmat,
                af=af, Jf=jf, cond_Z=cond_z)


def _probe(q, k, v, beta, ell, c, g, dtype, gauge, fixed):
    q, k, v, beta, ell, c = (x.to(dtype) for x in (q, k, v, beta, ell, c))
    p = _prepare_common(q, k, beta, ell)
    for j in range(v.shape[-1] // g):
        _prepare_group(p, v[:, j*g:(j+1)*g], c / g, gauge,
                       None if fixed is None else fixed[:, j].to(dtype), False)


def _plan_head(q, k, v, beta, ell, state, c, g, chunk_size, dtype, gauge, fixed, output_scale):
    result = []
    cancellation_bits = source_roundoff_bits(q, k, v, state, output_scale)
    radius = 0.0
    if c.item() > 0:
        groups = state.reshape(state.shape[0], -1, g).transpose(0, 1).flatten(1)
        radius = torch.exp2(0.5 * (math.log2(c.item()) - math.log2(g)) + _log_norm(groups, -1)).max().item()
    for start in range(0, len(k), chunk_size):
        end = min(start + chunk_size, len(k))
        fx = None if fixed is None else fixed[start:end]
        bound = _input_bound(k[start:end], v[start:end], beta[start:end], c, g, dtype, fx, radius)
        bound['requested_amplitude_bits'] = bound['amplitude_bits']
        bound['requested_inverse_bound_bits'] = bound['inverse_bound_bits']
        bound['state_readout_source_bits'] = cancellation_bits
        reasons = list(bound['reasons']) + list(source_roundoff_reason(cancellation_bits, dtype))
        maximum = max(x.detach().abs().max().item() for x in (k[start:end], v[start:end], state, c))
        source_bits = math.log2(maximum) if maximum else 0.
        qmax = q[start:end].detach().abs().max().item()
        query_bits = math.log2(qmax) + math.log2(abs(output_scale)) if qmax and output_scale else -math.inf
        source_bits = max(source_bits, query_bits)
        if query_bits + math.log2(torch.finfo(dtype).eps) > math.log2((2e-5 if dtype == torch.float32 else 1e-10) / 8):
            reasons.append('readout_source_roundoff')
        if source_bits > (24 if dtype == torch.float32 else 200):
            reasons.append('source_amplitude_range')
        if dtype == torch.float32 and radius > .95:
            reasons.append('entry_ball_margin')
        if not reasons:
            try:
                _probe(q[start:end], k[start:end], v[start:end], beta[start:end], ell[start:end], c, g, dtype, gauge, fx)
            except ReferenceRangeError as error:
                reasons.append(str(error))
        if not reasons:
            result.append((start, end, str(dtype), False, (), bound))
            continue
        pos = start
        while pos < end:
            last_error = None
            for stop in range(end, pos, -1):
                fx = None if fixed is None else fixed[pos:stop]
                strict = _input_bound(k[pos:stop], v[pos:stop], beta[pos:stop], c, g, torch.float64, fx, radius)
                strict['requested_amplitude_bits'] = bound['amplitude_bits']
                strict['requested_inverse_bound_bits'] = bound['inverse_bound_bits']
                strict['state_readout_source_bits'] = cancellation_bits
                strict['reasons'] += source_roundoff_reason(cancellation_bits, torch.float64)
                if source_bits > 200:
                    strict['reasons'] += ('source_amplitude_range',)
                if query_bits + math.log2(torch.finfo(torch.float64).eps) > math.log2(1e-10 / 8):
                    strict['reasons'] += ('readout_source_roundoff',)
                if strict['reasons']:
                    last_error = ','.join(strict['reasons'])
                    continue
                try:
                    _probe(q[pos:stop], k[pos:stop], v[pos:stop], beta[pos:stop], ell[pos:stop], c, g,
                           torch.float64, gauge, fx)
                except ReferenceRangeError as error:
                    last_error = str(error)
                    continue
                result.append((pos, stop, str(torch.float64), True, tuple(reasons), strict))
                pos = stop
                break
            else:
                stop = min(pos + 4, end)
                strict = _input_bound(k[pos:stop], v[pos:stop], beta[pos:stop], c, g, torch.float64,
                                      None if fixed is None else fixed[pos:stop], radius)
                strict['requested_amplitude_bits'] = bound['amplitude_bits']
                strict['requested_inverse_bound_bits'] = bound['inverse_bound_bits']
                strict['state_readout_source_bits'] = cancellation_bits
                result.append((pos, stop, 'mpmath', True, tuple(dict.fromkeys([*reasons, last_error])), strict))
                pos = stop
    return result


def _online(p, groups, state, incoming_y, next_p):
    ops = arithmetic(p['k'])
    g = groups[0]['v'].shape[-1]
    states, next_y, outputs, tape = [], [], [], []
    if next_p is None:
        fn = p['F'].new_empty((0, p['F'].shape[1]))
    else:
        fn = like(next_p['F'], p['F'])
    k, gamma, tau = p['k'][-1], p['gamma'][-1], p['decay'][-1]
    e = (fn - gamma * (fn @ k)[:, None] * k[None]) * tau[None]
    hcarry = fn @ p['P']
    for j, a in enumerate(groups):
        s = state[:, j*g:(j+1)*g]
        y = p['F'] @ s if incoming_y is None else incoming_y[j]
        h = (a['v'] * y).sum(-1)
        if a['af'] is None:
            w = ops.linalg.solve_triangular(a['Z'], (a['nu'] * (p['eta'] - a['kappa'] * h))[:, None],
                                            upper=False)[:, 0]
        else:
            w = a['af'] + a['Jf'] @ h
        u = w[:, None] * a['v'] - a['nu'][:, None] * (p['L'] @ y)
        prev = w.new_ones(())
        rrows, trows, zprev = [], [], []
        for i in range(len(w)):
            r = (1 - p['beta'][i]) / a['lam'][i] * prev
            t = p['n'][i] * w[i]
            zprev.append(prev)
            rrows.append(r)
            trows.append(t)
            prev = r + t
        r, t = ops.stack(rrows), ops.stack(trows)
        if not (ops.isfinite(r + t) & (r + t > 0) & (r >= 0) & (t >= 0)).all().item():
            raise ReferenceRangeError('Nonpositive or nonfinite hatted mass; no clipping is permitted.')
        ro = a['nu'][:, None] * (p['Qer'] @ s) + (p['To'] * a['rho']) @ u
        target_o = p['aq'][:, None] * a['v']
        output = _positive(ro, target_o, r, t)
        ss = a['nu'][-1] * s
        us = a['rho'][-1, :-1, None] * u[:-1]
        rs = _erase(k, gamma, tau[:, None] * ss) + p['P'] @ us
        ts = ops.outer(k, a['v'][-1]) / p['n'][-1]
        sn = _positive(rs, ts, r[-1], t[-1])
        ry = e @ ss + hcarry @ us
        ty = (fn @ k / p['n'][-1])[:, None] * a['v'][-1]
        yn = _positive(ry, ty, r[-1], t[-1])
        if isinstance(w, MPArray):
            residual = a['Z'] @ w - a['nu'] * (p['eta'] - a['kappa'] * h)
        else:
            residual = a['Z'].detach().double() @ w.detach().double() - (
                a['nu'].detach().double() * (p['eta'].detach().double() - a['kappa'].detach().double() * h.detach().double()))
        if not all(ops.isfinite(x).all().item() for x in (output, sn, yn, residual)):
            raise ReferenceRangeError('Nonfinite chunk result or solve residual.')
        states.append(sn)
        next_y.append(yn)
        outputs.append(output)
        tape.append(dict(**a, S=s, Y=y, h=h, w=w, U=u, r=r, t=t, zprev=ops.stack(zprev), RO=ro, TO=target_o,
                         Ss=ss, Us=us, RS=rs, TS=ts, Sout=sn, RY=ry, TY=ty, solve_residual=float(abs(residual).max().item())))
    return ops.cat(outputs, -1), ops.cat(states, -1), next_y, dict(groups=tape, Fnext=fn, H=hcarry)


def _arguments(q, k, v, beta, log_decay, curvature, initial_state, scale, cu_seqlens, group_size,
               check_inputs, chunk_size, carry_mode, gauge, gauge_values, feedback_precomposition, reanchor_interval):
    bounds, scale = _validate(q, k, v, beta, log_decay, curvature, initial_state, scale, cu_seqlens, group_size, False)
    if q.device.type != 'cpu':
        raise ValueError('The chunk references support CPU only.')
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError('chunk_size must be a positive integer.')
    if carry_mode not in ('no_carry', 'y_carry'):
        raise ValueError('carry_mode must be no_carry or y_carry.')
    if gauge not in ('none', 'fixed', 'zero_live', 'zero_detached'):
        raise ValueError('Unknown gauge mode.')
    if not isinstance(feedback_precomposition, bool):
        raise ValueError('feedback_precomposition must be bool.')
    if isinstance(reanchor_interval, bool) or not isinstance(reanchor_interval, int) or reanchor_interval < 0:
        raise ValueError('reanchor_interval must be a nonnegative integer.')
    sources = (q, k, v, beta, log_decay) + (() if initial_state is None else (initial_state,))
    sources += (curvature,) if isinstance(curvature, torch.Tensor) else ()
    dtype = torch.float64 if any(x.dtype == torch.float64 for x in sources) else torch.float32
    c = curvature if isinstance(curvature, torch.Tensor) else torch.as_tensor(curvature, dtype=dtype, device=q.device)
    B, T, H, K = q.shape
    state = q.new_zeros((len(bounds) - 1, H, K, v.shape[-1]), dtype=dtype) if initial_state is None else initial_state
    if gauge_values is not None:
        shape = (*q.shape[:3], v.shape[-1] // group_size)
        if gauge != 'fixed' or gauge_values.shape != shape or gauge_values.device != q.device:
            raise ValueError(f'Frozen gauge_values require gauge=fixed and shape {shape}.')
        if gauge_values.requires_grad or not (torch.isfinite(gauge_values) & (gauge_values > 0)).all().item():
            raise ValueError('gauge_values must be finite, positive and frozen.')
    if check_inputs:
        _domain(q, k, v, beta, log_decay, c.detach(), state, group_size)
    options = dict(bounds=bounds, scale=scale, group_size=group_size, chunk_size=chunk_size, dtype=dtype,
                   carry_mode=carry_mode, gauge=gauge, gauge_values=gauge_values,
                   feedback_precomposition=feedback_precomposition, reanchor_interval=reanchor_interval)
    return (q, k, v, beta, log_decay, state, c), options


def _execute(sources, options, telemetry, execution_plan=None, replay=None):
    q, k, v, beta, ell, s0, c = sources
    B, T, H, K = q.shape
    V, g = v.shape[-1], options['group_size']
    flat = [x.reshape(B * T, *x.shape[2:]) for x in (q, k, v, beta, ell)]
    fixed = options['gauge_values']
    if fixed is not None:
        fixed = fixed.reshape(B * T, H, V // g)
    output_sequences, final_sequences, tapes, plans, records = [], [], [], [], []
    if execution_plan is not None and len(execution_plan) != (len(options['bounds']) - 1) * H:
        raise ValueError('Frozen execution_plan has the wrong sequence/head count.')
    direct_fs = carry_edges = 0
    plan_index = 0
    mp_context = None if replay is None else replay[1]
    for seq, (bos, eos) in enumerate(zip(options['bounds'], options['bounds'][1:])):
        output_heads, final_heads = [], []
        for head in range(H):
            x = [z[bos:eos, head] for z in flat]
            fx = None if fixed is None else fixed[bos:eos, head]
            if replay is None:
                with torch.no_grad():
                    plan = _plan_head(*x, s0[seq, head], c, g, options['chunk_size'],
                                      options.get('working_dtype', options['dtype']), options['gauge'], fx, options['scale'])
            else:
                plan = replay[0][plan_index]['plan']
            if any(row[2] == 'mpmath' for row in plan):
                if torch.is_grad_enabled():
                    from fla.ops.key_gated_hyperbolic_delta_rule.reference_backward import _ManualChunk
                    output, final = _ManualChunk.apply(*sources, options, telemetry, execution_plan)
                    if telemetry is not None:
                        telemetry['ordinary_entry_extended_manual'] = True
                    return output, final, []
                if mp_context is None:
                    precision_sources = (*sources, torch.tensor(options['scale'], dtype=torch.float64))
                    if fixed is not None:
                        precision_sources += (fixed,)
                    mp_context = precision_context(precision_sources)
            if execution_plan is not None:
                frozen = execution_plan[plan_index]
                signature = tuple((a, b, dt, fb, why) for a, b, dt, fb, why, _ in plan)
                if tuple(frozen) != signature:
                    raise ValueError('Frozen execution_plan does not match this input/options schedule.')
            plans.append(tuple((a, b, dt, fb, why) for a, b, dt, fb, why, _ in plan))
            plan_index += 1
            prepared = []
            for start, stop, precision, fallback, reasons, bound in plan:
                precision = precision if replay is None else 'mpmath'
                dt = torch.float64 if precision == 'torch.float64' else torch.float32
                xx = [extended(z[start:stop], mp_context) if precision == 'mpmath' else z[start:stop].to(dt) for z in x]
                p = _prepare_common(xx[0] * options['scale'], xx[1], xx[3], xx[4])
                groups = [_prepare_group(p, xx[2][:, j*g:(j+1)*g], like(c, xx[1]) / g, options['gauge'],
                                         None if fx is None else like(fx[start:stop, j], xx[1]),
                                         options['feedback_precomposition']) for j in range(V // g)]
                prepared.append(dict(p=p, groups=groups, start=start, stop=stop, fallback=fallback))
                records.append(dict(sequence=seq, head=head, start=start, stop=stop, precision=precision,
                                    fallback=fallback, reasons=list(reasons),
                                    **{a: b for a, b in bound.items() if a != 'reasons'},
                                    cond_L=p['cond_L'], cond_Z=None if precision == 'mpmath' else max(
                                        a['cond_Z'] for a in groups),
                                    precision_bits=mp_context.prec if precision == 'mpmath' else torch.finfo(dt).bits))
            state, y = s0[seq, head].to(options['dtype']), None
            output_chunks, head_tape = [], []
            for index, part in enumerate(prepared):
                p = part['p']
                state = like(state, p['k'])
                outgoing = options['carry_mode'] == 'y_carry' and index + 1 < len(prepared)
                outgoing = outgoing and not part['fallback'] and not prepared[index + 1]['fallback']
                if options['reanchor_interval'] and (index + 1) % options['reanchor_interval'] == 0:
                    outgoing = False
                next_p = prepared[index + 1]['p'] if outgoing else None
                output, state, next_y, tape = _online(p, part['groups'], state, y, next_p)
                tape.update(p=p, start=part['start'], stop=part['stop'], direct=y is None, outgoing=outgoing)
                direct_fs += int(y is None)
                carry_edges += int(outgoing)
                head_tape.append(tape)
                y = next_y if outgoing else None
                if replay is None:
                    output_chunks.append(tensor(output, q.dtype))
            output_heads.append(torch.cat(output_chunks) if output_chunks else v.new_empty((0, V)))
            if options['output_final_state'] and replay is None:
                final_heads.append(tensor(state, options['dtype']))
            tapes.append(dict(sequence=seq, head=head, bos=bos, eos=eos, blocks=head_tape, plan=plan))
        output_sequences.append(torch.stack(output_heads, dim=1))
        if options['output_final_state'] and replay is None:
            final_sequences.append(torch.stack(final_heads))
    if replay is not None:
        return None, None, tapes
    anchor = sum(x.reshape(-1)[:0].sum() for x in sources)
    output = (torch.cat(output_sequences).reshape(B, T, H, V) + anchor).to(q.dtype)
    final = torch.stack(final_sequences) + anchor if options['output_final_state'] else None
    if not torch.isfinite(output).all().item() or (final is not None and not torch.isfinite(final).all().item()):
        raise ReferenceRangeError('A public output or final state is not representable in its return dtype.')
    if telemetry is not None:
        telemetry.update(execution_plan=tuple(plans), partition=records, direct_FS=direct_fs, carry_edges=carry_edges,
                         fallback_subblocks=sum(r['fallback'] for r in records),
                         fallback_chunks=len({(r['sequence'], r['head'], r['start'] // options['chunk_size'])
                                              for r in records if r['fallback']}),
                         backend='cpu_chunk_reference', gauge=options['gauge'], carry_mode=options['carry_mode'],
                         feedback_precomposition=options['feedback_precomposition'],
                         max_solve_residual=max((a['solve_residual'] for t in tapes for b in t['blocks']
                                                for a in b['groups']), default=0.0),
                         extended_subblocks=sum(r['precision'] == 'mpmath' for r in records),
                         extended_precision_bits=None if mp_context is None else mp_context.prec,
                         min_mass=min((float((a['r'] + a['t']).min().item()) for t in tapes for b in t['blocks']
                                       for a in b['groups']), default=None))
    return output, final, tapes


def naive_chunk_key_gated_hyperbolic_delta_rule(
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
    """Evaluate the CPU chunk graph with ordinary AD, or an explicit extended manual route.

    The canonical interface matches naive_recurrent_key_gated_hyperbolic_delta_rule.
    Gauge modes are none, fixed (optional positive [B,T,H,G] values), zero_live and zero_detached.
    Flagged chunks use maximal safe FP64 subblocks, then short arbitrary-precision blocks if needed.
    Extended arithmetic uses the complete manual first-order VJP, including through this AD entry.
    Telemetry records the actual partition; execution_plan freezes it for checkpoint replay.
    Output follows q.dtype; public state retains FP64 if any source is FP64, otherwise FP32.
    """
    sources, options = _arguments(q, k, v, beta, log_decay, curvature, initial_state, scale, cu_seqlens, group_size,
                                  check_inputs, chunk_size, carry_mode, gauge, gauge_values,
                                  feedback_precomposition, reanchor_interval)
    options['output_final_state'] = output_final_state
    with torch.autocast(device_type='cpu', enabled=False):
        output, final, tapes = _execute(sources, options, telemetry, execution_plan)
        if torch.is_grad_enabled() and tapes and any(x.requires_grad for x in sources):
            from fla.ops.key_gated_hyperbolic_delta_rule.reference_backward import _OrdinaryCotangent
            output, final = _OrdinaryCotangent.apply(output, final, *sources, options, tapes, telemetry)
    return output, final
