# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Measure operand-rounding errors for the pending HDN precision RFC on CPU.

This development analysis reuses the independently checked VJP in the CPU test
module. It requires the test dependencies. It neither launches Triton kernels
nor models FP32 accumulator rounding, GPU performance, or model training.
"""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch


def _load_math_reference():
    path = Path(__file__).resolve().parents[1] / 'tests/ops/test_hyperbolic_delta_rule_precision.py'
    spec = importlib.util.spec_from_file_location('hyperbolic_precision_math', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _evaluate(reference, base, kind, dtype, chunk, do, dh):
    q, k, v, beta, state, c = base
    records, outputs = [], []
    for start in range(0, len(q), chunk):
        end = min(start + chunk, len(q))
        qc, kc, vc, bc = (x[start:end] for x in (q, k, v, beta))
        qr = qc.to(dtype).double() if kind else qc
        kr = kc.to(dtype).double() if kind else kc
        kg = kr if kind == 'qk_gram' else kc
        inputs = (qc, kc, vc, bc, state, c, qr, kr, kg)
        output, state, _ = reference._wy_from_edges(*inputs)
        records.append((start, end, inputs))
        outputs.append(output)
    final = state
    dq, dk, dv, dbeta = (torch.zeros_like(x) for x in (q, k, v, beta))
    dc, ds = torch.zeros_like(c), dh
    for start, end, inputs in reversed(records):
        gradients = reference._wy_edge_vjp(inputs, do[start:end], ds)
        dq[start:end] = gradients[0] + gradients[6]
        dk[start:end] = gradients[1] + gradients[7] + gradients[8]
        dv[start:end], dbeta[start:end], ds = gradients[2:5]
        dc += gradients[5]
    return torch.cat(outputs), final, dq, dk, dv, dbeta, ds, dc


def _metrics(candidate, reference, atol, rtol):
    error = (candidate - reference).abs()
    budget = atol + rtol * reference.abs()
    return {
        'max_abs': error.max().item(),
        'relative_rms': ((candidate - reference).square().mean().sqrt()
                         / reference.square().mean().sqrt().clamp_min(1e-30)).item(),
        'max_error_over_budget': (error / budget).max().item(),
        'finite': bool(torch.isfinite(candidate).all()),
        'passes_budget': bool((error <= budget).all()),
        'atol': atol,
        'rtol': rtol,
    }


def _parse_case(text):
    try:
        length, key_dim, value_dim, chunk = (int(x) for x in text.split(':'))
    except ValueError as error:
        raise argparse.ArgumentTypeError('case must be T:K:V:chunk, for example 257:64:16:32') from error
    if min(length, key_dim, value_dim) <= 0 or chunk not in (16, 32):
        raise argparse.ArgumentTypeError('T/K/V must be positive and chunk must be 16 or 32')
    return length, key_dim, value_dim, chunk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', type=_parse_case,
                        default=[(17, 16, 8, 16), (257, 64, 16, 32), (4096, 128, 16, 16), (16384, 128, 16, 16)])
    parser.add_argument('--json', type=Path, default=Path('profile/hyperbolic-opt/precision-cpu.json'))
    parser.add_argument('--require-all-pass', action='store_true',
                        help='return nonzero when any experimental policy fails the unchanged numerical budgets')
    args = parser.parse_args()
    torch.set_num_threads(1)
    reference = _load_math_reference()
    rows = []
    with torch.no_grad():
        for length, key_dim, value_dim, chunk in args.cases:
            base = tuple(x.detach() for x in reference._inputs(length, key_dim, value_dim, 0.5, None))
            torch.manual_seed(889)
            do = torch.randn(length, value_dim, dtype=torch.float64)
            dh = torch.randn(key_dim, value_dim, dtype=torch.float64)
            baseline = _evaluate(reference, base, None, torch.float64, chunk, do, dh)
            for kind in ('qk', 'qk_gram'):
                for name, dtype in (('fp16', torch.float16), ('bf16', torch.bfloat16)):
                    candidate = _evaluate(reference, base, kind, dtype, chunk, do, dh)
                    metrics = {}
                    for index, (label, actual, expected) in enumerate(zip(
                        ('output', 'final_state', 'dq', 'dk', 'dv', 'dbeta', 'dh0', 'dc'), candidate, baseline,
                    )):
                        if index < 2:
                            atol, rtol = 8e-4, 8e-3
                        elif label == 'dc':
                            atol, rtol = 5e-3, 5e-2
                        else:
                            atol, rtol = 2.4e-3, 2.4e-2
                        metrics[label] = _metrics(actual, expected, atol, rtol)
                    failures = [name for name, result in metrics.items() if not result['passes_budget']]
                    row = {
                        'T': length, 'K': key_dim, 'V': value_dim, 'chunk': chunk,
                        'policy': f'{kind}_{name}', 'status': 'rejected' if failures else 'passes_cpu_rounding_budget',
                        'failed_outputs': failures, 'metrics': metrics,
                    }
                    rows.append(row)
                    print(json.dumps({key: value for key, value in row.items() if key != 'metrics'}))
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True, check=True,
    ).stdout.strip()
    payload = {
        'status': 'CPU_ROUNDING_STUDY_ONLY',
        'source_baseline_commit': '405835cc0a44ab4b82d7984f6f5b33e98210ee9b',
        'current_commit': revision,
        'torch': torch.__version__, 'device': 'CPU', 'input_seed': 883, 'cotangent_seed': 889,
        'curvature': 0.5, 'scale': 1.0, 'nonzero_initial_state': True,
        'arithmetic': 'FP64 WY algebra with independently rounded GEMM operands and FP64 analytic adjoints',
        'tolerance_source': 'tests/ops/test_hyperbolic_delta_rule.py: tensor_core forward/backward and curvature tests',
        'GPU': 'NOT RUN', 'production_precision_policy': 'NOT IMPLEMENTED: required RFC issue unavailable',
        'rows': rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + '\n')
    print(f'Wrote CPU rounding study to {args.json}; this is not Triton correctness or GPU performance evidence.')
    if args.require_all_pass and any(row['status'] == 'rejected' for row in rows):
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
