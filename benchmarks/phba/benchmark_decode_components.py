# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""One-layer component diagnostics, separate from full trained-model latency."""

import argparse
import json
from pathlib import Path

import torch

from run_decode_acceptance import check_manifest
from fla.ops.phba.decode import PHBALayerCache
from fla.utils import assert_close


def timed(fn, prepare, repeats):
    values = []
    for i in range(repeats + 3):
        prepare()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        if i >= 3:
            values.append(start.elapsed_time(end))
    return {'mean_ms': sum(values)/len(values), 'min_ms': min(values), 'max_ms': max(values), 'raw_ms': values}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-report', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--length', type=int, default=32768)
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    check_manifest(args.validation_report)
    if args.length < 128 or args.length % 128 or args.repeats < 1:
        raise ValueError('length must be a multiple of 128 and repeats positive')
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype, h, d, m, c, k = torch.bfloat16, 16, 128, 128, 128, 8
    length = args.length
    x = [torch.randn(1, length+c, h, d, device='cuda', dtype=dtype)*.1 for _ in range(3)]
    gate = torch.randn(1, length+c, h, m, device='cuda', dtype=dtype)*.3
    q, qr = [torch.randn(1, h, d, device='cuda', dtype=dtype)*.1 for _ in range(2)]

    def make(stage):
        cache = PHBALayerCache(1, length+c, h, d, m, c, k, dtype=dtype, stage=stage)
        cache.prefill(x[0][:, :length], x[1][:, :length], x[2][:, :length], gate[:, :length])
        cache.keys[:, length:].copy_(x[1][:, length:])
        cache.values[:, length:].copy_(x[2][:, length:])
        cache.raw_tail.copy_(x[0][:, length:])
        log, write = cache._gates(gate[:, length:])
        cache.log_tail.copy_(log)
        cache.write_tail.copy_(write)
        return cache

    results = {'length': length, 'scope': 'one-layer synthetic component diagnostics; not model/rebuttal latency',
               'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'measurements': {}}
    eager, delayed = make('p0'), make('p1_state')
    anchors = [(cache.state_k.clone(), cache.state_v.clone()) for cache in (eager, delayed)]
    for cache, anchor in zip((eager, delayed), anchors):
        def prepare(cache=cache, anchor=anchor):
            cache.state_k.copy_(anchor[0])
            cache.state_v.copy_(anchor[1])
        def update(cache=cache):
            if cache.stage == 'p0':
                for n in range(1, c+1):
                    cache._refresh_live(length//c, n)
            cache._finish_block(length//c)
        results['measurements'][cache.stage+'_one_block_state'] = timed(update, prepare, args.repeats)
    assert_close('boundary K', eager.state_k, delayed.state_k, .006)
    assert_close('boundary V', eager.state_v, delayed.state_v, .006)
    del eager, delayed, anchors
    pooled, cached = make('p1_state'), make('p1_route')
    reference_routes = pooled.route(qr, length).clone()
    actual_routes = cached.route(qr, length).clone()
    if not torch.equal(reference_routes, actual_routes):
        raise RuntimeError('route mismatch: inspect near-ties/reduction precision before benchmarking')
    for cache in (pooled, cached):
        results['measurements'][cache.stage+'_route'] = timed(lambda: cache.route(qr, length), lambda: None, args.repeats)
    del pooled
    split = make('p1_split')
    a = cached.read(q, qr, length, reference_routes)
    b = split.read(q, qr, length, reference_routes)
    assert_close('fixed-route split readout', a.float(), b.float(), .03)
    for cache in (cached, split):
        results['measurements'][cache.stage+'_fixed_route_readout'] = timed(
            lambda: cache.read(q, qr, length, reference_routes), lambda: None, args.repeats)
    results['persistent_bytes'] = split.memory_bytes()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
