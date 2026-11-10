"""Recompute single-H20 latency, memory and throughput from individual observations."""
import argparse
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path


def summarize(values):
    return dict(median=statistics.median(values), p95=sorted(values)[math.ceil(.95 * len(values)) - 1],
                minimum=min(values), maximum=max(values), stdev=statistics.stdev(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--details', action='store_true', help='emit all per-trial distributions as JSON')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'manifest.json').read_bytes())
    data = gzip.decompress((root / 'samples.jsonl.gz').read_bytes())
    assert hashlib.sha256(data).hexdigest() == manifest['uncompressed_sha256']
    rows = [dict(zip(manifest['columns'], json.loads(line), strict=True)) for line in data.splitlines()]
    assert len(rows) == manifest['records'] == 8000
    results = []
    for run in manifest['runs']:
        for mode in run['mode_order']:
            observations = [r for r in rows if r['run'] == run['id'] and r['mode'] == mode]
            assert [r['index'] for r in observations] == list(range(40))
            for i, row in enumerate(observations):
                assert row['phase'] == ('warmup' if i < 10 else 'sample') and row['sample_id'] == i
                assert row['valid_labels'] == run['batch'] * (run['length'] - 1)
                assert row['input_slots'] == row['nonpadding_tokens'] == 16384
            steady = observations[10:]
            item = {k: run[k] for k in ('id', 'series', 'length', 'batch', 'method', 'trial', 'gpu_uuid')}
            for key in ('wall_ms', 'cuda_ms', 'peak_allocated_bytes', 'peak_reserved_bytes'):
                item[key] = summarize([r[key] for r in steady])
            for key in ('input_slots', 'valid_labels'):
                item[key + '_per_second'] = summarize([r[key] * 1000 / r['wall_ms'] for r in steady])
            item.update(mode=mode, baseline_memory_bytes=run['initial_memory_bytes'][mode])
            results.append(item)
    if args.details:
        print(json.dumps(results, indent=2, allow_nan=False))
        return
    methods = ('kda', 'comba', 'hdn_tc', 'hdn_staged')
    print('T       ' + ' '.join(f'{m:>12}' for m in methods) + '  (step wall ms)')
    for length in (2048, 4096, 8192, 16384):
        values = []
        for name in methods:
            v = [r['wall_ms']['median'] for r in results if r['series'] == 'baseline' and
                 r['length'] == length and r['method'] == name and r['mode'] == 'step']
            assert len(v) == 3
            values.append(statistics.median(v))
        print(f'{length:<7} ' + ' '.join(f'{value:12.2f}' for value in values))
    pair = {r['method']: r for r in results if r['series'] == 'same_card' and r['mode'] == 'step'}
    assert len(pair) == 2 and pair['kda']['gpu_uuid'] == pair['khdn_fs6']['gpu_uuid']
    print(f"Single same-card KHDN/KDA step latency ratio: {pair['khdn_fs6']['wall_ms']['median'] / pair['kda']['wall_ms']['median']:.4f}")


if __name__ == '__main__':
    main()
