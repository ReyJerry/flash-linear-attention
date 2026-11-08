# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Verify archived research records without executing or qualifying GPU kernels."""

import gzip
import hashlib
import json
import math
import statistics
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def check_metrics(rows):
    for row in rows:
        assert row.get('passed', True) and row.get('repeat_exact', True)
        for metric in row['metrics']:
            assert metric['finite'] and metric['passed']
    return len(rows)


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert sha((ROOT / name).read_bytes()) == digest, name
    binding = json.loads((ROOT / 'source-binding.json').read_text())
    seen = set()
    for name in ('operator-sources.tar.gz', 'model-sources.tar.gz'):
        with tarfile.open(ROOT / name) as archive:
            for member in archive:
                path = Path(member.name)
                assert member.isfile() and not path.is_absolute() and '..' not in path.parts
                assert member.name not in seen
                seen.add(member.name)
                assert sha(archive.extractfile(member).read()) == binding['source_files'][member.name]
                assert path.name != 'test_key_gated_hyperbolic_precision.py'
    assert seen == set(binding['source_files'])
    evidence = json.loads(gzip.decompress((ROOT / 'evidence.json.gz').read_bytes()))
    for name, row in evidence.items():
        if name == 'model_diagnostics':
            continue
        receipt = row['receipt']
        assert receipt['terminal'] and receipt['reaped'] and receipt['group_drained']
        assert receipt['exit'] == 0 and not receipt['apps_after']
        assert sha(row['result_text'].encode()) == receipt['result_sha256']
        assert json.loads(row['result_text']) == row['result']

    selected = evidence['56-compact-selective-candidate']['result']['parts']
    counts = {name: check_metrics(part['result']['rows']) for name, part in selected.items()}
    assert counts == {'four_chunks': 8, 'original12': 24, 'c0': 8, 'boundary': 2}
    assert check_metrics(evidence['57-historical-strong-gate']['result']['rows']) == 16
    boundary = evidence['66-selected-boundary-alias']['result']['rows']
    assert check_metrics(boundary) == 56
    assert all(row['cross_stream_exact'] for row in boundary if row['mode'] != 'independent')

    samples = 0
    for experiment in ('58-selective-chunk-cost', '61-selected-long-cost'):
        rows = evidence[experiment]['raw']
        assert check_metrics(evidence[experiment]['result']['inverse_control']['rows']) == 12
        weights = {rows[name]['qualification.json']['model']['initial_weights_sha256']
                   for name in ('c64', 'c64_i3', 'c128')}
        assert len(weights) == 1
        batches = []
        print(experiment)
        for arm, data in rows.items():
            observations, result = data['samples.json'], data['result.json']
            assert len(observations) == 10 and result['physical_optimizer_steps'] == 6
            assert result['parameters'] == 1412912512
            assert result['optimizer_backend'] == 'triton_fp32_adamw'
            batches.append([(r['mode'], r['index'], r['samples']) for r in observations])
            for mode in ('fb', 'step'):
                steady = [r for r in observations if r['mode'] == mode and not r['warmup']]
                assert len(steady) == 3
                assert statistics.median(r['total_wall_ms'] for r in steady) == result['medians_ms'][mode]
            samples += len(observations)
            print(f"  {arm}: F+B {result['medians_ms']['fb']:.3f} ms; step {result['medians_ms']['step']:.3f} ms")
        assert all(batch == batches[0] for batch in batches)
    assert samples == 80

    gradient = evidence['67-selected-full-gradient']['result']['result']
    records = gradient['gradients']
    assert len(records) == 459 and sum(r['elements'] for r in records.values()) == 1412912512
    error, actual, reference, dot = [math.fsum(r[key] for r in records.values()) for key in
                                     ('error_squared', 'actual_squared', 'reference_squared', 'dot')]
    assert dot / math.sqrt(actual * reference) == gradient['gradient_cosine']
    assert math.sqrt(actual / reference) == gradient['gradient_norm_ratio']
    assert math.sqrt(error / reference) == gradient['gradient_relative_L2']
    assert gradient['gradient_cosine'] >= .99 and .9 <= gradient['gradient_norm_ratio'] <= 1.1
    assert sum(r['old_budget_bad'] for r in records.values()) == gradient['old_budget_bad'] == 15325
    assert gradient['physical_optimizer_updates'] == 0 and gradient['weights_unchanged']
    assert gradient['execution_configuration']['chunk_size'] == 64
    print('Verified 114 operator executions, 80 timing observations and 459 gradient records.')
    print('Research evidence only; model regressions and final engineering qualification remain open.')


if __name__ == '__main__':
    main()
