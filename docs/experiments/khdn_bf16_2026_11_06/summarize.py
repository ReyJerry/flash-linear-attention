# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Verify published source artifacts and recompute timing statistics without a GPU."""

import gzip
import hashlib
import json
import math
import statistics
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def stats(values):
    assert len(values) == 30 and all(math.isfinite(value) and value > 0 for value in values)
    return dict(median=statistics.median(values), p95=sorted(values)[28], minimum=min(values),
                maximum=max(values), stdev=statistics.stdev(values))


def main():
    evidence = json.loads((ROOT/'evidence.json').read_bytes())
    for name, record in evidence['artifacts'].items():
        data = (ROOT/name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == record['sha256'] and len(data) == record['bytes'], name
    with tarfile.open(ROOT/'timing-sources.tar.gz') as archive:
        for experiment, record in evidence['experiments'].items():
            for name, digest in record['files'].items():
                value = archive.extractfile(experiment+'/'+name).read()
                assert hashlib.sha256(value).hexdigest() == digest
                if name.endswith('.py'):
                    compile(value, experiment+'/'+name, 'exec')
    samples = json.loads(gzip.decompress((ROOT/'samples.json.gz').read_bytes()))
    count = 0
    for comparison in ('native', 'matched'):
        pairs = []
        for trial in range(3):
            verified = evidence[comparison]['trials'][trial]
            assert verified['trial'] == trial
            for method in ('kda', 'khdn_no_carry'):
                for mode in ('fwd', 'bwd', 'fb', 'step'):
                    rows = samples[comparison][str(trial)][method][mode]
                    assert len(rows) == 40 and [row['index'] for row in rows] == list(range(40))
                    for index, row in enumerate(rows):
                        assert row['phase'] == ('warmup' if index < 10 else 'sample')
                        assert row['input_slots'] == 16384 and row['valid_labels'] == 16376
                        assert math.isfinite(row['loss'])
                    for kind in ('wall_ms', 'cuda_ms'):
                        measured = stats([row[kind] for row in rows[10:]])
                        expected = verified['methods'][method]['modes'][mode][kind]
                        for key, value in measured.items():
                            assert math.isclose(value, expected[key], rel_tol=1e-12, abs_tol=1e-9)
                    count += len(rows)
            medians = {method: verified['methods'][method]['modes']['step']['wall_ms']['median']
                       for method in ('kda', 'khdn_no_carry')}
            pairs.append(medians['khdn_no_carry']/medians['kda'])
            print(f'{comparison} trial {trial}: KDA={medians["kda"]:.6f} ms, KHDN={medians["khdn_no_carry"]:.6f} ms')
        ratio = statistics.median(pairs)
        assert math.isclose(ratio, evidence[comparison]['aggregate']['step']['median_paired_time_ratio'])
        print(f'{comparison}: median paired time ratio={ratio:.9f}, throughput ratio={1/ratio:.9f}')
    assert count == 1920
    print(f'Checked {count} observations and all published artifact hashes; GPU training was not rerun.')


if __name__ == '__main__':
    main()
