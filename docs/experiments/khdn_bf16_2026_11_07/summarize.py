# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


"""Verify the public optimization release and recompute all1440 observations."""
import ast
import gzip
import hashlib
import io
import json
import math
import statistics
import tarfile
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[2]


def digest(value):
    return hashlib.sha256(value).hexdigest()


def read(name):
    return json.loads((DIRECTORY / name).read_bytes())


def main():
    for name, expected in read('manifest.json').items():
        assert digest((DIRECTORY / name).read_bytes()) == expected, name
    evidence = read('evidence.json')
    archives = {}
    for name, expected in evidence['archive_members'].items():
        with tarfile.open(fileobj=io.BytesIO((DIRECTORY / name).read_bytes()), mode='r:gz') as archive:
            contents = {}
            for member in archive:
                assert member.isfile() and member.name not in contents
                assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
                contents[member.name] = archive.extractfile(member).read()
        assert {name: digest(value) for name, value in contents.items()} == expected
        archives[name] = contents
    overlay = archives['measured-source-overlay.tar.gz']
    for name, expected in evidence['measured_files'].items():
        assert digest(overlay[name]) == expected
        actual = (ROOT / name).read_bytes()
        assert digest(actual) == evidence['published_files'][name]
        assert ast.dump(ast.parse(actual), include_attributes=False) == ast.dump(
            ast.parse(overlay[name]), include_attributes=False)
    raw = json.loads(gzip.decompress((DIRECTORY / 'raw-timings.json.gz').read_bytes()))
    summary = read('summary.json')
    assert len(raw) == 36 and sum(map(len, raw.values())) == summary['raw_observations'] == 1440
    medians = {}
    for variant in ('kda', 'kda_matched', 'khdn'):
        for mode in ('fwd', 'bwd', 'fb', 'step'):
            values, peaks = [], []
            for trial in range(3):
                rows = raw[f'trial{trial}/{variant}/{mode}']
                assert [row['index'] for row in rows] == list(range(40))
                assert [row['phase'] for row in rows] == ['warmup'] * 10 + ['sample'] * 30
                assert all(math.isfinite(row['wall_ms']) and row['wall_ms'] > 0 for row in rows)
                assert all(row['input_slots'] == 16384 and row['global_valid_labels'] == 16376 for row in rows)
                values.append(statistics.median(row['wall_ms'] for row in rows[10:]))
                peaks.append(max(row['training_memory']['peak_allocated_bytes'] for row in rows[10:]))
            recorded = summary['rows'][variant][mode]
            assert values == recorded['trial_medians_ms']
            assert statistics.median(values) == recorded['median_ms']
            assert min(values) == recorded['min_ms'] and max(values) == recorded['max_ms']
            assert max(peaks) == recorded['max_peak_allocated_bytes']
            if mode == 'step':
                medians[variant] = values
    for variant in ('kda', 'kda_matched'):
        paired = [x / y for x, y in zip(medians['khdn'], medians[variant])]
        recorded = summary['paired_time_ratios'][variant]
        assert paired == recorded['trials'] and statistics.median(paired) == recorded['median']
        assert 1 / statistics.median(paired) == recorded['KHDN_throughput_fraction']
    print(json.dumps(dict(status='VERIFIED', observations=1440, source_AST='identical',
                          full_step_ms={name: statistics.median(v) for name, v in medians.items()},
                          paired_ratios=summary['paired_time_ratios']), indent=2))


if __name__ == '__main__':
    main()
