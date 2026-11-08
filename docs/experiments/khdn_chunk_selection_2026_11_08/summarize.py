# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Verify published records without treating incomplete numerical gates as passed."""

import gzip
import hashlib
import json
import statistics
import tarfile
from collections import defaultdict
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    records = json.loads(gzip.decompress((root / 'observations.json.gz').read_bytes()))
    binding = json.loads((root / 'source-binding.json').read_text())['experiments']
    with tarfile.open(root / 'result-receipts.tar.gz') as archive:
        receipts = {item.name: archive.extractfile(item).read() for item in archive if item.isfile()}
    for experiment, identity in binding.items():
        raw = receipts[experiment + '/result.json']
        assert hashlib.sha256(raw).hexdigest() == identity['result_sha256']
        assert json.loads(raw) == records[experiment]['result.json']
        receipt = records[experiment]['supervisor.json']
        assert receipt['terminal'] and receipt['reaped'] and receipt['group_drained']
        assert receipt['exit'] == 0 and not receipt['apps_after']
    samples = 0
    for experiment in ('26-chunk-selection', '27-chunk-long-endpoint', '31-dead-input-stores'):
        record = records[experiment]
        summary = record['result.json']
        for arm, result in summary['arms'].items():
            rows = record[arm + '/samples.json']
            assert len(rows) == 10
            for mode in ('fb', 'step'):
                group = [row for row in rows if row['mode'] == mode]
                assert [row['index'] for row in group] == list(range(5))
                assert [row['warmup'] for row in group] == [True, True, False, False, False]
                value = statistics.median(row['wall_ms'] for row in group if not row['warmup'])
                assert value == result['medians_ms'][mode]
            assert result['physical_optimizer_steps'] == 6
            samples += len(rows)
            print(experiment, arm, result['medians_ms'])
    for experiment in ('28-original-numeric-ablations', '30-preparation-storage-diagnosis'):
        counts = defaultdict(lambda: [0, 0, 0, 0])
        for row in records[experiment]['result.json']['rows']:
            key = row['variant'] + ':' + str(row['chunk'])
            counts[key][0] += 1
            counts[key][1] += sum(metric['elementwise_bad'] for metric in row['metrics'])
            counts[key][2] += sum(metric['rms'] > metric['rms_limit'] for metric in row['metrics'])
            counts[key][3] += not row['repeat_exact']
            assert all(metric['finite'] for metric in row['metrics'])
            assert not row['passed']
        print(experiment, 'cases/bad/RMS failures/repeat failures', dict(counts))
    boundary = records['29-raw-boundary']['result.json']['rows']
    assert len(boundary) == 5
    for row in boundary:
        assert row['raw_margin_fp32'] > 0
        if row['variant'] == 'diag32':
            assert row['passed'] and all(metric['passed'] for metric in row['metrics'])
        else:
            assert not row['passed'] and 'nonfinite candidate forward' in row['error']
    gate = records['31-dead-input-stores']['result.json']['gate']
    assert gate['all_passed'] and gate['baseline_exact'] and len(gate['rows']) == 8
    assert len(gate['poison_checks']) == 24
    assert samples == 90
    print('Verified 90 timing observations; original numerical failures remain open.')


if __name__ == '__main__':
    main()
