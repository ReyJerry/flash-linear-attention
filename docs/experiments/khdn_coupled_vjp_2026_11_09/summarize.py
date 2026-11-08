# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Verify retained raw samples, source hashes and research screen medians."""
import gzip
import hashlib
import json
import statistics
from pathlib import Path


def main():
    directory = Path(__file__).resolve().parent
    root = directory.parents[2]
    evidence = json.loads(gzip.decompress((directory / 'evidence.json.gz').read_bytes()))
    for name, digest in evidence['execution_source_sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    count = 0
    for name, experiment in evidence['experiments'].items():
        assert experiment['terminal'] and experiment['reaped'] and experiment['group_drained'], name
        if experiment['exit'] != 0:
            assert name == '84-coupled-source-owner'
            continue
        for arm, payload in experiment.get('arms', {}).items():
            samples, result = payload['samples'], payload['result']
            assert len(samples) == 14 and result['timed_optimizer_steps'] == 7
            for phase in ('fb', 'step'):
                steady = [x['wall_ms'] for x in samples if x['mode'] == phase and not x['warmup']]
                assert len(steady) == 5
                assert statistics.median(steady) == result['medians_ms'][phase], (name, arm, phase)
            print(f"{name}/{arm}: FB {result['medians_ms']['fb']:.3f} ms, step {result['medians_ms']['step']:.3f} ms")
            count += len(samples)
    print(f'Verified execution sources and {count} raw timing observations; final qualification remains incomplete.')


if __name__ == '__main__':
    main()
