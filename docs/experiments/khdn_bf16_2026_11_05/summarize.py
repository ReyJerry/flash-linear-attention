# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Recompute the exported BF16 kernel-chain cost statistics without accelerator dependencies."""

import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path


def summarize(root: Path) -> dict:
    evidence = json.loads((root / 'evidence.json').read_bytes())
    raw = gzip.decompress((root / 'cost-samples.json.gz').read_bytes())
    if hashlib.sha256(raw).hexdigest() != evidence['cost_samples_uncompressed_sha256']:
        raise ValueError('cost sample hash mismatch')
    samples = json.loads(raw)
    if len(samples) != 120:
        raise ValueError('expected three arms with ten warmups and thirty steady samples each')
    result = {}
    for arm in ('direct', 'compensated', 'residual'):
        rows = [row for row in samples if row['arm'] == arm]
        if sorted(row['index'] for row in rows) != list(range(40)):
            raise ValueError('duplicate or missing observation')
        if any(row['kernel_calls'] != 6 or arm not in row['order'] for row in rows):
            raise ValueError('incorrect kernel count or interleaving record')
        if any(row['phase'] != ('warmup' if row['index'] < 10 else 'sample') for row in rows):
            raise ValueError('incorrect warmup/steady partition')
        values = sorted(row['wall_ms'] for row in rows if row['phase'] == 'sample')
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError('nonfinite or nonpositive latency')
        p95 = values[math.ceil(.95 * len(values)) - 1]
        median = statistics.median(values)
        recorded = evidence['cost']['result']['arms'][arm]
        if not math.isclose(median, recorded['wall_median_ms'], abs_tol=1e-10, rel_tol=0):
            raise ValueError('median differs from audited record')
        if not math.isclose(p95, recorded['wall_p95_ms'], abs_tol=1e-10, rel_tol=0):
            raise ValueError('p95 differs from audited record')
        result[arm] = dict(median_ms=median, p95_ms=p95, numerical_payload_pass=recorded['payload_numeric_pass'])
    return dict(
        scope=evidence['cost']['result']['scope'],
        arms=result,
        compensated_over_direct=result['compensated']['median_ms'] / result['direct']['median_ms'],
        residual_over_direct=result['residual']['median_ms'] / result['direct']['median_ms'],
        full24_zero_update=evidence['full24_zero_update'],
        tiny_ddp=evidence['tiny_ddp'],
        production_accepted=evidence['production_accepted'],
    )


if __name__ == '__main__':
    print(json.dumps(summarize(Path(__file__).resolve().parent), indent=2))
