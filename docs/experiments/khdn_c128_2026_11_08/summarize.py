# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Verify the published observations and source overlay without importing CUDA."""

import ast
import gzip
import hashlib
import json
import math
import statistics
import tarfile
from pathlib import Path


def main():
    directory = Path(__file__).resolve().parent
    repository = directory.parents[2]
    manifest = json.loads((directory / 'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest, name
    binding = json.loads((directory / 'source-binding.json').read_text())
    with tarfile.open(directory / 'measured-sources.tar.gz') as archive:
        sources = {}
        for member in archive:
            assert member.isfile() and member.name in binding['measured_files'] and member.name not in sources
            sources[member.name] = archive.extractfile(member).read()
    assert set(sources) == set(binding['measured_files'])
    assert not set(sources).intersection(binding['excluded_unrelated_files'])
    for name, value in sources.items():
        assert hashlib.sha256(value).hexdigest() == binding['measured_files'][name], name
    for name in binding['production_AST_equal_paths']:
        measured = ast.dump(ast.parse(sources[name].decode()), include_attributes=False)
        published = ast.dump(ast.parse((repository / name).read_text()), include_attributes=False)
        assert measured == published, name
    data = json.loads(gzip.decompress((directory / 'observations.json.gz').read_bytes()))
    timing = {}
    for arm, observed in data['performance'].items():
        samples, result = observed['samples'], observed['result']
        assert len(samples) == 10 and result['physical_optimizer_steps'] == 6
        timing[arm] = {}
        for mode in ('fb', 'step'):
            warm = [row for row in samples if row['mode'] == mode and row['warmup']]
            steady = [row['wall_ms'] for row in samples if row['mode'] == mode and not row['warmup']]
            assert len(warm) == 2 and len(steady) == 3 and all(math.isfinite(x) and x > 0 for x in steady)
            timing[arm][mode] = statistics.median(steady)
        assert timing[arm] == result['medians_ms']
    assert data['performance']['base']['qualification']['model']['initial_weights_sha256'] == (
        data['performance']['pair']['qualification']['model']['initial_weights_sha256'])
    operator = data['operator_gate']
    assert operator['status'] == 'M64_INTERVAL_GATE_PASS' and len(operator['rows']) == 8
    assert all(metric['passed'] for row in operator['rows'] for metric in row['metrics'])
    gradient = data['initial_gradient']
    params = gradient['gradients']
    assert len(params) == 459 and sum(row['elements'] for row in params.values()) == 1412912512
    aa = math.fsum(row['actual_squared'] for row in params.values())
    bb = math.fsum(row['reference_squared'] for row in params.values())
    dot = math.fsum(row['dot'] for row in params.values())
    cosine, ratio = dot / math.sqrt(aa * bb), math.sqrt(aa / bb)
    assert cosine == gradient['gradient_cosine'] and ratio == gradient['gradient_norm_ratio']
    assert cosine >= .99 and .9 <= ratio <= 1.1
    assert sum(row['old_budget_bad'] for row in params.values()) == gradient['old_budget_bad'] == 15447
    assert gradient['physical_optimizer_updates'] == 0 and gradient['weights_unchanged']
    cli = data['cli']
    assert cli['world_size'] == 1 and not cli['ddp_accepted'] and len(cli['global_samples']) == 16
    for key in ('all_parameter_gradient_errors', 'all_parameter_update_errors', 'resume_parameter_errors'):
        assert len(cli[key]) == 43
        assert all(row['max_abs'] == row['relative_rms'] == 0 for row in cli[key].values())
    assert all(row['positive_observation']['attested_before_optimizer_step'] for row in cli['update_guards'])
    failures = {}
    diagnostics = data['operator_diagnostics']['rows']
    assert len(diagnostics) == 12
    for route in ('c128', 'c16'):
        rows = [metric for row in diagnostics for metric in row['metrics'] if metric['route'] == route]
        assert len(rows) == 108 and all(row['finite'] for row in rows)
        failures[route] = {'elements': sum(row['elements'] for row in rows),
                           'elementwise': sum(row['bad'] for row in rows),
                           'rms': sum(not row['rms_pass'] for row in rows)}
        if route == 'c128':
            assert {row['source'] for row in rows if not row['rms_pass']} == {'dc'}
    assert failures['c128'] == {'elements': 5336112, 'elementwise': 133, 'rms': 4}
    assert failures['c16'] == {'elements': 5336112, 'elementwise': 0, 'rms': 0}
    for receipt in data['receipts'].values():
        supervisor = receipt['supervisor']
        assert supervisor['terminal'] and supervisor['reaped'] and supervisor['group_drained']
        assert not supervisor['apps_after'] and supervisor['archive'] == receipt['execution']['archive']
    summary = {'timing_ms': timing,
               'candidate_step_over_base': timing['pair']['step'] / timing['base']['step'],
               'candidate_step_over_matched_kda': timing['pair']['step'] / timing['kda']['step'],
               'initial_gradient_cosine': cosine, 'initial_gradient_norm_ratio': ratio,
               'retained_operator_failures': failures,
               'scope': 'Published hashes, source ASTs and recorded data verified; no new GPU or final acceptance.'}
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
