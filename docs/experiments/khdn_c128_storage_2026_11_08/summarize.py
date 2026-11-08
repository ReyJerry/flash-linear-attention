# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Verify recorded observations and explicit measured/published source boundaries on CPU."""

import ast
import gzip
import hashlib
import json
import math
import statistics
import tarfile
from pathlib import Path


class Normalize(ast.NodeTransformer):
    def visit_Module(self, node):
        node.body = [item for item in node.body if not isinstance(item, (ast.Import, ast.ImportFrom))]
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        unused = {'g', 'nu'} if node.name == 'shared_feedback_system_bwd' else {'rows'} if node.name == 'backward' else set()
        reduction_correction = node.name in ('erase', 'erase_vjp')

        class NormalizeBody(ast.NodeTransformer):
            def visit_Assign(self, item):
                if len(item.targets) == 1 and isinstance(item.targets[0], ast.Name) and item.targets[0].id in unused:
                    return None
                return self.generic_visit(item)

            def visit_Call(self, item):
                item = self.generic_visit(item)
                if (reduction_correction and isinstance(item.func, ast.Attribute) and item.func.attr == 'sum'
                        and isinstance(item.func.value, ast.Name) and item.func.value.id == 'tl' and item.args):
                    value = item.args[0]
                    if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == 'to'
                            and len(value.args) == 1 and ast.unparse(value.args[0]) == 'tl.float32'):
                        item.args[0] = value.func.value
                return item

        return NormalizeBody().visit(node)


def main():
    directory = Path(__file__).resolve().parent
    repository = directory.parents[2]

    def read(path):
        return json.loads(path.read_text())

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = read(directory/'manifest.json')
    for name, expected in manifest['files'].items():
        assert digest(directory/name) == expected, name
    binding = read(directory/'source-binding.json')
    for name, expected in binding['published_files'].items():
        assert digest(repository/name) == expected, name
    with tarfile.open(directory/'measured-sources.tar.gz') as archive:
        sources = {}
        for member in archive:
            assert member.isfile() and member.name in binding['measured_files'] and member.name not in sources
            sources[member.name] = archive.extractfile(member).read()
    assert set(sources) == set(binding['measured_files'])
    assert not set(sources).intersection(binding['excluded_unrelated_files'])
    for name, content in sources.items():
        assert hashlib.sha256(content).hexdigest() == binding['measured_files'][name]
    for measured, published in binding['mapping'].items():
        before = ast.dump(Normalize().visit(ast.parse(sources[measured].decode())), include_attributes=False)
        after = ast.dump(Normalize().visit(ast.parse((repository/published).read_text())), include_attributes=False)
        assert before == after, published
    kernel = ast.parse((repository/'fla/ops/key_gated_hyperbolic_delta_rule/chunk128_kernels.py').read_text())
    for name, count in (('erase', 1), ('erase_vjp', 2)):
        function = next(node for node in kernel.body if isinstance(node, ast.FunctionDef) and node.name == name)
        casts = [node for node in ast.walk(function) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == 'to'
                 and len(node.args) == 1 and ast.unparse(node.args[0]) == 'tl.float32']
        assert len(casts) == count
    data = json.loads(gzip.decompress((directory/'observations.json.gz').read_bytes()))
    times = {}
    for arm, value in data['performance'].items():
        assert len(value['samples']) == 10 and value['result']['physical_optimizer_steps'] == 6
        times[arm] = {}
        for mode in ('fb', 'step'):
            steady = [row['wall_ms'] for row in value['samples'] if row['mode'] == mode and not row['warmup']]
            assert len(steady) == 3 and all(math.isfinite(x) and x > 0 for x in steady)
            times[arm][mode] = statistics.median(steady)
        assert times[arm] == value['result']['medians_ms']
    assert data['performance']['base']['initial_weights_sha256'] == data['performance']['pair']['initial_weights_sha256']
    gate = data['operator_gate']
    assert len(gate['rows']) == 8
    for route in ('baseline', 'paired'):
        failed = [(row['case'], metric['source']) for row in gate['rows'] for metric in row['metrics']
                  if metric['route'] == route and not metric['passed']]
        assert failed == [(4, 'dbeta')]
    assert all(metric['finite'] and metric['bad'] == 0 for row in gate['rows'] for metric in row['metrics'])
    gradient = data['initial_gradient']
    records = gradient['gradients']
    assert len(records) == 459 and sum(row['elements'] for row in records.values()) == 1412912512
    actual, reference, dot = [math.fsum(row[key] for row in records.values())
                              for key in ('actual_squared', 'reference_squared', 'dot')]
    cosine, ratio = dot/math.sqrt(actual*reference), math.sqrt(actual/reference)
    assert cosine == gradient['gradient_cosine'] and ratio == gradient['gradient_norm_ratio']
    assert cosine >= .99 and .9 <= ratio <= 1.1
    assert sum(row['old_budget_bad'] for row in records.values()) == gradient['old_budget_bad'] == 18632
    assert gradient['physical_optimizer_updates'] == 0 and gradient['weights_unchanged']
    quality = data['quality']
    continuous, resumed = quality['continuous'], quality['resume']
    assert continuous['physical_optimizer_updates'] == 16 and resumed['physical_optimizer_updates'] == 8
    assert continuous['training'][8:] == resumed['training']
    assert continuous['checkpoints']['16'] == resumed['checkpoints']['16']
    assert continuous['final_rng_sha256'] == resumed['final_rng_sha256']
    assert resumed['restored_shared_state_sha256'] == continuous['checkpoints']['8']['shared_state_sha256']
    assert all(continuous['validation'][key] == value for key, value in resumed['validation'].items())
    paired = []
    for step in (0, 4, 8, 12, 16):
        sets = [quality['reference_validation'][str(step)], continuous['validation'][str(step)]]
        assert [(row['sample_id'], row['batch_sha256'], row['valid_labels']) for row in sets[0]] == [
            (row['sample_id'], row['batch_sha256'], row['valid_labels']) for row in sets[1]]
        means = [math.fsum(row['mean_ce']*row['valid_labels'] for row in rows)/sum(row['valid_labels'] for row in rows)
                 for rows in sets]
        paired.append(dict(step=step, reference_ce=means[0], candidate_ce=means[1],
                           ppl_difference_percent=100*math.expm1(means[1]-means[0])))
    assert paired == quality['result']['paired_validation'] and quality['result']['recovery_passed']
    assert not quality['result']['training_quality_accepted']
    public = data['public_gpu']
    assert public['result']['cli_exit'] == 1 and len(public['tests']) == 16
    assert sum(row['failed'] for row in public['tests']) == 12
    assert not any(row['error'] or row['skipped'] for row in public['tests'])
    for receipt in data['receipts'].values():
        assert receipt['terminal'] and receipt['reaped'] and receipt['group_drained'] and receipt['exit'] == 0
    print(json.dumps(dict(timing_ms=times, measured_step_over_matched_kda=times['pair']['step']/times['kda']['step'],
                          initial_gradient_cosine=cosine, initial_gradient_norm_ratio=ratio,
                          synthetic_ppl_differences=paired, public_numeric_failures=12,
                          current_gpu_revalidation='pending after FP32 reduction correction',
                          scope='Recorded evidence and explicit source boundaries verified; no new GPU qualification.'), indent=2))


if __name__ == '__main__':
    main()
