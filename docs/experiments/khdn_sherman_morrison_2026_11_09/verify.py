# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Verify the archived research results without running accelerator work."""

import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(relative):
    return json.loads((ROOT / relative).read_text())


def verify():
    binding = read('source-binding.json')
    summary = read('summary.json')
    source = binding['measured_snapshot_sha256']
    assert summary['source'] == source
    assert binding['matched_tracked_file_count'] == len(binding['matched_tracked_files']) == 883
    for item in read('evidence-manifest.json')['files'] + binding['candidate_files']:
        relative = item.get('archive_path', item.get('path'))
        data = (ROOT / relative).read_bytes()
        assert hashlib.sha256(data).hexdigest() == item['sha256'], relative
        assert len(data) == item['bytes'], relative

    observations = updates = 0
    for shape in ('short', 'long'):
        raw = {}
        qualification = {}
        for arm in ('block', 'sm', 'kda_matched'):
            prefix = f'evidence/{shape}/{arm}'
            rows = read(f'{prefix}/samples.json')
            result = read(f'{prefix}/result.json')
            qualification[arm] = read(f'{prefix}/qualification.json')
            raw[arm] = rows
            assert len(rows) == 28
            assert result['parameters'] == 1_412_912_512
            assert result['physical_optimizer_steps'] == 7
            assert result['source_identity']['snapshot_sha256'] == source
            assert result['checkpoint_policy']['policy'] == 'save_mm'
            assert result['optimizer_backend'] == 'triton_fp32_adamw'
            assert result['fallback_calls'] == 0
            gradients = read(f'{prefix}/gradient-sha.json')
            assert len(gradients) == 459
            expected = summary['arms'][shape][arm]
            for mode in ('fwd', 'bwd', 'fb', 'step'):
                selected = [row for row in rows if row['mode'] == mode]
                assert [row['index'] for row in selected] == list(range(7))
                assert [row['warmup'] for row in selected] == [True, True, False, False, False, False, False]
                assert all(row['commit_inside_timed_window'] for row in selected)
                assert all(math.isfinite(row['wall_ms']) and row['wall_ms'] > 0 for row in selected)
                if mode == 'step':
                    assert all(row['commit_before_optimizer'] for row in selected)
                median = statistics.median(row['wall_ms'] for row in selected[2:])
                assert median == result['medians_ms'][mode] == expected['medians_ms'][mode]
            rate = 16_384_000 / result['medians_ms']['step']
            assert math.isclose(rate, expected['tokens_per_second'], rel_tol=1e-12)
            assert result['peak_allocated'] == expected['peak_allocated']
            assert result['peak_reserved'] == expected['peak_reserved']
            assert result['source_vjp_actual'] == expected['source_vjp_actual']
            trace = result['kernel_trace']
            if arm == 'sm':
                assert trace['sm_fwd_kernel'] == 48
                assert trace['sm_bwd_kernel'] == trace['sm_reduce_kernel'] == 24
                assert result['source_vjp_actual'] == 'sm_rank_one_manual'
            elif arm == 'block':
                assert trace['state_fwd_kernel'] == 48
                assert trace['coupled_sources_bwd'] == 96
            else:
                assert trace['kda_tilelang_bwd'] == 24
            observations += len(rows)
            updates += result['physical_optimizer_steps']
        assert qualification['block']['model'] == qualification['sm']['model']
        assert qualification['block']['batch_sha256'] == qualification['sm']['batch_sha256']
        for arm in ('sm', 'kda_matched'):
            for baseline, candidate in zip(raw['block'], raw[arm]):
                assert (baseline['mode'], baseline['index'], baseline['samples']) == (
                    candidate['mode'], candidate['index'], candidate['samples'],
                )
        arms = summary['arms'][shape]
        block_step = arms['block']['medians_ms']['step']
        sm_step = arms['sm']['medians_ms']['step']
        kda_step = arms['kda_matched']['medians_ms']['step']
        assert math.isclose((sm_step / block_step - 1) * 100, arms['sm_time_change_percent'], abs_tol=1e-10)
        assert math.isclose(kda_step / sm_step * 100, arms['sm_throughput_vs_kda_percent'], abs_tol=1e-10)
        assert sm_step > block_step
        print(f'{shape}: block {block_step:.3f} ms; SM {sm_step:.3f} ms; matched KDA {kda_step:.3f} ms')
    assert observations == summary['observations'] == 168
    assert updates == summary['physical_optimizer_updates'] == 42
    gate = read('gpu-gate.json')
    assert gate['source'] == source and gate['gpu_cases'] == 8
    assert gate['actual_tiny_optimizer_updates'] == 1
    assert gate['fp64_training_arithmetic'] is False
    print(f'Verified {observations} observations, {updates} optimizer updates and five exact archived source files.')
    print('Decision: retain the block training path; the SM candidate has no full-step gain.')


if __name__ == '__main__':
    verify()
