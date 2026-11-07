# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import json

import torch

from benchmarks.ops.registry import generate_inputs, get_op, list_ops


def test_phba_registration_and_packed_factory():
    assert 'chunk_phba' in list_ops()
    config = get_op('chunk_phba')
    assert config.import_path == 'fla.ops.phba'
    assert config.output_is_tuple and not config.skip_backward
    inputs = generate_inputs(config, B=1, T=8, H=2, D=16, M=16, lengths='3,5', device='cpu')
    assert inputs['length_plan'].lengths == (3, 5)
    assert inputs['gate'].shape == (1, 8, 2, 16)
    assert all(inputs[name].requires_grad for name in ('q', 'k', 'v', 'q_rope', 'k_rope', 'gate'))
    assert inputs['q'].dtype == torch.bfloat16


def test_phba_packed_benchmark_result_key():
    from benchmarks.ops.run import _make_result_key

    config = get_op('chunk_phba')
    keys = []
    for shape in config.default_shapes.values():
        row = {'op': 'chunk_phba', 'mode': 'fwdbwd', **shape, 'median_ms': 1.}
        keys.append(_make_result_key(json.loads(json.dumps(row))))
    assert len(set(keys)) == 2
