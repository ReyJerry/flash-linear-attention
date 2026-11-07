# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Real CPU policy/guard/autograd/checkpoint tests; simulated native execution, no KHDN/GPU numerical claim."""

import __future__

import ast
import contextlib
import gc
import importlib.util
import inspect
import json
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from test_khdn_bf16_fs6_harness import FLAGS
from test_khdn_bf16_fs6_harness import source_modules as source_modules
from torch.utils.checkpoint import checkpoint

from benchmarks import benchmark_hyperbolic_training as benchmark
from benchmarks import hyperbolic_benchmark_utils as utils
from benchmarks import hyperbolic_training_distributed as training


class PretendCuda(torch.Tensor):
    @property
    def is_cuda(self):
        return True


@pytest.fixture
def native_boundary(monkeypatch, request):
    # Release completed negative checkpoint graphs before replacing their fake module namespace.
    gc.collect()
    chunk, kernels = request.getfixturevalue('source_modules')
    chunk.__name__, kernels.__name__ = utils.KHDN_PREFIX + '.chunk', utils.KHDN_PREFIX + '.chunk_kernels'
    controller = {'root_calls': 0, 'options': []}

    class Jit:
        def __init__(self, fn):
            self.fn = fn

        def __getitem__(self, grid):
            return lambda *a, **kw: None

    for name, wrapper in list(vars(kernels).items()):
        if hasattr(wrapper, 'fn'):
            setattr(kernels, name, Jit(wrapper.fn))
    for name in utils.KHDN_STAGES:
        setattr(chunk, name, getattr(kernels, name))

    def launches(names, options, phase):
        for name in names:
            if controller.get('skip_stage') == name:
                continue
            kernel = getattr(chunk, name)
            flags = dict(C=16, GS=32, K=128, G=32, KD=128, VD=128, H=2, NG=4,
                         CARRY=False, MIXED=options.get('mixed', False), BF16_FS6=options.get('bf16_fs6', False))
            if name != 'state_fwd_kernel':
                flags['BF16_FS6'] = False
            if controller.get('bad_flag_phase') == phase and name == ('state_fwd_kernel' if phase == 'forward' else 'state_bwd_kernel'):
                flags.update(controller['bad_flags'])
            values = {key: flags.get(key, parameter.default if parameter.default is not inspect.Parameter.empty else None)
                      for key, parameter in inspect.signature(kernel.fn).parameters.items()}
            kernel[(1,)](**values)

    class Native(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, beta, ell, state, curve, options, telemetry):
            telemetry = {} if telemetry is None else telemetry
            ctx.options, ctx.telemetry = options, telemetry
            controller['options'].append(dict(options))
            ctx.save_for_backward(q)
            launches(utils.KHDN_STAGES[:4], options, 'forward')
            telemetry.update(precision_route='experimental_bf16_fs6', fallback_subblocks=0)
            if controller.get('wrong_context') and controller.get('first_ctx') is not None:
                ctx.telemetry = controller['first_ctx'].telemetry
            if controller.get('wrong_context'):
                controller.setdefault('first_ctx', ctx)
            return q.clone(), None

        @staticmethod
        def backward(ctx, do, ds):
            _saved = ctx.saved_tensors
            launches(utils.KHDN_STAGES[4:], ctx.options, 'backward')
            ctx.telemetry.update(backward_precision_route='experimental_bf16_fs6', backward_precision='device_float64',
                                 backward='manual_triton_chunk_source_vjp')
            return do, None, None, None, None, None, None, None, None

    class TorchRuntimeProxy:
        __version__ = '2.9.0+cu128'
        cuda = SimpleNamespace(get_device_capability=lambda d: (9, 0), device=lambda d: contextlib.nullcontext())

        def __getattr__(self, name):
            return getattr(torch, name)

    chunk.__dict__.update(torch=TorchRuntimeProxy(), math=math, triton=SimpleNamespace(__version__='3.5.0'), _Chunk=Native,
                          _exception_reason=lambda *a: None)

    def validate(*args):
        q, cu = args[0], args[8]
        bounds = cu.tolist() if cu is not None else [index * q.shape[1] for index in range(q.shape[0] + 1)]
        return bounds, 1.
    chunk._validate = validate
    names = {'chunk_key_gated_hyperbolic_delta_rule', '_source_alias_groups', '_reset_telemetry',
             '_qualification_reason', '_bf16_fs6_qualification_reason', '_small_model_geometry'}
    tree = ast.parse(Path(chunk.__file__).read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), chunk.__file__, 'exec',
                 flags=__future__.annotations.compiler_flag), vars(chunk))

    def reject(*args, **kwargs):
        raise AssertionError('unobserved substitute')
    package = ModuleType(utils.KHDN_PREFIX)
    package.chunk_key_gated_hyperbolic_delta_rule = chunk.chunk_key_gated_hyperbolic_delta_rule
    recurrent, naive = ModuleType(utils.KHDN_PREFIX + '.fused_recurrent'), ModuleType(utils.KHDN_PREFIX + '.naive')
    chunk._host_call = chunk._host_backward = chunk._strict_device_replay = reject
    recurrent.fused_recurrent_key_gated_hyperbolic_delta_rule = reject
    naive.naive_recurrent_key_gated_hyperbolic_delta_rule = reject
    for module in (chunk, kernels, package, recurrent, naive):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    class Attention(torch.nn.Module):
        def __init__(self, index):
            super().__init__()
            self.A_log, self.dt_bias = torch.nn.Parameter(torch.zeros(())), torch.nn.Parameter(torch.zeros(()))
            self.f_proj, self.precision_policy, self.index = torch.nn.Identity(), 'bf16_fs6', index
            self.group_size, self.curvature, self.decay_function = 32, .6, 'cpu_dispatch_fixture'

        def forward(self, value, cu=None):
            if controller.get('skip_layer') == self.index:
                return value
            q = (value * .001 + self.A_log * .001 + self.dt_bias * .001 + .01).as_subclass(PretendCuda)
            kwargs = dict(curvature=torch.tensor(.6), cu_seqlens=cu, chunk_size=16, group_size=32,
                          carry_mode='no_carry', check_inputs=False)
            policy = controller.get('policy', 'bf16_fs6')
            if controller.get('attack_root') == controller['root_calls']:
                policy = 'strict'
            if policy is not None:
                kwargs['precision_policy'] = policy
            output, _ = package.chunk_key_gated_hyperbolic_delta_rule(
                q, torch.ones_like(q) / math.sqrt(128), torch.ones_like(q) * .01,
                torch.full(q.shape[:3], .5), torch.full_like(q, -.1), **kwargs)
            return output

    class Model(torch.nn.Module):
        def __init__(self, use_checkpoint=True):
            super().__init__()
            self.layers = torch.nn.ModuleList([Attention(0), Attention(1)])
            self.use_checkpoint = use_checkpoint
            self.config = SimpleNamespace(num_hidden_layers=2, precision_policy='bf16_fs6', num_heads=2, head_dim=128)
            self.config.to_dict = lambda: {'precision_policy': 'bf16_fs6'}

        def forward(self, input_ids, labels, cu_seqlens=None, num_items_in_batch=1, **kwargs):
            controller['root_calls'] += 1
            value = input_ids.float()[..., None, None].expand(*input_ids.shape, 2, 128) * .001
            if controller.get('zero_calls'):
                return SimpleNamespace(loss=sum(p.sum() for p in self.parameters()) * 0.)
            for layer in self.layers:
                value = (checkpoint(layer, value, cu_seqlens, use_reentrant=False)
                         if self.use_checkpoint else layer(value, cu_seqlens))
            return SimpleNamespace(loss=value.sum() * labels[:, 1:].ne(-100).sum() / num_items_in_batch)

    binding = utils._khdn_fs6_source_binding(chunk, kernels)
    return SimpleNamespace(chunk=chunk, kernels=kernels, package=package, controller=controller, Model=Model, binding=binding)


def batch_args(checkpointed=True):
    args = benchmark.parse_arguments([*FLAGS, '--layers', '2'])
    args.no_layer_checkpoint = not checkpointed
    return args


def run_guard(core, *, stage='accumulated_update', checkpointed=True, supplied_batch=None):
    args = batch_args(checkpointed)
    model = core.Model(checkpointed)
    batches = [training.make_batch(args, 0, small=True)]
    records, steps = [], []
    with training.khdn_update_guard(args, 'khdn_no_carry', stage, records, model=model, batches=batches,
                                    source_binding=core.binding, arm='cpu_candidate'):
        training.weighted_loss(model, supplied_batch or batches[0], args, batches[0].valid_labels, 1).backward()
    steps.append('step')
    return records, steps


def test_fs6_kernel_contract_and_current_public_source_binding(request):
    modules = request.getfixturevalue('source_modules')
    binding = utils._khdn_fs6_source_binding(*modules)
    fixtures = Path(__file__).parent / 'fixtures'
    retained = json.loads((fixtures / 'khdn-runtime-source-binding.json').read_text())
    current = json.loads((fixtures / 'khdn-public-fs6-source-binding.json').read_text())
    assert binding == current
    for key in ('jit_functions', 'selected_forward_callee', 'backward_primal'):
        assert binding[key] == retained[key]
    assert len(binding['jit_functions']) == 23
    assert utils._binding_hash(retained) == 'b97aa379fc039ad484cdcc107f17be28013d41090e262ebea3c3b18e0a793d95'
    assert utils._binding_hash(binding) == '80a1f5488e2d4534bac07dbf75b5dabd4ce9b458e571775db51c35886c4f6581'
    assert all(inspect.getsource(getattr(modules[1], name).fn).startswith('@triton.jit\n')
               for name in binding['jit_functions'])


@pytest.mark.parametrize('stage', ['accumulated_update', 'uninterrupted_recovery', 'resumed_recovery'])
@pytest.mark.parametrize('checkpointed', [False, True])
def test_positive_public_native_counts_and_real_checkpoint_before_step(native_boundary, stage, checkpointed):
    records, steps = run_guard(native_boundary, stage=stage, checkpointed=checkpointed)
    proof = records[0]['positive_observation']
    assert steps == ['step'] and records[0]['guard']['events'] == []
    assert proof['original_graphs'] == 2 and proof['recompute_graphs'] == (2 if checkpointed else 0)
    assert proof['attested_before_optimizer_step'] and len(proof['root_calls']) == 1


@pytest.mark.parametrize('stage,policy', [('accumulated_update', None), ('uninterrupted_recovery', 'mixed_readout'),
                                          ('resumed_recovery', 'strict')])
def test_reproduces_parent_empty_events_false_pass_then_new_pre_step_rejection(native_boundary, stage, policy):
    core = native_boundary
    # Load the retained parent implementation, not an imitation of its negative sentinel.
    parent = Path(__file__).parent / 'fixtures/khdn_parent_exception_guard.py'
    spec = importlib.util.spec_from_file_location('retained_parent_utils_cpu', parent)
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    core.controller['policy'] = policy
    args, model = batch_args(False), core.Model(False)
    batch = training.make_batch(args, 0, small=True)
    with old.KHDNExceptionGuard('bf16_fs6') as guard:
        training.weighted_loss(model, batch, args, batch.valid_labels, 1).backward()
    assert guard.events == [] and not core.controller['options'][-1].get('bf16_fs6', False)
    records, steps = [], []
    with pytest.raises(AssertionError, match='precision|policy'):
        with training.khdn_update_guard(args, 'khdn_no_carry', stage, records, model=model, batches=[batch],
                                        source_binding=core.binding):
            training.weighted_loss(model, batch, args, batch.valid_labels, 1).backward()
        steps.append('step')
    assert not steps and 'failure' in records[0]


@pytest.mark.parametrize('attack', ['zero_calls', 'skip_layer', 'skip_stage', 'wrong_batch', 'public_stub', 'wrong_context',
                                    'forward_flag', 'backward_flag', 'strict', 'mixed_readout', 'missing_policy'])
def test_rejects_silent_substitution_missing_identity_and_wrong_flags_before_step(native_boundary, monkeypatch, attack):
    core = native_boundary
    if attack == 'zero_calls':
        core.controller['zero_calls'] = True
    elif attack == 'skip_layer':
        core.controller['skip_layer'] = 1
    elif attack == 'skip_stage':
        core.controller['skip_stage'] = 'group_af_fwd_kernel'
    elif attack == 'public_stub':
        monkeypatch.setattr(core.chunk._Chunk, 'apply', lambda *args: (args[0], None))
    elif attack == 'wrong_context':
        core.controller['wrong_context'] = True
    elif attack.endswith('_flag'):
        core.controller.update(bad_flag_phase=attack.split('_')[0], bad_flags={'MIXED': True})
    elif attack in ('strict', 'mixed_readout', 'missing_policy'):
        core.controller['policy'] = None if attack == 'missing_policy' else attack
    supplied = training.make_batch(batch_args(), 0, small=True) if attack == 'wrong_batch' else None
    with pytest.raises((AssertionError, RuntimeError)):
        run_guard(core, supplied_batch=supplied)


@pytest.mark.parametrize('root,stage,expected_updates', [(1, 'accumulated_update', 0), (17, 'accumulated_update', 0),
                                                         (33, 'uninterrupted_recovery', 2), (34, 'resumed_recovery', 3), (None, None, 4)])
def test_small_gate_rejects_each_arm_and_recovery_before_corresponding_optimizer(native_boundary, monkeypatch,
                                                                                 root, stage, expected_updates):
    core = native_boundary
    core.controller['attack_root'] = root
    counts = {'updates': 0}
    args = batch_args()
    original = training.clip_and_step

    def step(*a, **kw):
        original(*a, **kw)
        counts['updates'] += 1

    monkeypatch.setattr(benchmark, 'build_model', lambda *a, **kw: (core.Model(), {'configuration': {'cpu_fixture': True}}))
    monkeypatch.setattr(training, 'trace_route', lambda *a, **kw: {'source_binding': core.binding, 'cpu_fixture': True})
    monkeypatch.setattr(training, 'khdn_model_oracle_gate', lambda *a, **kw: {'cpu_fixture': True})
    monkeypatch.setattr(training, 'clip_and_step', step)

    class Runtime:
        device, rank, world_size = torch.device('cpu'), 0, 1
        wrap = staticmethod(lambda model: model)
        global_count = staticmethod(lambda batches: sum(batch.valid_labels for batch in batches))
    if root is None:
        result = training.small_gate(args, 'khdn_no_carry', Runtime(), benchmark)
        records = result['update_route_guards']
        assert [r['positive_observation']['original_graphs'] for r in records] == [32, 32, 2, 2]
        assert [r['positive_observation']['recompute_graphs'] for r in records] == [32, 32, 2, 2]
        assert [r['arm'] for r in records] == ['ddp', 'serial', 'ddp', 'ddp']
        assert counts['updates'] == 4
        return
    with pytest.raises(AssertionError) as caught:
        training.small_gate(args, 'khdn_no_carry', Runtime(), benchmark)
    records = caught.value.khdn_evidence['update_route_guards']
    assert records[-1]['stage'] == stage and 'failure' in records[-1]
    assert counts['updates'] == expected_updates
