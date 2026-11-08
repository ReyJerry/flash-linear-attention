# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Correctness gates and synchronized measurements for HDN operator benchmarks."""

import ast
import contextlib
import functools
import gc
import hashlib
import importlib
import inspect
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves


@dataclass
class Method:
    name: str
    run: object
    environment: dict
    group: str
    expected_kernel: str
    expected_flags: dict | None = None
    baseline: str | None = None


@contextlib.contextmanager
def environment(values):
    saved = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def percentile(values, quantile):
    if not values:
        raise ValueError('At least one timing sample is required.')
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    return ordered[lower] + (ordered[math.ceil(index)] - ordered[lower]) * (index - lower)


def summarize(samples):
    return {
        'median': statistics.median(samples), 'p95': percentile(samples, 0.95),
        'min': min(samples), 'max': max(samples), 'stdev': statistics.pstdev(samples),
    }


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(payload, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def validated_source_snapshot(root, manifest_path):
    """Verify the pack_snapshot.py identity and every declared byte before trusting archive provenance."""
    root, manifest_path = Path(root).resolve(), Path(manifest_path).resolve()
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    base, files, modes = (manifest[key] for key in ('base_commit', 'files', 'modes'))
    if not isinstance(base, str) or re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', base) is None:
        raise ValueError('source manifest base_commit must be a full declared commit hash')
    if not files or set(files) != set(modes):
        raise ValueError('source manifest file/mode inventory mismatch')
    identity = {'base_commit': base, 'files': files, 'modes': modes}
    snapshot = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if snapshot != manifest['snapshot_sha256']:
        raise ValueError('source manifest snapshot identity mismatch')
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or '.git' in relative.parts:
            raise ValueError(f'unsafe manifest path: {name}')
        path = root / relative
        if not path.resolve().is_relative_to(root):
            raise ValueError(f'source manifest path escapes root: {name}')
        if path.is_symlink():
            actual = 'symlink:' + os.readlink(path)
        elif path.is_file():
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            actual_mode = path.stat().st_mode & 0o777
            declared = modes[name] & 0o777
            if actual_mode not in (declared, declared & ~0o222):
                raise ValueError(f'source mode mismatch: {name}')
        else:
            raise ValueError(f'source manifest file missing: {name}')
        if actual != expected:
            raise ValueError(f'source content mismatch: {name}')
    actual_files = {str(path.relative_to(root)) for path in root.rglob('*')
                    if (path.is_file() or path.is_symlink()) and '__pycache__' not in path.parts
                    and '.git' not in path.relative_to(root).parts and path.resolve() != manifest_path}
    if actual_files != set(files):
        raise ValueError(f'source inventory mismatch: {sorted(actual_files.symmetric_difference(files))[:10]}')
    overlay = manifest.get('selected_working_files', {})
    for name, expected in overlay.items():
        if (expected is None and name in files) or (expected is not None and files.get(name) != expected):
            raise ValueError(f'declared source overlay mismatch: {name}')
    return {'source_kind': 'immutable_source_manifest', 'revision': None, 'declared_base_commit': base,
            'source_tree_sha256': snapshot, 'snapshot_sha256': snapshot,
            'source_manifest_sha256': hashlib.sha256(raw).hexdigest(), 'source_manifest': str(manifest_path),
            'declared_overlay_sha256': hashlib.sha256(json.dumps(overlay, sort_keys=True).encode()).hexdigest(),
            'working_tree_dirty': None, 'dirty_patch_sha256': None, 'untracked_file_sha256': {},
            'declared_file_count': len(files), 'git_history': 'not present or asserted by archive metadata',
            'benchmark_source_sha256': {name: digest for name, digest in files.items()
                                        if name.startswith('benchmarks/') and name.endswith('.py')}}


def source_metadata(root):
    manifest_path = os.environ.get('FLA_SOURCE_MANIFEST')
    if manifest_path:
        return validated_source_snapshot(root, manifest_path)
    location = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=root,
                              text=True, capture_output=True, check=False)
    if location.returncode or Path(location.stdout.strip()).resolve() != root.resolve():
        raise ValueError('source is not a Git checkout; set FLA_SOURCE_MANIFEST to its immutable snapshot.json')
    revision = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=root, text=True, capture_output=True, check=True,
    ).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=root, text=True, capture_output=True, check=True).stdout
    patch = subprocess.run(['git', 'diff', '--binary', 'HEAD'], cwd=root, capture_output=True, check=True).stdout
    untracked = subprocess.run(['git', 'ls-files', '--others', '--exclude-standard', '-z'],
                               cwd=root, capture_output=True, check=True).stdout
    sources = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(root.glob('benchmarks/*hyperbolic*.py'))}
    untracked_hashes = {os.fsdecode(name): hashlib.sha256((root / os.fsdecode(name)).read_bytes()).hexdigest()
                        for name in untracked.split(b'\0') if name and (root / os.fsdecode(name)).is_file()}
    tree = hashlib.sha256()
    for path in sorted([*root.glob('fla/**/*.py'), *root.glob('benchmarks/*.py')]):
        tree.update(str(path.relative_to(root)).encode())
        tree.update(path.read_bytes())
    return {'source_kind': 'git_checkout', 'source_tree_sha256': tree.hexdigest(),
            'dirty_patch_sha256': hashlib.sha256(patch).hexdigest(), 'untracked_file_sha256': untracked_hashes,
            'benchmark_source_sha256': sources, 'revision': revision, 'working_tree_dirty': bool(dirty)}


def metadata(arguments):
    root = Path(__file__).resolve().parents[1]
    result = {
        **source_metadata(root), 'arguments': vars(arguments).copy(),
        'command': sys.argv, 'python': platform.python_version(), 'torch': torch.__version__,
        'cuda_runtime': torch.version.cuda, 'strict_tolerances': True,
        'environment': {key: value for key, value in os.environ.items()
                        if key.startswith(('FLA_', 'TRITON_', 'NCCL_', 'TORCH_NCCL_', 'CUDA_'))},
    }
    result['arguments'] = {key: str(value) if isinstance(value, Path) else value for key, value in result['arguments'].items()}
    if torch.cuda.is_available():
        import triton
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        result.update({
            'triton': triton.__version__, 'gpu': properties.name, 'compute_capability': [properties.major, properties.minor],
            'device_total_memory': properties.total_memory, 'cuda_device': torch.cuda.current_device(),
            'gpu_uuid': str(getattr(properties, 'uuid', 'unavailable')),
            'nccl': torch.cuda.nccl.version() if torch.distributed.is_nccl_available() else None,
            'tf32_matmul_allowed': torch.backends.cuda.matmul.allow_tf32,
        })
    return result


def validate_arguments(parser, args):
    if min(args.batch, args.seq_len, args.heads, args.key_dim, args.value_dim) <= 0:
        parser.error('batch, seq-len, heads, key-dim and value-dim must be positive')
    if args.warmup < 1 or args.repeats < 2:
        parser.error('warmup must be >=1 and repeats must be >=2')
    if args.modes is None:
        args.modes = ['fwd_bwd'] if args.backward else ['fwd']
    if args.backward and args.modes != ['fwd_bwd']:
        parser.error('--backward is the compatibility alias for --modes fwd_bwd')
    if len(set(args.modes)) != len(args.modes):
        parser.error('modes must not contain duplicates')


class _ObservedKernel:
    def __init__(self, kernel, name, calls):
        self.kernel, self.name, self.calls = kernel, name, calls

    def __getattr__(self, name):
        return getattr(self.kernel, name)

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def observed(*args, **kwargs):
            flags = tuple(sorted((key, value) for key, value in kwargs.items()
                                 if key in ('FAST', 'HALF', 'BF16', 'SHARED_INVERSE', 'BT', 'BV', 'BK', 'C', 'NEED_DC')))
            self.calls[(self.name, flags)] += 1
            return launch(*args, **kwargs)
        return observed


class KernelTrace:
    """Observe actual entry-kernel launches during correctness, outside timing."""

    def __init__(self):
        self.calls, self.saved = Counter(), []

    def __enter__(self):
        prefix = 'fla.ops.hyperbolic_delta_rule.'
        for suffix in ('tensor_core_kernels', 'kernels', 'protected_kernels', 'protected_value_tile_kernels',
                       'protected_fused_kernels', 'protected_fused_backward', 'staged', 'protected_staged'):
            if importlib.util.find_spec(prefix + suffix) is not None:
                importlib.import_module(prefix + suffix)
        observed = {}
        for module_name, module in list(sys.modules.items()):
            if not module_name.startswith(prefix) or module is None:
                continue
            for name, value in list(vars(module).items()):
                is_entry = name.endswith('_kernel') or name in (
                    'hyperbolic_delta_tc_chunk_fwd', 'hyperbolic_delta_tc_chunk_bwd',
                )
                if is_entry and hasattr(value, '__getitem__') and hasattr(value, 'fn'):
                    if id(value) not in observed:
                        original_module = getattr(value.fn, '__module__', module_name)
                        observed[id(value)] = _ObservedKernel(value, f'{original_module}.{name}', self.calls)
                    self.saved.append((module, name, value))
                    setattr(module, name, observed[id(value)])
        return self

    def __exit__(self, exc_type, exc, traceback):
        for module, name, value in reversed(self.saved):
            setattr(module, name, value)

    def records(self):
        return [{'kernel': name, 'flags': dict(flags), 'launches': count}
                for (name, flags), count in sorted(self.calls.items())]


def attest(method, records):
    matching = [entry for entry in records if entry['kernel'].endswith(method.expected_kernel)]
    if not matching:
        raise AssertionError(f'{method.name}: required kernel {method.expected_kernel} was not launched')
    for key, value in (method.expected_flags or {}).items():
        if any(entry['flags'].get(key) != value for entry in matching):
            raise AssertionError(f'{method.name}: actual launch flag {key} differs from expected {value}')
    return records


def _float_outputs(outputs):
    return tuple(x for x in outputs if isinstance(x, torch.Tensor) and x.is_floating_point())


def evaluate(run, inputs, seed, trace=False):
    leaves = tuple(x.detach().clone().requires_grad_() for x in inputs)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    with KernelTrace() if trace else contextlib.nullcontext() as observed:
        returned = run(leaves)
        outputs = _float_outputs(returned)
        cotangents = tuple((torch.randint(0, 2, x.shape, generator=generator) * 2 - 1).to(x) for x in outputs)
        gradients = torch.autograd.grad(outputs, leaves, cotangents, allow_unused=False)
    result = {
        'outputs': tuple(x.detach().float().cpu() for x in outputs),
        'gradients': tuple(x.detach().float().cpu() for x in gradients),
        'output_dtypes': tuple(x.dtype for x in outputs),
        'input_dtypes': tuple(x.dtype for x in inputs),
        'discrete_outputs': tuple(x.detach().cpu() for x in returned
                                  if isinstance(x, torch.Tensor) and not x.is_floating_point()),
    }
    return result, observed.records() if trace else []


def compare(actual, expected, output_tolerance, gradient_tolerance, cast_reference=False):
    errors = {}
    if len(actual['discrete_outputs']) != len(expected['discrete_outputs']):
        raise AssertionError('Discrete return length differs')
    for actual_count, reference_count in zip(actual['discrete_outputs'], expected['discrete_outputs']):
        torch.testing.assert_close(actual_count, reference_count, atol=0, rtol=0)
    for category, tolerance, dtypes in (
        ('outputs', output_tolerance, actual['output_dtypes']),
        ('gradients', gradient_tolerance, actual['input_dtypes']),
    ):
        if len(actual[category]) != len(expected[category]):
            raise AssertionError(f'{category}: return length differs')
        for index, (value, reference, dtype) in enumerate(zip(actual[category], expected[category], dtypes)):
            if cast_reference:
                reference = reference.to(dtype).float()
            if not torch.isfinite(value).all() or not torch.isfinite(reference).all():
                raise AssertionError(f'{category}[{index}]: non-finite result')
            difference = (value - reference).abs()
            atol, rtol = tolerance
            errors[f'{category}_{index}'] = {
                'max_abs': difference.max().item() if difference.numel() else 0.0,
                'relative_rms': ((value - reference).square().mean().sqrt()
                                 / reference.square().mean().sqrt().clamp_min(1e-30)).item() if value.numel() else 0.0,
                'atol': atol, 'rtol': rtol,
            }
            torch.testing.assert_close(value, reference, atol=atol, rtol=rtol)
    return errors


def _prepare(method, inputs, mode):
    leaves = tuple(x.detach().requires_grad_() for x in inputs)
    if mode == 'bwd':
        outputs = _float_outputs(method.run(leaves))
        loss = sum(x.float().square().mean() for x in outputs)
        return lambda: torch.autograd.grad(loss, leaves, allow_unused=False)
    if mode == 'fwd':
        return lambda: method.run(leaves)

    def forward_backward():
        outputs = _float_outputs(method.run(leaves))
        loss = sum(x.float().square().mean() for x in outputs)
        return torch.autograd.grad(loss, leaves, allow_unused=False)
    return forward_backward


def _sample(method, inputs, mode):
    with environment(method.environment):
        operation = _prepare(method, inputs, mode)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        start.record()
        result = operation()
        end.record()
        torch.cuda.synchronize()
        wall = (time.perf_counter() - started) * 1000
        cuda = start.elapsed_time(end)
        del result, operation
    return {'wall_ms': wall, 'cuda_ms': cuda}


def _memory(method, inputs, mode):
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    before_allocated, before_reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    with environment(method.environment):
        operation = _prepare(method, inputs, mode)
        result = operation()
        torch.cuda.synchronize()
        allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        del result, operation
    return {
        'peak_allocated_bytes': allocated, 'peak_reserved_bytes': reserved,
        'starting_allocated_bytes': before_allocated, 'starting_reserved_bytes': before_reserved,
        'additional_peak_allocated_bytes': allocated - before_allocated,
        'measurement': 'separate untimed memory trial; CUDA cache emptied before trial only',
        'includes_forward_graph_and_setup': mode == 'bwd',
    }


def measure(methods, inputs_by_group, modes, warmup, repeats, tokens):
    rows = []
    for mode in modes:
        samples = {method.name: [] for method in methods}
        startup, warming = {}, {}
        for method in methods:
            inputs = inputs_by_group[method.group]
            startup[method.name] = _sample(method, inputs, mode)
            warming[method.name] = [_sample(method, inputs, mode) for _ in range(warmup)]
        for repeat in range(repeats):
            offset = repeat % len(methods)
            for method in methods[offset:] + methods[:offset]:
                samples[method.name].append(_sample(method, inputs_by_group[method.group], mode))
        for method in methods:
            raw = samples[method.name]
            wall, cuda = summarize([x['wall_ms'] for x in raw]), summarize([x['cuda_ms'] for x in raw])
            row = {
                'method': method.name, 'mode': mode, 'environment': method.environment, 'raw_samples': raw,
                'wall_ms': wall, 'cuda_ms': cuda, 'tokens_per_second_wall': tokens * 1000 / wall['median'],
                'tokens_per_second_cuda': tokens * 1000 / cuda['median'],
                'first_sample_including_possible_compilation': startup[method.name],
                'warmup_samples': warming[method.name],
                'memory': _memory(method, inputs_by_group[method.group], mode),
                'baseline_method': method.baseline,
                'timing_scope': ('backward only; forward graph and loss prepared before timing' if mode == 'bwd' else
                                 'training forward with graph creation' if mode == 'fwd' else
                                 'training forward, sum of mean-square floating outputs, and backward'),
            }
            if method.baseline:
                baseline = samples[method.baseline]
                row['wall_speedup'] = statistics.median(x['wall_ms'] for x in baseline) / wall['median']
                row['cuda_speedup'] = statistics.median(x['cuda_ms'] for x in baseline) / cuda['median']
            rows.append(row)
            print(f"{method.name} {mode}: wall {wall['median']:.3f} ms (p95 {wall['p95']:.3f}), "
                  f"CUDA {cuda['median']:.3f} ms, {row['tokens_per_second_wall']:.0f} tokens/s; "
                  f"allocated {row['memory']['peak_allocated_bytes'] / 2**30:.3f} GiB, "
                  f"reserved {row['memory']['peak_reserved_bytes'] / 2**30:.3f} GiB")
    return rows


KHDN_PREFIX = 'fla.ops.key_gated_hyperbolic_delta_rule'
KHDN_STAGES = ('common_fwd_kernel', 'group_fwd_kernel', 'group_af_fwd_kernel', 'state_fwd_kernel',
               'state_bwd_kernel', 'common_bwd_kernel')
KHDN_BF16_FORWARD_STAGES = ('common_base_fwd_kernel', 'common_interval_fwd_kernel', 'common_system_fwd_kernel',
                            'group_fwd_kernel', 'group_af_fwd_kernel', 'state_fwd_kernel', 'output_fwd_kernel')
KHDN_BF16_BACKWARD_STAGES = ('output_bwd_kernel', 'state_bwd_kernel', 'parameter_bwd_kernel',
                             'gauge_bwd_kernel', 'common_matrix_bwd_kernel', 'common_interval_bwd_kernel')


def _khdn_stages(precision_policy, backward=True, chunk_size=16, source_vjp='expanded'):
    if precision_policy == 'bf16_selective':
        from benchmarks.khdn_selective_trace import stages
        return stages(chunk_size, backward, source_vjp)
    if precision_policy in ('bf16', 'bf16_selective') and chunk_size == 128:
        from benchmarks.khdn_c128_trace import BACKWARD, FORWARD
        return FORWARD + (BACKWARD if backward else ())
    if precision_policy in ('bf16', 'bf16_selective'):
        return KHDN_BF16_FORWARD_STAGES + (KHDN_BF16_BACKWARD_STAGES if backward else ())
    return KHDN_STAGES if backward else KHDN_STAGES[:4]


def _replace_aliases(replacements, saved):
    for module_name, module in list(sys.modules.items()):
        if module is None or not module_name.startswith('fla.'):
            continue
        for name, value in list(vars(module).items()):
            replacement = replacements.get(id(value))
            if replacement is not None:
                saved.append((module, name, value))
                setattr(module, name, replacement)


def _restore_aliases(saved):
    for owner, name, value in reversed(saved):
        setattr(owner, name, value)


def _checkpoint_frames():
    frames = []
    frame = inspect.currentframe()
    try:
        while frame is not None:
            if frame.f_globals.get('__name__') == 'torch.utils.checkpoint':
                frames.append(frame.f_code.co_name)
            frame = frame.f_back
    finally:
        del frame
    return frames


def _tensor_descriptions(value):
    if isinstance(value, torch.Tensor):
        return {'shape': list(value.shape), 'dtype': str(value.dtype), 'device': str(value.device)}
    if isinstance(value, (tuple, list)):
        return [_tensor_descriptions(item) for item in value]
    return value if isinstance(value, (bool, int, float, str)) else None


def _compiler_without_comments(source):
    tokens = r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*[\s\S]*?\*/'
    return re.sub(tokens, lambda match: ' ' + '\n' * match[0].count('\n') if match[0].startswith('/') else match[0], source)


def _mlir_without_location(line):
    match = re.search(r'(?<!\S)loc\(', line)
    if match is None:
        return line.strip()
    depth = 1
    tail = line[match.end():]
    for token in re.finditer(r'"(?:\\.|[^"\\])*"|[()]', tail):
        if token[0].startswith('"'):
            continue
        depth += 1 if token[0] == '(' else -1
        if depth == 0:
            if tail[token.end():].strip():
                raise AssertionError('KHDN unrecognized trailing location syntax in compiled IR')
            return line[:match.start()].strip()
    raise AssertionError('KHDN unterminated/multiline location syntax in compiled IR')


def _parse_mlir_contraction_types(ir):
    source = _compiler_without_comments(ir)
    value = r'%[-\w.$]+(?:#\d+)?'
    encoding = r'#[\w.]+(?:<\{[^{}<>\n]*\}>)?'
    type_pattern = rf'(?:f\d+|bf16|(?:tensor|vector)<(?:\d+x)+(?:f\d+|bf16)(?:,\s*{encoding})?>)'
    types = []
    for line in source.splitlines():
        if not re.search(r'\b(?:arith\.(?:mulf|addf)|math\.fma)\b', line):
            continue
        line = _mlir_without_location(line)
        match = re.fullmatch(
            rf'(?:{value}\s*=\s*)?(arith\.(?:mulf|addf)|math\.fma)\s+'
            rf'({value}(?:\s*,\s*{value})+)\s*:\s*({type_pattern})', line,
        )
        if match is None or len(match[2].split(',')) != (3 if match[1] == 'math.fma' else 2):
            raise AssertionError('KHDN unrecognized operation/type syntax in compiled IR')
        dtype = re.fullmatch(r'(f\d+|bf16)', match[3])
        element = dtype[1] if dtype else re.match(r'(?:tensor|vector)<(?:\d+x)+(f\d+|bf16)', match[3])[1]
        types.append(element)
    return types


def _mlir_contraction_types(ir):
    types = _parse_mlir_contraction_types(ir)
    if not types or any(dtype != 'f64' for dtype in types):
        raise AssertionError('KHDN strict contraction operand/accumulator dtype missing or not FP64 in compiled IR')
    return types


def _strict_compiler_evidence(compiled):
    assembly = getattr(compiled, 'asm', {})
    ptx = assembly.get('ptx', '')
    if not ptx:
        raise AssertionError('KHDN strict launch has no compiled PTX evidence')
    ir = assembly.get('ttir') or assembly.get('ttgir', '')
    contractions = _mlir_contraction_types(ir)
    instructions = re.sub(r'"(?:\\.|[^"\\])*"', '""', _compiler_without_comments(ptx))
    opcodes = re.findall(r'\b(?:add|sub|mul|fma|mad|div|sqrt|rcp)(?:\.[a-z0-9]+)+\b', instructions)
    if any(not re.fullmatch(r'(?:f\d+(?:x\d+)?|bf16(?:x\d+)?|[sbu]\d+)', opcode.split('.')[-1])
           for opcode in opcodes):
        raise AssertionError('KHDN unrecognized arithmetic opcode type in compiled PTX')
    floating = [opcode for opcode in opcodes if re.search(r'\.(?:f\d+(?:x\d+)?|bf16(?:x\d+)?)\b', opcode)]
    tensor_cores = len(re.findall(r'\b(?:mma|wmma|wgmma|tcgen05)\.[\w.]+', instructions))
    if (not floating or any(re.findall(r'\.(f\d+(?:x\d+)?|bf16(?:x\d+)?)\b', opcode) != ['f64']
                            for opcode in floating) or tensor_cores):
        raise AssertionError('KHDN inspected PTX floating arithmetic must be FP64 with zero Tensor Core opcodes')
    return {'ptx_sha256': hashlib.sha256(ptx.encode()).hexdigest(), 'fp64_arithmetic_instructions': len(floating),
            'contraction_ir_sha256': hashlib.sha256(ir.encode()).hexdigest(),
            'multiply_add_operand_accumulator_dtypes': ['float64'], 'multiply_add_ir_operations': len(contractions),
            'ptx_inspected_floating_opcodes': dict(Counter(floating)), 'tensor_core_instructions': tensor_cores,
            'interpretation': 'parsed same-type arith.mulf/addf and math.fma clauses; independent PTX '
                              'add/sub/mul/fma/mad/div/sqrt/rcp floating opcode check; no claim about all PTX '
                              'instructions, conversions, library calls or per-source operation counts'}


def _khdn_kernel_source_hash(kernel):
    return hashlib.sha256(inspect.getsource(kernel.fn).encode()).hexdigest()


def _binding_hash(binding):
    return hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()


def _khdn_fs6_source_binding(chunk, kernels):
    functions = {name: inspect.getsource(value.fn) for name, value in vars(kernels).items() if hasattr(value, 'fn')}
    forward = ast.parse(functions['state_fwd_kernel'])
    selected = [node for node in ast.walk(forward) if isinstance(node, ast.If)
                and isinstance(node.test, ast.Name) and node.test.id == 'BF16_FS6']
    if len(selected) != 1 or not any(isinstance(node, ast.Call) and ast.unparse(node) == 'bf16_fs6(f, s, C, K, G)'
                                     for node in ast.walk(selected[0])):
        raise AssertionError('KHDN BF16_FS6 branch/callee source contract differs')
    backward = ast.parse(functions['state_bwd_kernel'])
    calls = [ast.unparse(node) for node in ast.walk(backward) if isinstance(node, ast.Call)]
    if ('load_matrix(SavedY, block, C, G)' not in calls or 'load_matrix(SavedS, block, K, G)' not in calls
            or any(call.startswith('bf16_fs6(') for call in calls)):
        raise AssertionError('KHDN strict saved-primal VJP source contract differs')
    return {'files': {name: hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest()
                      for name, module in (('chunk', chunk), ('chunk_kernels', kernels))},
            'jit_functions': {name: hashlib.sha256(source.encode()).hexdigest() for name, source in functions.items()},
            'selected_forward_callee': 'bf16_fs6', 'backward_primal': 'SavedS/SavedY; strict physical VJP'}


def _attest_fs6_sources(row, interrupted):
    sources = row.get('executed_sources', [])
    if len(sources) != 7 or any(value.get('shape', [])[-1:] != [128] for value in sources[:3]):
        raise AssertionError('KHDN bf16_fs6 unverified K/V geometry')
    dtype = sources[0].get('dtype')
    if dtype not in ('torch.float32', 'torch.bfloat16') or any(value.get('dtype') != dtype for value in sources[:3]):
        raise AssertionError('KHDN bf16_fs6 unqualified q/k/v dtype')
    if any(value.get('dtype') != 'torch.float32' for value in sources[3:]):
        raise AssertionError('KHDN bf16_fs6 control/state source dtype differs')
    if (not interrupted and row.get('output_all_finite') is not True
            or row['backward_calls'] and row.get('gradients_all_finite') is not True):
        raise AssertionError('KHDN bf16_fs6 missing finite output/gradient observations')
    fwd = next((launch for launch in row['launches'] if launch['kernel'] == 'state_fwd_kernel'), {})
    if fwd.get('y_diagnostics', {}).get('saved_state_y_fp64') is not True:
        raise AssertionError('KHDN bf16_fs6 saved state/Y must remain FP64')


def _khdn_bf16_source_binding(module, kernels, source_vjp='expanded'):
    c128 = module.__name__.endswith('.chunk128')
    selective = module.__name__.endswith('.bf16_selective')
    name = 'bf16_selective' if selective else 'chunk128' if c128 else 'bf16'
    files = {label: hashlib.sha256(Path(inspect.getfile(value)).read_bytes()).hexdigest()
             for label, value in ((name, module), (name + '_kernels', kernels))}
    if c128 or selective:
        if selective:
            from benchmarks.khdn_selective_trace import ARITHMETIC
        else:
            from benchmarks.khdn_c128_trace import ARITHMETIC
        root = Path(inspect.getfile(module)).resolve()
        for label, path in (('layer', root.parents[2]/'layers/key_gated_hyperbolic_deltanet.py'),
                            ('fused_prepare', root.parent/'fused_prepare.py'),
                            ('fused_prepare_kernels', root.parent/'fused_prepare_kernels.py')):
            files[label] = hashlib.sha256(path.read_bytes()).hexdigest()
    if c128 or selective:
        files['fixed_curvature'] = hashlib.sha256((root.parent/'fixed_curvature.py').read_bytes()).hexdigest()
    functions = {key: value for key, value in vars(kernels).items() if hasattr(value, 'fn')}
    if source_vjp != 'expanded':
        if not selective:
            raise AssertionError('Coupled source binding requires selective arithmetic')
        coupled = importlib.import_module(KHDN_PREFIX + '.bf16_coupled_kernels')
        files['bf16_coupled_kernels'] = hashlib.sha256(Path(inspect.getfile(coupled)).read_bytes()).hexdigest()
        for key, value in vars(coupled).items():
            if hasattr(value, 'fn'):
                if key in functions and functions[key] is not value:
                    raise AssertionError('Conflicting coupled kernel/helper ownership')
                functions[key] = value
    return {'files': files,
            'jit_functions': {key: _khdn_kernel_source_hash(value) for key, value in functions.items()},
            'source_vjp': source_vjp,
            'selected_arithmetic': ARITHMETIC if c128 or selective else 'bf16_output_source_single_c16',
            'backward_primal': 'BF16 SavedS/SavedY' if c128 else 'FP32 SavedS/SavedY'}


def _attest_bf16_sources(row, interrupted):
    sources = row.get('executed_sources', [])
    if len(sources) != 7 or any(value.get('shape', [])[-1:] != [128] for value in sources[:3]):
        raise AssertionError('KHDN bf16 training trace requires K=V128')
    if any(value.get('dtype') not in ('torch.float32', 'torch.bfloat16') for value in sources[:3]):
        raise AssertionError('KHDN bf16 source dtype differs')
    if any(value.get('dtype') != 'torch.float32' for value in sources[3:]):
        raise AssertionError('KHDN bf16 training controls/state must be FP32')
    if (not interrupted and row.get('output_all_finite') is not True
            or row['backward_calls'] and row.get('gradients_all_finite') is not True):
        raise AssertionError('KHDN bf16 finite output/gradient observations missing')
    forward = next((launch for launch in row['launches'] if launch['kernel'] == 'state_fwd_kernel'), {})
    c128 = row['requested']['chunk_size'] == 128 and row['requested'].get('precision_policy', 'bf16') != 'bf16_selective'
    expected_primal = 'saved_state_y_bf16' if c128 else 'saved_state_y_fp32'
    if forward.get('y_diagnostics', {}).get(expected_primal) is not True:
        raise AssertionError('KHDN bf16 saved primal state/Y precision differs')
    flags = forward.get('arguments', {})
    for name, count in (('Starts', len(row['expected_chunk_lengths'])), ('Ends', len(row['expected_chunk_lengths'])),
                        ('SeqChunks', len(row['sequence_bounds']))):
        if flags.get(name, {}).get('shape') != [count] or flags.get(name, {}).get('dtype') != 'torch.int64':
            raise AssertionError('KHDN bf16 chunk/sequence metadata shape or dtype differs')
    for name in ('P', 'Gp', 'SavedS', 'SavedY', 'Carry', 'SavedU', 'SavedR', 'SavedT', 'SavedW', 'SavedZP'):
        buffers = flags.get(name)
        buffers = buffers if isinstance(buffers, list) else [buffers]
        bf16_indices = set()
        if c128:
            if name == 'P':
                bf16_indices = set(range(18)) - {2, 3, 4}
            elif name == 'Gp':
                bf16_indices = {0, 1, 3, 4}
            elif name in ('SavedS', 'SavedY', 'SavedU'):
                bf16_indices = {0}
        if not buffers or any(not isinstance(value, dict) or value.get('dtype') != (
                'torch.bfloat16' if index in bf16_indices else 'torch.float32') for index, value in enumerate(buffers)):
            raise AssertionError('KHDN bf16 prepared/saved buffer precision differs')
    if c128 or row['requested'].get('precision_policy') == 'bf16_selective':
        output = next((launch for launch in row['launches'] if launch['kernel'] == 'output_fwd_kernel'), {})
        expected_dtype = 'torch.bfloat16' if c128 else 'torch.float32'
        if output.get('arguments', {}).get('SavedRO', {}).get('dtype') != expected_dtype:
            raise AssertionError('KHDN saved output residual precision differs')
        if not c128 and any(flags['P'][index]['shape'][0] != 0 for index in (0, 14, 15, 22)):
            raise AssertionError('Selective BF16 unexpectedly materializes removed preparation buffers')
        if not c128:
            needs_curvature = row.get('source_requires_grad', [])[6:7]
            if (len(needs_curvature) != 1 or output['arguments'].get('NEED_DC') is not needs_curvature[0]
                    or row['telemetry'].get('readout_compensated') is not needs_curvature[0]):
                raise AssertionError('Selective readout precision differs from the actual curvature-gradient requirement')


def _compact_khdn_assembly(evidence):
    # Raw compiler text is stored once; each launch keeps both independently checked hashes.
    import copy
    result = copy.deepcopy(evidence)
    store = {}
    for row in result['operator_calls']:
        for launch in row['launches']:
            inventory = launch['compiled_precision_inventory']
            key = inventory['ttir_sha256'] + ':' + inventory['ptx_sha256']
            raw = inventory.pop('raw_assembly')
            if key in store and store[key] != raw:
                raise AssertionError('KHDN inconsistent raw compiler evidence')
            store[key] = raw
            launch['compiled_assembly_ref'] = key
    result['compiled_assemblies'] = store
    return result


def reparse_khdn_trace(evidence, *, expected_source_binding):
    """Reparse a serialized BF16-family trace against a separately frozen source binding."""
    import copy
    policy = evidence.get('precision_requested', 'bf16_fs6')
    if policy not in ('bf16_fs6', 'bf16', 'bf16_selective'):
        raise AssertionError('KHDN serialized precision policy is unsupported')
    if evidence.get('source_binding') != expected_source_binding:
        raise AssertionError('KHDN frozen source binding differs')
    calls = copy.deepcopy(evidence['operator_calls'])
    for row in calls:
        for launch in row['launches']:
            key = launch['compiled_assembly_ref']
            launch['compiled_precision_inventory']['raw_assembly'] = evidence['compiled_assemblies'][key]
    return attest_khdn(calls, evidence.get('execution_chunk_size', 16), 32, 'no_carry', evidence['backward_calls'], evidence['expected_layers'],
                       policy, expected_source_binding, evidence['expected_checkpoint'], evidence.get('source_vjp', 'expanded'))


class _ObservedKHDNKernel:
    def __init__(self, kernel, trace):
        self.kernel, self.trace = kernel, trace
        self.contractions = _khdn_contraction_sites(kernel, importlib.import_module(kernel.fn.__module__))

    def __getattr__(self, name):
        return getattr(self.kernel, name)

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def observed(*args, **kwargs):
            current = getattr(self.trace.local, 'current', None)
            if current is None:
                raise AssertionError('KHDN kernel launched outside observed operator/autograd entry')
            names = inspect.signature(self.kernel.fn).parameters
            values = dict(zip(names, args))
            values.update({key: value for key, value in kwargs.items() if key in names})
            compiled = launch(*args, **kwargs)
            from benchmarks.khdn_compiler_precision import compiled_precision_inventory
            row = {'kernel': self.kernel.fn.__name__, 'phase': current[1],
                   'arguments': {key: _tensor_descriptions(value) for key, value in values.items()},
                   'compiler': {'raw_assembly': {name: getattr(compiled, 'asm', {}).get(name) for name in ('ttir', 'ptx')}},
                   'kernel_source_sha256': _khdn_kernel_source_hash(self.kernel),
                   'thread': threading.get_ident(), 'gemm_source_sites': self.contractions}
            if self.trace.precision_policy in ('bf16_fs6', 'bf16', 'bf16_selective'):
                row['source_binding_sha256'] = _binding_hash(self.trace.source_binding)
                row['selected_fs_callee'] = ('bf16_fs6' if values.get('BF16_FS6') is True else None)
            current[0]['launches'].append(row)
            inventory = compiled_precision_inventory(compiled, allow_data_movement=(
                self.trace.chunk_size == 128 and row['kernel'] == 'gauge_collect'))
            row['compiled_precision_inventory'] = inventory
            row['compiler'] = ({'ptx_sha256': inventory['ptx_sha256'], 'contraction_ir_sha256': inventory['ttir_sha256']}
                               if self.trace.precision_policy in ('bf16', 'bf16_selective') or values.get('MIXED') is True or values.get('BF16_FS6') is True
                               else _strict_compiler_evidence(compiled))
            if row['kernel'] == 'group_fwd_kernel':
                row['prepared_diagnostics'] = _khdn_buffer_diagnostics(values)
            if row['kernel'] == 'state_fwd_kernel':
                row['y_diagnostics'] = _khdn_y_diagnostics(values, self.trace.precision_policy)
            if row['kernel'] == 'carry_fwd_kernel':
                nxt, heads = values['Next'], values['H']
                terminal = (nxt < 0).repeat_interleave(heads)
                row['terminal_rows'] = int(terminal.sum())
                row['future_carry_tail_zero'] = all(bool((buffer[terminal] == 0).all()) for buffer in values['Carry'])
                row['observed_edges'] = int((nxt >= 0).sum()) * heads
            return compiled
        return observed


class KHDNTrace:
    """Retain real operator telemetry through autograd and observe its compiled launches, outside timing."""

    kernel_observer = _ObservedKHDNKernel

    def __init__(self, chunk_size, group_size, carry_mode, expected_backward_calls=1, precision_policy='strict',
                 source_vjp='expanded'):
        self.chunk_size, self.group_size, self.carry_mode = chunk_size, group_size, carry_mode
        if precision_policy not in ('strict', 'mixed_readout', 'bf16_fs6', 'bf16', 'bf16_selective'):
            raise ValueError('unknown KHDN precision policy')
        self.precision_policy = precision_policy
        if source_vjp not in ('expanded', 'coupled', 'coupled_state', 'coupled_fused') or (
                source_vjp != 'expanded' and (precision_policy != 'bf16_selective' or chunk_size != 64)):
            raise ValueError('Unsupported KHDN source VJP trace geometry')
        self.source_vjp = source_vjp
        if precision_policy == 'bf16_fs6' and (chunk_size, group_size, carry_mode) != (16, 32, 'no_carry'):
            raise ValueError('bf16_fs6 trace requires C16/g32/no_carry')
        allowed_chunks = (16, 32, 64, 128) if precision_policy == 'bf16_selective' else (16, 128)
        if precision_policy in ('bf16', 'bf16_selective') and (chunk_size not in allowed_chunks or (group_size, carry_mode) != (32, 'no_carry')):
            raise ValueError('bf16 training trace requires C16 or C128/g32/no_carry')
        self.expected_checkpoint = None
        self.expected_backward_calls = expected_backward_calls
        self.saved, self.calls, self.by_telemetry = [], [], {}
        self.hooks, self.layer_names = [], []
        self.local = threading.local()

    def bind_model(self, model):
        if model.config.precision_policy != self.precision_policy:
            raise AssertionError('KHDN model precision policy differs from request')
        if getattr(model.config, 'source_vjp', 'expanded') != self.source_vjp:
            raise AssertionError('KHDN model source VJP differs from request')
        attentions = [(name, attention) for name, attention in model.named_modules()
                      if hasattr(attention, 'f_proj') and hasattr(attention, 'A_log')]
        if any(attention.precision_policy != self.precision_policy
               or getattr(attention, 'source_vjp', 'expanded') != self.source_vjp for _, attention in attentions):
            raise AssertionError('KHDN layer precision policy differs from request')
        if len(attentions) != self.expected_backward_calls:
            raise AssertionError('KHDN trained-layer count differs from model configuration')
        for name, attention in attentions:
            self.layer_names.append(name)

            def before(module, args, layer=name):
                self.local.layer = layer

            def after(module, args, output):
                self.local.layer = 'operator'
            self.hooks.extend((attention.register_forward_pre_hook(before),
                               attention.register_forward_hook(after, always_call=True)))
        return self

    def _entry(self, function):
        @functools.wraps(function)
        def observed(*args, **kwargs):
            bound = inspect.signature(function).bind(*args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            telemetry = values.get('telemetry')
            if telemetry is None:
                telemetry = {}
                bound.arguments['telemetry'] = telemetry
            elif telemetry:
                raise AssertionError('KHDN telemetry must be fresh, not prepopulated')
            q, ell, v, k = (values[key] for key in ('q', 'log_decay', 'v', 'k'))
            cu = values['cu_seqlens']
            bounds = ([i * q.shape[1] for i in range(q.shape[0] + 1)] if cu is None else cu.detach().cpu().tolist())
            lengths = [min(self.chunk_size, end - pos) for start, end in zip(bounds, bounds[1:])
                       for pos in range(start, end, self.chunk_size)]
            chunks_per_sequence = [(end - start + self.chunk_size - 1) // self.chunk_size
                                   for start, end in zip(bounds, bounds[1:])]
            curvature = values['curvature']
            c_value = float(curvature.detach()) if isinstance(curvature, torch.Tensor) else float(curvature)
            diagnostic_dtype = torch.float32 if self.precision_policy in ('bf16', 'bf16_selective') else torch.float64
            n = k.detach().to(diagnostic_dtype).square().sum(-1)
            delta = c_value / values['group_size'] * v.detach().to(diagnostic_dtype).reshape(
                *v.shape[:-1], -1, values['group_size']).square().sum(-1) / n[..., None]
            state = values['initial_state']
            radius2 = (0. if state is None else float((c_value / values['group_size'] * state.detach().to(diagnostic_dtype).reshape(
                *state.shape[:-1], -1, values['group_size']).square().sum((-3, -1))).max()))
            row = {'requested': {key: values[key] for key in ('chunk_size', 'group_size', 'carry_mode', 'check_inputs', 'precision_policy', 'source_vjp')},
                   'sources': {key: _tensor_descriptions(values[key]) for key in
                               ('q', 'k', 'v', 'beta', 'log_decay', 'curvature', 'initial_state')},
                   'sequence_bounds': bounds, 'expected_chunk_lengths': lengths,
                   'expected_edges': sum(max(0, count - 1) for count in chunks_per_sequence) * q.shape[2],
                   'heads': q.shape[2], 'launches': [], 'telemetry': telemetry,
                   'forward_thread': threading.get_ident(), 'backward_calls': 0,
                   'checkpoint_frames': _checkpoint_frames(), 'layer': getattr(self.local, 'layer', 'operator'),
                   'consumed': {'curvature': c_value, 'log_decay_min': float(ell.detach().min()),
                                'log_decay_max': float(ell.detach().max()),
                                'all_finite': all(bool(torch.isfinite(values[key]).all())
                                                  for key in ('q', 'k', 'v', 'beta', 'log_decay')),
                                'key_norm2_min': float(n.min()), 'key_norm2_max': float(n.max()),
                                'feedback_delta_max': float(delta.max()), 'initial_state_radius2_max': radius2,
                                'beta_min': float(values['beta'].detach().min()),
                                'beta_max': float(values['beta'].detach().max())}}
            self.calls.append(row)
            self.by_telemetry[id(telemetry)] = row
            previous = getattr(self.local, 'current', None)
            self.local.current = (row, 'forward')
            try:
                if values['precision_policy'] != self.precision_policy or values.get('source_vjp', 'expanded') != self.source_vjp:
                    raise AssertionError('KHDN operator entry precision policy differs from request')
                returned = function(*bound.args, **bound.kwargs)
                row['returned'] = _tensor_descriptions(returned)
                row['forward_complete'] = True
                if self.precision_policy in ('bf16_fs6', 'bf16', 'bf16_selective'):
                    row['output_all_finite'] = all(value is None or bool(torch.isfinite(value).all()) for value in returned)
                return returned
            except BaseException as error:
                row['forward_interrupted'] = type(error).__name__
                raise
            finally:
                self.local.current = previous
        return observed

    def __enter__(self):
        module = importlib.import_module(KHDN_PREFIX + '.chunk')
        function = module.chunk_key_gated_hyperbolic_delta_rule
        replacements = {id(function): self._entry(function)}
        native_bf16 = self.precision_policy in ('bf16', 'bf16_selective')
        native_name = 'bf16_selective' if self.precision_policy == 'bf16_selective' else 'chunk128' if self.chunk_size == 128 else 'bf16'
        kernels = importlib.import_module(KHDN_PREFIX + ('.' + native_name + '_kernels' if native_bf16 else '.chunk_kernels'))
        arithmetic_module = importlib.import_module(KHDN_PREFIX + '.' + native_name) if native_bf16 else module
        autograd_class = arithmetic_module._ChunkBF16 if native_bf16 else module._Chunk
        self.kernel_module = kernels
        if self.precision_policy == 'bf16_fs6':
            self.source_binding = _khdn_fs6_source_binding(module, kernels)
        elif native_bf16:
            self.source_binding = _khdn_bf16_source_binding(arithmetic_module, kernels, self.source_vjp)
        self.kernel_owners = {name: value for name, value in vars(kernels).items() if hasattr(value, 'fn')}
        if self.source_vjp != 'expanded':
            coupled = importlib.import_module(KHDN_PREFIX + '.bf16_coupled_kernels')
            self.kernel_owners.update({name: value for name, value in vars(coupled).items() if hasattr(value, 'fn')})
        for name in dict.fromkeys((*_khdn_stages(self.precision_policy, chunk_size=self.chunk_size,
                                                 source_vjp=self.source_vjp), 'carry_fwd_kernel')):
            kernel = self.kernel_owners[name]
            replacements[id(kernel)] = self.kernel_observer(kernel, self)
        _replace_aliases(replacements, self.saved)
        original_forward, original_backward = vars(autograd_class)['forward'], vars(autograd_class)['backward']

        @functools.wraps(original_forward.__func__)
        def forward(ctx, *args, **kwargs):
            previous = self.local.current
            row = previous[0]
            if args[-1] is not row['telemetry']:
                nested = {'telemetry': args[-1], 'launches': [], 'backward_calls': 0}
                row.setdefault('strict_replay_calls', []).append(nested)
                row = nested
                self.by_telemetry[id(args[-1])] = row
            self.local.current = (row, 'forward')
            if self.precision_policy in ('bf16_fs6', 'bf16', 'bf16_selective'):
                row['autograd_context_id'] = id(ctx)
            row['executed_options'] = dict(args[-2])
            row['executed_sources'] = [_tensor_descriptions(value) for value in args[:7]]
            if self.precision_policy in ('bf16', 'bf16_selective'):
                row['source_requires_grad'] = list(ctx.needs_input_grad[:7])
            try:
                return original_forward.__func__(ctx, *args, **kwargs)
            finally:
                self.local.current = previous

        @functools.wraps(original_backward.__func__)
        def backward(ctx, *args, **kwargs):
            row = self.by_telemetry.get(id(ctx.telemetry))
            if row is None:
                raise AssertionError('KHDN backward lost its forward telemetry identity')
            if self.precision_policy in ('bf16_fs6', 'bf16', 'bf16_selective') and row.get('autograd_context_id') != id(ctx):
                raise AssertionError('KHDN backward context differs from its public forward graph')
            previous = getattr(self.local, 'current', None)
            self.local.current = (row, 'backward')
            row['backward_options'] = dict(ctx.opts if native_bf16 else ctx.options)
            row['backward_thread'] = threading.get_ident()
            try:
                returned = original_backward.__func__(ctx, *args, **kwargs)
                row['backward_calls'] += 1
                row['gradient_dtypes'] = _tensor_descriptions(returned[:7])
                if self.precision_policy in ('bf16_fs6', 'bf16', 'bf16_selective'):
                    row['gradients_all_finite'] = all(value is None or bool(
                        torch.isfinite(value).all()) for value in returned[:7])
                return returned
            finally:
                self.local.current = previous
        self.saved.extend(((autograd_class, 'forward', original_forward), (autograd_class, 'backward', original_backward)))
        autograd_class.forward, autograd_class.backward = staticmethod(forward), staticmethod(backward)
        return self

    def __exit__(self, exc_type, error, tb):
        _restore_aliases(self.saved)
        for hook in self.hooks:
            hook.remove()
        if error is not None:
            error.khdn_evidence = self.calls

    def attest(self):
        try:
            if self.precision_policy == 'bf16_fs6':
                current = _khdn_fs6_source_binding(importlib.import_module(KHDN_PREFIX + '.chunk'), self.kernel_module)
                if current != self.source_binding:
                    raise AssertionError('KHDN source/helper binding changed during trace')
            elif self.precision_policy in ('bf16', 'bf16_selective'):
                current = _khdn_bf16_source_binding(importlib.import_module(
                    KHDN_PREFIX + ('.bf16_selective' if self.precision_policy == 'bf16_selective' else '.chunk128' if self.chunk_size == 128 else '.bf16')), self.kernel_module, self.source_vjp)
                if current != self.source_binding:
                    raise AssertionError('KHDN bf16 source/helper binding changed during trace')
            for row in self.calls:
                for launch in row['launches']:
                    kernel = self.kernel_owners[launch['kernel']]
                    if launch['kernel_source_sha256'] != _khdn_kernel_source_hash(kernel):
                        raise AssertionError('KHDN kernel source changed during trace')
            evidence = attest_khdn(self.calls, self.chunk_size, self.group_size, self.carry_mode,
                                   self.expected_backward_calls, self.layer_names or None, self.precision_policy,
                                   getattr(self, 'source_binding', None), self.expected_checkpoint, self.source_vjp)
        except BaseException as error:
            error.khdn_evidence = self.calls
            raise
        if self.precision_policy in ('bf16_fs6', 'bf16', 'bf16_selective'):
            evidence = _compact_khdn_assembly(evidence)
        else:
            evidence['source_binding'] = {name: hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest()
                                          for name in ('chunk', 'chunk_kernels')
                                          for module in (importlib.import_module(KHDN_PREFIX + '.' + name),)}
        return evidence


def attest_khdn(calls, chunk_size, group_size, carry_mode, expected_backward_calls=1, expected_layers=None,
                precision_policy='strict', source_binding=None, expected_checkpoint=None, source_vjp='expanded'):
    from benchmarks.khdn_compiler_precision import attest_compiled_policy
    if precision_policy not in ('strict', 'mixed_readout', 'bf16_fs6', 'bf16', 'bf16_selective'):
        raise AssertionError('unknown KHDN precision policy')
    mixed = precision_policy == 'mixed_readout'
    fs6 = precision_policy == 'bf16_fs6'
    bf16 = precision_policy in ('bf16', 'bf16_selective')
    arithmetic = 'bf16_fs6_projection' if fs6 else 'strict_readout_tf32x3_state_products' if mixed else 'device_float64'
    selective = precision_policy == 'bf16_selective'
    c128 = bf16 and chunk_size == 128 and not selective
    if selective:
        from benchmarks.khdn_selective_trace import ARITHMETIC, backward_stages
        BACKWARD = backward_stages(chunk_size, source_vjp)
    if c128:
        from benchmarks.khdn_c128_trace import ARITHMETIC, BACKWARD, attest_schedule
    arithmetic = (ARITHMETIC if c128 or selective else 'bf16_output_source_single_c16') if bf16 else arithmetic
    backward_arithmetic = 'device_float32' if bf16 else 'device_float64' if fs6 else arithmetic
    route = 'bf16' if bf16 else 'experimental_bf16_fs6' if fs6 else 'experimental_mixed_readout' if mixed else 'strict'
    if (fs6 or bf16) and ((chunk_size not in ((16, 32, 64, 128) if selective else (16, 128) if bf16 else (16,)))
                          or (group_size, carry_mode) != (32, 'no_carry') or not source_binding):
        raise AssertionError('KHDN BF16 geometry/source binding missing or different')
    if (fs6 or bf16) and expected_layers is not None and expected_checkpoint is not None:
        original = Counter(row.get('layer') for row in calls if row['backward_calls'])
        recompute = Counter(row.get('layer') for row in calls if not row['backward_calls'])
        if original != Counter(expected_layers) or recompute != (Counter(expected_layers) if expected_checkpoint else Counter()):
            raise AssertionError('KHDN missing/duplicated per-layer checkpoint graph')
    if not calls or sum(row['backward_calls'] for row in calls) != expected_backward_calls:
        raise AssertionError('KHDN missing operator/backward observations for the actual model depth')
    if expected_layers is not None:
        coverage = Counter(row.get('layer') for row in calls for _ in range(row['backward_calls']))
        if coverage != Counter(expected_layers):
            raise AssertionError('KHDN missing or duplicated actual backward per trained layer')
    for row in calls:
        if row['backward_calls'] not in (0, 1):
            raise AssertionError('KHDN duplicate backward on one operator graph')
        if (not row['backward_calls'] and 'recompute_fn' not in row.get('checkpoint_frames', [])
                and row.get('forward_interrupted') != '_StopRecomputationError'):
            raise AssertionError('KHDN forward-only record has no observed checkpoint recomputation')
        requested, telemetry, options = row['requested'], row['telemetry'], row.get('executed_options')
        if requested.get('source_vjp', 'expanded') != source_vjp or (
                options and options.get('source_vjp', 'expanded') != source_vjp):
            raise AssertionError('KHDN source VJP request/execution differs')
        if requested['chunk_size'] != chunk_size or requested['group_size'] != group_size or requested['carry_mode'] != carry_mode:
            raise AssertionError('KHDN requested C/g/carry differs from the experiment')
        policies = (('c', chunk_size), ('g', group_size), ('carry', carry_mode == 'y_carry' if bf16 else carry_mode))
        if (not options or options.get('bounds') != row['sequence_bounds']
                or any(options.get(key) != value for key, value in policies)):
            raise AssertionError('KHDN executed C/g/carry metadata missing or different')
        if (requested.get('precision_policy') != precision_policy or options.get('mixed', False) is not mixed
                or telemetry.get('precision_requested') != precision_policy
                or telemetry.get('precision_eligible') is not (mixed or fs6 or bf16)
                or options.get('bf16_fs6', False) is not fs6):
            raise AssertionError(f'KHDN requested/executed precision policy differs: {telemetry}')
        if any(source['device'].split(':')[0] != 'cuda' for source in row['executed_sources']):
            raise AssertionError('KHDN substituted host source')
        interrupted = row.get('forward_interrupted') == '_StopRecomputationError' and not row['backward_calls']
        if (telemetry.get('exception_reason') or telemetry.get('backward_exception_reason')
                or telemetry.get('strict_replay_details') or row.get('strict_replay_calls')
                or (not interrupted and telemetry.get('fallback_subblocks') != 0)
                or telemetry.get('backward_fallback_subblocks', 0) != 0
                or (not interrupted and (telemetry.get('backend') != 'triton_chunk'
                                         or telemetry.get('precision_route') != route
                                         or telemetry.get('arithmetic') != arithmetic
                                         or (not bf16 and telemetry.get('tensor_cores') is not (mixed or fs6))))):
            raise AssertionError(f'KHDN {precision_policy} device route/fallback/precision attestation failed: {telemetry}')
        if not bf16 and not interrupted and telemetry.get('chunk_lengths') != row['expected_chunk_lengths']:
            raise AssertionError('KHDN chunk lengths/tail metadata differs from actual sequence bounds')
        edges = row['expected_edges'] if carry_mode == 'y_carry' else 0
        if not bf16 and not interrupted and (telemetry.get('carry_edges') != edges
                                             or telemetry.get('direct_FS') != len(row['expected_chunk_lengths']) * row['heads'] - edges):
            raise AssertionError('KHDN carry edges or direct-FS count differs')
        consumed = row['consumed']
        if (not consumed['all_finite'] or not math.isfinite(consumed['curvature']) or consumed['curvature'] <= 0
                or not consumed['log_decay_min'] <= consumed['log_decay_max'] < 0
                or consumed['feedback_delta_max'] > .25 or consumed['key_norm2_min'] <= 2.**-80
                or consumed.get('key_norm2_max', float('inf')) > 2.**80
                or consumed.get('initial_state_radius2_max', float('inf')) > .5
                or not 0 <= consumed.get('beta_min', -1) <= consumed.get('beta_max', 2) <= 1):
            raise AssertionError('KHDN consumed gate/domain observations failed')
        if fs6:
            _attest_fs6_sources(row, interrupted)
        elif bf16:
            _attest_bf16_sources(row, interrupted)
            if (options.get('arithmetic') != 'bf16' or telemetry.get('host_fallback') is not False
                    or telemetry.get('state_control_precision') != 'fp32'
                    or telemetry.get('contraction_precision') != 'bf16_operands_fp32_accumulation'):
                raise AssertionError('KHDN bf16 source arithmetic/control route differs')
        launches = row['launches']
        required = _khdn_stages(precision_policy, backward=bool(
            row['backward_calls']), chunk_size=chunk_size, source_vjp=source_vjp)
        if row.get('forward_interrupted'):
            if row['forward_interrupted'] != '_StopRecomputationError':
                raise AssertionError('KHDN unexpected interrupted forward')
        for name in set(required):
            matching = [launch for launch in launches if launch['kernel'] == name]
            if len(matching) != required.count(name):
                raise AssertionError(f'KHDN missing or duplicated actual kernel launch: {name}')
        if (fs6 or bf16) and [launch['kernel'] for launch in launches] != list(required):
            raise AssertionError('KHDN BF16 stage order/phase differs')
        if selective:
            from benchmarks.khdn_selective_trace import attest_schedule as attest_selective_schedule
            attest_selective_schedule(launches, bool(row['backward_calls']), chunk_size, source_vjp)
        elif c128:
            attest_schedule(launches, bool(row['backward_calls']))
        for launch in launches:
            flags = launch['arguments']
            if fs6 or bf16:
                if (launch.get('source_binding_sha256') != _binding_hash(source_binding)
                        or launch['kernel_source_sha256'] != source_binding['jit_functions'].get(launch['kernel'])
                        or launch.get('phase') != ('backward' if (launch['kernel'] in BACKWARD if c128 or selective
                                                                  else launch['kernel'].endswith('bwd_kernel')) else 'forward')):
                    raise AssertionError('KHDN bf16_fs6 source/callee/phase binding differs')
                selected = 'bf16_fs6' if fs6 and launch['kernel'] == 'state_fwd_kernel' else None
                if launch.get('selected_fs_callee') != selected:
                    raise AssertionError('KHDN bf16_fs6 selected callee differs')
            if flags.get('C') != chunk_size or ('GS' in flags and flags['GS'] != group_size):
                raise AssertionError('KHDN kernel C/g differs from request')
            if launch['kernel'] in ('state_fwd_kernel', 'state_bwd_kernel') and (
                    flags.get('GS') != group_size or flags.get('CARRY') is not (carry_mode == 'y_carry')):
                raise AssertionError('KHDN actual forward/backward carry policy differs')
            if launch['kernel'] == 'group_fwd_kernel':
                diagnostics = launch.get('prepared_diagnostics', {})
                if (not diagnostics.get('all_prepared_buffers_finite')
                        or any(diagnostics.get(key, 0) <= 0 for key in
                               ('prepared_denominator_min', 'positive_mass_min', 'positive_normalizer_min'))):
                    raise AssertionError('KHDN missing/invalid consumed denominator or positive mass diagnostics')
            if launch['kernel'] == 'state_fwd_kernel' and not launch.get('y_diagnostics', {}).get('all_saved_state_y_finite'):
                raise AssertionError('KHDN missing/invalid saved state/Y diagnostics')
            attest_compiled_policy(launch, precision_policy)
            compiler = launch['compiler']
            if not (bf16 or (mixed and launch['kernel'] in ('state_fwd_kernel', 'state_bwd_kernel'))
                    or (fs6 and launch['kernel'] == 'state_fwd_kernel')):
                if (not compiler.get('ptx_sha256') or compiler.get('fp64_arithmetic_instructions', 0) <= 0
                        or compiler.get('tensor_core_instructions') != 0
                        or not compiler.get('contraction_ir_sha256')
                        or compiler.get('multiply_add_operand_accumulator_dtypes') != ['float64']):
                    raise AssertionError('KHDN missing strict compiled arithmetic evidence')
        carry = [launch for launch in launches if launch['kernel'] == 'carry_fwd_kernel']
        if carry_mode == 'y_carry':
            if len(carry) != 1 or carry[0].get('observed_edges') != edges or not carry[0].get('future_carry_tail_zero'):
                raise AssertionError('KHDN missing carry launch/edges/zero future carry tail')
        elif carry:
            raise AssertionError('KHDN no_carry unexpectedly launched carry preparation')
        if row['backward_calls']:
            if (row.get('backward_options') != options or telemetry.get('backward') != 'manual_triton_chunk_source_vjp'
                    or telemetry.get('backward_precision_route') != route
                    or telemetry.get('backward_precision') != backward_arithmetic):
                raise AssertionError('KHDN missing/manual backward route or precision differs')
    return {'route': route if mixed or fs6 or bf16 else 'strict_device_fp64_simt',
            'precision_requested': precision_policy, 'execution_chunk_size': chunk_size, 'source_vjp': source_vjp,
            'tensor_core_claim': mixed or fs6 or bf16,
            'forward_arithmetic': arithmetic, 'backward_arithmetic': backward_arithmetic,
            'source_binding': source_binding, 'expected_checkpoint': expected_checkpoint,
            'expected_layers': expected_layers,
            'numerical_policy_qualified': False, 'performance_accepted': False, 'operator_calls': calls,
            'backward_calls': sum(row['backward_calls'] for row in calls),
            'backward_layer_coverage': dict(Counter(row.get('layer', 'operator') for row in calls if row['backward_calls'])),
            'checkpoint_recompute_observed': any('recompute_fn' in row.get('checkpoint_frames', [])
                                                 or row.get('forward_interrupted') == '_StopRecomputationError' for row in calls),
            'policy_scope': 'actual options and launch arguments; no invented gauge/feedback policy flags',
            'gemm_precision_scope': 'per-family compiled arithmetic, public/buffer dtypes and source sites; '
                                    'source sites are not executed counts; no numerical/performance qualification'}


class _ObservedKHDNUpdateKernel:
    """Observe native launch ownership/flags; the separate initial full trace attests compiler arithmetic."""

    def __init__(self, kernel, trace):
        self.kernel, self.trace = kernel, trace

    def __getattr__(self, name):
        return getattr(self.kernel, name)

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def observed(*args, **kwargs):
            current = getattr(self.trace.local, 'current', None)
            if current is None:
                raise AssertionError('KHDN update launch outside its public forward/backward graph')
            signature = inspect.signature(self.kernel.fn)
            values = signature.bind(*args, **{key: value for key, value in kwargs.items() if key in signature.parameters})
            values.apply_defaults()
            values = values.arguments
            name, phase = self.kernel.fn.__name__, current[1]
            selective = self.trace.precision_policy == 'bf16_selective'
            c128 = self.trace.chunk_size == 128 and not selective
            if selective:
                from benchmarks.khdn_selective_trace import backward_stages
                BACKWARD = backward_stages(self.trace.chunk_size, self.trace.source_vjp)
            if c128:
                from benchmarks.khdn_c128_trace import BACKWARD
            backward = name in BACKWARD if c128 or selective else name.endswith('bwd_kernel')
            if name not in _khdn_stages(self.trace.precision_policy, chunk_size=self.trace.chunk_size, source_vjp=self.trace.source_vjp) or phase != (
                    'backward' if backward else 'forward'):
                raise AssertionError('KHDN update native stage/phase differs')
            fs6_forward = self.trace.precision_policy == 'bf16_fs6' and name == 'state_fwd_kernel'
            if (values.get('C') != self.trace.chunk_size or values.get('MIXED', False) is not False
                    or values.get('BF16_FS6', False) is not fs6_forward):
                raise AssertionError('KHDN update native C/MIXED/BF16_FS6 flag differs')
            if name in ('state_fwd_kernel', 'state_bwd_kernel'):
                expected = dict(K=128, G=32, KD=128, VD=128, GS=32, CARRY=False)
                if any(values.get(key) != value for key, value in expected.items()):
                    raise AssertionError('KHDN update native state geometry/carry differs')
            digest = _khdn_kernel_source_hash(self.kernel)
            if digest != self.trace.source_binding['jit_functions'][name]:
                raise AssertionError('KHDN update selected kernel source differs')
            compiled = launch(*args, **kwargs)
            current[0]['launches'].append({'kernel': name, 'phase': phase, 'kernel_source_sha256': digest,
                                          'arguments': {key: values[key] for key in
                                                        ('C', 'GS', 'K', 'G', 'KD', 'VD', 'CARRY', 'MIXED', 'BF16_FS6',
                                                         'MODE', 'B', 'N', 'SINGLE', 'RHO_MATERIALIZED', 'NEED_DC',
                                                         'COUPLED_STATE', 'COUPLED', 'SOURCE_OWNER', 'PI_READY')
                                                        if key in values}})
            return compiled
        return observed


class KHDNUpdateTrace(KHDNTrace):
    """Require each expected batch/layer graph before its optimizer step, outside the explicit Oracle."""

    kernel_observer = _ObservedKHDNUpdateKernel

    def __init__(self, model, batches, layouts, *, expected_checkpoint, expected_source_binding, arm,
                 precision_policy='bf16_fs6'):
        if precision_policy not in ('bf16_fs6', 'bf16', 'bf16_selective'):
            raise ValueError('KHDN positive update trace requires a supported BF16 policy')
        super().__init__(getattr(model.config, 'chunk_size', 16), 32, 'no_carry',
                         model.config.num_hidden_layers, precision_policy, getattr(model.config, 'source_vjp', 'expanded'))
        if not batches or len(batches) != len(layouts) or not expected_source_binding:
            raise AssertionError('KHDN update requires positive batch and independent initial-trace source contracts')
        self.model, self.batches, self.layouts, self.arm = model, batches, layouts, arm
        self.expected_source_binding, self.expected_checkpoint = expected_source_binding, expected_checkpoint
        self.root_calls, self.active_batch = [], None

    def __enter__(self):
        native_bf16 = self.precision_policy in ('bf16', 'bf16_selective')
        native_name = 'bf16_selective' if self.precision_policy == 'bf16_selective' else 'chunk128' if self.chunk_size == 128 else 'bf16'
        chunk = importlib.import_module(KHDN_PREFIX + ('.' + native_name if native_bf16 else '.chunk'))
        kernels = importlib.import_module(KHDN_PREFIX + ('.' + native_name + '_kernels' if native_bf16 else '.chunk_kernels'))
        bind = _khdn_bf16_source_binding if native_bf16 else _khdn_fs6_source_binding
        actual = bind(chunk, kernels, self.source_vjp) if native_bf16 else bind(chunk, kernels)
        if actual != self.expected_source_binding:
            raise AssertionError('KHDN update differs from the initial full-trace source binding')
        try:
            self.bind_model(self.model)

            def before(module, args, kwargs):
                index = len(self.root_calls)
                if module is not self.model or args or index >= len(self.batches):
                    raise AssertionError('KHDN update root model or microbatch count differs')
                expected = self.batches[index].inputs
                if any(kwargs.get(key) is not value for key, value in expected.items()):
                    raise AssertionError('KHDN update actual batch tensor identity/order differs')
                self.active_batch = index
                self.root_calls.append({'batch_index': index, 'model_identity': id(module),
                                        'batch_sha256': self.batches[index].sha256})

            self.hooks.append(self.model.register_forward_pre_hook(before, with_kwargs=True))
            return super().__enter__()
        except BaseException:
            _restore_aliases(self.saved)
            for hook in self.hooks:
                hook.remove()
            raise

    def _entry(self, function):
        @functools.wraps(function)
        def checked(*args, **kwargs):
            values = inspect.signature(function).bind(*args, **kwargs)
            values.apply_defaults()
            values = values.arguments
            if (values['precision_policy'] != self.precision_policy or values['chunk_size'] != self.chunk_size
                    or values['group_size'] != 32 or values['carry_mode'] != 'no_carry'
                    or values.get('source_vjp', 'expanded') != self.source_vjp):
                raise AssertionError('KHDN update public policy/C/g/carry differs')
            row = self.local.current[0]
            if self.active_batch is None or row['layer'] not in self.layer_names:
                raise AssertionError('KHDN update public entry lacks actual root batch/layer ownership')
            row.update(batch_index=self.active_batch, model_identity=id(self.model))
            return function(*args, **kwargs)
        return super()._entry(checked)

    def attest(self):
        try:
            native_bf16 = self.precision_policy in ('bf16', 'bf16_selective')
            bind = _khdn_bf16_source_binding if native_bf16 else _khdn_fs6_source_binding
            native_name = 'bf16_selective' if self.precision_policy == 'bf16_selective' else 'chunk128' if self.chunk_size == 128 else 'bf16'
            binding = bind(importlib.import_module(
                KHDN_PREFIX + ('.' + native_name if native_bf16 else '.chunk')), self.kernel_module, self.source_vjp) if native_bf16 else bind(
                importlib.import_module(KHDN_PREFIX + '.chunk'), self.kernel_module)
            route = 'bf16' if native_bf16 else 'experimental_bf16_fs6'
            backward_precision = 'device_float32' if native_bf16 else 'device_float64'
            if binding != self.expected_source_binding or binding != self.source_binding:
                raise AssertionError('KHDN update source/helper identity changed')
            if len(self.root_calls) != len(self.batches):
                raise AssertionError('KHDN update missing actual root microbatch calls')
            expected = Counter((index, layer) for index in range(len(self.batches)) for layer in self.layer_names)
            original, recompute = Counter(), Counter()
            for row in self.calls:
                index, layer = row['batch_index'], row['layer']
                if row['model_identity'] != id(self.model) or (index, layer) not in expected:
                    raise AssertionError('KHDN update graph model/batch/layer identity differs')
                layout = self.layouts[index]
                if (row['sequence_bounds'] != layout['sequence_bounds']
                        or any(row['sources'][key]['shape'][:2] != layout['operator_batch_time']
                               for key in ('q', 'k', 'v', 'beta', 'log_decay'))):
                    raise AssertionError('KHDN update graph differs from actual batch token domain')
                options, telemetry = row.get('executed_options', {}), row['telemetry']
                if (options.get('bf16_fs6', False) is not (not native_bf16) or options.get('mixed', False) is not False
                        or (options.get('c'), options.get('g'), options.get('carry')) != (self.chunk_size, 32, False if native_bf16 else 'no_carry')
                        or options.get('bounds') != row['sequence_bounds'] or 'autograd_context_id' not in row
                        or options.get('source_vjp', 'expanded') != self.source_vjp):
                    raise AssertionError('KHDN update missing or substituted native forward/options')
                if (row.get('strict_replay_calls') or telemetry.get('exception_reason') or telemetry.get('backward_exception_reason')
                        or telemetry.get('strict_replay_details') or telemetry.get('backward_fallback_subblocks', 0)):
                    raise AssertionError('KHDN update exceptional/substituted route')
                if native_bf16 and (options.get('arithmetic') != 'bf16' or telemetry.get('host_fallback') is not False):
                    raise AssertionError('KHDN update BF16 arithmetic or host route differs')
                interrupted = row.get('forward_interrupted') == '_StopRecomputationError' and not row['backward_calls']
                if not interrupted and (not row.get('forward_complete') or row.get('output_all_finite') is not True
                                        or telemetry.get('precision_route') != route
                                        or telemetry.get('fallback_subblocks') != 0):
                    raise AssertionError('KHDN update missing completed native forward evidence')
                if row['backward_calls'] == 1:
                    original[index, layer] += 1
                    if (row.get('backward_options') != options or row.get('gradients_all_finite') is not True
                            or telemetry.get('backward_precision_route') != route
                            or telemetry.get('backward_precision') != backward_precision
                            or telemetry.get('backward') != 'manual_triton_chunk_source_vjp'):
                        raise AssertionError('KHDN update missing native strict backward evidence')
                    stages = _khdn_stages(self.precision_policy, chunk_size=self.chunk_size, source_vjp=self.source_vjp)
                elif row['backward_calls'] == 0 and ('recompute_fn' in row['checkpoint_frames'] or interrupted):
                    recompute[index, layer] += 1
                    stages = _khdn_stages(self.precision_policy, backward=False,
                                          chunk_size=self.chunk_size, source_vjp=self.source_vjp)
                else:
                    raise AssertionError('KHDN update missing/duplicated backward or unowned recompute')
                if self.precision_policy == 'bf16_selective':
                    from benchmarks.khdn_selective_trace import attest_schedule as attest_selective_schedule
                    attest_selective_schedule(row['launches'], bool(row['backward_calls']), self.chunk_size, self.source_vjp)
                elif self.chunk_size == 128:
                    from benchmarks.khdn_c128_trace import attest_schedule
                    attest_schedule(row['launches'], bool(row['backward_calls']))
                if [launch['kernel'] for launch in row['launches']] != list(stages):
                    raise AssertionError('KHDN update missing/duplicated native F/B stage calls')
            if original != expected or recompute != (expected if self.expected_checkpoint else Counter()):
                raise AssertionError('KHDN update positive batch/layer/backward/checkpoint counts differ')
        except BaseException as error:
            error.khdn_evidence = self.calls
            raise
        return {'arm': self.arm, 'precision_requested': self.precision_policy,
                'model_identity': id(self.model), 'source_binding': binding,
                'expected_checkpoint': self.expected_checkpoint, 'expected_layers': self.layer_names,
                'root_calls': self.root_calls, 'original_graphs': len(list(original.elements())),
                'recompute_graphs': len(list(recompute.elements())), 'operator_calls': self.calls,
                'attested_before_optimizer_step': True, 'compiled_arithmetic_scope': 'initial full trace required separately'}


class _NoFP64Training(TorchDispatchMode):
    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if any(isinstance(value, torch.Tensor) and value.dtype == torch.float64 for value in tree_leaves((args, kwargs))):
            raise RuntimeError('KHDN bf16 training observed an FP64 tensor input')
        result = function(*args, **kwargs)
        if any(isinstance(value, torch.Tensor) and value.dtype == torch.float64 for value in tree_leaves(result)):
            raise RuntimeError('KHDN bf16 training observed an FP64 tensor result')
        return result


class KHDNExceptionGuard:
    """Reject fallback/replay entries and experimental-policy ineligibility during execution."""

    def __init__(self, precision_policy='strict'):
        if precision_policy not in ('strict', 'mixed_readout', 'bf16_fs6', 'bf16', 'bf16_selective'):
            raise ValueError('unknown KHDN precision policy')
        self.precision_policy = precision_policy
        self.saved, self.events = [], []

    def __enter__(self):
        module = importlib.import_module(KHDN_PREFIX + '.chunk')
        recurrent = importlib.import_module(KHDN_PREFIX + '.fused_recurrent')
        naive = importlib.import_module(KHDN_PREFIX + '.naive')
        targets = (module._host_call, module._host_backward, recurrent.fused_recurrent_key_gated_hyperbolic_delta_rule,
                   naive.naive_recurrent_key_gated_hyperbolic_delta_rule)
        targets = (*targets, module._strict_device_replay)
        replacements = {}
        if self.precision_policy in ('mixed_readout', 'bf16_fs6'):
            helper = '_bf16_fs6_qualification_reason' if self.precision_policy == 'bf16_fs6' else '_qualification_reason'
            qualify = getattr(module, helper)

            @functools.wraps(qualify)
            def require_mixed(*args, **kwargs):
                reason = qualify(*args, **kwargs)
                if reason is not None:
                    self.events.append({'entry': helper, 'reason': reason,
                                        'thread': threading.get_ident()})
                    raise RuntimeError(f'KHDN native {self.precision_policy} measurement rejected strict fallback: {reason}')
                return reason
            replacements[id(qualify)] = require_mixed
        for function in targets:
            def reject(*args, _name=function.__name__, **kwargs):
                self.events.append({'entry': _name, 'thread': threading.get_ident()})
                raise RuntimeError(f'KHDN measurement rejected unexpected fallback/substitution: {_name}')
            replacements[id(function)] = reject
        _replace_aliases(replacements, self.saved)
        if not self.saved:
            raise AssertionError('KHDN exception sentinel could not be installed')
        self.dtype_guard = _NoFP64Training() if self.precision_policy in ('bf16', 'bf16_selective') else contextlib.nullcontext()
        self.dtype_guard.__enter__()
        return self

    def __exit__(self, *exception):
        try:
            self.dtype_guard.__exit__(*exception)
        finally:
            _restore_aliases(self.saved)

    def evidence(self):
        return {'method': 'raise_before_host_forward_or_backward_or_recurrent_or_reference',
                'events': self.events, 'precision_requested': self.precision_policy,
                'fp64_tensor_guard': self.precision_policy in ('bf16', 'bf16_selective'),
                'normal_path_wrappers': int(self.precision_policy in ('mixed_readout', 'bf16_fs6')),
                'scope': 'exception/replay entries and requested-policy eligibility rejection installed outside timer; '
                         'original core range and eligibility checks remain timed'}


def _khdn_buffer_diagnostics(values):
    p, groups = values['P'], values['Gp']
    count, heads, ng = p[5].shape[0], values['H'], values['NG']
    lengths = (values['Ends'] - values['Starts']).repeat_interleave(heads * ng)
    valid = torch.arange(values['C'], device=p[5].device)[None, :] < lengths[:, None]
    n = p[18].repeat_interleave(ng, 0)
    denominator = n - float(values['Curvature'].detach()) / values['GS'] * groups[4].diagonal(dim1=-2, dim2=-1)
    return {'prepared_denominator_min': float(denominator[valid].min()),
            'positive_mass_min': float(groups[6][valid].min()), 'positive_normalizer_min': float(groups[7][valid].min()),
            'all_prepared_buffers_finite': all(bool(torch.isfinite(value).all()) for value in (*p, *groups[:5], *groups[6:])),
            'chunk_heads': count, 'groups_per_head': ng}


def _khdn_y_diagnostics(values, precision_policy='strict'):
    p, s, y, ng = values['P'], values['SavedS'], values['SavedY'], values['NG']
    maximum, finite = 0., True
    native_bf16 = precision_policy in ('bf16', 'bf16_selective')
    residual_only = native_bf16 or values.get('MIXED', False) or values.get('BF16_FS6', False)
    for first in range(0, s.shape[0], 32):
        last = min(first + 32, s.shape[0])
        owners = torch.arange(first, last, device=s.device) // ng
        with torch.autocast(device_type=s.device.type, enabled=False):
            projection_dtype = torch.float32 if native_bf16 else s.dtype
            expected = torch.bmm(p[5][owners].to(projection_dtype), s[first:last].to(projection_dtype))
        if not residual_only:
            torch.testing.assert_close(y[first:last], expected, atol=1e-10, rtol=1e-9)
        maximum = max(maximum, float((y[first:last] - expected).abs().max()))
        finite = finite and bool(torch.isfinite(s[first:last]).all() & torch.isfinite(y[first:last]).all())
    result = {'all_saved_state_y_finite': finite,
              'saved_y_vs_actual_F_times_S_max_abs': maximum,
              'diagnostic_atol_rtol': None if residual_only else [1e-10, 1e-9],
              'residual_only_not_numerical_gate': bool(residual_only),
              'saved_state_y_fp64': s.dtype == y.dtype == torch.float64,
              'scope': 'untimed FP64 buffer projection in at most 32 chunk/head/group blocks; no timed projection'}
    if native_bf16:
        result.update(saved_state_y_fp32=s.dtype == y.dtype == torch.float32,
                      saved_state_y_bf16=s.dtype == y.dtype == torch.bfloat16,
                      scope='untimed FP32 buffer projection, autocast disabled; no timed projection or numerical gate')
    return result


def _khdn_contraction_sites(kernel, kernel_module):
    pending, seen, sites = [kernel.fn.__name__], set(), []
    native_bf16 = getattr(kernel_module, '__name__', '').endswith(
        ('.bf16_kernels', '.chunk128_kernels', '.bf16_selective_kernels', '.bf16_coupled_kernels'))
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        function = getattr(kernel_module, name, None)
        if not hasattr(function, 'fn'):
            continue
        tree = ast.parse(inspect.getsource(function.fn))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == 'tl' and node.func.attr == 'sum'):
                sites.append({'owner': name, 'line_in_function_source': node.lineno, 'call': ast.unparse(node),
                              'implementation': 'SIMT reduction; compiled launch arithmetic recorded separately'})
            if not isinstance(node.func, ast.Name):
                continue
            if node.func.id in ('mm', 'mm_state_product', 'bf16_fs6', 'mm_three',
                                'mm_precise', 'mm_backward', 'mm_left_two', 'mm_right_two'):
                sites.append({'owner': name, 'line_in_function_source': node.lineno, 'call': ast.unparse(node),
                              'implementation': ('BF16 operands/FP32 accumulation or FP32 SIMT; compiled products recorded separately'
                                                 if native_bf16 else
                                                 '21 radix-256 BF16 dots, FP32 digit accumulation, FP64 reconstruction'
                                                 if node.func.id == 'bf16_fs6' else
                                                 'selected FP32 TF32x3 or FP64 SIMT by MIXED and dimensions'
                                                 if node.func.id == 'mm_state_product' else 'FP64 SIMT multiply/add loop'),
                              'evidence_scope': 'static source site; not an executed count or per-site compiler attribution'})
            elif hasattr(getattr(kernel_module, node.func.id, None), 'fn'):
                pending.append(node.func.id)
    return sites
