# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Correctness gates and synchronized measurements for HDN operator benchmarks."""

import contextlib
import gc
import hashlib
import importlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch


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
