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


def _mlir_contraction_types(ir):
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


class _ObservedKHDNKernel:
    def __init__(self, kernel, trace):
        self.kernel, self.trace = kernel, trace
        self.contractions = _khdn_contraction_sites(kernel, trace.kernel_module)

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
            row = {'kernel': self.kernel.fn.__name__, 'phase': current[1],
                   'arguments': {key: _tensor_descriptions(value) for key, value in values.items()},
                   'compiler': _strict_compiler_evidence(compiled), 'thread': threading.get_ident(),
                   'gemm_source_sites': [{**site, 'operand_dtypes': ['float64', 'float64'],
                                          'accumulator_dtype': 'float64',
                                          'evidence': 'supported compiled IR multiply/add/FMA type clauses are f64; '
                                                      'source-site attribution is not an execution count'}
                                         for site in self.contractions]}
            if row['kernel'] == 'group_fwd_kernel':
                row['prepared_diagnostics'] = _khdn_buffer_diagnostics(values)
            if row['kernel'] == 'state_fwd_kernel':
                row['y_diagnostics'] = _khdn_y_diagnostics(values)
            if row['kernel'] == 'carry_fwd_kernel':
                nxt, heads = values['Next'], values['H']
                terminal = (nxt < 0).repeat_interleave(heads)
                row['terminal_rows'] = int(terminal.sum())
                row['future_carry_tail_zero'] = all(bool((buffer[terminal] == 0).all()) for buffer in values['Carry'])
                row['observed_edges'] = int((nxt >= 0).sum()) * heads
            current[0]['launches'].append(row)
            return compiled
        return observed


class KHDNTrace:
    """Retain real operator telemetry through autograd and observe its compiled launches, outside timing."""

    def __init__(self, chunk_size, group_size, carry_mode, expected_backward_calls=1):
        self.chunk_size, self.group_size, self.carry_mode = chunk_size, group_size, carry_mode
        self.expected_backward_calls = expected_backward_calls
        self.saved, self.calls, self.by_telemetry = [], [], {}
        self.hooks, self.layer_names = [], []
        self.local = threading.local()

    def bind_model(self, model):
        for name, attention in model.named_modules():
            if not hasattr(attention, 'f_proj') or not hasattr(attention, 'A_log'):
                continue
            self.layer_names.append(name)

            def before(module, args, layer=name):
                self.local.layer = layer

            def after(module, args, output):
                self.local.layer = 'operator'
            self.hooks.extend((attention.register_forward_pre_hook(before),
                               attention.register_forward_hook(after, always_call=True)))
        if len(self.layer_names) != self.expected_backward_calls:
            raise AssertionError('KHDN trained-layer count differs from model configuration')
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
            n = k.detach().double().square().sum(-1)
            delta = c_value / values['group_size'] * v.detach().double().reshape(
                *v.shape[:-1], -1, values['group_size']).square().sum(-1) / n[..., None]
            state = values['initial_state']
            radius2 = (0. if state is None else float((c_value / values['group_size'] * state.detach().double().reshape(
                *state.shape[:-1], -1, values['group_size']).square().sum((-3, -1))).max()))
            row = {'requested': {key: values[key] for key in ('chunk_size', 'group_size', 'carry_mode', 'check_inputs')},
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
                returned = function(*bound.args, **bound.kwargs)
                row['returned'] = _tensor_descriptions(returned)
                row['forward_complete'] = True
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
        kernels = importlib.import_module(KHDN_PREFIX + '.chunk_kernels')
        self.kernel_module = kernels
        for name in (*KHDN_STAGES, 'carry_fwd_kernel'):
            kernel = getattr(kernels, name)
            replacements[id(kernel)] = _ObservedKHDNKernel(kernel, self)
        _replace_aliases(replacements, self.saved)
        original_forward, original_backward = vars(module._Chunk)['forward'], vars(module._Chunk)['backward']

        @functools.wraps(original_forward.__func__)
        def forward(ctx, *args, **kwargs):
            row = self.local.current[0]
            row['executed_options'] = dict(args[-2])
            row['executed_sources'] = [_tensor_descriptions(value) for value in args[:7]]
            return original_forward.__func__(ctx, *args, **kwargs)

        @functools.wraps(original_backward.__func__)
        def backward(ctx, *args, **kwargs):
            row = self.by_telemetry.get(id(ctx.telemetry))
            if row is None:
                raise AssertionError('KHDN backward lost its forward telemetry identity')
            previous = getattr(self.local, 'current', None)
            self.local.current = (row, 'backward')
            row['backward_options'] = dict(ctx.options)
            row['backward_thread'] = threading.get_ident()
            try:
                returned = original_backward.__func__(ctx, *args, **kwargs)
                row['backward_calls'] += 1
                row['gradient_dtypes'] = _tensor_descriptions(returned[:7])
                return returned
            finally:
                self.local.current = previous
        self.saved.extend(((module._Chunk, 'forward', original_forward), (module._Chunk, 'backward', original_backward)))
        module._Chunk.forward, module._Chunk.backward = staticmethod(forward), staticmethod(backward)
        return self

    def __exit__(self, exc_type, error, tb):
        _restore_aliases(self.saved)
        for hook in self.hooks:
            hook.remove()
        if error is not None:
            error.khdn_evidence = self.calls

    def attest(self):
        try:
            evidence = attest_khdn(self.calls, self.chunk_size, self.group_size, self.carry_mode,
                                   self.expected_backward_calls, self.layer_names or None)
        except BaseException as error:
            error.khdn_evidence = self.calls
            raise
        evidence['source_binding'] = {name: hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest()
                                      for name in ('chunk', 'chunk_kernels')
                                      for module in (importlib.import_module(KHDN_PREFIX + '.' + name),)}
        return evidence


def attest_khdn(calls, chunk_size, group_size, carry_mode, expected_backward_calls=1, expected_layers=None):
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
        if requested['chunk_size'] != chunk_size or requested['group_size'] != group_size or requested['carry_mode'] != carry_mode:
            raise AssertionError('KHDN requested C/g/carry differs from the experiment')
        policies = (('c', chunk_size), ('g', group_size), ('carry', carry_mode))
        if (not options or options.get('bounds') != row['sequence_bounds']
                or any(options.get(key) != value for key, value in policies)):
            raise AssertionError('KHDN executed C/g/carry metadata missing or different')
        if any(source['device'].split(':')[0] != 'cuda' for source in row['executed_sources']):
            raise AssertionError('KHDN substituted host source')
        interrupted = row.get('forward_interrupted') == '_StopRecomputationError' and not row['backward_calls']
        if (telemetry.get('exception_reason') or (not interrupted and telemetry.get('fallback_subblocks') != 0)
                or telemetry.get('backward_fallback_subblocks', 0) != 0
                or (not interrupted and (telemetry.get('backend') != 'triton_chunk'
                                         or telemetry.get('arithmetic') != 'device_float64'
                                         or telemetry.get('tensor_cores') is not False))):
            raise AssertionError('KHDN strict device route/fallback/precision attestation failed')
        if not interrupted and telemetry.get('chunk_lengths') != row['expected_chunk_lengths']:
            raise AssertionError('KHDN chunk lengths/tail metadata differs from actual sequence bounds')
        edges = row['expected_edges'] if carry_mode == 'y_carry' else 0
        if not interrupted and (telemetry.get('carry_edges') != edges
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
        launches = row['launches']
        required = KHDN_STAGES if row['backward_calls'] else KHDN_STAGES[:4]
        if row.get('forward_interrupted'):
            if row['forward_interrupted'] != '_StopRecomputationError':
                raise AssertionError('KHDN unexpected interrupted forward')
        for name in required:
            matching = [launch for launch in launches if launch['kernel'] == name]
            if len(matching) != 1:
                raise AssertionError(f'KHDN missing or duplicated actual kernel launch: {name}')
        for launch in launches:
            flags = launch['arguments']
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
            compiler = launch['compiler']
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
                    or telemetry.get('backward_precision') != 'device_float64'):
                raise AssertionError('KHDN missing/manual backward route or precision differs')
    return {'route': 'strict_device_fp64_simt', 'tensor_core_claim': False, 'operator_calls': calls,
            'backward_calls': sum(row['backward_calls'] for row in calls),
            'backward_layer_coverage': dict(Counter(row.get('layer', 'operator') for row in calls if row['backward_calls'])),
            'checkpoint_recompute_observed': any('recompute_fn' in row.get('checkpoint_frames', [])
                                                 or row.get('forward_interrupted') == '_StopRecomputationError' for row in calls),
            'policy_scope': 'actual options and launch arguments; no invented gauge/feedback policy flags',
            'gemm_precision_scope': 'SIMT FP64 contractions; per-source sites, public/buffer dtypes, compiled IR and PTX evidence'}


class KHDNExceptionGuard:
    """Reject exception/recurrent/reference entries without touching the normal timed operator path."""

    def __init__(self):
        self.saved, self.events = [], []

    def __enter__(self):
        module = importlib.import_module(KHDN_PREFIX + '.chunk')
        recurrent = importlib.import_module(KHDN_PREFIX + '.fused_recurrent')
        naive = importlib.import_module(KHDN_PREFIX + '.naive')
        targets = (module._host_call, module._host_backward, recurrent.fused_recurrent_key_gated_hyperbolic_delta_rule,
                   naive.naive_recurrent_key_gated_hyperbolic_delta_rule)
        replacements = {}
        for function in targets:
            def reject(*args, _name=function.__name__, **kwargs):
                self.events.append({'entry': _name, 'thread': threading.get_ident()})
                raise RuntimeError(f'KHDN measurement rejected unexpected fallback/substitution: {_name}')
            replacements[id(function)] = reject
        _replace_aliases(replacements, self.saved)
        if not self.saved:
            raise AssertionError('KHDN exception sentinel could not be installed')
        return self

    def __exit__(self, *exception):
        _restore_aliases(self.saved)

    def evidence(self):
        return {'method': 'raise_before_host_forward_or_backward_or_recurrent_or_reference',
                'events': self.events, 'normal_path_wrappers': 0,
                'scope': 'exception entries replaced outside timer; original core range checks remain timed'}


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


def _khdn_y_diagnostics(values):
    p, s, y, ng = values['P'], values['SavedS'], values['SavedY'], values['NG']
    maximum, finite = 0., True
    for first in range(0, s.shape[0], 32):
        last = min(first + 32, s.shape[0])
        owners = torch.arange(first, last, device=s.device) // ng
        expected = torch.bmm(p[5][owners], s[first:last])
        torch.testing.assert_close(y[first:last], expected, atol=1e-10, rtol=1e-9)
        maximum = max(maximum, float((y[first:last] - expected).abs().max()))
        finite = finite and bool(torch.isfinite(s[first:last]).all() & torch.isfinite(y[first:last]).all())
    return {'all_saved_state_y_finite': finite,
            'saved_y_vs_actual_F_times_S_max_abs': maximum,
            'diagnostic_atol_rtol': [1e-10, 1e-9],
            'scope': 'untimed FP64 buffer projection in at most 32 chunk/head/group blocks; no timed projection'}


def _khdn_contraction_sites(kernel, kernel_module):
    pending, seen, sites = [kernel.fn.__name__], set(), []
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
            if node.func.id == 'mm':
                sites.append({'owner': name, 'line_in_function_source': node.lineno, 'call': ast.unparse(node),
                              'implementation': 'SIMT multiply/add loop; compiled launch arithmetic recorded separately'})
            elif hasattr(getattr(kernel_module, node.func.id, None), 'fn'):
                pending.append(node.func.id)
    return sites
