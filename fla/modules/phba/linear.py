# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import atexit
import base64
import ctypes
import hashlib
import importlib.util
import json
import os
import subprocess
import tempfile
import threading
import warnings
from functools import cache
from pathlib import Path

import torch
import torch.nn.functional as F
from filelock import FileLock
from torch import nn

from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

_BACKEND = None
_INITIALIZED = False
_LOCK = threading.Lock()


@cache
def _supported_device(index: int) -> bool:
    return torch.cuda.get_device_name(index) == 'NVIDIA H20'


class _ExactBackend:
    def __init__(self, library: Path, configuration: dict):
        from torch.utils.cpp_extension import CUDA_HOME, get_default_build_root, load_inline

        source = Path(__file__).parent / 'csrc'
        build_key = hashlib.sha256(
            (source / 'lt_exact.cpp').read_bytes() + torch.__version__.encode()
        ).hexdigest()[:16]
        cache = Path(get_default_build_root()) / ('phba_' + build_key)
        cache.mkdir(parents=True, exist_ok=True)
        binary = cache / 'lt_exact.so'
        with FileLock(str(cache / 'build.lock')):
            if not binary.exists():
                if CUDA_HOME is None:
                    raise RuntimeError('The PHBA exact GEMM option requires CUDA development headers')
                with tempfile.TemporaryDirectory(dir=cache) as temporary:
                    output = Path(temporary) / binary.name
                    subprocess.run(
                        [
                            os.environ.get('CXX', 'c++'), '-std=c++17', '-O2', '-shared', '-fPIC',
                            '-I' + str(Path(CUDA_HOME) / 'include'),
                            '-I' + str(library.parent.parent / 'include'),
                            str(source / 'lt_exact.cpp'), str(library),
                            '-Wl,-rpath,' + str(library.parent), '-o', str(output),
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    output.replace(binary)
        self.autocast = load_inline(
            name='phba_cached_autocast_' + build_key,
            cpp_sources=(source / 'native_autocast.cpp').read_text(),
            functions=['cached_bf16'],
            extra_cflags=['-O2'],
            with_cuda=False,
            verbose=False,
        )
        self.configuration = configuration
        self.library = ctypes.CDLL(str(binary))
        pointer = ctypes.c_void_p
        integer = ctypes.c_int
        self.library.make_session.argtypes = [ctypes.c_int64] * 7 + [ctypes.c_size_t, ctypes.POINTER(integer)]
        self.library.make_session.restype = pointer
        self.library.install_algo.argtypes = [pointer, pointer, ctypes.c_size_t]
        self.library.run_algo.argtypes = [pointer, integer, pointer, pointer, pointer, pointer, pointer]
        self.library.drop_session.argtypes = [pointer]
        self.sessions = {}
        atexit.register(self.close)

    def close(self):
        for handle in self.sessions.values():
            self.library.drop_session(handle)
        self.sessions.clear()

    def weight_gradient(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        signature = (tuple(a.shape), tuple(b.shape), tuple(a.stride()), tuple(b.stride()))
        algorithm = next(
            (entry for entry in self.configuration['algorithms']
             if signature == ((entry['m'], entry['k']), (entry['k'], entry['n']),
                              tuple(entry['astride']), tuple(entry['bstride']))),
            None,
        )
        if algorithm is None:
            return torch.mm(a, b)
        stream = torch.cuda.current_stream(a.device)
        key = (a.device.index, stream.cuda_stream, signature)
        if key not in self.sessions:
            status = ctypes.c_int()
            handle = self.library.make_session(
                algorithm['m'], algorithm['n'], algorithm['k'], *a.stride(), *b.stride(),
                self.configuration['workspace_limit'], ctypes.byref(status),
            )
            if status.value:
                self.library.drop_session(handle)
                raise RuntimeError(f'PHBA cuBLASLt descriptor creation failed: {status.value}')
            encoded = base64.b64decode(algorithm['algorithm'])
            if hashlib.sha256(encoded).hexdigest() != algorithm['algorithm_sha256']:
                self.library.drop_session(handle)
                raise RuntimeError('PHBA cuBLASLt algorithm descriptor is corrupted')
            buffer = ctypes.create_string_buffer(encoded, len(encoded))
            if self.library.install_algo(handle, buffer, len(encoded)):
                self.library.drop_session(handle)
                raise RuntimeError('PHBA cuBLASLt algorithm was rejected by the validated runtime')
            self.sessions[key] = handle
        output = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
        workspace = torch.empty(max(4, algorithm['workspace']), device=a.device, dtype=torch.uint8)
        status = self.library.run_algo(
            self.sessions[key], 0, ctypes.c_void_p(a.data_ptr()), ctypes.c_void_p(b.data_ptr()),
            ctypes.c_void_p(output.data_ptr()), ctypes.c_void_p(workspace.data_ptr()),
            ctypes.c_void_p(stream.cuda_stream),
        )
        if status:
            raise RuntimeError(f'PHBA cuBLASLt weight-gradient execution failed: {status}')
        return output


def exact_backend(device: torch.device) -> _ExactBackend | None:
    """Load the screened H20 implementation, or retain the native linear path."""
    global _BACKEND, _INITIALIZED
    if (device.type != 'cuda' or torch.__version__.split('+')[0] != '2.9.0'
            or torch.version.cuda != '12.8' or not _supported_device(device.index)):
        return None
    if _INITIALIZED:
        return _BACKEND
    with _LOCK:
        if _INITIALIZED:
            return _BACKEND
        try:
            specification = importlib.util.find_spec('nvidia.cublas')
            locations = [] if specification is None else list(specification.submodule_search_locations or [])
            libraries = [Path(folder) / 'lib/libcublasLt.so.12' for folder in locations]
            configuration = json.loads((Path(__file__).parent / 'exact_gemm_algorithms.json').read_text())
            library = next((p for p in libraries if p.is_file()), None)
            if library is not None and hashlib.sha256(library.read_bytes()).hexdigest() == configuration['library_sha256']:
                _BACKEND = _ExactBackend(library=library, configuration=configuration)
        except (ImportError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
            warnings.warn(f'PHBA exact GEMM unavailable; using native linear operations: {error}', stacklevel=2)
        _INITIALIZED = True
    return _BACKEND


def exact_weight_gradient(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16 or torch.is_grad_enabled():
        return torch.mm(a, b)
    backend = exact_backend(a.device)
    return torch.mm(a, b) if backend is None else backend.weight_gradient(a=a, b=b)


class _Linear(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        return F.linear(x, weight)

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        inputs = x.reshape(-1, x.shape[-1])
        grad = grad.reshape(-1, weight.shape[0])
        dx = torch.mm(grad, weight).reshape(x.shape) if ctx.needs_input_grad[0] else None
        dw = exact_weight_gradient(a=grad.t(), b=inputs) if ctx.needs_input_grad[1] else None
        return dx, dw


class PHBALinear(nn.Linear):
    """PHBA projection with an optional byte-validated H20 weight-gradient path."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eligible = (
            not torch.compiler.is_compiling() and type(x) in (torch.Tensor, nn.Parameter)
            and type(self.weight) in (torch.Tensor, nn.Parameter)
            and self.bias is None and self.training and torch.is_grad_enabled() and x.is_cuda
            and torch.is_autocast_enabled('cuda') and torch.get_autocast_dtype('cuda') == torch.bfloat16
            and x.is_contiguous() and self.weight.is_contiguous()
            and x.shape[-1] == 2048 and x.numel() == 8192 * 2048 and self.out_features in (2048, 4272)
        )
        backend = exact_backend(x.device) if eligible else None
        if backend is None:
            return F.linear(x, self.weight, self.bias)
        return _Linear.apply(backend.autocast.cached_bf16(x), backend.autocast.cached_bf16(self.weight))
