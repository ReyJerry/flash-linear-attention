# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Inspect offline cuBLASLt candidates without changing production descriptors."""
import argparse
import base64
import ctypes
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import torch
from torch.utils.cpp_extension import CUDA_HOME

from fla.ops.phba.backends.cublaslt import exact_backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    backend = exact_backend(torch.device('cuda', torch.cuda.current_device()))
    if backend is None or CUDA_HOME is None:
        raise RuntimeError('Candidate inspection requires the validated H20 runtime and CUDA development headers')
    import nvidia.cublas

    library = Path(next(iter(nvidia.cublas.__path__))) / 'lib/libcublasLt.so.12'
    source = Path(__file__).parent / 'phba_tools/csrc/lt_screen.cpp'
    pointer, integer, size = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
    records = []
    with tempfile.TemporaryDirectory() as directory:
        binary = Path(directory) / 'screen.so'
        subprocess.run(
            [os.environ.get('CXX', 'c++'), '-std=c++17', '-O2', '-shared', '-fPIC',
             '-I' + str(Path(CUDA_HOME) / 'include'), '-I' + str(library.parent.parent / 'include'),
             str(source), str(library), '-Wl,-rpath,' + str(library.parent), '-o', str(binary)],
            check=True,
        )
        api = ctypes.CDLL(str(binary))
        api.make_session.argtypes = [ctypes.c_int64] * 7 + [size, ctypes.POINTER(integer)]
        api.make_session.restype = pointer
        api.count_algos.argtypes = [pointer]
        api.describe_algo.argtypes = [pointer, integer] + \
            [ctypes.POINTER(integer)] * 3 + [ctypes.POINTER(size), ctypes.POINTER(ctypes.c_float)]
        api.algo_bytes.restype = size
        api.copy_algo.argtypes = [pointer, integer, pointer, size]
        api.drop_session.argtypes = [pointer]
        for geometry in backend.configuration['algorithms']:
            status = integer()
            handle = api.make_session(
                geometry['m'], geometry['n'], geometry['k'], *geometry['astride'], *geometry['bstride'],
                backend.configuration['workspace_limit'], ctypes.byref(status),
            )
            try:
                if status.value:
                    raise RuntimeError(f'cuBLASLt enumeration failed: {status.value}')
                for index in range(api.count_algos(handle)):
                    identity, split, reduction, workspace, waves = integer(), integer(), integer(), size(), ctypes.c_float()
                    result = api.describe_algo(handle, index, ctypes.byref(identity), ctypes.byref(split),
                                               ctypes.byref(reduction), ctypes.byref(workspace), ctypes.byref(waves))
                    blob = ctypes.create_string_buffer(api.algo_bytes())
                    if api.copy_algo(handle, index, blob, len(blob)):
                        raise RuntimeError('Cannot export candidate descriptor')
                    records.append(dict(
                        m=geometry['m'], n=geometry['n'], k=geometry['k'], index=index, status=result,
                        algorithm_id=identity.value, split_k=split.value, reduction=reduction.value,
                        workspace_bytes=workspace.value, waves=waves.value,
                        descriptor=base64.b64encode(blob.raw).decode(), descriptor_sha256=hashlib.sha256(blob.raw).hexdigest(),
                    ))
            finally:
                api.drop_session(handle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(candidates=records, validated=False), indent=2) + '\n')
    print(f'Inspected {len(records)} candidates; numerical equality and complete-call timing are still required.')


if __name__ == '__main__':
    main()
