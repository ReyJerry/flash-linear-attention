# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Profile PHBA stages; instrumented timings are not throughput measurements."""
from contextlib import ExitStack
from functools import wraps
from unittest.mock import patch

import torch
from benchmark_phba import parse_args, run

from fla.ops.phba import parallel, prefix


def instrument(function, name):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with torch.profiler.record_function(name):
            return function(*args, **kwargs)
    return wrapped


def main():
    args = parse_args()
    stages = (
        (parallel, 'local_attention'), (parallel, 'build_window_metadata'),
        (parallel, 'launch_gather'), (parallel, 'launch_history'), (parallel, 'launch_owner'),
        (prefix, '_prefix_forward'), (prefix, '_prefix_backward_pair'),
    )
    with ExitStack() as stack:
        for module, name in stages:
            stack.enter_context(patch.object(module, name, instrument(getattr(module, name), 'phba::' + name)))
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        ) as trace:
            run(args)
    path = args.output.with_suffix('.trace.json')
    trace.export_chrome_trace(str(path))
    print(f'Instrumented PHBA trace: {path}; use benchmark_phba.py for throughput.')


if __name__ == '__main__':
    main()
