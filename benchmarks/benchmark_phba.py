# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Synchronized PHBA training-compute benchmark with optional validation hooks."""
import argparse
import hashlib
import importlib
import json
import statistics
import time
from itertools import accumulate
from pathlib import Path

import torch

from fla.models.phba import PHBAConfig, PHBAForCausalLM


def model_options(args):
    options = dict(
        hidden_size=2048, num_hidden_layers=24, num_heads=16, num_kv_heads=16,
        intermediate_size=4272, vocab_size=32000, gsa_num_slots=128,
        max_position_embeddings=args.seq_len, moba_chunk_size=args.chunk_size, moba_topk=args.topk,
        use_cache=False, qk_norm=False, use_output_gate=False, use_phba=True,
        fuse_norm=True, fuse_swiglu=True, fuse_cross_entropy=True, fuse_linear_cross_entropy=False,
    )
    if args.config is not None:
        options.update(json.loads(args.config.read_text()))
    return options


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError('PHBA training benchmarks require CUDA')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(args.device)
    observer = None if args.observer is None else importlib.import_module(args.observer).Observer(args)
    if observer is not None:
        observer.configure()
    options = model_options(args)
    model = PHBAForCausalLM(PHBAConfig(**options)).cuda().train()
    parameters = list(model.parameters())
    assert {p.dtype for p in parameters} == {torch.float32}
    decay, no_decay = [], []
    for parameter in parameters:
        (no_decay if parameter.ndim < 2 or getattr(parameter, '_no_weight_decay', False) else decay).append(parameter)
    module_name, class_name = args.optimizer.rsplit(':', maxsplit=1)
    optimizer_class = getattr(importlib.import_module(module_name), class_name)
    optimizer = optimizer_class(
        [dict(params=decay, weight_decay=.1), dict(params=no_decay, weight_decay=0.)],
        lr=args.learning_rate, betas=(.9, .95), fused=True,
    )
    tokens = torch.randint(
        0, options['vocab_size'], (args.batch_size, args.seq_len), device='cuda',
        generator=torch.Generator(device='cuda').manual_seed(args.seed),
    )
    labels = tokens
    forward_kwargs = {}
    lengths = None if args.lengths is None else [int(value) for value in args.lengths.split(',')]
    if lengths is not None:
        if args.batch_size != 1 or min(lengths) < 1 or sum(lengths) != args.seq_len:
            raise ValueError('Packed lengths must be positive, sum to seq-len, and use batch-size1')
        cumulative = [0, *accumulate(lengths)]
        forward_kwargs['cu_seqlens'] = torch.tensor(cumulative, device='cuda', dtype=torch.int32)
        labels = tokens.clone()
        labels[:, cumulative[1:-1]] = -100
    result = dict(
        options=options, parameters=sum(p.numel() for p in parameters), parameter_tensors=len(parameters),
        batch_size=args.batch_size, sequence_length=args.seq_len, seed=args.seed,
        learning_rate=args.learning_rate, optimizer=args.optimizer,
        warmup_steps=args.warmup_steps, measure_steps=args.steps, activation_checkpointing=False,
        parameter_dtype='float32', compute_dtype='bfloat16_autocast',
        torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
        gpu=torch.cuda.get_device_name(args.device), warmup=[], steps=[],
        lengths=lengths,
    )
    if observer is not None:
        result['labels_sha256'] = hashlib.sha256(labels.cpu().contiguous().numpy().tobytes()).hexdigest()
        observer.start(model, optimizer, tokens, result)
    for iteration in range(args.warmup_steps + args.steps):
        if iteration == 2:
            torch.cuda.reset_peak_memory_stats()
        if iteration == args.warmup_steps and observer is not None:
            observer.before_measure(model, optimizer, result)
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        events[0].record()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = model(input_ids=tokens, labels=labels, use_cache=False, return_dict=True, **forward_kwargs).loss
        events[1].record()
        loss.backward()
        events[2].record()
        if iteration == 0 and observer is not None:
            observer.raw_gradients(model, result)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        optimizer.step()
        events[3].record()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        row = dict(
            iteration=iteration, seconds=elapsed, loss=float(loss.detach()), grad_norm=float(norm),
            forward_ms=events[0].elapsed_time(events[1]), backward_ms=events[1].elapsed_time(events[2]),
            clip_optimizer_ms=events[2].elapsed_time(events[3]), cuda_step_ms=events[0].elapsed_time(events[3]),
        )
        result['warmup' if iteration < args.warmup_steps else 'steps'].append(row)
        if iteration == 0 and observer is not None:
            observer.first_step(model, optimizer, row, result)
    result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    result['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    times = [entry['seconds'] for entry in result['steps']]
    result['step_seconds_mean'] = statistics.fmean(times)
    result['tokens_per_second'] = args.batch_size * args.seq_len / result['step_seconds_mean']
    if observer is not None:
        observer.finish(model, optimizer, result)
    result['status'] = 'success'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('parameters', 'step_seconds_mean', 'tokens_per_second')}))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--seq-len', type=int, default=8192)
    parser.add_argument('--lengths', help='Comma-separated packed sequence lengths summing to seq-len')
    parser.add_argument('--chunk-size', type=int, default=128)
    parser.add_argument('--topk', type=int, default=8)
    parser.add_argument('--warmup-steps', type=int, default=20)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--learning-rate', type=float, default=0.)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--optimizer', default='torch.optim:AdamW', help='Optimizer factory as module:class')
    parser.add_argument('--observer', help='Optional module providing untimed validation and paired-run synchronization')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch_size, args.seq_len, args.steps) < 1 or args.warmup_steps < 3:
        parser.error('Require positive workload/measurement sizes and at least three warmup steps')
    return args


if __name__ == '__main__':
    run(parse_args())
