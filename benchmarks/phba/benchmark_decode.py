# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Within-checkpoint PHBA stage comparisons; not a five-architecture benchmark."""

import argparse
import csv
import gc
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch

from run_decode_acceptance import check_manifest
from fla.models.phba import PHBAForCausalLM
from fla.models.phba.inference import PHBADecoder
from fla.ops.phba.decode import STAGES
from fla.utils import assert_close


def stats(values):
    if not values:
        return {}
    ordered = sorted(values)
    def quantile(q):
        return ordered[min(len(ordered)-1, round(q*(len(ordered)-1)))]
    return {'mean': statistics.mean(values), 'median': statistics.median(values),
            'p95': quantile(.95), 'p99': quantile(.99), 'max': max(values)}


def peak():
    return {'allocated': torch.cuda.max_memory_allocated(), 'reserved': torch.cuda.max_memory_reserved()}


@torch.inference_mode()
def verify_short(model, stage, dtype):
    c = model.config.moba_chunk_size
    torch.manual_seed(42)
    tokens = torch.randint(model.config.vocab_size, (1, c+3), device='cuda')
    decoder = PHBADecoder(model, c+3, stage=stage)
    with torch.autocast('cuda', dtype=dtype):
        expected = model(tokens, use_cache=False, logits_to_keep=1).logits
        decoder.prefill(tokens[:, :c-1])
        for pos in range(c-1, c+3):
            actual = decoder.decode(tokens[:, pos:pos+1])
    assert_close('real checkpoint short cached logits', expected.float(), actual.float(),
                 .03 if dtype == torch.bfloat16 else .006)
    error = (expected.float()-actual.float()).square().mean().sqrt().item()
    decoder.clear()
    return error


@torch.inference_mode()
def measure(model, stage, tokens, prompt, steps, dtype, greedy):
    decoder = PHBADecoder(model, prompt+steps, stage=stage)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    wall_start = time.perf_counter()
    with torch.autocast('cuda', dtype=dtype):
        logits = decoder.prefill(tokens[:, :prompt])
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - wall_start
    prefill_peak = peak()
    cache_start = decoder.memory_bytes()
    current = logits.argmax(-1)
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(steps)]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    wall_start = time.perf_counter()
    with torch.autocast('cuda', dtype=dtype):
        for i, (start, end) in enumerate(events):
            next_ids = current if greedy else tokens[:, prompt+i:prompt+i+1]
            start.record()
            logits = decoder.decode(next_ids)
            current = logits.argmax(-1)
            end.record()
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - wall_start
    decode_peak = peak()
    if not torch.isfinite(logits).all():
        raise RuntimeError('non-finite model logits')
    times = [start.elapsed_time(end) for start, end in events]
    c = model.config.moba_chunk_size
    closing = [i for i in range(steps) if (prompt+i+1) % c == 0]
    ordinary = [t for i, t in enumerate(times) if i not in closing]
    boundaries = [times[i] for i in closing]
    row = {'stage': stage, 'prompt': prompt, 'incremental_forward_steps': steps,
           'prefill_s': prefill_s, 'decode_wall_ms_per_step': wall_s*1000/steps,
           'decode_wall_tokens_per_s': steps/wall_s, 'event_ms': stats(times),
           'ordinary_event_ms': stats(ordinary), 'boundary_event_ms': stats(boundaries),
           'boundary_step_indices': closing, 'per_step_event_ms': times,
           'prefill_peak_bytes': prefill_peak, 'decode_peak_bytes': decode_peak,
           'peak_allocated_gib': max(prefill_peak['allocated'], decode_peak['allocated'])/2**30,
           'cache_after_prefill_bytes': cache_start, 'cache_after_decode_bytes': decoder.memory_bytes()}
    decoder.clear()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='local trained PHBA checkpoint directory')
    parser.add_argument('--validation-report', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--stages', nargs='+', choices=STAGES, default=list(STAGES))
    parser.add_argument('--lengths', nargs='+', type=int, default=[32768, 65536])
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--dtype', choices=['bf16', 'fp16'], default='bf16')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tokens', help='optional .pt dense token tensor, loaded with weights_only=True')
    parser.add_argument('--greedy', action='store_true', help='default: shared teacher-forced continuation across stages')
    args = parser.parse_args()
    check_manifest(args.validation_report)
    if not torch.cuda.is_available() or not Path(args.checkpoint).is_dir():
        raise RuntimeError('requires CUDA and a local checkpoint')
    if min(args.steps, args.repeats, *args.lengths) < 1 or args.warmup < 1:
        raise ValueError('positive lengths/steps/repeats and at least one warmup are required')
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float16
    model = PHBAForCausalLM.from_pretrained(args.checkpoint, torch_dtype=dtype, local_files_only=True).cuda().eval()
    if model.config.phba_ablation != 'full':
        raise ValueError('main decode comparison must use full PHBA, not a quality-ablation checkpoint')
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'raw.jsonl').exists():
        raise FileExistsError('use a new output directory; do not mix independent benchmark runs')
    metadata = {'args': vars(args), 'model_config': model.config.to_dict(), 'torch': torch.__version__,
                'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
                'parameter_count': sum(p.numel() for p in model.parameters()),
                'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'note': 'P0 is an eager boundary-compatible reference, not an optimized recurrent decode baseline; '
                        'these are within-checkpoint stage comparisons, not superiority claims over other architectures.'}
    (directory / 'environment.json').write_text(json.dumps(metadata, indent=2, default=str))
    all_rows = []
    for length in args.lengths:
        generator = torch.Generator().manual_seed(args.seed)
        tokens = (torch.load(args.tokens, map_location='cpu', weights_only=True) if args.tokens else
                  torch.randint(model.config.vocab_size, (1, length+args.steps), generator=generator))
        if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] < length+args.steps:
            raise ValueError('tokens must be [1, >=prompt+steps]')
        tokens = tokens[:, :length+args.steps].to('cuda', dtype=torch.long)
        for stage in args.stages:
            status = {'stage': stage, 'prompt': length}
            try:
                status['short_reference_rmse'] = verify_short(model, stage, dtype)
                for _ in range(args.warmup):
                    measure(model, stage, tokens, length, args.steps, dtype, args.greedy)
                for repeat in range(args.repeats):
                    gc.collect()
                    torch.cuda.empty_cache()
                    row = measure(model, stage, tokens, length, args.steps, dtype, args.greedy)
                    row['repeat'] = repeat
                    all_rows.append(row)
                    with (directory / 'raw.jsonl').open('a') as stream:
                        stream.write(json.dumps(row)+'\n')
                status['status'] = 'passed'
            except torch.cuda.OutOfMemoryError as error:
                status.update(status='OOM', error=str(error))
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as error:
                status.update(status='FAILED', error=repr(error))
                (directory / f'{stage}_{length}_status.json').write_text(json.dumps(status, indent=2))
                raise
            (directory / f'{stage}_{length}_status.json').write_text(json.dumps(status, indent=2))
    with (directory / 'summary.csv').open('w', newline='') as stream:
        fields = ['stage', 'prompt', 'repeat', 'prefill_s', 'decode_wall_ms_per_step', 'peak_allocated_gib']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in all_rows)
    lines = ['| Stage | Prompt | Runs | Prefill s | Decode wall ms/step | Peak allocated GiB |',
             '|---|---:|---:|---:|---:|---:|']
    for length in args.lengths:
        for stage in args.stages:
            rows = [r for r in all_rows if r['stage'] == stage and r['prompt'] == length]
            if rows:
                vals = [statistics.median([r[k] for r in rows]) for k in
                        ('prefill_s', 'decode_wall_ms_per_step', 'peak_allocated_gib')]
                lines.append(f'| {stage} | {length} | {len(rows)} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} |')
    (directory / 'summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
