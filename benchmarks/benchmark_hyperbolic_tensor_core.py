"""Benchmark unprotected HyperbolicDeltaNet old/recurrent/Tensor-Core chunk paths.

Example:
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_hyperbolic_tensor_core.py --seq-len 4096 --backward
"""
import argparse
import statistics
import torch
from fla.ops.hyperbolic_delta_rule.chunk import chunk_hyperbolic_delta_rule
from fla.ops.hyperbolic_delta_rule.fused_recurrent import fused_recurrent_hyperbolic_delta_rule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--seq-len', type=int, default=4096)
    parser.add_argument('--key-dim', type=int, default=64)
    parser.add_argument('--value-dim', type=int, default=64)
    parser.add_argument('--dtype', choices=('fp16','bf16','fp32'), default='bf16')
    parser.add_argument('--chunk-size', type=int, choices=(16,32), default=16)
    parser.add_argument('--backward', action='store_true')
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('Requires CUDA/Triton.')
    B,H,T,K,V=args.batch,args.heads,args.seq_len,args.key_dim,args.value_dim
    dtype=dict(fp16=torch.float16,bf16=torch.bfloat16,fp32=torch.float32)[args.dtype]
    q=torch.randn(B,T,H,K,device='cuda',dtype=dtype)
    k=torch.nn.functional.normalize(torch.randn_like(q).float(),dim=-1).to(dtype)
    v=.03*torch.randn(B,T,H,V,device='cuda',dtype=dtype)
    b=torch.rand(B,T,H,V,device='cuda',dtype=torch.float32)
    s=torch.zeros(B,H,K,V,device='cuda',dtype=torch.float32)
    methods={
        'old_chunk':lambda *a:chunk_hyperbolic_delta_rule(*a[:4],curvature=.5,initial_state=a[4],chunk_size=args.chunk_size,output_final_state=True),
        'tensor_core_wy':lambda *a:chunk_hyperbolic_delta_rule(*a[:4],curvature=.5,initial_state=a[4],chunk_size=args.chunk_size,output_final_state=True,tensor_core=True),
        'fused_recurrent':lambda *a:fused_recurrent_hyperbolic_delta_rule(*a[:4],curvature=.5,initial_state=a[4],checkpoint_size=args.chunk_size,output_final_state=True),
    }
    def once(fn):
        a=[x.detach().requires_grad_(args.backward) for x in (q,k,v,b,s)]
        y,final=fn(*a)
        if args.backward:
            (y.float().square().mean()+final.square().mean()).backward()
    print('GPU:',torch.cuda.get_device_name(),'mode:',args.dtype,'T:',T,'K:',K,'V:',V)
    for name,fn in methods.items():
        for _ in range(args.warmup):
            once(fn)
        torch.cuda.synchronize()
        samples=[]
        for _ in range(args.repeats):
            start=torch.cuda.Event(enable_timing=True)
            stop=torch.cuda.Event(enable_timing=True)
            start.record()
            once(fn)
            stop.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(stop))
        ms=statistics.median(samples)
        print(f'{name}: {ms:.3f} ms, {B*T*1000.0/ms:.0f} tokens/s')


if __name__ == '__main__':
    main()
