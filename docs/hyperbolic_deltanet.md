# HyperbolicDeltaNet (experimental)

This implements the base hyperbolic KV projection rule and a slow explicit
protected-association reference mode. The name is provisional. The
construction uses established Lorentz hyperplane projection mathematics; this
implementation does not claim that the underlying projection is novel or that
hyperbolic geometry universally reduces interference. The reference tests include
both an interference-cancellation example and a counterexample.

## Layout

- Layer: `fla/layers/hyperbolic_deltanet.py` (`HyperbolicDeltaNet`).
- Hugging Face configuration/model: `fla/models/hyperbolic_deltanet/`.
- All operators: `fla/ops/hyperbolic_delta_rule/`.
- Operator tests: `tests/ops/test_hyperbolic_delta_rule*.py`.
- Integration tests: `tests/layers/test_hyperbolic_deltanet.py` and
  `tests/models/test_hyperbolic_deltanet.py`.
- Benchmark: `benchmarks/benchmark_hyperbolic_delta_rule.py`.

The layer and model follow the GSA2 interfaces at base commit
`bac3b4615ea0516de8d8ebc68fe8febaded7a738` on `add-gsa2`, rather than the older
fork main branch. Existing model/kernel implementations are unchanged; only the
public layer/model export lists are extended. The model is registered with
`AutoConfig`, `AutoModel`, and `AutoModelForCausalLM`.

The integration includes optional short convolution and output gating, grouped
value heads, padding/packed input, recurrent caching, shared FLA hybrid-attention
configuration and AttnRes. The causal-LM wrapper excludes cross-sequence targets
from the shared shifted next-token loss for packed input. These integration paths
have tests but have not been run in the development environment described below.

## Exact rule and state convention

Inputs are q/k `[B,T,H,K]`, v `[B,T,H,V]`, and beta `[B,T,H]` or `[B,T,H,V]`.
Initial/final state has shape `[N,H,K,V]`: N=B for dense input, or the number of
packed sequences for B=1 and `cu_seqlens`. Empty sequences preserve their state.

For one head the original homogeneous update is:

```text
a = sum(k*k) - c*v*v
b = beta * (z*v - M^T k) / a
M_new = M + k b^T
z_new = z + c*v*b
output = scale * (M_new^T q) / z_new
```

M is stored as `[K,V]`, transposed relative to a `W k` convention. Vector
multiplication and division outside contractions are elementwise.

To avoid growth of homogeneous coordinates, the public operator/cache stores
**S = M / z**, broadcasting z along K. Resetting z to one is an exact projective
gauge change, not clipping or an added forgetting rule:

```text
r = v - S^T k
u = beta*r / a
den = 1 + c*v*u
S_new = (S + k u^T) / den
output = scale * S_new^T q
```

A returned state is therefore not an unnormalized M. Initial/final-state gradients
refer to this canonical S representation. Readout uses the post-update state
(inclusive causality). For c=0 the rule becomes normalized Delta, with step
beta/||k||^2. For beta=1 the current association is written exactly in exact
arithmetic, subject to the feasible domain.

The required domain is beta in [0,1], `||k||^2-c*v^2 > 0`, and
`c*sum_K(S*S) < 1` for c>0. The layer normalizes keys in FP32, maps zero keys to
a basis vector, and bounds values with tanh and a default margin of 0.5. Queries
are normalized before readout. Curvature defaults to c=1. With `learnable_curvature=True`, a positive scalar is
parameterized by softplus and trained through the PyTorch reference recurrence;
the current Triton kernels do not compute a curvature gradient. With
`protection_rank>0`, the layer retains recent realized KV associations and
projects new constraints into their Lorentz-orthogonal complement. Protection
is also a slow reference path, supports dense unpadded batches only, and retains
at most the configured number of recent associations. If no independent
spacelike direction remains, that channel is left unchanged.

Low-level operators do not implicitly normalize or clamp arbitrary inputs.
`check_inputs=True` validates input values but synchronizes the CUDA device.
Callers disabling checks must maintain the domain and the packed-boundary
contract: integer cu_seqlens on the input device, starting at zero, ending at T,
and nondecreasing. PyTorch references validate values by default.

## Algorithms

`naive_recurrent_hyperbolic_delta_rule` is an independent PyTorch recurrence with
float64 support and ordinary autograd. `naive_chunk_hyperbolic_delta_rule` is an
independent triangular-solve reference.

`fused_recurrent_hyperbolic_delta_rule` implements the forward recurrence in
Triton. `chunk_hyperbolic_delta_rule` has a separate native Triton chunk forward;
it is not a Python loop calling the recurrent forward. Both paths use an analytic
Triton backward with boundary checkpoints and interval replay. Backward does not
invert token transitions, which can be singular at beta=1, and does not retain
every token's full state. It supports gradients of q/k/v/beta and initial/final
states, but not double backward or gradients of the Python curvature scalar.

For each value channel, chunk forward solves:

```text
alpha_i = beta_i / (||k_i||^2 - c*v_i^2)
L_ij = alpha_i * (k_i^T k_j - c*v_i*v_j), j < i
(I + L) b = alpha * (v - K S_in)
z_i = 1 + c * prefix_sum(v*b)_i
```

Causal QK products give the output numerator; dividing by z_i gives the readout.
The state is normalized at chunk boundaries. Backward differentiates the
algebraically equivalent rational recurrence, with expected FP32 rounding
differences from the chunk solve.

### Implementation limits

This is an untuned, correctness-first implementation, not a claim of FLA-level
throughput. Chunk forward repeats q/k Gram calculations per value channel and
uses forward substitution. It has not been optimized to share Gram computation
or use a specialized batched triangular solver. Backward stores one replay
interval per sequence/value tile and reduces q/k partial gradients in PyTorch
without atomics. Runtime memory depends on sequence count and head dimensions.

The intended CUDA contract is FP16/BF16/FP32 input, FP32 state/accumulation,
K<=256, and chunk/checkpoint size 16 or 32. Near-boundary values can make the
triangular systems ill-conditioned; start with the default margin and short
chunks. Learned curvature, explicit protected-memory subspaces, double backward,
other devices, and torch.compile/CUDA-graph compatibility are not implemented or
certified. Model quality and long-training stability have not been measured.

## Validation status

Development environment: Python 3.13, PyTorch 2.10.0+cpu, no CUDA device, and no
Triton/Transformers installation. Package downloads were unavailable.

**Executed: 159 CPU tests passed.** They cover independent homogeneous versus
canonical-gauge equations, recurrent versus triangular-chunk output and all first
derivatives, finite-difference gradcheck, the analytic VJP used by backward
(transcribed in PyTorch), dense/packed/empty sequences, nonzero initial states,
split-cache equivalence, noncontiguous inputs, causality, c=0, beta=0/1, invalid
domains, long-sequence finiteness and checkpoint address bounds. Python syntax
compilation was also performed on the implementation and tests.

**Not executed:** Triton JIT compilation, actual GPU forward/backward correctness,
CUDA memory/race sanitizers, full FLA layer/HF model integration, generation and
GPU benchmarks. The three CUDA-dependent test modules were skipped, not passed.
No H20, B200 or other GPU result is claimed. The CUDA suite must pass before
this branch should be used for GPU training. Repository-wide lint/pre-commit and
regression suites were not run in this environment.

```bash
# CPU algebra, without the repository's GPU-dependent conftest.
python -m pytest --noconftest -q tests/ops/test_hyperbolic_delta_rule_reference.py

# Run in an existing working FLA CUDA environment.
python -m pytest -q tests/ops/test_hyperbolic_delta_rule.py
python -m pytest -q tests/layers/test_hyperbolic_deltanet.py tests/models/test_hyperbolic_deltanet.py

# Optional diagnostics after the ordinary correctness tests.
compute-sanitizer --tool memcheck python -m pytest -xq tests/ops/test_hyperbolic_delta_rule.py
compute-sanitizer --tool racecheck python -m pytest -xq tests/ops/test_hyperbolic_delta_rule.py

# Benchmark only after CUDA correctness passes.
python benchmarks/benchmark_hyperbolic_delta_rule.py
python benchmarks/benchmark_hyperbolic_delta_rule.py --backward
```

## Small model example

Run this only after the CUDA tests pass. It initializes random weights, not a
pretrained model.

```python
import torch
from fla.models import HyperbolicDeltaNetConfig, HyperbolicDeltaNetForCausalLM

config = HyperbolicDeltaNetConfig(
    hidden_size=256, head_dim=64, num_heads=4, num_hidden_layers=2,
    intermediate_size=512, vocab_size=256, attn_mode="chunk",
    curvature=1.0, value_margin=0.5, chunk_size=16,
    use_short_conv=True, fuse_cross_entropy=False,
)
model = HyperbolicDeltaNetForCausalLM(config).cuda().train()
ids = torch.randint(0, config.vocab_size, (2, 128), device="cuda")
loss = model(ids, labels=ids, use_cache=False).loss
loss.backward()
```

`attn_mode="naive"` selects the slow reference operator for debugging. It does not
remove the other FLA runtime dependencies. No pretrained checkpoint or task-quality
improvement is provided by the architecture code alone.
