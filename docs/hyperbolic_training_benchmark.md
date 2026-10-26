# HyperbolicDeltaNet full-model training benchmark

The runner in [`benchmarks/benchmark_hyperbolic_training.py`](../benchmarks/benchmark_hyperbolic_training.py)
compares complete dense causal-LM training steps with FP32 parameters, an explicitly selected autocast policy,
and a fused AdamW optimizer. It measures the unprotected TC/staged routes, the protected baseline/shared-inverse/staged
routes, and GatedDeltaNet (GDN). GDN is a performance reference with a different update rule, not a numerical oracle for HDN.

The initial legacy-route implementation was validated in a CPU-only environment; those historical checks below do not
establish GPU performance. Subsequent native GDN W1/W8/W16 tiny checks and strict KHDN validation are described in
[the distributed harness](hyperbolic_ddp_harness.md) and the KHDN section below. Example commands are not execution evidence.

## Default configuration and comparison contract

Run commands from the repository root in an environment with the repository's CUDA, PyTorch, Triton, Transformers,
and any selected backend dependencies installed. Choose a GPU with sufficient memory for the requested shape;
the script does not shrink the batch, model, or sequence automatically after an allocation failure.

| Setting | Default and behavior |
| --- | --- |
| Layers / hidden width | 24 / 2048 |
| Heads / Key width / Value width | 16 / 128 / 128; `expand_v=1.0` |
| FFN intermediate width / vocabulary | 5184 / 32000 |
| Batch / sequence length | 8 / 2048 |
| Parameters | FP32 for every method; checked after model construction |
| Autocast | BF16; `--dtype fp32` disables autocast |
| Optimizer | `AdamW(fused=True, betas=(0.9, 0.95))`, learning rate `1e-4`, weight decay `0.1` |
| Outer layer checkpoint | Enabled with `use_reentrant=False`; disable with `--no-layer-checkpoint` |
| Protected inner checkpoint | Disabled; enable with `--inner-checkpoint` |
| Short convolution | Enabled; disable with `--no-short-conv` |
| Fused operations | Norm, SwiGLU, and cross entropy enabled; fused linear cross entropy disabled |
| HDN chunk size | 16; `--chunk-size 32` is also accepted |
| Protected bank rank / initial curvature | 4 / 1.0 |
| Curvature training | Fixed by default; enable with `--learnable-curvature` |
| Initialization seed / data seed | 42 / 43 |
| Warmup / measured / extra continuous steps | 10 / 30 / 10 |
| Recurrent cache | `use_cache=False`; dense, unpadded training inputs |

Model dimensions are configurable through the CLI. This comparison requires
`hidden_size = heads * head_dim`, `head_dim` in `{16,32,...,128}`, rank in `[1,16]`, and initial curvature in `(0,2]`.
It requires at least one warmup step and two measured steps. The actual parameter count and complete model configuration
are saved per method; use those values when describing model size.

The same model-shape fields and optimizer options are used for GDN, but GDN retains its own algorithm, gating,
and operator scheduling. In particular, `--chunk-size` controls HDN and its small-model validation shape; it does not
override GDN's native chunk implementation. Parameter counts need not be identical between the two model families.

The runner sets `torch.backends.cuda.matmul.allow_tf32=False`. This setting applies to PyTorch matrix multiplication;
it does not override explicit precision choices inside Triton. BF16 autocast does not mean every HDN internal GEMM
uses BF16 operands. The strict TC path and the staged path retain their existing explicit precision contracts.

### Method selection and environment flags

Every method explicitly sets `FLA_CI_ENV=0` and `FLA_HYPERBOLIC_TC_FAST=0`, and restores the previous environment afterward.
The following flags are set for both forward and backward, so inherited shell flags cannot silently select another HDN route.

| Method | Group | `FLA_HYPERBOLIC_STAGED` | `FLA_HYPERBOLIC_SHARED_INVERSE` | `FLA_HYPERBOLIC_PROTECTED_STAGED` | Speedup baseline |
| --- | --- | ---: | ---: | ---: | --- |
| `hdn_tc` | Unprotected HDN | 0 | 0 | 0 | None; strict TC baseline |
| `hdn_staged` | Unprotected HDN | 1 | 0 | 0 | `hdn_tc` |
| `protected_value` | Protected HDN | 0 | 0 | 0 | None; existing Value-parallel baseline |
| `protected_shared_inverse` | Protected HDN | 0 | 1 | 0 | `protected_value` |
| `protected_staged` | Protected HDN | 0 | 0 | 1 | `protected_value` |
| `gdn` | GDN | 0 | 0 | 0 | None; performance comparison only |

All six are selected by default. `--methods` selects a subset and the order of full-size measurements.
The shared-inverse flag controls the existing Value-parallel path. Its value of zero for `protected_staged` does not
disable the staged implementation's own shared Bank-Key Gram computation.

Speedups are emitted only when the named baseline was also measured in the same run. The small-model gate automatically
adds an omitted HDN baseline for correctness, but it does not add that baseline to full-size performance measurements.
GDN-relative throughput is emitted only when GDN was measured; it is not an equivalence comparison.

The default, inner-checkpoint-disabled setting avoids an additional replay-policy difference. With
`--inner-checkpoint`, the existing protected route checkpoints each chunk, while the staged protected route checkpoints
the complete staged trajectory. Both honor the public checkpoint request, but the checkpoint partitions and replay
costs differ. Report this setting when comparing those methods. Outer layer checkpointing remains a separate choice.

## What is verified before timing

### Small-model numerical and route gate

The first GPU phase constructs two-layer models with hidden width 32, two heads, Key/Value width 16,
FFN width 64, vocabulary 97, batch 2, and sequence length `2 * chunk_size + 1`.
It preserves the requested autocast, curvature, bank rank, fused-operation, short-convolution, and checkpoint settings.
The sequence crosses chunk boundaries and has a partial final chunk.

Within each HDN group, baseline and candidate must start with identical parameter hashes. The gate compares:

- Scalar loss with the existing model tolerance `atol=1e-4, rtol=1e-3`.
- Every trainable parameter gradient with the existing layer tolerance `atol=2e-3, rtol=2e-2`.
- Parameter names, presence of gradients, and finite gradient values.
- Actual forward and backward execution evidence for the requested implementation.

The tolerances come from
[`tests/models/test_hyperbolic_deltanet.py`](../tests/models/test_hyperbolic_deltanet.py) and
[`tests/layers/test_hyperbolic_deltanet.py`](../tests/layers/test_hyperbolic_deltanet.py).
They are not relaxed by this runner. A failed gate prevents performance measurement and records the failure.
GDN receives loss/gradient finiteness and route checks, but its gradients are not compared with HDN.

These are baseline/candidate model checks, not a new high-precision model reference. Independent mathematical,
reference, packed-input, nonzero-state, curvature-VJP, streaming, and cache checks belong to the relevant operator
and model tests. This runner does not replace them.

### Full-size continuous training checks

Each full-size method first executes one untimed forward/backward pass to attest the actual route and check finite loss
and gradients. It does not perform an optimizer update in that pass. Then warmup, measured, and extra stability steps
all update the same model and optimizer continuously, with no parameter reset at the warmup boundary.

Every step checks finite loss and the norm of all trainable parameter gradients, and rejects missing gradients.
An untimed final check verifies that every parameter remains finite after the last optimizer update, including an
optional profiling step. The raw loss and gradient-norm sequences are retained.

**Full-size baseline/candidate losses and parameter-gradient tensors are not compared for numerical equivalence.**
The small-model gate provides that comparison. Finite continuous training is a different check and does not prove
long-term convergence or equal training trajectories. Continuous steps refer to optimizer updates; recurrent state
and protected-bank cache are not carried from one training batch to the next because `use_cache=False`.

### Actual route evidence

HDN entry kernels are observed during the untimed forward/backward pass, including checkpoint replay.
The trace is removed before timing. A missing required stage or mismatched flag raises an error instead of reporting
the fallback implementation as the requested optimization.

| Method | Required evidence |
| --- | --- |
| `hdn_tc` | `hyperbolic_delta_tc_chunk_fwd` and `hyperbolic_delta_tc_chunk_bwd`, both with `FAST=False` |
| `hdn_staged` | All seven `hyperbolic_staged_*_kernel` stages: `prepare_fwd`, `state_fwd`, `output_fwd`, `output_bwd`, `state_bwd`, `values_bwd`, `parameters_bwd` |
| `protected_value` | Value-tile forward/backward with `SHARED_INVERSE=False`, plus soft bank-update backward |
| `protected_shared_inverse` | Value-tile forward/backward with `SHARED_INVERSE=True`, soft bank-update backward, and `shared_key_gram_fwd_kernel` |
| `protected_staged` | Shared/value trajectory forward/backward, Value and bank partial-gradient stages, and `shared_key_gram_fwd_kernel` |
| `gdn` | Actual `chunk_gated_delta_rule_fwd` and `chunk_gated_delta_rule_bwd` calls, plus observed common-backend calls |

GDN tracing directly wraps known entry/backend functions so that autograd worker threads are also observed.
Additional Python backend calls are recorded. `tilelang_backward_observed=true` is positive evidence of an executed
TileLang backward entry. A false value is not proof of a pure Triton implementation. Use `--profile` for an additional
CUDA/CPU kernel trace; do not infer the executed backend merely from an installed package or compiler log.

## Reproducibility, timing, and memory boundaries

### Weights and inputs

Before every model construction, the runner resets the initialization seed. It hashes the initial CPU state dictionary,
including tensor names, shapes, dtypes, and contents, before moving that model to CUDA. Full-size hashes must agree
within each HDN group, even when the user changes the order of methods.

The input generator uses the independent data seed. All input batches are generated and transferred before timed
steps, using token IDs from `[3, vocab_size)`. The same tensor is reused by all full-size methods. Its SHA256 is saved.
Small-model tokens have their own recorded hash. A fresh optimizer is created for every method.

Only one full-size model and optimizer are alive on the GPU at a time. Models are measured sequentially and released
between methods; the allocator cache is emptied between methods, not between timed steps. The script does not randomize
method order. Repeat runs with reversed `--methods` order when investigating clock, thermal, or order effects.

Hashes are identity checks, not saved training checkpoints. JSON does not contain full weights, optimizer state,
gradient tensors, or all intermediate activations. Reproduce an error by using the recorded revision, software,
arguments, flags, and seeds; the recorded error summaries alone cannot reconstruct gradient tensors.

### Timed interval

For each optimizer step, `zero_grad(set_to_none=True)`, a pre-step CUDA synchronization, and event construction happen
before the wall clock starts. Four CUDA events delimit:

1. Model forward, including the language-model loss.
2. Backward, including any outer and inner checkpoint replay.
3. The fused AdamW update.

There is no inserted synchronization between those three phases. After the final event, the runner synchronizes CUDA
and closes the wall interval. The result contains per-phase CUDA milliseconds, total CUDA milliseconds, and synchronized
total wall milliseconds. Host launch overhead and the final synchronization are represented in the wall measurement.
No separate synchronized wall duration is claimed for each phase.

Loss extraction and gradient health checks run after that interval. Data generation, weight hashing, model setup,
kernel-route tracing, and optional profiling are also outside the measured samples. The initial untimed route pass
and warmup keep compilation, autotuning, and first optimizer-state allocation out of steady-state summaries, but the
runner does not claim to measure pure compilation time. `untimed_trace_wall_ms` includes route-observation overhead.

`warmup_samples`, `samples`, and `stability_samples` are separate arrays. Extra stability steps use the same step helper
and therefore retain diagnostic timings, but they are excluded from reported latency/throughput distributions.
Default continuous optimizer updates total 50: 10 warmup, 30 measured, and 10 extra. Profiling adds one further update.

Throughput is `batch * seq_len * 1000 / total_wall_ms` for each measured step. It counts input tokens, not only the
shifted, nonignored loss targets. Summaries contain median, linearly interpolated p95, min, max, and population standard
deviation. `tokens_per_second.median` is the median of per-step throughputs; phase speedups use ratios of median latencies.
For an even sample count, a median throughput need not be exactly the reciprocal of a median latency.

### Allocated and reserved memory

Peak counters are reset after optimizer warmup and captured after the measured loop, before extra stability steps,
profiling, and the final parameter-finiteness check. Reported memory includes the FP32 model, gradients, optimizer state,
activations/temporary buffers, and all pregenerated input batches. Per-step gradient-health operations are untimed but
occur within this peak-counter window. This is whole-process training memory, not isolated HDN-kernel workspace.

`start_allocated_bytes` and `start_reserved_bytes` describe the warmed state at the start of measurement.
`peak_allocated_bytes` is the PyTorch allocator's live-allocation peak; `peak_reserved_bytes` includes its reserved pool.
These are not device-wide memory measurements and do not enumerate every non-PyTorch CUDA allocation. Convert bytes
to GiB by dividing by `2**30`. The script does not subtract the starting values to manufacture an operator-only peak.

## Commands

All examples below use actual parser options. The default output directory is under the ignored `profile/` tree.

### Host inspection and small-model GPU gate

```bash
python benchmarks/benchmark_hyperbolic_training.py --help

python benchmarks/benchmark_hyperbolic_training.py --dry-run \
  --methods hdn_tc hdn_staged protected_value protected_shared_inverse protected_staged gdn

python benchmarks/benchmark_hyperbolic_training.py --check-only \
  --methods hdn_tc hdn_staged protected_value protected_shared_inverse protected_staged gdn \
  --output profile/hyperbolic-opt/training-correctness.json
```

`--help` and `--dry-run` work without CUDA. Dry run prints validated configuration JSON to stdout; it does not construct
models, execute correctness gates, or write the `--output` file. `--check-only` runs only the small-model GPU gate,
regardless of the requested full-size batch/sequence/model dimensions. It does not time the large model.

When CUDA is unavailable, a nondry execution writes `GPU NOT RUN: CUDA device unavailable` to the JSON status and exits 2.
Invalid arguments also exit 2, but fail before JSON generation. A valid dry run exits 0 and reports `DRY RUN; GPU NOT RUN`.

### Four original training shapes

Each command measures all six methods, with the same 24-layer model defaults and 16,384 input tokens per step.
Select a smaller model or batch explicitly if a device cannot fit a requested configuration, and label it as a different
comparison. These commands have not been executed on a GPU in this environment.

```bash
python benchmarks/benchmark_hyperbolic_training.py --batch 8 --seq-len 2048 \
  --warmup 10 --steps 30 --stability-steps 10 --seed 42 --data-seed 43 \
  --output profile/hyperbolic-opt/training-2k-b8.json

python benchmarks/benchmark_hyperbolic_training.py --batch 4 --seq-len 4096 \
  --warmup 10 --steps 30 --stability-steps 10 --seed 42 --data-seed 43 \
  --output profile/hyperbolic-opt/training-4k-b4.json

python benchmarks/benchmark_hyperbolic_training.py --batch 2 --seq-len 8192 \
  --warmup 10 --steps 30 --stability-steps 10 --seed 42 --data-seed 43 \
  --output profile/hyperbolic-opt/training-8k-b2.json

python benchmarks/benchmark_hyperbolic_training.py --batch 1 --seq-len 16384 \
  --warmup 10 --steps 30 --stability-steps 10 --seed 42 --data-seed 43 \
  --output profile/hyperbolic-opt/training-16k-b1.json
```

Use `CUDA_VISIBLE_DEVICES` externally to choose a device if needed. This is a single-device runner, not a distributed
training launcher. Different precision, checkpoint, rank, or chunk settings should use separate output files.

### Profiling and configuration variants

```bash
python benchmarks/benchmark_hyperbolic_training.py --batch 8 --seq-len 2048 \
  --methods hdn_tc hdn_staged gdn \
  --profile profile/hyperbolic-opt/traces-2k-b8 \
  --output profile/hyperbolic-opt/training-2k-b8-profiled.json

python benchmarks/benchmark_hyperbolic_training.py --check-only --dtype fp32 \
  --learnable-curvature --chunk-size 32 \
  --methods hdn_tc hdn_staged protected_value protected_shared_inverse protected_staged \
  --output profile/hyperbolic-opt/training-fp32-c32-learned-correctness.json

python benchmarks/benchmark_hyperbolic_training.py --batch 8 --seq-len 2048 \
  --inner-checkpoint --methods protected_value protected_shared_inverse protected_staged \
  --output profile/hyperbolic-opt/training-protected-inner-checkpoint.json
```

Profiling executes one additional optimizer step after measurement and stability checks, reusing the last data batch.
It writes `<method>.json` Chrome traces inside the chosen directory; those samples are excluded from speed and memory
summaries. A CUDA/CPU timeline is not an Nsight Compute register/occupancy report. Hardware-specific dtype/backend
support still has to pass the actual gate; a syntactically valid command is not evidence that a GPU configuration passed.

## Inspecting the JSON evidence

Check the top-level status before using any numbers. The runner saves the completed small-model gate and then each
completed full-size method. If a later method fails, earlier results remain, but the overall status becomes
`GPU VALIDATION OR TRAINING BENCHMARK FAILED` and `failure` contains the traceback. A failing method's partially collected
step list is not checkpointed to JSON. Do not treat an overall failure or a partial run as an all-method pass.

| JSON location | What to check |
| --- | --- |
| `metadata` | Revision, dirty-tree flag, Python/Torch information, GPU runtime/device information when available, arguments, command, and relevant environment |
| `methods` | Method names, comparison groups, explicit environment flags, and named baselines |
| `training_contract` | Parameter/autocast dtype, optimizer, checkpoint policy, and stated validation/timing scope |
| `correctness.methods.<name>` | Small-model loss, finite gradient norm, model/hash, and actual route evidence |
| `correctness.methods.<name>.errors_against_baseline` | Loss error, per-parameter max-absolute and relative-RMS gradient errors, and unchanged tolerances |
| `correctness.tokens_sha256` / `data_sha256` | Small and full input identities |
| `measurements.<name>.model` | Actual parameter count, initial weight SHA256, FP32 parameter dtype, full configuration |
| `measurements.<name>.route_evidence` | Full-size required kernel launches/flags, or observed GDN backend calls |
| `measurements.<name>.samples` | Measured per-step timings, loss, gradient norm, current allocator counts |
| `warmup_samples` / `stability_samples` within each method | Continuous updates outside the steady-state summary |
| `summary_ms` / `tokens_per_second` within each method | Latency and throughput distributions |
| `memory` / `final_parameter_health` within each method | Measurement-window peaks and final weight finiteness |
| `speedup_against_baseline` | Same-run ratio of baseline/candidate median latency; omitted without the measured baseline |
| `training_throughput_fraction_of_gdn` | GDN median wall latency divided by HDN median wall latency; a performance comparison |
| `profile_trace` | Optional post-measurement Chrome trace path |

For example, independently recompute the wall-time median and p95 from the raw measured samples:

```bash
python - <<'PY'
import json
import math
import statistics
from pathlib import Path

report = json.loads(Path('profile/hyperbolic-opt/training-2k-b8.json').read_text())
print('status:', report['status'])
print('revision:', report['metadata']['revision'], 'dirty:', report['metadata']['working_tree_dirty'])
if 'failure' in report:
    print(report['failure'])
for name, result in report.get('measurements', {}).items():
    wall = sorted(row['total_wall_ms'] for row in result['samples'])
    index = 0.95 * (len(wall) - 1)
    low = math.floor(index)
    p95 = wall[low] + (wall[math.ceil(index)] - wall[low]) * (index - low)
    print(name, 'samples:', len(wall), 'wall median/p95:', statistics.median(wall), p95)
    print('stored:', result['summary_ms']['total_wall_ms'])
    print('initial weights:', result['model']['initial_weights_sha256'])
    print('peak allocated/reserved GiB:', result['memory']['peak_allocated_bytes'] / 2**30,
          result['memory']['peak_reserved_bytes'] / 2**30)
    print('route:', result['route_evidence'])
PY
```

Relative RMS is supplementary error information; it cannot overrule an elementwise tolerance failure. Recheck the
recorded revision and dirty-tree flag before attributing a result to a particular commit. Compare precision, data,
checkpoint settings, shape, measured sample count, and actual routes before using speedup fields.

## Verification performed on the CPU host

At preparation of this document, the following actual command passed:

```bash
FLA_CI_ENV=0 python -m pytest --noconftest -q \
  tests/benchmarks/test_hyperbolic_training_benchmark.py
```

Result: **26 passed, 0 failed, 0 skipped, 1 expected CPU-platform warning, 8.39 seconds** in the task's
PyTorch 2.8.0 CPU environment. This is the standalone host-suite result before any later merged regression run.

The suite verifies parser/defaults, invalid inputs, explicit no-CUDA status, method flags, weight identity and
gradient-error rejection, final parameter checks, required forward/backward evidence, worker-thread tracing,
and comparison arithmetic. A real-constructor integration test builds all six small FLA/Transformers models,
temporarily replacing only `nn.Module.cuda` with identity. It verifies CPU/FP32 parameters, matching group hashes,
and two real `GradientCheckpointingLayer` blocks with nonreentrant checkpoint hooks. It executes no model forward
or backward and no CUDA kernel.

`--help` and `--dry-run` were checked on CPU. An actual `--check-only` invocation without CUDA wrote the explicit
`GPU NOT RUN` status and exited 2. Ruff, autopep8 diff, compileall, and diff-whitespace checks were also clean.
None of these results establish GPU correctness, throughput, peak GPU memory, long-sequence stability, or convergence.

## Relation to the optimization work

The scheduling and sharing implementations remain opt-in; see
[`hyperbolic_staged.md`](hyperbolic_staged.md),
[`hyperbolic_shared_inverse.md`](hyperbolic_shared_inverse.md), and
[`hyperbolic_protected_staged.md`](hyperbolic_protected_staged.md) for their supported scope.
The runner's route checks are intended to prevent a fallback or another precision path from being mislabeled as one
of these optimizations. Operator microbenchmarks remain useful for attributing a full-model change to forward,
backward, launch, or workspace behavior.

**A2 per-GEMM precision selection is not implemented by this runner.** `--dtype` changes autocast, not a per-GEMM
HDN policy. The existing global fast TC path is deliberately disabled in all six methods. The A2 mathematical study,
negative numerical examples, cast-gradient contract questions, and required RFC blocker are documented in
[`hyperbolic_precision.md`](hyperbolic_precision.md). Do not describe these measurements as testing an unimplemented
QK-only or Gram-specific low-precision kernel.

GPU acceptance still requires successful operator/reference checks, the actual small-model gate, representative
full-size measurements on the same hardware/software, and longer training evidence appropriate to the intended use.
The current host evidence supports running that verification workflow; it does not establish readiness of the new
GPU paths for production training.

## Experimental strict KHDN integration

`khdn_no_carry` remains the comparison baseline. Both optimized KHDN methods require `--khdn-strict`; this selects the existing FP64 SIMT device implementation, with FP32 parameters and the requested public/autocast dtype. It makes no internally FP32 or Tensor Core claim. C64 is permitted for KHDN only; the legacy HDN routes retain C16/C32. KHDN has no protected inner-checkpoint policy, so `--inner-checkpoint` is rejected. Missing compiled IR/PTX, stage launches, backward, C/g/carry metadata, domain diagnostics, or an exceptional route rejects the gate.

The untimed trace wraps the real exported operator and all loaded FLA aliases, captures executed autograd options and seven source descriptors, retains the injected telemetry dictionary until original-context backward completes, and records each real JIT launch. Per-layer backward coverage is required. Nonreentrant recomputation may create additional forward-only dictionaries; these are retained without inventing backward telemetry. Replay must actually be observed for every trained layer when outer checkpointing is requested. Each stage records bound positional/keyword launch arguments, pointer/buffer dtypes, compiled IR/PTX hashes and inspected Tensor Core opcodes. The IR parser strips comments/location metadata and parses supported same-type scalar/tensor/vector arith.mulf/addf and math.fma type clauses; unknown or multiline operation syntax fails closed. The independent PTX check rejects non-FP64 floating add/sub/mul/fma/mad/div/sqrt/rcp opcodes. This does not certify every instruction, conversion or library call. Source contraction sites are identified as source sites, not runtime counts or profiler attribution. Real preparation buffers supply denominator, positive mass and normalizer summaries. An additional bounded FP64 projection checks saved Y against the actual F and saved canonical S; terminal carry buffers must contain exact zeros, including when zero carry edges are expected.

Before model timing, dense and packed small g32 KHDN models compare loss and every parameter gradient with the explicit independent dense Oracle. This supplemental witness enables learned positive curvature and requires f_proj, A_log, dt_bias and curvature_raw gradients. It preserves the existing loss and parameter-gradient budgets. A separate operator witness compares output, final state and all seven source gradients against unrounded dense FP64 truth, using the existing operator elementwise and relative-RMS budgets. DDP serial-update, valid-label weighting, zero-label handling, checkpoint generation and recovery contracts remain separate gates. A full-size untimed route pass uses the actual configured model and batch before any measured sample. Every original and recompute record must match the actual dense B/T or the exact packed/unpadded token count and sequence boundaries. Explicit cu_seqlens takes precedence over a padding mask, matching the layer; mask-derived boundaries retain empty documents.

Normal timed execution has no diagnostic/projection hooks or launch wrappers. Before each sample, the harness replaces only exceptional host forward/backward and reference/recurrent entry aliases with raising sentinels. An unexpected route fails before fallback work; failure journals retain the event. The production core's original range/finiteness checks remain inside the measured operator. This is an exception monitor, not a proof that arbitrary future tensors remain in a normal numerical domain. Reserved-memory accounting retains the existing process-allocator scope.

A first true W1 NCCL/DDP check, after independent review and source packaging, can use the following launch. Strong mode requires CLI batch8/seq2048, while check-only still runs tiny gate models/batches. The executor must deploy the matching canonical manifest and place all writable outputs, logs, optional profiles and compiler caches outside the immutable source tree. This example assumes source deployment under /tmp/khdn-training-fixed/source and external run storage:

```bash
FLA_SOURCE_MANIFEST=/tmp/khdn-training-fixed/snapshot.json \
TRITON_CACHE_DIR=/tmp/khdn-training-fixed/cache/triton \
TORCHINDUCTOR_CACHE_DIR=/tmp/khdn-training-fixed/cache/inductor \
CUDA_CACHE_PATH=/tmp/khdn-training-fixed/cache/cuda \
TORCH_EXTENSIONS_DIR=/tmp/khdn-training-fixed/cache/extensions \
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=1 --max-restarts=0 \
  --log-dir /mnt/cephfs/akseli/hyperbolic-validation/khdn-training-fixed/torchrun \
  /tmp/khdn-training-fixed/source/benchmarks/benchmark_hyperbolic_training.py \
  --methods khdn_no_carry --khdn-strict --check-only --dtype fp32 --scaling strong \
  --layers 2 --hidden-size 64 --heads 2 --head-dim 32 --intermediate-size 128 --vocab-size 97 \
  --batch 8 --seq-len 2048 --chunk-size 16 --group-size 32 --learnable-curvature \
  --loss-backend plain --ddp-timeout 90 --task-timeout 900 \
  --output /mnt/cephfs/akseli/hyperbolic-validation/khdn-training-fixed/khdn-w1.json
```

### Operator pilot and actual long shapes

The operator sweep is a separate bounded single-device path in the same script; it does not initialize DDP. B=H=1 and K=V=`--head-dim`; every JSON/profile row records the actual T. It uses fresh graphs for backward-only and F+B, raw samples, separate warmup and a separate optional profiler sample. It does not stand in for full-model profiling. Default lengths 17/33/65 form the first pilot:

```bash
python benchmarks/benchmark_hyperbolic_training.py \
  --methods khdn_no_carry khdn_y_carry --khdn-strict --operator-sweep --dtype fp32 \
  --hidden-size 32 --heads 1 --head-dim 32 \
  --modes fwd_bwd --chunk-size 16 --group-size 32 --operator-lengths 17 33 65 \
  --warmup 1 --steps 2 \
  --output /mnt/cephfs/akseli/hyperbolic-validation/khdn-training-fixed/operator-pilot.json
```

Long/full-model expansion requires a separately reviewed run scope. The CLI accepts explicit `--operator-lengths 2048 8192 16384` for actual long inputs, and `--modes fwd`, `bwd`, or `fwd_bwd` in separate invocations. `--check-only` performs correctness without timing. `--profile DIR` exports an additional real-shape Chrome trace. The scope remains B=H=1; full-model B8/H16 and all-layer coverage must come from the model path. A first long pilot can request T2048 alone before proceeding to T8192/T16384; no automatic shape reduction occurs on failure.

Long correctness uses nonreentrant checkpointing around independent dense-Oracle temporal chunks (`--oracle-chunk-size`, default16). Canonical S connects adjacent chunks, retaining the complete q/k/v/beta/log_decay/S0/c gradient chain. Each case first compares this harness against the uncheckpointed independent dense Oracle on a bounded witness. It then evaluates actual long FP64 truth without retaining every token's dense-solve graph. The truth never calls production chunk kernels. This reduces retained AD activation memory; it does not remove the dense Oracle's compute cost or claim long-shape feasibility without GPU measurement. High-precision exceptional routes remain outside strict timing acceptance.

The fixed integration has passed 217 CPU cases and targeted independent review of compiled-type parsing and actual token-domain binding. Actual H20 FP32/BF16 tiny W1 checks passed for both carry modes, including same-model Oracle parity, all-parameter gradients, checkpoint replay and zero fallback. Two unchanged 24-layer BF16/plain-CE B8x2048 pilots each completed five finite updates and checkpoint reload with the actual 1,412,912,536-parameter learned-curvature model. Their two measured step samples per method are feasibility evidence, not the required 10-warmup/30-sample/three-repeat performance acceptance. Multi-rank full KHDN training, broader precision coverage and formal performance remain pending. Separately discovered extreme-query and c0/large-state numerical limits are under repair; normal model checks do not certify that whole input domain.
