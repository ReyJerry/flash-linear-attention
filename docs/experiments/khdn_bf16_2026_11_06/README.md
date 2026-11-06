# BF16 KHDN full-model code and H20 timing

This release publishes the three-product BF16 KHDN implementation, its public API integration, and the exact frozen scripts behind the measured 24-layer B8/T2048 comparison. Candidate contractions use BF16 operands with FP32 accumulation; state and control arithmetic use FP32. Candidate training never falls back to FP64. Explicit untimed high-precision references remain separate.

## Same-card full training results

All trials used the same H20 (`GPU-e70e1a7f-5eaf-94b2-ecde-06ec00c56e92`), 24 layers, hidden2048, H16/K=V128, FFN5184, vocabulary32000, FP32 parameters, BF16 autocast, ShortConv, nonreentrant layer checkpointing, plain cross-entropy and AdamW. KHDN uses g32/C16/no_carry, fixed positive curvature c=1 and trainable nonidentity key-channel gates. Each mode has 10 warmups and 30 steady observations, repeated in three fresh-process trials with varied execution order. No long-tail observation was discarded.

| Model/comparison | Parameters | Trial step medians, ms | Median paired KHDN time ratio |
| --- | ---: | --- | ---: |
| Native KDA | 1,324,881,280 | 1673.302 / 1673.616 / 1672.798 | 2.582592 |
| KHDN in native comparison | 1,412,912,512 | 4321.261 / 4322.266 / 4321.852 | — |
| KDA with matched output projection | 1,412,912,512 | 1759.423 / 1760.009 / 1759.364 | 2.456452 |
| KHDN in matched comparison | 1,412,912,512 | 4322.526 / 4323.180 / 4321.794 | — |

The matched KDA control replaces only its 24 output-gate projections with full 2048×2048 bias-free Linear layers. It is a separately named research variant, not native KDA. Both models retain sigmoid gated RMSNorm. The native parameter difference is 88,031,232, arising from this projection difference. The matched comparison gives KHDN about 40.71% of KDA's throughput; it does not prove an intrinsic architectural slowdown.

The measured step includes forward, loss, backward, gradient clipping, AdamW and the candidate's current input/nonfinite checks. It excludes data transfer, `zero_grad` and untimed diagnostic checks. Forward-only, backward-only and directly measured F+B observations are also included; their medians are not added to synthesize a training-step result. Inputs contain 16384 token slots and 16376 valid next-token labels per microstep. Pilot peak allocated memory was 24.435 GiB for native KDA, 25.587 GiB for matched KDA and 30.204 GiB for KHDN.

Run the CPU-only audit from the repository root:

```bash
python docs/experiments/khdn_bf16_2026_11_06/summarize.py
```

It verifies artifact hashes, frozen script hashes, all 1920 original observations, and the per-trial wall/CUDA median, p95, min/max and sample standard deviation. `evidence.json` preserves qualification, failure and provenance records. This command does not rerun training or independently re-audit remote gradient/checkpoint tensors.

## Public API and measured source identity

The public implementation is in `fla/ops/key_gated_hyperbolic_delta_rule/bf16.py` and `bf16_kernels.py`. Select it explicitly:

```python
from fla.models import KeyGatedHyperbolicDeltaNetConfig

config = KeyGatedHyperbolicDeltaNetConfig(
    attn_mode='chunk', precision_policy='bf16', chunk_size=16, group_size=32,
)
```

Low-level chunk and recurrent entry points also accept `precision_policy='bf16'`. Short eval/decode automatically preserves this policy. The route accepts BF16/FP32 tensors, returns FP32 canonical state, supports first-order source gradients, and rejects unsupported runtimes or FP64 inputs instead of replaying through the old strict path. The currently qualified environment is NVIDIA SM90, PyTorch 2.9.0+cu128, Triton 3.5.0, with patched `libtriton.so` SHA256 `1f48e0f0f7fb841418645069b378b787e41338192b5fd1ac81f95e1ffb05ed32`. A version string alone is insufficient; the compiler patch is under `scripts/patches/`. Other builds require qualification, not removal of the guard.

The timing above used an isolated private wrapper bound into the model, with kernel SHA256 `758b1b059e76a264ec5d5dc0c9627c60f013378f8d4bd99116d547b6512a1baa` and wrapper SHA256 `5800eb90482c1e441f0e9af7e8e5e895518d8946331652396cd50ba40804e544`. The public kernel's executable AST is identical; packaging, runtime rejection, autocast/input decorators and compact telemetry differ in the public wrapper. **The public wrapper has not yet been re-timed end to end.** Do not describe 4.323 seconds as a new measurement of this checkout's integrated API.

The public API remains opt-in while full integration validation continues. The legacy default is still `strict`, which contains FP64; selecting BF16 autocast alone does not select the new no-FP64 policy. This staged publication does not complete the BF16-first default transition or the full engineering objective.

## Exact experiment sources and replay requirements

`timing-sources.tar.gz` contains four byte-preserved experiment directories: `native/pilot`, `native/repeats`, `matched/pilot`, and `matched/repeats`. Each includes the actual `measure.py`, `timing_runtime.py`, `runner.py`, `kernels_direct.py`, `operator_direct.py`, model/timing plans, contract and release hash manifest. The matched directories also contain `control_model.py`; repeat directories contain `repeat_runner.py`. These archived historical scripts retain their original comments and machine-bound paths. The completed status is established by the verified results, not an old pre-submission label in a release plan.

`timed-source-overlay.tar.gz` supplies the exact 16 source overrides plus `snapshot.json`, the isolated `run_project.py` entry point and recorded runtime settings. Reconstruct the measured repository from Git base `e1a1970ba51715951eff28feb03c97211937977e` plus these overrides and apply the manifest's recorded file modes; its manifest identity is `9e8e944207fe29c6dabef646ce0539d41a01c173b3a59ae62ee11a796f9247d3`. All file contents and symlink targets were checked against this reconstruction. Use an isolated directory, never overwrite an active checkout. The archived entry point validates every source hash and the compiler before importing the project.

`replay-inputs.tar.gz` includes the finite/recovery admission receipts, the original cache index, the ten required shared autotune choices and the matched KDA control's sealed tuning choices. Paths are relative to the original `/mnt/cephfs/akseli/hyperbolic-validation` root. The shared `triton-cache.tar.gz` is deliberately reduced to the hash-identical autotune JSON files; compiled kernels are regenerated. These are evidence/tuning inputs, not model weights, a corpus or a bundled compiler binary.

On the existing qualified runtime, the trial command used this form after staging one frozen experiment directory and reserving an otherwise idle GPU/node:

```bash
CUDA_VISIBLE_DEVICES=GPU-e70e1a7f-5eaf-94b2-ecde-06ec00c56e92 \
KHDN_RUN_ID=khdn-replay-unique \
BF16_PROBE_OUT=/absolute/new-empty-result-directory \
BF16_PROBE_SHARD=0 TIMING_TRIAL=0 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 FLA_CI_ENV=0 FLA_CONV_BACKEND=triton \
/opt/conda/bin/python3 -I /tmp/khdn-named-fs7-9e8e944207fe29c6/run_project.py \
-B /absolute/extracted/matched/pilot/runner.py
```

The runner creates fresh processes for both models. For trials 1/2 use the corresponding repeats directory's `repeat_runner.py`. Use a new run ID/output directory for an intentional new measurement; do not reuse an uncertain live submission. The GPU/runtime paths above record the measured environment, not portable defaults. Another installation must explicitly provision the qualified dependency/compiler environment and relocate recorded input paths; a new machine's throughput must be measured independently. Source inspection and the CPU audit require no private cluster access.

## Validation and remaining limits

Public integration has 19 host precision/config/decode-wiring tests and 63 distinct native H20 operator tests passing, with zero failures or skips. The initial 43 GPU cases cover all seven source gradients, output/state/joint cotangents, dense/packed, C16/C32 and Y-carry, plus C1 empty/one/multitoken recurrence and FP64 rejection. A separate 20-case run adds aliased/nonleaf/noncontiguous/independent-storage inputs, three backward requests on one retained graph, strong decay (including K128/g32), nonzero dc at c=0, double-backward rejection and compiler rejection. Repository NaN allocation poisoning and candidate no-FP64 tensor checks apply. Raw JUnit/log/supervisor records are in `public-test-results.tar.gz`. The two source snapshots differ only by the added tests; final publication changes only two documentation strings afterward.

The private candidate's independent 108-case operator matrix retains 105 passes and **three numerical failures** in scalar C64 dc RMS. The user explicitly accepted those three bounded exceptions to avoid six-product compensation; no other tolerance was relaxed. Original elementwise failures remain recorded. The candidate uses three products, not the retired six-product variant.

Three seeds completed 16-position short full-model trajectories (one retained prefix update plus 15 continuation updates), same-current-weight gradient comparisons, heldout checks and exact fresh-process 8→16 recovery. The final heldout PPL differences were +0.02456%, +0.02773% and −0.00672%. This is synthetic engineering validation, not real-corpus quality or long-term stability. The earlier `mixed_readout` 24-layer gradient failure remains a failure.

A separate full-model profile attributed 50.9% of summed GPU kernel time to the four common/state forward/backward kernels, with high register pressure. Increasing warp counts, globally capping registers and one early-store reorder did not improve the measured kernel chain; none is promoted. Profile percentages and compiler spill metadata are not causal speedup proofs.

Public full-model cache/reorder/save-load regression, integrated full-model numerical/timing qualification, required full 1/8/16-GPU strong/weak scaling, longer-shape finite training, BF16 Y-carry full-model comparison and final external review remain incomplete. PR #2 stays draft; this release is not a claim of full production acceptance or completion.
