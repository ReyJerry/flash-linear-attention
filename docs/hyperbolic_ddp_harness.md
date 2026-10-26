# Hyperbolic training/DDP harness

The recovery, evidence-retention and archive-metadata fixes have passed targeted independent review. Native GDN FP32 tiny-model update/recovery checks have passed on H20 at W1/W8/W16. Fixed KHDN FP32/BF16 tiny W1 checks and two finite full-recipe pilots have also passed; formal full-model performance and KHDN multi-rank scaling acceptance remain pending.

`benchmark_hyperbolic_training.py` retains all previous options and the six default methods. `--rank` remains the protected-memory rank; torchrun supplies global rank through `RANK` and device rank through `LOCAL_RANK`. `--local-rank`/`--local_rank` optionally select the local device. A multi-rank invocation must explicitly choose `--scaling strong` or `weak`.

## Availability and configuration

GDN, KDA, Comba, HDN TC/staged and the existing protected baselines use their native CausalLMs. GDN2 has native operators and a layer but no native CausalLM/config, so a full-model GDN2 request fails. KHDN requests (`khdn_no_carry`/`khdn_y_carry`) require explicit `--khdn-strict` and actual operator/autograd/kernel, precision, token-domain and checkpoint-replay checks. The current device path uses FP64 SIMT; exceptional host/reference/recurrent substitutions are rejected from strict timing. `khdn_naive` requires `--check-only` and constructs only a tiny model with head dimension/group size 32 by default; it never produces full-model timing.

The full recipe is 24 layers, hidden 2048, H16, K/V128, FFN5184, vocab32000, expand_v1, short convolution, output gate, FP32 parameters, BF16 autocast, and outer nonreentrant checkpointing. Default fixed-curvature counts from the source inventory are HDN 1,400,280,064; GDN 1,401,067,264; KDA 1,324,881,280; Comba 1,401,068,032; KHDN 1,412,912,512. Learned KHDN curvature adds 24 parameters. KHDN adds 12,632,448 parameters over HDN. KDA uses its native low-rank output gate, so matched width/depth is not equal parameter count. `--parameter-counts` cross-checks actual meta constructors when native dependencies exist; when Triton is absent it explicitly reports source inventory without constructor verification. Every full CUDA model's actual count must agree before timing.

KDA's native forward explicitly selects fused linear cross entropy when `fuse_cross_entropy` and training are enabled, independently of the serialized `fuse_linear_cross_entropy=False` flag. Route evidence records this effective backend. This native model behavior is unchanged and belongs to the native fastest-qualified comparison category. `--loss-backend plain` sets both `fuse_cross_entropy=False` and `fuse_linear_cross_entropy=False` for every selected model, yielding a separate controlled common-CE category. No native model code is changed. Fusion moves work between phases, so native KDA backward-only timings are not equal-work plain-CE comparisons.

## Work and loss

Strong scaling requires 16,384 input slots per local microstep. W1/8/16 use accumulation16/2/1 and exactly 262,144 global input slots per step. Sample index is `step*(W*accumulation) + microstep*W + rank`; sample content depends only on data seed and that index. Weak scaling uses accumulation1 and 16,384 slots/rank. For a fixed strong step, sorted global sample IDs, hashes and masks agree across W1/8/16. The persisted cursor is an absolute global microbatch offset: each call consumes `count*W` distinct consecutive IDs and advances by that amount. The legacy field name `next_data_step` now stores this offset under checkpoint schema 2; the optimizer counter remains separate. Old unmanifested checkpoints are rejected. Keep seed, layout and all workload flags fixed when comparing runs.

Dense, right-padded and packed synthetic data have explicit shifted-label masks. Packed document-start targets are ignored before the shared helper shifts labels. Padding slots, nonpadding tokens and valid loss labels have separate counts and throughput fields. Protected memory accepts dense unpadded inputs only. There is no external corpus loader or quality claim.

Before timing, all ranks sum valid labels over the complete optimizer step. Each local CE uses summed reduction divided by `N/W`. There is no additional division by accumulation. `no_sync()` encloses forward and backward for every nonfinal microstep; the final backward synchronizes even with zero labels. A globally empty step fails. Clipping occurs once after the assembled gradient and before one AdamW step. `_no_weight_decay` groups are preserved for serial comparison copies and reapplied after optimizer resume.

## Timing, evidence and recovery

Default sampling is 10 warmups and 30 samples; shorter explicit runs are exploratory. `--final-validation` requires at least those counts and `--trials 3`. Trials independently initialize model/optimizer and reverse method order on alternating trials. Legacy method-level summaries and comparisons are explicitly labeled `trial0 only`; the raw `trials` array does not imply aggregate confirmation. Main owns cross-trial confirmation. Trials share the already initialized process group; cross-node reruns and frozen GPU correctness gates remain separate requirements.

Select exactly one of `--modes fwd`, `bwd`, `fwd_bwd` or `step` per invocation. All mixed-mode invocations are rejected, including single-device runs. Phase-only experiments consume one microbatch per rank; they first initialize Adam with one complete configured update before phase warmup, then keep model/optimizer fixed through phase measurements. They neither run stability training nor publish/reload checkpoints; `--resume` and `--checkpoint-dir` are rejected for phase modes. Separate processes/outputs isolate their trajectories from full-step strong/weak experiments.

Training forward constructs a graph using the local unwrapped model; backward-only constructs one fresh DDP graph before each timer. Full step includes the configured accumulation, loss, checkpoint replay, backward/DDP communication, clipping and optimizer. Data preparation, zero_grad, label-count and reporting collectives, diagnostics and durable journal writes stay outside the timer. Devices synchronize before and after it. Full-step medians come from actual full-step samples, never sums of phase medians.

Each active trial has an atomically replaced, flushed rank-local journal `OUTPUT_STEM.METHOD.trialN.rankR.json`, created before model construction and updated before each work stage and after each successful sample. It retains warmups, raw measured samples, configuration, method/trial/mode/sample position, absolute data cursor and completed optimizer counter even if later checkpoint/reload/profile work fails. Partial failures are labeled `FAILED; PARTIAL EVIDENCE ONLY`. Counters track completed sample calls; a failed call can have partially updated state and is not a resumable cutoff. The parent failure JSON also retains the active trial. On completed runs, per-rank raw samples remain in `OUTPUT.rankR.json`; rank0's `OUTPUT.json` contains max-rank per-step wall times and rank result paths. Launch with torchrun logging enabled to retain per-rank stdout/stderr. GPU UUID, rank/device/PIDs, available job/Pod/image identity, source-tree and patch/untracked hashes, arguments, Torch/CUDA/Triton/NCCL versions and environment switches are recorded. Missing external identity fields are null; the scheduler must supply `JOB_ID`, `POD_UID`, `IMAGE_DIGEST` where available. Ranks must agree on source/software identity. A source change during a run rejects the evidence. Use a frozen source snapshot during future GPU work.

Memory records post-warmup optimizer-initialized allocated/reserved baselines and sample peaks. Allocated peaks are captured before that sample's diagnostics. Reserved memory is explicitly process-caching-allocator scope: earlier diagnostics, methods, trials and checkpoint/profile allocations can remain resident. Resetting peak counters never establishes a diagnostics-free reserved peak.

For a separate memory experiment, start a fresh process with one method, one trial, `--modes step --memory-only`, no profile/resume/final-validation. This runs the existing correctness gates, defers interleaved health diagnostics through warmup/sampling, releases inactive cache once before the memory series, and checks final gradients/parameters/optimizer afterward. No empty_cache occurs within steady-state samples. Its timings are labeled memory-only observations and do not produce legacy speedup comparisons. Persistent runtime allocations still belong to the process; do not call the reserved measurement a universal isolated kernel footprint. Normal runs retain per-sample finite checks.

Before full timing, each actual small model compares serial accumulation with DDP over the same 16 batches, unequal labels and a zero-label final batch. Every gradient and parameter update delta is checked; KHDN names include `f_proj`, `A_log`, `dt_bias`, and learned curvature. Existing HDN/protected small-model parity gates also run. The small-model recovery gate replays the next update after restoring model, optimizer, per-rank Python/NumPy/Torch/CUDA RNG and data/step position. Each full-step trial saves, reloads and executes a finite next update after measurements, under `OUTPUT_STEM.checkpoints/trialN/` or `--checkpoint-dir/trialN/`. The saved checkpoint is the pre-replay state; result counters include the extra recovery update. A save creates a shared generation ID, durable rank files and `METHOD.complete.json`. The manifest binds every rank file hash plus the shared model/Adam/scheduler/config/position digest. Completion is published only after every rank agrees; rank RNG is deliberately independent. On load, each rank verifies its local file/manifest and all ranks collectively compare generation, manifest identity, shared state and positions **before applying state or wrapping the full model in DDP**. Missing/partial/stale/mixed generations fail collectively; parameters are never broadcast as a repair for inconsistent optimizer/data state. Rank-local or node-local storage is supported by replicating the same completion manifest.

`--resume` points to the directory containing `METHOD.rankR.pt` and `METHOD.complete.json`; it requires one step-mode method/trial. Compatibility binds physical batch/sequence shape, absolute cursor schema, layout/seed/scaling, requested and saved actual model config, outer/reentrant checkpoint policy, FP32/autocast/TF32 policy, effective loss backend, optimizer/clipping and supported backend environment. Rank-specific manifest/cache/dump paths are metadata locations, not shared execution-policy identity; the validated source snapshot ID is bound instead. The loader also rejects actual-config drift. No recurrent cache survives an optimizer update. No scheduler is used by this recipe; checkpoint helpers support an explicitly supplied scheduler.

Process-group timeout defaults to 300 seconds and task deadline to 14,400 seconds. The harness destroys only its own process group. Use torchrun with `--max-restarts=0`; its worker group owns local peer termination. External orchestration owns both launcher jobs and must terminate only PIDs/Pods verified against the recorded run identity. The harness never scans or kills unrelated processes. Do not resubmit after a lost observation without first locating the existing run ID.

## Explicit archive provenance

A `.git`-free frozen source tree must set `FLA_SOURCE_MANIFEST` to the immutable `snapshot.json` emitted by `pack_snapshot.py`. No fake Git initialization or synthetic history is used. For example, main can supply the discovered snapshot path through the launch environment:

```bash
export FLA_SOURCE_MANIFEST="$KHDN_FROZEN_PARENT/snapshot.json"
```

The utility validates the full declared `base_commit/files/modes` identity using the packer's exact canonical JSON hash, then checks every content hash or symlink and the complete source inventory. Original modes or their read-only hardened equivalents are allowed. It rejects missing/modified/extra files and manifest identity drift; only generated `__pycache__` entries are ignored. Keep result files and caches outside the immutable source root. `base_commit` is explicitly **declared provenance**, `revision` and dirty-Git fields are null, and `source_kind=immutable_source_manifest`; content verification does not claim Git ancestry/authentication. Normal real checkouts retain their actual Git metadata. Every rank and the end-of-run guard revalidate snapshot and manifest identity.

The previous frozen H20 source failed at `git rev-parse` before model execution. It remains untouched. The manifest support above has CPU archive/tamper tests, not a claimed H20 rerun.

## Separate controlled-loss and memory recipes

Run after the matching correctness and route checks pass, in separate fresh processes with separate outputs. These memory-only examples describe supported flags, not executed memory-isolation results:

```bash
python benchmarks/benchmark_hyperbolic_training.py --methods gdn kda comba hdn_tc hdn_staged \
  --loss-backend plain --modes step --output /absolute/results/common-ce.json
python benchmarks/benchmark_hyperbolic_training.py --methods gdn --modes step --trials 1 \
  --memory-only --stability-steps 0 --output /absolute/results/gdn-memory.json
```

For a second method's memory run, exit the first process, launch a new one, and use a different output. Leave native/default loss runs in their separate category. The absolute result paths above must be chosen by the scheduler and outside the frozen source.

## CPU commands already exercised

From the repository, use the sibling `../venv/bin/python`:

```bash
FLA_CI_ENV=0 OMP_NUM_THREADS=1 ../venv/bin/python -m pytest --noconftest \
  tests/benchmarks/test_hyperbolic_training_benchmark.py \
  tests/benchmarks/test_hyperbolic_training_distributed.py \
  tests/benchmarks/test_hyperbolic_benchmarks.py \
  tests/benchmarks/test_hyperbolic_tensor_core_benchmark.py \
  -k 'not real_model_constructors' -q
../venv/bin/python benchmarks/benchmark_hyperbolic_training.py --help
../venv/bin/python benchmarks/benchmark_hyperbolic_training.py --dry-run --parameter-counts --methods gdn kda comba hdn_tc hdn_staged
../venv/bin/python benchmarks/benchmark_hyperbolic_training.py --dry-run --scaling strong --methods gdn
```

The local macOS venv lacks Triton. `--noconftest` avoids the unrelated FLA-wide accelerator fixture, and the real-constructor test is explicitly deselected. The two-rank Gloo test launches real processes using a loopback rendezvous, checks unequal/zero-label updates, and is bounded with cleanup restricted to its own process session. It is supplementary CPU evidence only.

## Future GPU commands: not executed

After independent Astra review, the existing native baselines can be exercised using the target environment's Python. First run `--check-only`. Use identical commands and separate outputs for W1/8; the W1 command below exercises a process group/DDP through torchrun.

```bash
python -m torch.distributed.run --standalone --nproc-per-node=1 --max-restarts=0 \
  --log-dir profile/khdn-ddp/gdn-w1-logs --redirects 3 --tee 3 \
  benchmarks/benchmark_hyperbolic_training.py --methods gdn --scaling strong \
  --warmup 10 --steps 30 --trials 3 --final-validation --modes step \
  --output profile/khdn-ddp/gdn-strong-w1.json
python -m torch.distributed.run --standalone --nproc-per-node=8 --max-restarts=0 \
  --log-dir profile/khdn-ddp/gdn-w8-logs --redirects 3 --tee 3 \
  benchmarks/benchmark_hyperbolic_training.py --methods gdn --scaling strong \
  --warmup 10 --steps 30 --trials 3 --final-validation --modes step \
  --output profile/khdn-ddp/gdn-strong-w8.json
```

For W16 run the following on each of the two existing nodes with scheduler-verified `KHDN_RDZV_ENDPOINT`, common unique `KHDN_RUN_ID`, and each node's `KHDN_NODE_RANK` supplied. These are placeholders for discovered runtime identity, not proposed addresses.

```bash
python -m torch.distributed.run --nnodes=2 --nproc-per-node=8 --max-restarts=0 \
  --node-rank "$KHDN_NODE_RANK" --rdzv-backend c10d \
  --rdzv-endpoint "$KHDN_RDZV_ENDPOINT" --rdzv-id "$KHDN_RUN_ID" \
  --log-dir profile/khdn-ddp/gdn-w16-logs --redirects 3 --tee 3 \
  benchmarks/benchmark_hyperbolic_training.py --methods gdn --scaling strong \
  --warmup 10 --steps 30 --trials 3 --final-validation --modes step \
  --ddp-timeout 300 --task-timeout 14400 --output profile/khdn-ddp/gdn-strong-w16.json
```

Use `--scaling weak` and separate output names for weak scaling. For the long endpoint use `--batch 1 --seq-len 16384`; other matched endpoints are batch4/4096 and batch2/8192. Phase measurements and `--profile` run separately; distributed profiling is rejected. The harness does not schedule manifests, compute cross-run scaling efficiency, or declare a winner. Review global sample hashes and calculate strong/weak efficiency from matching run files. KHDN's explicit strict opt-in does not waive numerical, actual-shape or fallback gates; it is not a performance endorsement.
