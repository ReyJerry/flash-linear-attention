# Grouped KHDN execution optimizations and public H20 measurements

This draft release moves output and local source-gradient work out of the chunk state scan, splits common preparation/backward, and removes unused no-carry buffers and expanded tail partials. It preserves the grouped canonical model, trainable key-channel gates, strong-decay semantics, g32 and supported first-order source gradients. Explicit `precision_policy='bf16'` uses BF16 contractions, FP32 state/control/accumulation, and no FP64 fallback. At C16 the two output-source VJP contractions use one BF16 product; larger-chunk contractions retain three products. No six-product compensation is introduced.

The production files correspond to measured snapshot `619f6549531d9e56738db626ac4ea8466d8cc472d56b989a0aee446baaf24788`. Publication changes only three comment initial letters in the kernel relative to that snapshot; executable ASTs are identical. The newer affine mass scan is an archived experiment, not installed in the operator. The overall engineering goal and production qualification remain incomplete; PR #2 stays draft. The legacy `strict` default remains, so BF16 autocast alone does not select the no-FP64 route.

## Public full-model results

All three paired trials used the same H20, GPU `0211455f-9dba-d074-4ce8-ac885029498e`, PyTorch 2.9.0+cu128, Triton 3.5.0 and qualified patched compiler SHA256 `1f48e0f0f7fb841418645069b378b787e41338192b5fd1ac81f95e1ffb05ed32`. The recipe is 24 layers, hidden2048, H16, K=V128, FFN5184, vocabulary32000, B8/T2048, FP32 parameters, BF16 autocast, ShortConv, layer checkpointing, plain CE and AdamW. KHDN uses C16/g32/no_carry, positive c=1 and trainable nonconstant gates.

Each mode/trial has 10 warmup and 30 steady observations, with separate fresh models per mode. Step trials perform 40 actual optimizer updates. The table gives the median of three trial medians; F+B is measured directly, not synthesized by adding medians.

| Model | Parameters | Forward ms | Backward ms | F+B ms | Full step ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native KDA | 1,324,881,280 | 372.15 | 1246.70 | 1617.74 | 1672.43 |
| Matched-output KDA | 1,412,912,512 | 392.21 | 1309.60 | 1700.73 | 1758.96 |
| Public KHDN | 1,412,912,512 | 832.28 | 2319.30 | 3147.08 | 3205.66 |

The median paired KHDN time ratio is **1.82252× matched KDA** (54.87% of its throughput), or 1.91664× native KDA. Matched KDA replaces only its output-gate projections with full 2048×2048 bias-free Linear layers; it is a distinct control, not native KDA.

The [November 6 release](../khdn_bf16_2026_11_06/README.md) reported about 4.323 s and 2.45645× matched KDA, using a frozen private wrapper on another H20. Those historical numbers and today's 3.206 s show progress, but are not one newly paired before/after experiment. A separate same-card structural comparison measured the public baseline at 4.343205 s versus the three-product structural candidate at 3.238770 s: 25.43% less time and 34.10% more throughput. That structural candidate precedes the current C16 output-source precision specialization.

The full step includes forward/loss/backward/clipping/optimizer and public wrapper guards; data transfer and `zero_grad` are outside the timing window. Each microbatch has 16384 input slots and 16376 valid next-token labels. All 1440 raw observations, including warmups and tails, are retained. KHDN step p95 is approximately 3.47 s, so its median is not a tail-latency guarantee. Peak step allocation is 30.02 GiB for KHDN, 25.59 GiB for matched KDA and 24.43 GiB for native KDA. Reserved-memory samples are separately retained.

## Six-issue status

| Issue | Implemented and measured | Remaining |
| --- | --- | --- |
| State serial chain | Separate output forward, output VJP and parameter VJP; necessary state dependence remains | Further reduce recurrence cost without excess saved state |
| Common preparation | Base/interval/system forward; gauge/matrix/interval backward; row-parallel products | Interval/matrix stages still cost time; retain direct local strong-gate intervals |
| Registers/live ranges | Direct tail loads, later loads and earlier stores; stage NCU measured fewer local loads/stores | Some kernels remain at 255 registers; final matched KDA hardware profile pending |
| Buffers/partials | Remove no-carry E/H and zero gfnext; compact two tail partials | Verified 1616 MiB operator peak saving is not a 24-layer multiplier; paired model saving was 184.5 MiB |
| Chunk size | Fixed g32, 720 observations: dense F+B C16/C32/C64 = 44.863/54.797/281.670 ms | C16 remains selected; larger chunks did not improve this implementation |
| Complete cost/precision | Public full-model three-trial comparison; bounded per-GEMM strategy | Four-shape final performance, scaling, current profile and carry comparison remain |

## Numerical and training evidence

- Publication checks passed 146 host tests with two GPU-only skips on the local machine, scoped pre-commit, and the 1440-observation audit. The exact integrated snapshot passed 213 native-runtime tests: 65 operator cases plus 148 harness/compiler checks, including two actual checkpoint on/off model/update traces. No failures or skips. Seven-source retained-graph gradients and C32 precision regressions are covered.
- The selected operator matrix has 105/108 passes and three preserved C64 scalar-curvature RMS failures, previously accepted as bounded exceptions. Their raw labels are not rewritten.
- Three synthetic full24 trajectories completed 16 updates and fresh-process 8→16 replay. Final heldout relative PPL differences were +0.036749%, −0.048551% and −0.067091%. Seed101's earlier step8 +0.122766% remains recorded. The user subsequently made 0.1% an observation line rather than an automatic veto in the absence of clear sustained quality deterioration. This does not relax mathematical, finite or same-precision DDP budgets.
- B8/T2048, B4/T4096, B2/T8192 and B1/T16384 have bounded finite-training/recovery evidence. The 4K +0.108873% PPL observation is explicitly accepted and retained. These synthetic checks do not establish real-corpus quality or long-term stability.
- Full24 NCCL W1 and W8 completed four strong-scaling updates and fresh 2→4 replay, with actual saved-gradient/update/checkpoint CPU readback. The W8 serial reference is evaluated at the current candidate weights and Adam state each step, without copying reference gradients or updates into the candidate. It passes the original gradient .002/.02, update 2e-6/.02 and recovery 2e-5/.0002 budgets; W8 replay was additionally byte-exact. The earlier independently drifting W8 trajectory's 48 failed coordinates remain failed.
- W16 completed four training updates with all459 matched gradient/update comparisons passing and fresh-process 2→4 recovery reported exact on all16 ranks; the final actual-tensor CPU audit is running at this release checkpoint. This is not a W16 throughput or full acceptance claim.
- Six historical exploratory cache/packed model cases failed sparse downstream-logit elementwise comparisons; two save/load-next-update cases passed. Later assertions were not reached in the failed cases. Those failures remain unresolved, and the user's quality policy is not presented as evidence that cache semantics passed.

The tested runtime rejects unqualified compiler builds and FP64 candidate inputs. Large raw gradient/checkpoint tensors remain in the experiment store; the published CPU audit records attest actual readback, not a new local reread of those tensors.

## Experimental parallel mass recurrence

`affine-mass-experiment.tar.gz` archives the separate C16 FP32 affine prefix/reverse scan candidate `6e52bfdb91f6de6ca2683cd0bb34011cd338002fcbdf789d5d4b07670c7749ae`, its proof, exact source, measurement script and records. It changes the order of the scalar mass recurrence without adding matrix products or buffers. Other chunk sizes retain their original executable branches.

It passed 213 native tests and memcheck/racecheck/synccheck, each with three native retained-graph/alias/tail witnesses. In 480 paired operator observations, dense F+B changed 43.447→40.764 ms (−6.17% time) and packed F+B 43.592→41.013 ms (−5.91%), with unchanged peak allocation. Compiler-reported spills increased in state kernels despite the measured speedup; this is not a claim of improved occupancy or lower dynamic traffic. The 108-case run and CPU tensor readback are verified: 105 pass and three unchanged prior C64 dc RMS failures remain, with zero new failed cases; 1373 compiled image instances and six CPU dense-oracle checks were audited. Full-model gradients, finite training, recovery and end-to-end speed are not yet qualified for this candidate, so its gain is not included in the full-model table.

## Reproduction and publication audit

Run from the repository root:

```bash
python docs/experiments/khdn_bf16_2026_11_07/summarize.py
```

This CPU-only command verifies artifact/member hashes, all 1440 raw observations and summary statistics, and the published source against the measured executable AST. It does not rerun GPU training or the remote tensor audits.

`timing-sources.tar.gz` preserves the executed drivers, model/timing plans, separate matched-KDA constructor, numerical contracts and admission receipt identities. `measured-source-overlay.tar.gz` contains the eight exact measured overrides and complete snapshot manifest, based on Git commit `a77663146289c5a6d7e331a5e869415ef166759d`. Reconstruct in an isolated checkout, preserve recorded file modes, and verify the snapshot before importing. The earlier release includes compiler patch guidance; qualified dependencies and frozen tuning/admission files must be provisioned explicitly. The scripts retain original machine-bound runtime/output paths: publication is not a portable environment installer or an instruction to bypass runtime/admission checks.

`validation-records.tar.gz` preserves native JUnit/logs, original numerical failures, approved policies, finite-training summaries and W1/W8 actual-readback receipts. Historical records may retain their original pre-submission or pre-policy status strings; the dated scope above distinguishes those records from subsequent accepted decisions. `evidence.json` binds the artifacts and lists incomplete requirements.

The highest executed scale is presently two nodes/16 H20. Required 1/8/16 strong and weak throughput, the long DDP endpoint, final same-precision full-model carry comparison, current KDA-matched profile, six model regression cases, final external review and the BF16 default transition are not closed. Optional 32/56-card extensions have not been run. This publication is a reviewable progress checkpoint, not completion of the full engineering objective.
