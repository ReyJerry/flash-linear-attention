# Sherman–Morrison research result, 2026-11-09

The rank-one token solve is mathematically valid, but this implementation regresses both representative full training steps. The experimental GPU wrapper, kernels and CUDA test have been removed from the live training/test path and preserved verbatim under `candidate/` with a `.txt` suffix. The untimed reference and CPU tests remain available. No operator default, model configuration or checkpoint format changes in this publication.

The engineering goal remains incomplete and paused. This publication records existing results; it launches no new experiments. Stage 2 producer/consumer fusion and stage 3 dedicated state backend did not proceed because the required full-step benefit was absent.

## Mathematical audit and actual implementation

The canonical per-token group system is `H = n I_g - kappa v v^T`, with `n = k^T k` and `kappa = c/g`. For `r = v - X^T k`, `s = v^T r` and `d = n - kappa v^T v`, the candidate uses:

```text
u = beta * (r/n + kappa*v*s/(n*d))
z = 1 + kappa*beta*s/d
S_next = (X + k*u^T)/z
```

The original C64 chunk graph already uses rank-one structure when preparing its per-token coefficients. Its cross-token feedback system `Zhat = (Qs - kappa*(A ⊙ VV^T)) ⊙ rho` is a different system and is generally not rank one. Replacing that entire system with a single rank-one solve would change the model.

The tested alternative performs a token-serial canonical recurrence, with state checkpoints every 64 tokens and a manual token VJP. Thus C64 here is a checkpoint interval, not 64 parallel token updates. FP32 controls `n/d/s/z` and the state; BF16/FP32 source tensors are accepted. The untimed oracle may explicitly use FP64; training and its fallback have no FP64 arithmetic. No numerical budget was relaxed.

The GPU guard admits the bounded Key-norm domain and requires `d/n > 2^-10` with safe normalization. Unsafe calls use the original block implementation for the whole call. Near-boundary parity is tested. Both full-step screens recorded zero fallback calls, so the negative timing comes from actual SM execution.

## Full training step results

Same H20 per endpoint, 24 layers, 1,412,912,512 parameters, H16/K=V128/g32/C64, fixed positive curvature, no_carry, ShortConv, plain cross-entropy, BF16 autocast, FP32 parameter masters and Triton FP32 AdamW. All arms use `save_mm` and deferred finite validation. Block KHDN actually uses `coupled_fused`; the candidate actually uses `sm_rank_one_manual`; matched KDA actually executes its TileLang backward.

Each F/B/direct F+B/step has two warmups and five steady observations. There are 168 raw observations and 42 real optimizer updates in total, including step warmups. F+B is directly measured. This short research screen does not establish continuous-training quality or formal throughput qualification.

| Shape     | Arm          | F ms    | B ms     | Direct F+B ms | Step ms  | Input tokens/s | Peak allocated GiB | Peak reserved GiB |
| --------- | ------------ | ------: | -------: | ------------: | -------: | -------------: | -----------------: | ----------------: |
| B8×T2048  | Block KHDN   | 521.122 | 1301.892 | 1821.011      | 1849.929 | 8856.56        | 43.7872            | 47.6445           |
| B8×T2048  | SM candidate | 573.021 | 2051.347 | 2621.684      | 2650.325 | 6181.88        | 43.7948            | 46.9238           |
| B8×T2048  | Matched KDA  | 392.496 | 1005.097 | 1396.574      | 1420.894 | 11530.77       | 43.7811            | 47.2500           |
| B1×T16384 | Block KHDN   | 551.188 | 1406.752 | 1955.560      | 1969.768 | 8317.73        | 43.7872            | 47.6270           |
| B1×T16384 | SM candidate | 969.526 | 3143.322 | 4112.337      | 4137.872 | 3959.52        | 43.7948            | 46.9258           |
| B1×T16384 | Matched KDA  | 407.173 | 1036.573 | 1438.980      | 1456.728 | 11247.12       | 43.7811            | 47.2500           |

SM step time regresses by **43.266% / 110.069%**. Its throughput is **53.612% / 35.205%** of matched KDA, versus block KHDN's **76.808% / 73.954%** in these screens. Input tokens/s counts all 16,384 input positions; valid next-token labels are 16,376 / 16,383. Peak allocated increases by 8,126,464 bytes. Lower reserved memory alone does not establish a useful memory saving.

The block arm reuses completed same-card, same-source baseline samples from experiment 149; corrected SM and KDA arms run consecutively in experiment 151. This is not a fresh single-process three-arm paired run. Experiment 149 later failed an observer-count assertion after valid block measurements; experiment 151 corrects observer import order while keeping the original trace contract. Earlier packaging/runner failures are retained locally and do not count as numerical or timing evidence.

## Why the candidate loses

The new recurrence removes the old block-solve execution but puts token updates back on the state dependency chain. The short shape has 512 sequence/head/group state CTAs; the long shape has 64 and eight times the per-sequence length. That organization explains why the long endpoint is especially exposed to serial work; a dynamic occupancy or DRAM attribution has not been measured for this candidate.

Backward replays each C64 segment into global scratch and reduces shared-source group partials. It bypasses the old Delta/Pi stages but does not deliver carry-only storage or complete producer/consumer fusion. Static compiler records show forward 95 registers/thread, backward 254 and reduction 77, with zero reported spills. Zero spills did not prevent a large end-to-end regression. No dedicated H20 shared-memory wide backend is claimed.

## Correctness and limits

- CPU: 10 tests pass against independent canonical dense solves/autograd and a manual seven-source VJP, including nonzero initial state, curvature, beta endpoints, packed empty segments and near-`d` fallback. The archived 78 comparisons have maximum absolute error `6.661338147750939e-16` under the existing FP64 oracle tolerances.
- H20: eight operator cases pass the unchanged BF16 budgets, covering FP32 sources, BF16 packed data, fixed point, repeated Key/beta=1, strong gates, c=0, alias gradients and near-`d` original-block parity. Repeated backward is bitwise stable. One real two-layer learned-curvature optimizer update passes. Eleven compiler images contain no FP64 training arithmetic.
- GPU per-quantity max/RMS errors were not captured; passing gates must not be presented as measured error magnitudes. Full-model finite gradients and their hashes were recorded, but equality of the two arithmetic graphs, full-model gradient cosine and unseen quality were not established. Initial same-batch loss differences are approximately -0.000431 / -0.000589 and do not establish model quality.
- Final-source continuous training/unseen quality, cache/packed model regressions, formal four-shape timing, recovery and 1/8/16-GPU acceptance remain open. Historical `1900.263 ms / 75.22%` is not current-source qualification.

## Source binding and review

Measured snapshot: `ae35de862dd6837be805e9836de59ca644e659cb614256bf39078ad0a9c89cc1`. The original manifest declares older base `256826f3b015fe600d5f710cd93e3e7e97c6db2f`; that label alone is not source identity. All **883 shared tracked files** match production commit `fba02a4c0c1fea25af2281b5c9a03ab1b089592f` byte for byte. The five new candidate source/test files match the measured manifest and are archived verbatim. Their original paths, hashes and mode-independent byte lengths are in [source-binding.json](source-binding.json). No protected parallel source/test contents are included.

The archived `.py.txt` files are reviewable source, deliberately excluded from imports and test collection. Recreating the experimental overlay requires the original paths and the production commit from the binding; the GPU route is not installed by this publication. The current live additions are only the untimed reference and CPU test. This archival commit is not a newly timed production revision.

[summary.json](summary.json) contains full precision results. [evidence-manifest.json](evidence-manifest.json) inventories the published evidence and original byte hashes. Raw samples and gradient hashes are byte-exact. Other JSON is pretty-printed with private runtime absolute paths replaced by a marker; source hashes, measurements and trace counts are retained. The full private execution environment and orchestration are not part of this package; the verifier checks evidence and does not launch or reproduce training.

Run from the repository root:

```bash
python docs/experiments/khdn_sherman_morrison_2026_11_09/verify.py
python -m pytest --noconftest -q tests/ops/test_key_gated_hyperbolic_sherman_morrison.py
```

The decision is to retain the original block training path and archive this SM candidate. Follow-up stages require a new structural proposal with an explicit whole-graph cost budget and evidence of full-step benefit.
