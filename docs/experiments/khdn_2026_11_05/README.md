# Full-model W1 measurements on H20

The serial B8/T2048 KHDN candidate took **65.634 s per complete step**, compared with **1.670 s for KDA** on the same H20. These are medians over 30 steady samples after 10 warmups. KHDN ran first. The 39.30× latency ratio is one paired trial; the data do not establish distributed scaling, a final three-trial KHDN result, or the performance of another kernel.

All models used 24 layers, width 2048, 16 heads, K=V128, FFN 5184, vocabulary 32000, FP32 parameters and BF16 autocast. Model parameters and projections differ by architecture; the manifest gives each parameter count and initial-weight digest. Each W1 microbatch had 16,384 dense input slots, no padding, and B×(T−1) valid next-token labels. The KHDN arm used trainable nonidentity Key gates, fixed curvature 1, g32, C16, no carry, ShortConv, checkpointing and plain cross-entropy. Only the short serial policy was published in [the experimental commit](https://github.com/ReyJerry/flash-linear-attention/commit/23fb428699c3d4866d4e55337cba62cd2949adbd). The separately tested long selector is not included in that commit.

| Length | KDA | Comba | Legacy HDN TC | Legacy HDN staged |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 1,671.61 | 1,710.19 | 4,567.05 | 3,206.43 |
| 4,096 | 1,674.58 | 1,712.82 | 4,846.44 | 3,301.42 |
| 8,192 | 1,675.14 | 1,713.88 | 4,837.28 | 3,300.12 |
| 16,384 | 1,686.44 | 1,724.30 | 4,818.39 | 3,290.49 |

Each cell is the median of three per-trial full-step medians in milliseconds, one trial on each of three H20s, distinct from the single same-card FS6/KDA trial above. Each phase begins with a fresh model and optimizer. Training forward, fresh-graph backward, combined forward/backward and full steps are independently synchronized. Full steps include loss, gradients, clipping and fused AdamW. Data preparation, zeroing gradients, per-sample health checks and the standalone route trace are outside the timed interval. Required runtime guards inside the measured path are included. An extra KHDN profile on a different card assigned 57.62% of summed GPU kernel time to state backward, 18.26% to state forward. State-forward kernel time includes both the original forward and checkpoint recomputation. These shares are not percentages of end-to-end wall time.

Run `python3 summarize.py` here with Python 3.10+ to reproduce the table and ratio from the 8,000 observations. Add `--details` for all 200 phase distributions, including wall and CUDA time, allocated/reserved peaks, the post-qualification memory baseline, and phase-normalized slot/label rates. Backward-only wall/CUDA timing excludes its fresh forward preparation, while the recorded memory peak includes that preparation. Rates for individual phases are not complete-update training throughput. Each JSONL row uses the column order in `manifest.json`; the manifest binds batch hashes, run/source identities, original sample/qualification file hashes and the uncompressed export. The sample export is a selection of timing fields, not the original training logs or a model checkpoint.

Short numeric evidence: two independently advancing optimizer steps checked all 459 KHDN parameter tensors; a separate fresh process replayed the fourth update bitwise from step three. By contrast, the full strong W1 accumulation-16 run exhausted memory in its first backward phase, before any update. No general KHDN training acceptance, long-sequence KHDN timing, or 8/16-GPU KHDN acceptance is claimed here.
