# KHDN experimental validation — 2026-11-04

The serial BF16 state-projection snapshot was built on fork commit `579315df36bb`. The separate strict parallel-backward snapshot was built on `15bdbe075b2b`; only documentation changed between those two Git commits. The snapshot IDs below identify exact source manifests, not Git commits. The branch's existing `mixed_readout` did not pass the 24-layer gradient gate: 4,329 elements in eight tensors exceeded tolerance. The new experiments do not change that result or the default precision policy.

## Two optimizer steps with the serial BF16 state projection

The opt-in `bf16_fs6` snapshot `f18372724acf` uses BF16 only for the forward state projection: six signed radix-256 digits, 21 BF16 products with FP32 accumulation, and FP64 reconstruction. Backward uses the saved forward projection and strict arithmetic. On one H20, the 24-layer comparison used hidden size 2048, 16 heads, K=V128, g32/C16, no carry, fixed c=1, trainable key-channel decay, ShortConv, layer checkpointing, FP32 parameters, BF16 autocast and plain cross-entropy. B8/T2048 supplied 16,376 valid labels per update.

Strict and candidate began with identical parameters and RNG, saw the same two minibatches, and advanced their **own** AdamW state. Each actual backward was used for the corresponding update. Both losses were equal to strict; all 459 trainable tensors passed the original gradient, parameter, update and Adam-moment checks.

| Update | Loss (both arms) | Max absolute gradient difference | Max absolute parameter difference after update |
| --- | ---: | ---: | ---: |
| 1 | 10.78660679 | 1.46e-11 | 0 |
| 2 | 10.78204155 | 8.57e-8 | 1.16e-10 |

A separate two-layer single-rank check gave zero errors in all 43 parameter-gradient, update and resumed-next-step comparisons. On eight ranks, using the same 16 tiny examples (126 valid labels, including a zero-label final example), the largest serial-versus-DDP gradient/update differences were 5.96e-8 and 3.73e-9. The default 16-rank check stopped **before any update**: rank 7's Layer0 output cotangent reached 146.9693 versus the qualified limit of 128. Its requested strict replay was rejected. A separate protocol kept the same global label denominator in each local backward and multiplied the FP32 gradient by world size after DDP averaging and before clipping. On the same tiny example the cotangent was 9.1856, inside the unchanged limit. Both W8 and W16 passed all 43 gradient, update and recovery checks; W16's largest gradient and update differences from serial were 4.47e-8 and 3.73e-9. This result does not lift the original-policy failure or establish full-model scaling.

A separate full single-rank run completed four continuous updates with losses 10.78660679, 10.78204155, 10.77416992 and 10.77620220. It saved a checkpoint after update three. A fresh process loaded it but stopped before replaying update four because a witness compared the model's integer label-map keys to JSON string keys. The four continuous updates are valid; this particular restart attempt did not establish next-step equality. A separately reviewed fresh process then loaded the **same** checkpoint, validated the native label map before encoding only its keys for the JSON witness, and replayed only update four. Its loss, all 459 gradients before and after clipping, resulting parameters, Adam state and RNG matched the uninterrupted fourth step bit for bit. This is single-rank recovery at B8/T2048, not a distributed restart test.

## Strict parallel backward: operator result

The separate strict parallel snapshot `81c371333f16` matched all 459 strict serial full-model gradient tensors bit for bit on the first B8/T2048 batch, without an optimizer update. Timings below use the actual Layer0 inputs and incoming cotangent, on the same H20 for both implementations. Each fresh process used 10 warmups plus 30 samples per phase; order was serial/parallel, parallel/serial, serial/parallel across trials. Backward-only used a fresh graph. F+B was measured directly.

| Independent trial | Strict serial F+B median (ms) | Strict parallel F+B median (ms) | Serial / parallel |
| --- | ---: | ---: | ---: |
| 1 | 2242.27 | 1298.48 | 1.7268 |
| 2 | 2241.99 | 1298.55 | 1.7265 |
| 3 | 2242.69 | 1298.63 | 1.7270 |

Forward stayed near 499 ms. Backward fell from 1740–1744 ms to 799–800 ms. Peak allocated memory increased from 15,230,075,904 to 17,664,869,376 bytes. This comparison is not a full-model speedup and does not qualify a combined parallel-backward/BF16-forward implementation.

## Other bounded results

- Long candidate `5239aece385d`: on real Layer0 sources and cotangents from strict 24-layer runs at B4/T4096, B2/T8192 and B1/T16384, the native operator met the original elementwise and RMS budgets against the independent dense FP64 recurrence. Full first-batch comparisons at 4K and 8K matched strict loss and passed all 459 gradients; no optimizer ran. The initial 16K whole-model comparison stopped on a strict reference reproducibility check before the candidate arm. A strict-only control reproduced the two losses by changing the shared backbone RMSNorm autotuning record; the original strict loss was retained. With the original RMSNorm record pinned for both arms, the 16K loss reproduced exactly (10.78878975), all shared backbone launches matched, and all 459 candidate gradients passed; maximum absolute difference was 3.64e-12. No optimizer update was made in any of these three long comparisons.
- Full KDA, Comba, old HDN TC and old HDN staged each completed four single-GPU updates at all four fixed 16,384-slot shapes and loaded the step-three checkpoint before the fourth update. Width/depth/FFN were fixed across architectures; parameter counts differ. These are finite-state and same-process reload checks, not throughput or distributed-restart measurements.
- GDN B8/T2048 strong scaling: W1 accumulation-16 completed the four-step reference. W8 accumulation-2 completed four updates and a reload, but failed the independent first-update comparison: all 435 gradients met their budget, while 7,465,596 parameter-delta coordinates across 355 tensors did not. A separate default TileLang backward replay reproduced differences in the gate derivative on identical physical inputs. A local shared-memory correction passed the tested kernel controls; it has not qualified the original whole-model W8 or W16 trajectories.

The [machine-readable measurements](validation-2026-11-04.json) include source and artifact hashes, per-step results, per-trial timing and memory distributions, and per-shape baseline losses. Full-model throughput, multi-rank optimizer equivalence and any default-policy change require their own evidence.
