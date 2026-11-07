# Experimental BF16 F@S policy and training harness

`bf16_fs6` is an explicit serial experimental policy. The default remains
`strict`; `mixed_readout` retains its TF32x3 products and existing checks.
The tested source manifest is `f18372724acf`; its production modules, benchmark
code and original tests are available in the first short-policy commit. The later
portability change affects only test selection and fallback coverage. On H20,
the full B8/T2048, fixed-c=1 comparison passed the two-step gradient, Adam and
parameter checks. A separate fresh process reproduced update four from the
step-three checkpoint. These bounded results do not qualify general training,
other lengths or distributed throughput. See the [validation records](experiments/khdn_2026_11_04/README.md).
The separately studied long-context selector is not included here.

The selector requires the entire existing forward/adjoint qualification plus
no_carry, C16, K=V128, g32 and at most 2,048 tokens per document. The native
qualification stack is SM90, torch 2.9.0+cu128 and Triton 3.5.0 with the
[validated compiler binary](key_gated_hyperbolic_precision.md#compiler-binding).
Version strings alone do not identify the tested binary. `_bf16_fs6_qualification_reason` retains the
original rejection reason before checking the additional geometry. The training
exception guard wraps that helper and rejects host, recurrent, reference or
strict-replay substitution before entering those paths. It covers the actual
route, seven-source gate, accumulated update, uninterrupted/resumed recovery,
and measured execution. The independent dense Oracle runs outside this guard.

Config → Model → Layer → chunk carries `precision_policy='bf16_fs6'`.
The operator uses `options.bf16_fs6=True`, `options.mixed=False`.
State forward receives `BF16_FS6=True`, `MIXED=False`, `CARRY=False`. Only F@S
in each nonempty chunk uses the 21 BF16 × BF16 → FP32 digit products with FP64
reconstruction. Readout and the remaining state arithmetic stay strict.
State backward has `MIXED=False`, consumes actual saved FP64 Y/S, and evaluates
the strict physical first-order VJP. Its route is `experimental_bf16_fs6` and
its arithmetic is `device_float64`. This is not a low-precision backward or a
strict replay. Short decode retains the original strict recurrent behavior.

The training CLI accepts `--khdn-precision-policy bf16_fs6` only with
`khdn_no_carry`, C16/K128/g32, BF16 autocast, plain CE, ShortConv and outer
checkpoint. Small/check-only models use two layers, hidden256, H2, K=V128,
FFN512, g32/C16 and learned positive curvature, including the matching strict
naive reference. Full-model fixed-c and other benchmark defaults are preserved.
This harness restriction does not expand the operator selector.

An untimed `KHDNTrace` binds loaded entry/helper source fingerprints and selected
callee before inspecting the actual Triton calls. It checks source stability,
phase and stage order, C/g/K/V/carry flags, real layer/checkpoint coverage,
batch token bounds, original gate/domain and prepared denominator/mass checks,
saved Y/S dtype/finite checks, and returned output/gradient finiteness.
Legitimate checkpoint early-stop may lack completion telemetry; it must retain
the corresponding four actual forward launches. Duplicate layer records cannot
replace another layer's recompute.

`attest_compiled_policy` reparses raw TTIR/PTX and verifies all recorded hashes.
For state forward it requires the independent BF16_FS6 flag, exactly 21
[16,128] BF16 × [128,32] BF16 → [16,32] FP32 dots, each with a zero FP32
accumulator and widening to FP64 before reconstruction, and native
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` instructions.
Scalar arithmetic stays FP64. The retained H20 specialization has 168 static
MMA instructions; this is not a dynamic count or a general compiler guarantee.
All other families, including backward, require strict FP64 arithmetic with
zero dot/MMA. The legacy strict/mixed verifier branches retain their checks.

New-policy trace results store raw TTIR/PTX once per hash pair in
`compiled_assemblies`; launches reference `compiled_assembly_ref`.
`reparse_khdn_trace(result, expected_source_binding=frozen_binding)` rechecks a
serialized record against the separately frozen source binding. A source or
compiler identity is not numerical acceptance. Numerical-policy qualification
and performance acceptance remain false after a successful trace.

For CPU-only tests use `--confcutdir=tests/benchmarks`: the parent operator
conftest requires Triton. Archive CLI/dry-run requires a complete immutable
`FLA_SOURCE_MANIFEST`; do not initialize a Git repository or fabricate a clean
revision for this archive. A retained prototype's `MIXED=True` is rejected as
new-policy identity even if its BF16 arithmetic inventory reparses successfully.

For requalification after a source or compiler change, use one final frozen source/compiler/import snapshot,
the original seven-source elementwise AND relative-RMS budgets, targeted
memcheck/INFO racecheck, then the original learned-curvature tiny W1
all-parameter update/recovery checks. Only after these prerequisites pass may
an independently adapted fixed-c1 full two-step comparator run. It must compare
loss and all preclip gradients before any candidate optimizer update, then
postclip gradients before stepping when separable, and retain the frozen Adam
moment envelope, parameter/update budgets, initial optimizer/RNG state and both
batch identities. Step two continues the candidate's own trajectory.

## Update and recovery observations

The external binding fixture retains `@triton.jit` and original line numbers on
all 23 Python functions. Its no-computation CPU decorator stores the incoming
function in `.fn`, matching Triton's source convention exactly. The in-tree small gate derives its update binding from the initial compiled
route. External acceptance also compares it with a separately frozen source
binding before the update trace is accepted.

For bf16_fs6, each accumulation arm has its own `KHDNUpdateTrace`, and each
uninterrupted/resumed recovery update has its own trace. The initial full
compiled trace's source binding is required. Root model hooks match every actual
input tensor and its order to the expected microbatch. Public operator entries
include defaults in policy/C/g/carry checks; forward options and context identity
link each graph's backward to its originating public call. Actual native launch
arguments enforce BF16_FS6=True/MIXED=False only in state forward, strict state
backward, the original geometry, and exact stage order. Each microbatch/layer
must have one original forward/backward and the configured nonreentrant
checkpoint recompute. Missing graphs or silent substitutions fail before the
corresponding optimizer step. Both accumulation arms finish these checks before
either arm steps. Strict/mixed retain their original sentinel and update order.

Update observations reuse the initial full compiler gate and collect lightweight
launch evidence rather than repeated TTIR/PTX text. They do not establish new
numerical correctness or performance acceptance. Explicit model/operator Oracles
remain outside candidate update guards. A new source package needs its own real
W1 update/recovery result; the parent package's successful pre-update trace and
Oracle, with zero optimizer updates, does not satisfy that gate.
