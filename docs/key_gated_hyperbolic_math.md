# KHDN stable CPU chunk and complete source backward

This document describes the mathematical CPU implementation in `fla/ops/key_gated_hyperbolic_delta_rule/reference_chunk.py`, `reference_backward.py`, and the small `reference_arithmetic.py` adapter. The independent oracle is the existing dense grouped recurrent solve in `naive.py`. These three paths implement the same canonical model in [design PR 1](https://github.com/ReyJerry/flash-linear-attention/pull/1). They are reference implementations for independent review and subsequent kernel design; no GPU kernel, low precision GPU mode, throughput result, or production default is established here.

## Interface and supported execution

`naive_chunk_key_gated_hyperbolic_delta_rule` uses ordinary PyTorch AD through the preparation and online graph for tensor arithmetic. A call requiring arbitrary precision explicitly uses the same complete manual VJP, with `ordinary_entry_extended_manual=True` in telemetry; this exceptional call is first-order only and is not an independent AD reference. `manual_chunk_key_gated_hyperbolic_delta_rule` runs that forward under `no_grad` and implements every source derivative explicitly. Its backward contains no automatic differentiation call, generic inverse backward, recurrent backward, or dense K-by-K state transform. Both return `(output, final_state_or_None)` and use the oracle's positional and keyword interface, followed by these CPU research controls:

| Option | Meaning |
| --- | --- |
| `chunk_size=32` | Requested time block size; independent of model group width |
| `carry_mode='no_carry'` | `no_carry` or exact `y_carry`; both use the same preparation, feedback, precision and range policy |
| `gauge='zero_detached'` | `none`, `fixed`, `zero_live`, or `zero_detached` |
| `gauge_values=None` | Optional finite positive frozen `[B,T,H,G]` values for `fixed`; omitted values use a deterministic nonmonotone gauge |
| `feedback_precomposition=True` | Prepare scaled `af/Jf` before the state chain; `False` solves the same scaled equation online |
| `reanchor_interval=0` | Optional periodic direct `Y=FS` initialization; zero means only required boundaries |
| `telemetry=None` | Caller-owned dictionary receiving partition, range reasons, precision, solve and backward ownership records |
| `execution_plan=None` | The immutable tuple from telemetry, for deterministic checkpoint replay; an inconsistent schedule raises |

The sources are q/k/ell `[B,T,H,K]`, v `[B,T,H,V]`, beta `[B,T,H]`, initial state `[N,H,K,V]`, and scalar curvature. `N=B` for dense input; packed input requires `B=1`, with `N=len(cu_seqlens)-1`. Repeated boundaries, all-empty calls, tails, noncontiguous inputs, optional initial/final state, fixed scales, mixed source dtypes and streaming gradients are supported. Tails have their actual length, so no invalid padded key is ever divided by its norm. A valid zero key remains an error, including at beta zero.

The mathematical domain is finite `c>=0`, `0<=beta<=1`, finite natural-log `ell<=0`, original `n=||k||²>0`, and `n-(c/g)||v_group||²>0`; for positive curvature, each incoming state group satisfies `(c/g)||S||²_F<1`. There is no input normalization, gate clamp, epsilon norm, curvature specialization, or freely learned replacement write gate. At c zero the state ball restriction is absent; c, beta and ell endpoint derivatives are retained.

Forward arithmetic follows the oracle: FP64 if any physical source is FP64, otherwise FP32, subject to explicit FP64 or arbitrary-precision exceptional execution. Output follows q.dtype; public final state follows the same FP64/FP32 rule. The working physical state is preserved between artificial subblocks at their selected precision, including mpmath-to-mpmath boundaries, with conversion only at an actual precision transition or public return. CPU autocast is disabled in both forwards and throughout manual backward. Ordinary PyTorch AD backward is invoked outside autocast, as in the normal PyTorch AMP workflow. The handwritten backward uses FP64 for tensor blocks and the saved arbitrary precision for extended blocks. When any extended block is present, source accumulation uses that precision until requested gradients are cast to their original source dtypes. A tensor curvature retains its original dtype at the custom-function boundary, including when other sources are FP64. This deliberately conservative CPU backward is not evidence for an FP32 GPU backward.

The manual function supports first-order differentiation only. A `create_graph=True` backward raises an explicit double-backward error, rather than returning a partially differentiable result. The non-extended ordinary AD route remains the research path for differentiating preparation and live gauge choices. Extended arithmetic has no second-order or generic-AD implementation.

## Canonical model and input-only preparation

For one head/group, set `kappa=c/g`, `X=diag(exp ell)S`, `Gv=nI-kappa vvᵀ`, `u=beta solve(Gv,v-Xᵀk)`, and

```text
Splus = (X + k uᵀ)/(1+kappa vᵀu)
o = scale qᵀ Splus.
```

The dense oracle uses this g-by-g solve. Its independent graph does not import chunk helpers. The chunk derivation instead uses homogeneous `S=M/z` and the exact erase/write identities

```text
Mplus = (I-gamma kkᵀ)D M + k w vᵀ,  gamma=beta/n
zplus = (1-beta)z + n w.
```

For a block, define `e[i,j]=exp(sum(ell[j+1:i+1]))`, `Gamma[i]=exp(sum(ell[:i+1]))`. These are direct local interval sums, never differences of cumulative logs. Preparation shared by every Value group of a head is

```text
GK[i,j] = sum(k[i]*k[j]*e[i,j]), j<i
L = I + diag(gamma) GK
A = solve_lower(L,I)
Ke = Gamma*K; B = diag(gamma)Ke; F = A B
Qs[i,i] = n[i]
Qs[i,j] = -beta[i] n[j] product(1-beta[j+1:i]), j<i
eta[i] = beta[i] product(1-beta[:i]).
```

`GK` is generally nonsymmetric. Its diagonal is excluded, unlike the current inclusive output contribution. Each Qs row is constructed with a reverse product scan, with saved partial products; beta zero and one require no division.

For each group, `R=A*(VVᵀ)` is a Hadamard product and `Z=Qs-kappa R`. The diagonal is the positive input margin. With `Y=FS` or incoming carried Y, `h=rowdot(V,Y)`, the feedback equation is `Zw=eta-kappa h`. Positivity of the diagonal gives exact solvability, not numerical conditioning.

## Consistent gauge and positive strict-past readout

For any positive finite gauge with `s0=1`, let `lambda[i]=s[i]/s[i-1]`, `nu[i]=1/s[i]`, `rho[i,j]=s[j]/s[i]`. The implemented graph is

```text
Zhat = Z*rho
afhat = solve_lower(Zhat,nu*eta)
Jfhat = -kappa solve_lower(Zhat,diag(nu))
what = afhat + Jfhat h
Uhat = diag(what)V - diag(nu)L Y
rhat[i] = (1-beta[i])/lambda[i] * zhat[i-1]
that[i] = n[i] what[i]
zhat[i] = rhat[i]+that[i], zhat[-1]=1.
```

The direct-feedback option solves `Zhat what=nu*(eta-kappa h)`. Neither route constructs an unscaled inverse first. The zero-entry gauge is prepared by a normalized history recurrence using the current block's inputs alone; factors nu and rho are products of local inverse lambda values. No unscaled w, z or absolute cumulative s is reconstructed.

For scaled q, define `aq=rowdot(q,k)/n`, `qr=q-beta*aq*k`, `Qer=Gamma*qr`, `Cr[i,j]=sum(qr[i]*k[j]*e[i,j])` for `j<i`, and `To=Cr A`. Write `erase(X)=X-gamma_last*k_last*(k_lastᵀ X)`, `tau=Gamma_last`, `P0=Ktail_pastᵀ A_past,past`, `P=erase(P0)`. Then

```text
ROhat = diag(nu)Qer S + (To*rho) Uhat
TO = diag(aq)V
Ss = nu_last S; Us = diag(rho_last,past) Uhat_past
RShat = erase(diag(tau)Ss) + P Us
TS = k_last v_lastᵀ/n_last
positive(R,T,r,t) = R/(r+t) + (t/(r+t))T
O = positive(ROhat,TO,rhat,that)
Sout = positive(RShat,TS,rhat_last,that_last).
```

C1 has an empty past matrix; it still retains its erased incoming state and current target. Residuals are computed directly and saved, never recovered by subtracting a target contribution from a normalized output. D0 is only the diagonal plus rank-one operation `erase(diag(tau)X)`; neither D0 nor its adjoint is allocated as a K-by-K matrix.

Every hatted term is its corresponding homogeneous term divided by its row's s. Thus the canonical output and state are independent of every legal gauge value. The manual first-order VJP fixes gauge selection while retaining every explicit nu/rho/lambda multiplier. Ordinary `zero_live` AD differentiates the selection as an independent check of this invariance; `zero_detached` fixes only the coordinate choice. No physical source is detached. Numerical truncation or lost factors would invalidate this argument, which is why range selection precedes preparation.

## Exact carry graph and its source owners

At a carried boundary, input-only shared preparation for the next block's F is

```text
E = [Fnext-gamma_last*(Fnext k_last) k_lastᵀ] diag(tau)
H = Fnext P
TY = (Fnext k_last/n_last) v_lastᵀ
RYhat = E Ss + H Us
Ynext = positive(RYhat,TY,rhat_last,that_last).
```

This computes the real second output branch; `Ynext=Fnext Sout` in exact arithmetic. The next block consumes it directly, with no second FS projection. Current public outputs and canonical state have no dependency on Fnext. The tail may have a different next block length. Carry ends at document boundaries, at the last block, at explicit reanchors, and on both sides of exceptional execution. Every new streaming call initializes Y from its canonical state and current inputs. No future-dependent public cache exists.

Backward retains independent physical-state and auxiliary-Y cotangents. Let `adj=bS+Fnextᵀ bY`, `z=r+t`, using the actual saved residuals:

```text
mn = <bS,RShat/z> + <bY,RYhat/z>
aT = <bS,TS> + <bY,TY>
bar_r += -(mn+(t/z)aT)/z
bar_t += ((r/z)aT-mn)/z
bar_Us = (Pᵀ bS+Hᵀ bY)/z
bar_P += (adj/z) Usᵀ
bar_TS += (t/z)adj
bar_Fnext += bY Soutᵀ.
```

The Fnext contribution includes all E/H/TY sources. Those derived caches are not differentiated a second time. D0's source VJP is implemented through its vector erase operation. `bar_Y` goes to the preceding carry edge; direct `bar_F+=bar_Y Sᵀ`, `bar_S+=Fᵀbar_Y` occurs only for an initializer/reanchor or each no_carry block. The reverse S/Y pass accumulates preparation slots first, then the preparation backward runs after the preceding block has supplied each F owner.

| Adjoint | Owner and reduction |
| --- | --- |
| v, what, Uhat, group Gram, Zhat, masses | Current sequence/head/block/group |
| S and Y | Distinct reverse edges; joined once by the outgoing fold |
| F at block j | Its local initializer if present, plus the folded source from block j-1 |
| A/L/Qs/eta/strict-past coefficients | One head/block buffer after all group and F-owner contributions |
| q/k/beta/ell | Shared head/token, summed across all groups |
| curvature | Sum of every group `bar_kappa/g` across sequences, heads and blocks |
| empty initial state | Identity final-state cotangent, with connected zero temporal and curvature gradients |

## Remaining handwritten source VJP

For `f(R,T,r,t)`, normalization uses

```text
z=r+t; mn=<bar_f,R/z>; aT=<bar_f,T>
bar_R=bar_f/z; bar_T=(t/z)bar_f
bar_r=-(mn+(t/z)aT)/z
bar_t=((r/z)aT-mn)/z.
```

This avoids a literal z squared, unnormalized large contractions, and zeroing derivatives at r or t zero. The mass scan reverses from the last row: combine local mass adjoints with the accumulated later z adjoint, propagate `bar_beta-=gr*zprev/lambda`, `bar_n+=gt*what`, `bar_what+=gt*n`, and `bar_zprev=gr*(1-beta)/lambda`.

The U branch contributes to what, V, L and Y. The feedback source derivative solves

```text
b = solve_upper(Zhatᵀ,bar_what)
bar_Z = -rho*tril(b whatᵀ)
d_rhs = nu*b
bar_eta += d_rhs; bar_h += -kappa*d_rhs
bar_Qs += bar_Z; bar_R += -kappa*bar_Z
bar_kappa += -<d_rhs,h>-<bar_Z,R>.
```

Both curvature terms remain at zero curvature; the implementation never divides Jf by kappa. The V Gram derivative uses `(BV+BVᵀ)V`, with `BV=bar_R*A`. This symmetry is specific to the Value Gram and is not applied to GK. Strict-past output and terminal state sources reverse through `To=Cr A`, `Qer=Gamma*qr`, aq, qr, P0, the rank-one erase and TS. The same erase VJP serves both terminal state and P:

```text
p=kᵀX; h=kᵀG
bar_X=G-gamma*k*h
bar_gamma=-<p,h>
bar_k=-gamma*(G pᵀ+X hᵀ).
```

After all group contributions, F=AB adds `bar_A+=bar_F Bᵀ`, `bar_B=Aᵀbar_F`. The explicit inverse derivative `bar_L-=Aᵀbar_A Aᵀ` executes once per head/block, then L is masked strictly below its constant diagonal. This shared step is O(C³); it is not presented as an O(C²) backward.

Qs/eta use the saved product recurrence. For row i, saved `p_j=product(1-beta[j+1:i])`, `pfinal=product(1-beta[:i])` give

```text
bar_n[i]+=GQ[i,i]; bar_beta[i]+=pfinal*ge[i]; bp=beta[i]*ge[i]
for j=0..i-1:
    bar_n[j]-=beta[i]*p_j*GQ[i,j]
    bar_beta[i]-=n[j]*p_j*GQ[i,j]
    bar_beta[j]-=p_j*bp
    bp=(1-beta[j])*bp-beta[i]*n[j]*GQ[i,j].
```

This is O(C²) and retains derivatives of zero products without dividing by beta or one minus beta. Direct interval backward scatters each causal pair's contribution onto exactly `j<t<=i`. Gamma's prefix intervals and Ktail's suffix intervals are added independently. No reverse prefix-difference or cancellation-prone difference-array reconstruction is used. Finish with `bar_beta+=bar_gamma/n`, `bar_n-=bar_gamma*gamma/n`, `bar_k+=2*bar_n*k`, and the original output scale on dq. Noncontiguous source gradients use explicitly contiguous owner buffers, so flattened accumulation remains a view.

## Source-sensitive precision selection and exceptional execution

Routing reads the same rounded binary sources as the mathematical reference. It never normalizes the public Key or replaces its original norm. All former magnitude, norm, gauge and amplification limits are **routing screens**, not terminal support exclusions. Invalid mathematical inputs and genuinely unrepresentable requested outputs/gradients still raise. A logarithmic norm test within 32 FP64 epsilons of the boundary is resolved with high-precision products of the original binary sources, so an ambiguous log comparison cannot itself reject a legal point.

For `delta=kappa ||v||²/n`, `y=sqrt(delta)`, the existing input-only homogeneous denominator bounds remain

```text
L_i = (1-beta_i)+beta_i/(1+y_i)
U_i = (1-beta_i)+beta_i*(1+y_i)/(1-delta_i).
```

The zero-entry gauge has `nu_i>=2^(-sum log2 U)`, and causal rho is bounded by products of `1/L`. The 32-bit amplification screen, triangular inverse majorant, original norm exponent screens (40/200), fixed-gauge exponent/spread screen (32), and source amplitude screens (24/200) avoid unsafe native arithmetic. They still select the retained C32 alternating-target witness as eight C4 FP64 blocks. The inverse majorant for `H=Z diag(1/n)` is unchanged:

```text
W_i = [1+beta_i sum(j<i)(1+sqrt(delta_i delta_j))*W_j]/(1-delta_i).
```

L and Zhat condition estimates remain before their native solves, with thresholds `2^10` for FP32 and `2^32` for FP64. Neither a condition estimate nor a tiny solve residual certifies source-gradient accuracy: a scalar system can have condition one while its coefficients and state feedback lose the digits needed by backward.

The additional source screen follows the subtractive quantities in preparation. For `a=n-kappa||v||²`, the relative componentwise subtraction condition is `(n+kappa||v||²)/a=(1+delta)/(1-delta)`. In state feedback, Cauchy–Schwarz and entry ball radius `R=sqrt(kappa)||S_entry||_F` give the conservative factor `(1+R)/(1-R)`; dropping `sqrt(delta)<=1` makes this screen independent of a favorable alignment. The implemented diagnostic is

```text
m = min_group,token(1-delta)
source_roundoff_screen = eps(dtype) * (2-m)/m * (1+R)/(1-R).
```

It must fit one eighth of the frozen relative budget (`2e-5` FP32, `1e-9` FP64). This reserves the other seven eighths for state/readout propagation and return rounding; it is a stage allocation, not a fitted margin cutoff. An unresolved native input margin directly selects extended arithmetic. A separate readout screen requires `eps(dtype)*maxabs(scale*q)` to fit one eighth of the absolute budget (`2e-5` or `1e-10`), because zero/canceling source derivatives do not acquire a relative-error allowance when q grows. These are conservative routing diagnostics, **not a uniform forward/backward error theorem**. They detect the R1 small-margin class before preparation loses precision, and the R2 small-margin/near-ball class even at C1. Casting an already rounded FP32 tape in backward cannot repair R1.

Absolute state cancellation must also be screened when curvature is zero or very small. The ball radius `sqrt(kappa)||S0||` is zero at c=0 regardless of the unscaled state, and a finite scalar solve does not bound the error in `S-gamma*k*(k@S)` or the curvature VJP. Before any preparation, define `Q=max(1,maxabs(scale*q))`, `S=max(1,maxabs(S0))`, and `W=max(1,max_token maxabs(v)/||k||)`, using logarithmic norm evaluation on the original binary sources. The additional diagnostic is

```text
state_readout_source_bits = log2(Q) + 2*log2(max(S,W)) + log2(W)
eps(working_dtype) * Q * max(S,W)^2 * W <= absolute_budget / 8.
```

The squared state/target factors cover the magnitudes of subtractive curvature-adjoint terms, including at c=0. This is a conservative magnitude screen for the bounded conditioning cases, not a uniform error bound or a replacement for the margin and inverse screens. It does not inspect beta for special treatment. Every source dependency, including dbeta at beta=1 and dc at c=0, remains in the complete graph. The sequence/head-wide source maxima are retained for every planned block, so a later large query or an earlier large write target cannot allow intervening state/tape conversions back to insufficient precision. Public streaming calls retain the declared state dtype and re-evaluate the screen from their supplied canonical state.

The CUDA wrapper applies the same screen at device FP64 working precision, including the public output scale, before launching preparation. Its original-source exponent screen applies to FP32 as well as FP64 sources. Exceptional calls use the reviewed CPU manual chunk route and return outputs/gradients on the original device. Device exceptions retain a private FP64 minimum preparation dtype even when all public sources and outputs are FP32; otherwise a later streaming call routed by a tiny c could downgrade to an FP32 CPU tape and violate the existing near-zero GPU gradient RMS budget. The minimum affects arithmetic selection only; original sources, requested-gradient ownership and public dtypes are preserved. No Triton kernel or tolerance changes are required. Normal g32/K=V128 C16/32/64 sources remain on the device path. Endpoint tests use exact full-write identities and independent 160/240-decimal canonical evaluation at identical binary inputs; the scalar identity is only a test witness and does not replace the production VJP.

For each flagged requested block, ends are tried longest first for FP64 preparation. If no FP64 prefix passes, at most four rows are prepared with mpmath arbitrary-exponent arithmetic. Four is a CPU work/partition choice, not a support cap: remaining rows continue in subsequent blocks. Every extended block uses the existing `_prepare_common`, `_prepare_group`, `_online`, `_reverse_group`, and `_reverse_common` formulas, with a narrow array adapter for reductions, matrix products and explicit forward/backward triangular substitution. There is no alternative recurrent model, dense state transform, generic AD engine, `mpmath.diff`, finite-difference VJP, or inverse-AD fallback in the implementation. The independent canonical mpmath differentiation lives only in tests.

The adapter uses an isolated context with `prec=max(300,128+4*E)` binary significand bits, where E is the nonzero input exponent span including fixed scale/gauge values. Arbitrary exponents remove hardware overflow/underflow of auxiliary coefficients; the extra significand span protects high-dynamic-range products and cancellation in the source algebra. This working-precision rule supplies substantial guard digits for the exercised cases; it does not establish a universal accuracy bound over every point of the unbounded open domain. Contexts are saved with the tape and are unaffected by another caller's global mpmath precision. No arbitrary exponent is rejected as proof of public unrepresentability. R3's `k,v -> 2^±101(k,v)`, frozen `s1=2^33`, `q=2^201`, and `a/n≈2^-34` all use this route. Their original n, all physical dependencies, and all source derivatives remain in the very same graph. No new Key scaling identity or missing scaling VJP is involved.

Every exception terminates carry on both sides, preserving canonical state and the distinct state adjoint, and restarting `Y=FS` with its full F/S source chain. Consecutive extended blocks do not round state or adjoints through FP64. Native carry segments keep their original E/H/target graph and next-F ownership. Checkpoint replay freezes the actual partition, precision kind and reasons; the saved tape fixes backward arithmetic. Public state still has the declared FP32/FP64 dtype, including at a new streaming call.

Backward computes all intermediate dependencies but checks/casts only `ctx.needs_input_grad` sources. It returns `None` for others. For R4, fixed c=0 therefore permits the finite output and dq even when an unrequested hypothetical dc exceeds FP32. Requesting dc still raises. The same check works when curvature is FP32 and other sources are FP64. Public output/state representability is checked after conversion, so the reviewed output near `6e38` still raises for FP32. A final state is converted and checked only when `output_final_state=True`; an output-only call retains its internal state and backward tape at working precision without constructing an unused public state.

Incoming cotangents also participate in accuracy selection before normalization. With the source diagnostic `G=Q*max(S,W)^2*W` above, `A=max(maxabs(dO),maxabs(dS_final))`, and the working route's absolute budget `b`, the required significand demand is `p_req=1+log2(G)+log2(A)-log2(b/8)`. This follows from the stage screen `2^(1-p)*G*A <= b/8`; A=0 needs no replay. FP32 and FP64 tapes supply 24 and 53 significand bits, respectively; extended tapes supply their saved context precision. The device route uses its FP64 working budget. This is an accuracy screen for the bounded source/cotangent cancellation classes, not an end-to-end condition-number theorem. The prior margin, norm, inverse and amplification screens remain.

If any saved block has fewer bits than this demand, backward **reprepares** the existing graph from original binary sources in one isolated context. The context includes all original sources, unscaled incoming output/final-state cotangents, fixed scale and any fixed gauges; its precision is at least `ceil(p_req)+32`, the source/cotangent exponent-span rule, and every saved extended context. Replay retains the forward partition, gauge/feedback options, and direct/carry edges, including each next-F source owner. It recalculates common/group coefficients, residuals, canonical states and carries using `_execute` and the same `_online` graph, then calls the existing complete manual reverse equations. Casting or rescaling the old tape is not used as an accuracy repair. Replay builds only an internal tape, with no new public state/output conversion or representability check; requested-source conversion and validation stay at the original owners. This preserves optional-state and selective-gradient contracts. Mixed heads/groups share the extended replay/accumulation context so a small incoming component is not erased while a large component is normalized.

Ordinary CPU AD keeps its independent native graph and higher-order behavior when the original tape meets the actual cotangent demand. An identity gradient router passes cotangents to those native output edges and returns no direct-source gradients. On an accuracy exception, it suppresses the native output edges and returns the complete original-source manual VJP through the direct-source edges. PyTorch accumulates aliased source positions normally. This exceptional branch is explicitly first-order-only and reports `ordinary_entry_cotangent_manual`; it does not provide independent AD evidence for the exceptional VJP. Manual/device routes retain their existing first-order contract. Telemetry distinguishes `backward_reprepared`, required bits, replay precision, unchanged block/edge partition and device `cotangent_accuracy` routing from forward fallback and the existing range-promotion fields.

A large incoming cotangent is normalized by a common power of two, restored at source owners before the requested dtype check. If any forward block uses extended arithmetic, the entire backward tape, source accumulators and incoming cotangents are promoted to one isolated context before scaling. Its precision covers the sources and cotangents and is at least that of every saved forward context. Native-only calls also promote backward when a scaled nonzero cotangent would enter the FP64 subnormal range. Promoting the complete backward prevents a later native-block conversion from deleting a small component whose product with a large saved coefficient is representable. This changes arithmetic representation, not the source graph or forward partition; normal native calls keep the existing FP64 backward. Telemetry separately reports cotangent-triggered promotion, backward extended block count and precision.

Telemetry records precision kind/bits, reasons, actual subblock lengths, fallback and extended counts, direct FS/carry counts, backward extended counts and requested source flags. Native condition estimates are `None` for extended blocks; they are not silently reported as passing native checks. The complete-call AD entry switches to the manual wrapper if it encounters an extended block; native blocks within that call still use native forward arithmetic. This is disclosed because ordinary and manual entries are not independent numerical witnesses on that route.

## Exceptional CPU cost and support status

The extended adapter requires installed mpmath and NumPy and executes object-array arithmetic on CPU. Extended blocks save the same complete preparation/residual/state tape and keep all source accumulation in arbitrary precision for that call. The serial triangular substitution is O(C²) per right-hand side; shared inverse/source preparation remains O(C³), interval scratch O(C²K), and state/output work O(CKV). These statements describe work/storage, not measured performance. There is no benchmark or GPU implementation in this work package. Normal tested K=V128/g32 C16/32/64 matrices retain their requested native partitions and actual carry graph with zero fallback.

The passing evidence establishes the bounded repair matrix and its unchanged budgets, including independent 90/160-decimal canonical truth for cancellation and scale cases. It does not certify arbitrary conditioning, a production FP32 backward, a GPU high-precision mapping, throughput, or every extreme streaming/cache scenario. Production mapping remains pending: choose and independently validate a compensated/scaled or higher-precision device representation implementing the same graph and full VJP. The CPU mpmath route must not be labeled GPU validated or used to claim normal-path speed.

## Validation and production mapping obligations

The formal suite is `tests/ops/test_key_gated_hyperbolic_delta_rule_chunk_reference.py`. Normal FP64 comparisons use the existing stricter `atol=1e-10, rtol=1e-9`, which also meets the author's `1e-9/1e-9` contract. FP32 uses frozen `2e-5/2e-5`. Every source records max absolute error, reference RMS and error RMS separately. Strong-decay and tiny mass derivatives use tighter scale-aware assertions. The author's GPU tolerance JSON is unchanged; no GPU mode is validated here, and its strict source RMS rule implies the reviewed `1e-8` near-zero RMS floor.

Coverage includes three independent complete paths; output-only, state-only and joint cotangents; beta endpoints and near one; c zero with analytic nonzero dc; ell zero; nonunit keys and nonzero state; C1/16/32/64 and tails; g1/16/32, multiple heads/groups and K=V128; fixed/live/detached gauges; precomposed/direct feedback; strong `[-1e8,-1]` decay with multiactive keys; retained inverse-overflow and nonscalar near-boundary cases; key-scale fallback; packed empty documents; noncontiguous, causal and streaming behavior; carry owner and reanchor checks; checkpoint replay; outer softplus gate and curvature/value parameter chains; explicit manual double-backward rejection; and a runtime prohibition on generic AD within manual backward.

Run the focused suite and independent oracle regression with the supplied CPU environment and `--noconftest`. Setting `KHDN_MANUAL_EVIDENCE` writes JSON metrics, dispatch records, environment, current HEAD and source hashes; pytest `--junitxml` records counts and failures. This isolation avoids initializing Triton or pretending a GPU exists. It does not establish production allocation-poisoning, accelerator, Layer/Model or distributed-training coverage.

The additional numerical regressions include all reviewed R1–R4 fixtures; FP32 same-rounded-source margin neighbors; g32 multiactive-Key multichunk cases; `2^±500` Key/Value scales with relative checks on tiny gradients; large-query full-write cancellations; mixed native/extended carry restart and checkpoint replay; extended packed empties; beta0/1 and trainable c0; selective requested sources; true FP32 overflow; and ambient CPU bf16-autocast isolation. Canonical test differentiation uses source-scaled perturbations, so its finite-difference step remains small relative to very small binary inputs. Raw 90/160-digit truth, input hex values and per-source error/RMS records are retained. The original test functions and tolerances are preserved.

This reference saves block preparations and all directly computed scaled residuals, U/Y/state checkpoints, masses and gauge factors. It pays for a no-grad scheduling probe and a differentiable/numerical preparation pass, plus FP64 tape conversion during manual backward. There is no claim of minimal memory or zero recomputation. CPU C²K interval scratch and direct interval scatters are expressly reference-only. Production mapping must tile interval products/reductions, preserve every source owner and precision decision, evaluate preparation and carry storage costs, and validate backward precision independently. No throughput conclusion follows from the exact carry identities or these CPU results.
