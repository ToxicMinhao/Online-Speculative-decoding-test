# K+ functional-step geometry: implementation and qualification

## Material Passport

- Date: 2026-11-08. ARS experiment-agent execution and evidence review.
- Status: **native qualification failed; main pilot not started**.
- Scientific implementation: `cdb4989026f6fa4e818ff70ae993945b008bbd76`; preserved first attempt: `f834dbd0da553d6231e39a4b5ff7a3b10a5dd623`.
- Base scientific implementation: `cdb640543d8fe3a43c8e63af4ccaaa002e09a599`; retained PR32 report: `18bd839cbd85f268a07f7da3abb9f9e14ba16bf0`.
- Publication: [PR #33](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/33). This report does not change the execution source or numerical protocol.
- Review provenance: same-team source and retained-record audits, including author overlap; not independent native replication.

## Outcome

The additive FSG updater, SCALAR control, continuation/data contracts and bounded qualifier are implemented. **109 CPU tests pass.** On A800, ordinary K+ and traced FSG with strength zero reproduce the same GSM8K weights, Adam state, RNG, counters, runtime and gradient-`None` pattern exactly. Enabled FSG also preserves the ordinary gradient and pre-filter Adam endpoint.

The small native chunked-versus-unchunked auxiliary-gradient oracle fails its predeclared coordinate tolerance. Qualification therefore stops before SCALAR, MathQA, the maximum-anchor resource test and native serving checks. No tolerance was changed, no further model execution was launched, and no benefit or harm to drafting is established.

## What is implemented

FSG preserves the original K+ forward-KL objective, normalization, sampling, native BF16 AdamW and its advanced moments. It filters the **actual native displacement**, using response-normalized pre-state matching-prefix sensitivity, the specified at-most-two-column sketch and FP64 small solve. The actual rounded candidate receives norm and old-draft-KL checks; finite rejection restores the exact native endpoint. SCALAR constructs its own hypothetical FSG norm and uses the specified 24-step rounded-endpoint search and common guard.

The code keeps historical event guards intact and authenticates each dataset's own full KPLUS4 origin. Fixed settings include strength 1, margin floor .05, native BF16 VJPs, FP32 cotangents and streamed FP64 reductions. There are no master FP32 weights, new damping, changed learning rate or additional optimizer applications inside the mechanism.

This milestone does **not** implement or launch the complete pilot orchestrator, all 16 counterfactual windows or the event-6 endpoint evaluator/report. Those integration pieces and full-pilot resource admission remain pending. The main pilot is not ready to launch while qualification is blocked.

## Two distinct qualification issues

### 1. Resolved recovery ownership defect

The first attempt completed one ordinary Adam update, saved and exactly decoded its endpoint, then failed while restoring the cached origin. PyTorch's optimizer loader shared a CPU counter tensor with the cached origin; the native step changed that in-memory counter from 512 to 513. The disk origin stayed unchanged. A focused real-optimizer CPU reproduction also exposed possible same-device moment aliasing.

The repair deep-copies the backend load payload. It changes ownership, not optimizer arithmetic, numerical tolerances or the algorithm. The repair reused the saved ordinary endpoint rather than repeating its Adam step. Two additional forwards/backwards recaptured its unavailable raw gradients without an optimizer application. Exact retained loss, anchor, response-observation, RNG/runtime and endpoint checks passed. Reconstructed gradients are not claimed to be original captured tensors.

The strength-zero case then passed exact final-state comparison over 239 tensors / 3,145,885,306 tensor coordinates, with no tensor or metadata differences. Enabled FSG's ordinary gradients also matched exactly over 58 tensors / 1,048,626,432 coordinates, and its pre-filter native endpoint/Adam state matched.

### 2. Unresolved native auxiliary-gradient discrepancy

The oracle uses the first two original anchors of the same saved response, without resampling. Input length is 183 tokens. It compares one-anchor head chunks with an unchunked two-anchor head calculation, retaining complete-block model conditioning.

| Quantity | Observed |
|---|---:|
| Fixed coordinate rule | `abs(error) <= 1e-5 + .016 * abs(unchunked)` |
| Coordinates exceeding that rule | 356,523 / 1,048,626,432 (0.0339990%) |
| Tensors containing violations | 55 / 58 |
| Global relative L2 discrepancy | 0.639685% |
| Separate global L2 ceiling | 1.6% |
| Chunked / unchunked gradient norms | 8.9374288 / 8.9415268 |
| Discrepancy norm | 0.0571976 |
| Largest absolute coordinate discrepancy | 0.0078125 |

Passing the global L2 condition does not override failing the coordinate condition. Prefix membership and normalized weights match exactly. Positive output-direction energy is present. Full and chunked FP64 risk reductions on the same native logits both give `0.07265389399933268`, with zero observed difference and deadband about `8.98e-10`.

The evidence localizes the failure to the auxiliary-gradient equivalence check. The oracle changes head batch size, activation checkpointing and a separately computed model graph together. Different BF16 head shapes or backward accumulation order are plausible explanations, **not confirmed causes**. The retained oracle does not separately compare hidden states, logits, cotangents and hidden-state VJPs, so it cannot yet exclude an implementation discrepancy or identify a specific kernel. No nonfinite failure is recorded.

The smallest useful next investigation would reuse the exact retained origin/native endpoint and original two anchors, with **zero new Adam steps**. First hold hidden tensors fixed and compare chunked/checkpointed, chunked/uncheckpointed and full/uncheckpointed head logits, cotangents and hidden adjoints. Only if those agree would repeated draft backwards with identical fixed adjoints be needed. That investigation has not been run or substituted for this failed gate. No tolerance relaxation is recommended from these results alone.

## Rounded-candidate observation

Before the oracle failed, the enabled disposable FSG proposal exercised the finite norm-rejection branch:

| Displacement | Norm |
|---|---:|
| Native K+ | 1.4583258594 |
| Continuous FSG | 1.4559686184 |
| BF16-realized FSG | 1.4584726072 |

Rounding made the realized norm **0.010063% larger** than native, so the hook restored native exactly and skipped candidate KL scoring. This is the prescribed behavior, not a guard failure. One disposable case does not estimate production fallback frequency or performance. Oracle cleanup also verified weights, RNG, modes, gradient pattern and optimizer tensor identity/versions were restored.

## Cumulative work and resource accounting

The second ledger contains the first ledger's exact bytes before appending new work. Failed attempts remain charged.

| Work across both attempts | Executed | Ceiling |
|---|---:|---:|
| Disposable native Adam applications | 3 | 8 |
| Ordinary response forwards / backwards | 6 / 6 | 16 / 16 |
| Additional response forwards | 18 | 96 |
| Additional backwards | 12 | 32 |
| Extra serving generations | 0 | 8 |

The enabled trial's Adam application completed before its oracle exception; this is not a successfully finalized continuation window. Five remaining Adam cases and all eight serving checks were unattempted. There were separately **four new shared feedback generations and four teacher preparations**, completed only once and reused unchanged. These packets remain the first two prescribed event-5 requests per dataset and count toward a future pilot's shared-feedback budget; they must not be recollected.

Attempt supervisors ran for 275 s and 402 s, respectively (677 s total, excluding intervening debugging and publication). Native qualification functions took 125.436 s and 299.933 s; these are nested within those supervisor totals. Feedback/teacher preparation took 53.756 s once. Final read-only integrity verification took 44.190 s separately. Qualification tracing, compilation, checkpoint round-trips and full-state comparisons are included where executed; these times are not deployment adaptation-cost estimates.

Maximum observed training allocation was **22.772 GB**, reservation **26.823 GB**. The 512-anchor/four-vector resource fixture was **not reached**; no maximum-shape memory claim is supported. Even its planned fixture retains actual response contexts and would not prove worst-case 2,272-token feasibility.

Repair admission had 62.802 GB free against 42.950 GB required, including an 8-GiB reserve. Post-run free space was 58.604 GB. Both GPUs are idle. The postflight rechecked all 3,402 sealed identities, origins/exports, models and runtime pins without change. No historical scientific artifact was deleted. Original failure evidence and the completed ordinary endpoint remain retained.

Disjoint repair counters sum to 20 draft forwards, 100 layer forwards, 376 ordinary objective chunks and 1,237 vocabulary-head calls. Nested oracle/hook counters are not added again. The original failed attempt did not persist its physical head/chunk counters before reset; those totals remain unavailable. Its response/backward/Adam counts are known. This gap is not filled by treating replayed counters as historical measurements.

After verification, only redundant source-transfer archives were removed (8.498 MB on A800 and 23.026 MB on Fenrir); deployed sources, Git revisions, seals, compact results and tensor evidence remain. Exact paths and before/after free space are retained in the cleanup receipts. No scientific replay capability was lost.

## Evidence and interpretation

- [Machine-readable summary](SUMMARY.json)
- [All 58 parameter comparisons](GRADIENT_COMPARISON.csv)
- [Private evidence identities and retrieval receipt](EVIDENCE_INDEX.json)
- [Same-team review receipts and interpretation checks](REVIEW_RECEIPTS.json)
- [A800 cleanup receipt](TRANSFER_CLEANUP_A800.json), [Fenrir cleanup receipt](TRANSFER_CLEANUP_FENRIR.json)
- Private compact copy: `research/dflash_functional_step_geometry_20261108/failure_002` (62 files / 806,238 bytes, all rehashed without differences). Original attempt is retained under `failure_001`; raw responses and tensors remain outside Git.

No efficacy estimates, p-values or confidence intervals were produced. Checks of significance/equivalence, pooling, multiplicity, selection and causal attribution are inapplicable as statistical findings here; qualification observations must not be promoted into such claims.

Lower protected KL is partly enforced and cannot establish success. For eventual evaluation: **if both FSG and SCALAR improve over KPLUS, but FSG has no demonstrated advantage over SCALAR, the necessity of directional geometry remains unestablished. This does not establish equivalence between them.** If SCALAR's improvement over KPLUS is also unresolved, the result is inconclusive. Off-fit buffered-response diagnostics remain training diagnostics, not held-out evidence.

**Decision:** keep the pilot stopped. Preserve this numerical failure and resolve the auxiliary-gradient discrepancy before completing native qualification. No performance conclusion or mechanism revision follows from this result.

## November 8 follow-up: retained-gradient diagnosis

The authorized zero-Adam diagnostic localized the numerical discrepancy to BF16 head batching; the original qualification remains failed. See [diagnostic report](GRADIENT_DIAGNOSTIC_REPORT.md) and its execution commit `41db7e31c3c80ee3220e5d919be327039caf2fb9`. No algorithm/tolerance change or main pilot followed.
