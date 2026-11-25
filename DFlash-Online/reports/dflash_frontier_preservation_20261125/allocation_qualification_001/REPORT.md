# Preservation-allocation numerical and native integration qualification

**The declared numerical and native integration gates passed. Production collection has started after storage admission; training and serving results remain pending.** The numerical audit made zero Adam updates; subsequent integration used four disposable updates and restored the parent after each. Neither phase measures learning or serving benefit.

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-11-25
- Verification Status: ANALYZED — native receipts authenticated and scalar arithmetic independently recomputed; no independent model rerun
- Version Label: native_token_contraction_v2_qualification_001

Execution source: `bf7c93455968ca1bba1c7cc179bde1abb6810444`. Native integration executed at `a7ae158dba4ef350ba061ccb3ec596da1652e894` on the same authenticated parent/window. Source, state, contract and retained raw-receipt identities are in [IDENTITIES.json](IDENTITIES.json).

## Declared target and prospective gates

The audit restores the exact PR59 PAIR1 warm parent (step 128, presentations 256, event 1, 58 warm Adam entries), using the same two retained training responses and frozen 512/269 anchors. Both responses enter the complete reference-gradient, FP32 optimizer-shadow, native BF16 preview assessment and fast-allocation calculation. Native K+ repair weights and PAIR remain fixed. No new feedback or held-out input is used.

For prefix position j in response r, the reference target is

```text
s_j = - dot_FP64(ordinary_native_gradient(unweighted_FKL_j), w_FP32) / (2 Z_r)
w = (d Phi / d g)^T r_preview
c_j = s_j - sum(beta_i s_i) / sum(beta_i)
a_j = beta_j (1 + 0.25 c_j / max_i |c_i|)
```

Here beta is the native positional weight exp(-(j-1)/7), Z is the native chunk-reduced sum over current prefix and first-error positions plus 1e-6, and g is the two-response teaching gradient with beta on prefix and unchanged K+PAIR repair. Phi is the declared continuous FP32 warm-AdamW shadow; r_preview is its native BF16 preview assessment gradient.

The actual native BF16 token-FKL gradient is computed with create_graph=False, contracted with represented FP32 w using FP64 products/reduction, then normalized once. Reference scores are cast to FP32 before the production allocator. Exact constant/singleton pools return beta. Stable centering subtracts a pool minimum before the same weighted centering. Preview assessment uses native logits and FP32 margin/softplus/reduction. The fast route casts w once to BF16 without rescaling; its direction-cast error is separately recorded and never replaces the primary FP32-w reference.

The [sealed contract](NUMERICAL_CONTRACT.json), SHA256 `f5c380be6ffb55684a76e8b2bdba3ffecd4268ea0703e735490ad60aaccc30e6`, fixes:

- **Gate A:** finite FP32 allocations, positive teaching weights, block mass/cap, unchanged repair coefficients, direct support P∪E, original native response denominators and factor 1/2 exactly once.
- **Gate B:** each fast allocation versus reference, and the two fast allocations versus each other, must satisfy `|delta a_j| <= 0.025 beta_j + t_j`, where `t_j=2e-7+2e-6|beta_j|`.
- **Gate C:** with FP64-centered FP32 reference scores, `G_fast >= 0.9 G_reference - sum(t_j|c_j|)`, where `G=sum((a-beta)c)`. Exact constant references instead require both fast allocations within t of beta. At least one nonconstant selected pool must move coefficients in both repeats.

Each response contributes its lowest eligible block and longest distinct eligible block, with lowest-index ties, selected before scores. A pool includes its entire prefix; eligibility requires at least two prefix positions. Coverage was 483/261 eligible pools and 2/2 selected pools. The four complete pools contain 58 positions, below the 60-oracle limit. All 58 serial oracle backwards completed on attempt 1; both fast evaluations were coefficient-identical.

| Response index | Native block index | Prefix positions | Maximum coefficient error / beta | Fast/reference local alignment | B / C |
|---|---|---:|---:|---:|---|
| 0 | 0 | 13 | 0.16822% | 100.05944% | Pass / Pass |
| 0 | 34 | 15 | 1.25920% | 101.47384% | Pass / Pass |
| 1 | 0 | 15 | 0.24996% | 100.04402% | Pass / Pass |
| 1 | 1 | 15 | 0.40266% | 99.84628% | Pass / Pass |

Both repeats pass every gate; all four pools are nonconstant and active. [POOLS.csv](POOLS.csv) gives all gate scalars and [TOKEN_ALLOCATIONS.csv](TOKEN_ALLOCATIONS.csv) all 58 position-level scores and coefficients, without token IDs. Native denominators are 2959.242919921875 and 1564.56884765625. The latter differs from a differently ordered full FP32 sum by 0.000244140625; native chunk reduction is preserved and the inherited mass tolerance passes.

## Numerical-audit state, work and resources

Both fast-stage state checks, the complete-audit check and final restoration pass at step 128/presentations 256 with exact weights, optimizer and RNG; no live gradient buffers remain. The constructor's initial resume receipt is also retained. Native work includes 70 full draft forwards (2 role-freeze + 2 reference + 2 preview + 4 fast + 58 oracle + 2 allocated-gradient diagnostic), 64 ordinary backwards (4 reference/allocated + 2 preview + 58 oracle), one FP32 shadow preview, and zero native Adam steps. Physical hooks record 350 layer entries and 1586 vocabulary-head calls, including checkpoint recomputation. Nested head/attention operations are not additional full-model passes. No work-ledger operation is pending.

Native elapsed time is 118.065277 s; supervised occupancy is 123.048981 card-seconds. Adding 267.424017 historical card-seconds gives 390.472998 cumulative. The new four-hour phase starts at epoch 1795615777.8640497 and is 123.769169 s old at this process receipt; the previous 3465.673397 s elapsed interval remains separately disclosed. The eight-card-hour cap includes historical work.

Peak after parent import is 37,604,353,024 B allocated and 48,368,713,728 B reserved; initialization peak is not separately established. End-receipt A800 free space is 429,778,173,952 B. Charged study allocation is 2,669,158,400 B across A800 and the measured local copy, including earlier attempts and compiler growth. This is a study total, not this audit's incremental writes. Limits remain 40 GB study allocation and 400 GB A800 free reserve; no inherited artifact was retired. Both local and A800 preflight suites passed 103 CPU tests. Exact counts, scopes and receipts are in [SUMMARY.json](SUMMARY.json).

## Native integration and attributable update activity

The second process independently restores the same parent for four one-step cases: native **CONTROL**, new runner **DISABLED**, cached native **BETA**, and cached **CANDIDATE**. Each ends at step 129/presentations 258, then restores step 128/presentations 256. These are disposable qualification updates, with no production endpoint retained.

CONTROL and DISABLED match exactly in normalized preclip gradients, BF16 weights, Adam moments/options/counters, RNG and scientific response rows. BETA's final native preclip gradient exactly equals the scoring reference g0. Cached CANDIDATE uses the same native anchor support as CONTROL; 8,961 coefficient positions change versus BETA, and its preclip gradient, BF16 parameters and both Adam moments change. This shows the assigned coefficients reach an actual native update; it does not establish improvement over CONTROL.

All four cases and final complete-state restoration pass. Four of the global eight disposable Adam attempts are used, on one fixed window. Integration adds no serial token oracle. It includes a discarded two-response startup with two backwards and zero Adam, eight cache-calculation draft forwards, and eight case forwards/backwards: **18 full draft forwards, 14 ordinary backwards and four Adam steps** overall. Physical hooks record 90 layer entries and 2,842 vocabulary-head calls including checkpoint recomputation. Nested attention/head calls are not extra full-model passes. The startup took 20.366560 s and is included in this process's costs. [INTEGRATION_CASES.csv](INTEGRATION_CASES.csv) and [INTEGRATION_SUMMARY.json](INTEGRATION_SUMMARY.json) retain scalar case, work and identity evidence.

The complete candidate-window timing is **21.769045 s = 17.145765 s sampling/scoring + 4.623280 s native teaching**. It excludes startup, parent/state hashing, restorations, other branches and the BETA shadow diagnostic; the native teaching component subtracts qualification-only preclip inspection/hashing. This is one fixed training window, with bounded ledger bookkeeping, not a full-event timing or end-to-end speedup.

The same-coefficient BETA diagnostic also exposes a substantial continuous-shadow/native-update difference: FP32 shadow displacement L2 1.968027, native BF16 displacement L2 1.230760, difference L2 1.759312 (0.893947 relative to shadow displacement). Native weights differ from the cast shadow at 1,420,195 coordinates. This is retained as a non-gating diagnostic of the declared continuous surrogate; qualification does not prove the shadow closely matches a finite BF16 update. The cast-shadow mismatch also prevents attributing the whole difference solely to final parameter quantization; no isolated arithmetic decomposition is established.

Integration native elapsed time is 153.585968 s and supervised occupancy 160.228055 card-seconds. Historical plus both new processes totals **550.701053 card-seconds**; the original four-hour allocation clock is 1,560.045084 s old at this receipt. Post-startup peak is 37,657,740,800 B allocated / 48,565,846,016 B reserved; initialization/startup peak is not separately measured. End-receipt A800 free space is 429,442,310,144 B; combined charged study storage is 3,004,755,968 B. Local full-suite 133 tests and A800 focused integration/training/supervision 14 tests pass. The remaining production forecast and staged live storage admission are separate requirements.

## Interpretation and remaining gates

The aggregate diagnostic gives native `-(g(a)-g(beta))·w = 0.167047969745`, versus centered score prediction 0.167018853066, difference 0.000029116679. This is a separate diagnostic, not an admission gate or finite-step benefit estimate. That numerical audit made no Adam update.

**The historical hidden-state check remains FAILED:** 2.047197% discrepancy exceeds its sealed 1.6% relative tolerance with 1e-5 absolute allowance. The user explicitly adopted this new allocation-level target and prospective gates; the new pass does not relabel that old result. It qualifies the declared allocation on one fixed training window, with no general accuracy, learning or serving claim. Native integration and attributable BF16 activity now pass on the same fixed window. Production source `225802657d630c0985abd68d6e7453f3e47c5b74` passed 140 focused CPU tests and starts the original shared-feedback event. Collection admission passed at 429.108 GB free with a 17.004 GB worst-case write bound. Remaining training storage is rechecked against the actual collected pool. The full-work forecast is 8,181.657 s including contingency; no serving outcome is yet available.

The [independent audit](AUDIT.json) rehashed 31 raw files and recomputed all 58 normalizations and all Gate B/C arithmetic. Gate A relies on authenticated runtime invariant receipts; unsaved full tensors cannot be independently reconstructed from scalars. The numerical audit covers all 11 ARS interpretation checks; these same no-efficacy and fixed-window limits also apply to integration. Repeats are numerical repeatability checks on identical inputs, not independent replicates. This public bundle excludes prompts, response tokens, anchor locations and feature tensors.

The [independent integration audit](INTEGRATION_AUDIT.json) authenticates all 45 retrieved files, the archive and shared reference/step/clock ledgers. It confirms four completed disposable steps, four remaining of eight, exact saved CONTROL/DISABLED endpoint identities and all initial/final restorations. It verifies saved identities and runtime equality receipts without reloading original model tensors.
