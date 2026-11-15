# Native K+ prefix-loss investigation — completed

## Material Passport

ARS-Codex 3.24.2, experiment-agent execution and validation. Scientific execution: **`2549f02d9b1edba8fb86f8e91a497e68c224d586`**, PR [#47](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/47), execution002. Finished **November 14, 2026, 19:41:26 America/Cambridge_Bay** (November 15, 02:41:26 UTC), exit 0. The original failed qualification remains preserved separately. Results and code are published together; the later report commit is not the execution commit.

Status: **completed and audited**. Statistical interpretation is ANALYZED; independent CPU reconstruction verifies the retained reductions and native token scoring. This is not an independent GPU replication. Project Stage 1 / project None remains unchanged.

## Finding and decision

**Withholding current correct-prefix KL gradients passes the prespecified local screen.** Error-only current gradients, E, outperform ordinary native K+, N, by **0.015625 accepted proposals per fixed block**, with conditional paired 95% interval **[+0.002441, +0.028809]**. Across 4,096 paired block/windows, E acquires **17 more prefix positions** and loses **47 fewer**, yielding **64 additional accepted positions**.

This benefit is small and local. E fits the source responses less strongly than N, and E's advantage over the optimizer-history-only control H remains unresolved. E also takes smaller parameter steps. The experiment therefore supports another carefully designed test of prefix withholding; it does **not** establish a prefix-specific directional explanation, a better final checkpoint, serving speedup or a deployable algorithm. No follow-up experiment was launched.

## What was compared

We reconstructed the exact native KPLUS6→7 trajectory, from Adam step 768 to 896, using the original 128 feedback responses and teacher features, two passes and 256 response presentations. Sixteen windows were fixed in advance: 8, 16, …, 128. At each window, every branch starts from the same complete pre-update weights, optimizer state and RNG. **Only N continues training.**

| State | Current correct-prefix gradients | Current first-error gradients | Optimizer behavior |
|---|---|---|---|
| P | None | None | Static pre-update state |
| N | On | On | Native K+ AdamW |
| E | Off | On | Same Adam history and decay |
| C | On | Off | Same Adam history and decay |
| H | Explicit zeros | Explicit zeros | Adam moments/counters/decay still advance |

P is the live pre-update state at each window, not released DFlash. H is a real update and must not be called a no-update baseline. C and H are explanatory controls, not alternative winners.

The model pair remains frozen-target Qwen3-8B and its qualified DFlash draft. Training retains BF16 weights/AdamW/KL, constant LR `3e-4`, weight decay `0.01`, clip norm `1`, Flex attention, up to 512 anchors per response, eight-block head chunks, decay gamma 7, microbatch 1 and accumulation 2. The inherited 7,424-step horizon and seed-17 RNG history remain intact; this continuation does not reseed. Teacher, embeddings and vocabulary head remain frozen.

E/C gate row gradients with `torch.where(active_role, token_loss, token_loss.detach())` in both the native numerator and K+ correction. Full displayed loss values, first-error priorities, total selected-weight denominator and reduction order remain unchanged. N/ALL use the original native tensor and full backward. Equal displayed losses therefore do not mean equal updates. H supplies explicit zero tensors for the native-present gradient pattern, preserving native `None` entries.

The independent panel is the unchanged **64-request reused development panel**, disjoint by ID and content from all 896 event-1–7 feedback requests. Four predetermined starts per request are evaluated in native serving geometry at every window/state: **20,480 outcomes**, all freshly scored. BF16, greedy block-16 decoding, complete saved histories, FA3, context 4,096 and the original 2,048-token/EOS boundaries are preserved. These are fixed-block diagnostic forwards, not newly generated responses. A correctly matched EOS counts; a matching suffix at a saved cap is censored. Every retained proposal is checked during reconstruction, including positions beyond the scorable suffix.

Exact model/runtime/configuration/input identities are in [EXECUTION_IDENTITIES.json](EXECUTION_IDENTITIES.json) and [CONFIGURATION.json](CONFIGURATION.json). Private prompts, tokens, teacher tensors and raw native traces remain outside Git.

## Independent local outcomes

| State | Mean accepted proposals/block | Acquired positions versus P | Lost positions versus P | Net versus P |
|---|---:|---:|---:|---:|
| P | 5.375000 | — | — | 0 |
| N | 5.385010 | 177 | 136 | +41 |
| E | 5.400635 | 194 | 89 | +105 |
| C | 5.386963 | 163 | 114 | +49 |
| H | 5.387939 | 154 | 101 | +53 |

Acquisition/loss entries count prefix-position occurrences across **4,096 paired block/windows**, not independent tokens or requests. E's advantage over N is **+17 acquisition +47 avoided losses = +64 net**. Separately, E repairs **57** old first blockers versus N's **52**, among **3,577** pre-update blocker opportunities. E introduces **40** earlier breaks versus N's **43**, among **3,511** blocks with a nonempty old matching prefix. Eight E breaks and seven N breaks occur at previously fully correct blocks without an old blocker. E also loses **eight of the 52** blocker repairs made by N. These block counts have different denominators from prefix-position gains.

| Contrast | Accepted proposals/block | Conditional paired 95% interval | Role |
|---|---:|---:|---|
| E − N | **+0.015625** | **[+0.002441, +0.028809]** | Prespecified primary |
| E − H | +0.012695 | [−0.005371, +0.028809] | Descriptive |
| C − H | −0.000977 | [−0.015869, +0.014404] | Descriptive |
| C − N | +0.001953 | [−0.013672, +0.018066] | Descriptive |
| H − P | +0.012939 | [−0.003424, +0.030273] | Descriptive |

The reduction equally averages four starts within each request/window, sixteen windows within each request, then 64 requests. Uncertainty uses 10,000 paired whole-request bootstrap draws, PCG64 seed 1707, linear 2.5/97.5 percentiles. Correlated starts, states and windows remain together. The 20,480 outcomes are not 20,480 independent samples; the interval is conditional on this one native trajectory and reused panel.

E−N request means improve on **22**, regress on **12** and tie on **30** requests. Windows improve/tie/regress on **11/2/3**; first- and second-pass mean differences are **+0.0078125** and **+0.0234375**. These are descriptive views of the same primary sample, not fresh replications. All per-request/per-window and raw scalar outcome rows are retained below.

## Source fitting differs from independent behavior

| State | Source positions acquired | Source positions lost | Source net versus P |
|---|---:|---:|---:|
| N | 2,948 | 139 | +2,809 |
| E | 2,119 | 153 | +1,966 |
| C | 1,051 | 162 | +889 |
| H | 383 | 359 | +24 |

E gains **843 fewer net source-prefix positions** than N: 829 fewer acquisitions and 14 more losses. Thus, the independent improvement is not explained by stronger immediate fit to the learning pair. It also does not prove which current gradient directions caused the independent difference.

The fitting population comprises 32 response/window presentations, **8,454 overlapping anchor occurrences**, 123,450 valid proposal occurrences, 66,993 pre-correct prefix occurrences and 6,356 old-blocker opportunities. All sampled slots are kept and eligible; no padding or zero-supervision slots occurred. The five states do not multiply exposure denominators. Native training across the full event uses 256 presentations and **556,152 selected supervised-position occurrences** out of 941,612 available occurrences. Disposable branch and observer work is reported separately.

## Optimizer history and update-size limits

E's realized BF16 parameter-displacement norm is **89.57%–97.00% of N's**, median **94.53%**, over the sixteen windows. Removing a current role changes magnitude and interaction with persistent Adam even though forward loss values and the denominator stay fixed. This is not a norm-matched directional comparison.

H moves parameters despite receiving zero current gradients. E−H and C−H intervals include zero: this leaves their additional independent benefit over that history-only update unresolved, and does **not** establish equivalence. No favorable C/H subgroup is promoted into a new primary outcome. Actual displacement, gradient, clipping and accounting summaries are retained in the execution review.

## Correctness, failure preservation and audits

Execution001 stopped because the H oracle returned an `OrderedDict` while the production checkpoint filter returned a plain `dict`; the strict checker stopped before tensor comparison. The narrow repair aligns that schema and retains exact BF16 tensor checks. It changes neither the algorithm nor tolerances. The original failed record is still [QUALIFICATION_REPORT.md](QUALIFICATION_REPORT.md), not retrospectively a pass.

The authorized execution002 passes **68 focused CPU tests** and all six disposable GPU cases: N, ALL, E, C, H and restored N. ALL and restored N match native losses, gradients, weights, Adam state, RNG and counters exactly. H now matches the direct zero-gradient Adam recurrence, including weights. The fixed pair has 291 active E gradient occurrences and 2,937 active C occurrences. Exact origin restoration passes.

All **176 transient state checks** pass. Final native weights, Adam, counters and RNG match the retained NATIVE7 endpoint exactly at 896 steps / 1,792 cumulative presentations. All **20,480 planned outcomes** are valid, with **zero invalid, unknown, missing, duplicate or unattempted rows**.

Independent CPU audit implementations reproduce **32,821 scalar/statistical checks**, reconcile **991,193 native work-ledger lines**, and rescore all 20,480 endpoints from raw native proposals/verdicts. A further **6,915 execution/cost checks** pass, including 32 forward-value comparisons and 64 exact response-loss/denominator checks. Reconstruction checks all **41,625 retained proposals in 2,775 historical blocks** plus four neutral qualification blocks. All 62,530 compact delivery files are authenticated. The token auditor authenticates its 61,653 used evidence files and 64 parent records. No GPU rerun is part of these audits.

All **11/11 ARS statistical interpretation categories** were assessed. Reused development data, many preceding research explorations, one trajectory, conditional uncertainty and unresolved update-scale attribution limit the strength of inference. The audits use separate implementations within the same team, with author overlap disclosed; they are not external or blinded replications.

## Work, duration and storage

| Work | Successful attempt | Cumulative including failed attempt |
|---|---:|---:|
| Adam applications | 182 = 128 N + 48 disposable + 6 qualification | **187 / 187 approved** |
| Training/qualification/source response forwards | 492 | 502 |
| Response backwards | 332 | 342 |
| Independent native endpoint blocks | 20,480 | 20,480 |
| New complete-response generations / new teacher preparation | 0 / 0 | 0 / 0 |
| Occupied-card time | 6,800.969879 s | **7,004.039942 s / 10,800 s** |

The successful attempt's 492 response forwards are 256 native +64 E/C +160 source observations +10 qualification +2 disposable startup. H needs no update forward/backward, but its 32 source-observation forwards are included in 160. Native execution separately records 23,425 draft blocks, 23,484 target calls, 448,601 context forwards and 84 model loads, including reconstruction and qualification. Projection/layer calls are nested work, not additional response presentations. Exact accounting is in [WORK_AND_COSTS.json](WORK_AND_COSTS.json).

Successful campaign wall time was **6,872.81 s (114.55 min)**. Cumulative occupied-card time including the failed attempt was **116.73 min**, inside the original three-hour cap and consistent with the approximate 1.5–2-hour estimate. The training process took 3,603.32 s, including qualification, state copying, exports and diagnostics; this is not a pure ordinary-K+ training-cost estimate. The enclosing independent native scoring phase took 2,707.38 s. These nested timings must not be added as disjoint costs. The earlier failure's five applications and 203.070064 seconds remain explicitly charged.

Main training peak allocated/reserved memory was **20,485,806,080 / 25,899,827,200 bytes** (19.08 / 24.12 GiB). Sampled native engine footprint was about 54.6 GiB, which is not an allocator peak measurement. Only physical GPU 0 was used; GPU 1 remained available. Both are idle after completion.

Before launching, safe cleanup removed **768.41 MB** of verified redundant transfer archives/generated bytecode on Fenrir and **4.01 MB** of stale empty entries/generated bytecode on A800. Scientific results, source/history, models, environments and recovery dependencies were retained. A800's original 210.053 GB peak-write-plus-reserve admission passed. After the planned exports it has **159.14 GB free**; Fenrir has approximately **54.06 GB free** after verified evidence retrieval. Exact deletion paths and retained-copy proofs remain in the private storage-maintenance manifests. No current branch exports or unique evidence were deleted to make this screen fit.

## Evidence and next boundary

- [SUMMARY.json](SUMMARY.json), [PER_REQUEST.csv](PER_REQUEST.csv), [PER_WINDOW.csv](PER_WINDOW.csv).
- [INDEPENDENT_TRADEOFF.csv](INDEPENDENT_TRADEOFF.csv), [PROBE_OUTCOMES.csv](PROBE_OUTCOMES.csv), [PROBE_KEYS.csv](PROBE_KEYS.csv).
- [SOURCE_TRADEOFF.csv](SOURCE_TRADEOFF.csv), [SOURCE_OUTCOMES.csv](SOURCE_OUTCOMES.csv).
- [QUALIFICATION_PASSED_002.json](QUALIFICATION_PASSED_002.json), [FINAL_NATIVE_PARITY.json](FINAL_NATIVE_PARITY.json), [DELIVERY_VERIFIED_002.json](DELIVERY_VERIFIED_002.json).
- [RESULT_AUDIT_002.md](RESULT_AUDIT_002.md), [RAW_PROBE_AUDIT_002.json](RAW_PROBE_AUDIT_002.json), [RESULT_INTERPRETATION_002.json](RESULT_INTERPRETATION_002.json).
- [FINAL_EXECUTION_COST_REVIEW.md](FINAL_EXECUTION_COST_REVIEW.md), [FINAL_EXECUTION_COST_SUMMARY.json](FINAL_EXECUTION_COST_SUMMARY.json), [FINAL_UPDATE_NORMS.csv](FINAL_UPDATE_NORMS.csv).

The direction earns the handoff's **next design step only**: determine whether its benefit exceeds an ordinary update-size alternative and whether it persists in an actual short continuation before measuring complete-response acceptance and latency on independent confirmation data. Neither follow-up is implemented or authorized by this result. The bounded experiment ends here.
