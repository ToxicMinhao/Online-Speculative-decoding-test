# KRC mechanism diagnostic: completed

## Material Passport

- Scope: two independent historical KRC replays, events 2 and 4; 20 fixed diagnostic windows and 80 temporary candidate slots. No serving or feedback generations.
- ARS workflow: experiment-agent run + validate. Historical endpoint reproduction is **VERIFIED**; the mechanistic interpretation is **ANALYZED**, not a demonstrated new method.
- Execution commit: `c9461345189b91d241ff21590027500e6ac7dab4`; main analysis implementation: `220c62e2c52489cf114125b679ae7188eea09218`. The numerical/cost supplement and independent audit use the separately published source scripts and hashes recorded in their JSON receipts. [PR #17](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/17).
- All 80 candidate slots are valid. Both decoded historical endpoint states match exactly; all original inputs and scientific sources remain unchanged.
- Stage 1 / project None is unchanged. The diagnostic has stopped; no replacement algorithm, tuning or additional event was run.

## Finding and decision

**The current correction often lowers its hinge loss without reliably restoring useful drafting prefixes. BF16 rounding weakens its intended displacement, and restoring some errors can erase progress elsewhere. Halving the correction or swapping the reference does not provide a consistent remedy.**

**Recommendation: prioritize ordinary K; this diagnostic does not justify selecting a replacement correction yet.** The current full correction produces −14 and +87 net prefix occurrences on its fitted references at the event-2 and event-4 probes, respectively, with losses on the separate diagnostic pair in both events. Its practical value remains inconclusive.

The clearest conditional hypothesis, if correction research is reopened, is to focus on the **first blocking, originally protected error**: hypothetically fixing that token recovers **581 of 599** occurrences obtained by fixing every protected error (97.0%). But the absolute opportunity is limited: only **177 of 9,923 reference-block occurrences** have a broken protected prefix; the all-error oracle adds about **0.060 prefix tokens per sampled block**, or 0.758% of K_ONLY's reference-prefix total. No numerical rejection threshold was prespecified, and these overlapping anchors are not independent examples. The concentration ratio does not establish that implementing this idea would be worthwhile. No new correction design is selected or implemented here.

## What was held fixed

The qualified Qwen3-8B/DFlash setup, BF16 weights/AdamW/KL, constant learning rate `3e-4`, weight decay `0.01`, clipping `1.0`, FlexAttention, block size 16, response-only supervision, per-response normalization, microbatch 1 and accumulation 2 were preserved. Each replay used its own complete historical parent state and its original 128-response O-generated feedback pool, two passes, natural presentation/anchor schedule, and inherited 7,424-step horizon. Sampling remained up to 512 anchors per response. No reseeding, forced anchors or optimizer substitution was used.

Event 2 restored KRC1 at Adam step 128 and ended at 256. Event 4 independently restored KRC3 at step 384 and ended at 512; event 3 was not rerun. The only continuing update was the original full KRC correction.

At windows **1, 16, 32, 48, 64, 65, 80, 96, 112, 128** of each event:

- **A** is the ordinary learning pair; **R** is the next pair within the same pass; **D** is the pair offset by 32. Their request identities are disjoint and their own retained anchors are fixed.
- **K_ONLY** is the actual post-K state. **R_FULL** applies the original correction. **R_HALF** halves its proposed FP32 displacement before BF16 casting. **D_FULL** constructs the same correction from D's own pre-K protected prefix and denominator.
- Each candidate is scored on all three pairs from the same post-K state. Only R_FULL is restored for continuation. D is buffered training data, not an unseen evaluation set; it is fitted data for D_FULL.

## 1. Does the correction realize its fitted-loss prediction?

**Not to the degree predicted by the unrounded linear model.** The full proposed FP32 displacement predicts essentially zero hinge, with absolute algebraic residuals below `4.6e-11` over these candidates. Casting the updated parameters to BF16 substantially reduces its effect along the measured gradient.

The entries below are medians of per-window ratios, each over ten fixed windows. “Rounded prediction” is `(V + g·realized_displacement)/V`, not an actual model evaluation. Actual hinge uses freshly recomputed native logits and strongest competitors.

| Event | Candidate | Realized/proposed displacement norm | Rounded predicted hinge / old hinge | Actual hinge / old hinge | Hinge decreases / increases / ties |
|---:|---|---:|---:|---:|---:|
| 2 | R_FULL | 57.0% | 75.5% | 74.4% | 8 / 1 / 1 |
| 2 | R_HALF | 41.5% | 93.5% | 90.3% | 6 / 4 / 0 |
| 2 | D_FULL | 54.9% | 77.1% | 78.1% | 8 / 2 / 0 |
| 4 | R_FULL | 57.7% | 72.5% | 59.6% | 8 / 2 / 0 |
| 4 | R_HALF | 48.2% | 91.1% | 79.1% | 7 / 2 / 1 |
| 4 | D_FULL | 53.3% | 78.3% | 72.8% | 7 / 3 / 0 |

R_HALF's norm denominator is its own half-sized proposal. All twenty R_HALF displacements remain nonzero after rounding; none is a quantized K_ONLY no-op. No candidate was silently rescaled or retried.

The difference between proposed and rounded linear predictions directly measures the effect of the realized parameter rounding on the same stored gradient. The remaining discrepancy between that prediction and the actual hinge can include nonlinear/nonsmooth changes and finite-precision forward/gradient effects; this experiment does **not** uniquely identify their individual causes. All windows, norms, coefficients and four predictions are in [PREDICTIONS.csv](analysis/PREDICTIONS.csv).

## 2. Does lower hinge restore native prefixes?

**Sometimes, but not reliably.** R_FULL lowers its fitted hinge in 16 of 20 windows. In six of those sixteen, its reference-prefix total does not improve: five decrease and one ties. Across all windows, its reference-prefix total improves in 10, regresses in 9 and ties in 1.

The table reports **summed matching-prefix token occurrences relative to K_ONLY**, not output tokens, verification-call savings or serving speed. Each event uses the same valid block set for every candidate. Event-2 denominators are A=5,065, R=4,978, D=5,027; event-4 denominators are A=4,306, R=4,945, D=4,372.

| Event | Candidate | Learning pair A | Fitted-reference pair R | Other buffered pair D | R windows improve / regress / tie |
|---:|---|---:|---:|---:|---:|
| 2 | R_FULL | +41 | −14 | −8 | 3 / 6 / 1 |
| 2 | R_HALF | +51 | −38 | +6 | 4 / 6 / 0 |
| 2 | D_FULL | +51 | +6 | +33 | 6 / 4 / 0 |
| 4 | R_FULL | −10 | +87 | −15 | 7 / 3 / 0 |
| 4 | R_HALF | +8 | +27 | −14 | 6 / 4 / 0 |
| 4 | D_FULL | −5 | −3 | −15 | 4 / 6 / 0 |

R is the fitted pair only for R_FULL/R_HALF; D_FULL instead fits D. Zero is the K_ONLY baseline throughout.

Gross changes expose cancellation. R_FULL's reference gains/losses are **115/129** in event 2 and **205/118** in event 4. It repairs **28/108** and **25/75** protected prediction errors, but also introduces **24** and **22** new protected errors. Its reference wrong-tie counts become **49** and **37**, compared with K_ONLY's **41** and **26**.

Ties are a real coverage limitation: a native incorrect token at equal canonical/rival logits has zero hinge and zero hinge derivative. For example, event-4 window 112 R_HALF reduces hinge to zero yet loses three prefix occurrences and retains three wrong protected ties. This does not show that all remaining problems are ties or that a tie-aware target alone would solve transfer.

## 3. Does correction undo useful learning or transfer to another pair?

Both happen locally. K's newly gained A-prefix occurrences total **2,110** and **1,704** at the two events' sampled windows. R_FULL subsequently loses **77** and **69** of those occurrences. Of K's **768** and **702** useful first-error repairs on A, it retains **743** and **684**, loses **25** and **18**, and obtains **44** and **29** additional repairs. The positive event-2 A total therefore does not mean every learned repair survives.

R_HALF improves the aggregate A total relative to R_FULL, but loses **85/74** of K's A-prefix gains, versus R_FULL's **77/69**. On R it loses **89/86** of K's newly gained prefix tokens, versus **68/51** for R_FULL, and its fitted-prefix totals are worse in both events. Its cross-D results are mixed. This does not provide consistent support for damping as the next trajectory-level fix.

R_FULL's cross-D totals are negative in both events. Reference-pair fitting and transfer therefore cannot be treated as interchangeable. Full repair identities, old-prefix loss, newly gained-prefix retention, native ties and strict violations are retained privately; their scalar totals are in [CANDIDATE_ROLE.csv](analysis/CANDIDATE_ROLE.csv).

## 4. How much does reference choice matter?

Swapping R for D changes the local results, but does not establish D as a better reference policy. In event 2, D_FULL improves D by +33 and R by +6, while R_FULL changes them by −8 and −14. In event 4, R_FULL improves R by +87, while D_FULL changes R by −3; both change D by −15.

The R/D gradient cosine has median **0.0317** in event 2 (range −0.0454 to 0.2336) and **0.0299** in event 4 (−0.0242 to 0.1175). Directions are weakly aligned, and their natural correction lengths also differ. This is evidence of reference sensitivity, not a direction-only or norm-matched intervention. D gradient-mode and final no-grad K_ONLY measurements agree exactly in all twenty windows; the separately recorded numerical comparison introduces no new acceptance gate.

## Oracle opportunity and the remaining hypothesis

The token oracle changes predictions directly, without any model update. “First” replaces only the earliest wrong originally protected token once per block; “all” replaces every wrong protected token. Neither is a realizable parameter update or a serving-improvement bound.

| Event | Role | Valid block occurrences | Blocks with broken protection | First-only gain | All-protected gain | First-only gain blocked by wrong ties |
|---:|---|---:|---:|---:|---:|---:|
| 2 | A | 5,065 | 25 | 89 | 89 | 22 |
| 2 | R | 4,978 | 104 | 291 | 301 | 97 |
| 2 | D | 5,027 | 98 | 253 | 258 | 99 |
| 4 | A | 4,306 | 13 | 56 | 56 | 39 |
| 4 | R | 4,945 | 73 | 290 | 298 | 83 |
| 4 | D | 4,372 | 82 | 175 | 182 | 82 |

Across all roles, first-only repair recovers 1,154/1,184 occurrences (97.5%); on the actual correction references it recovers 581/599 (97.0%). The R first blockers comprise **66 wrong ties and 111 strict errors**; ties account for **180/581** recovered occurrences. Keep the 9,746 zero-opportunity R blocks in the denominator.

First-blocker prioritization is the clearest conditional hypothesis in these records, but its high share of oracle opportunity must not be confused with a large absolute payoff. Merely making the present correction cheaper would not fix its observed prefix damage; simply halving it or substituting D has no consistent advantage. Wrong ties also expose a target limitation, but account for about 31% of R's first-only oracle gain, not all of it. Neither hypothesis is selected as a replacement mechanism by this report.

The opportunity is sparse and measured cross-pair transfer is unhelpful, so **prioritize K rather than automatically extending correction**. Reopening this branch would require an agreed test showing useful prefix gains without erasing K's repairs, followed by native held-out benefit. This conclusion does not prove that every possible correction would fail. This run authorizes no further test or automatic continuation.

## Pass and window variation

Each row covers five preselected windows. The totals below prevent a favorable aggregate from hiding a phase reversal; full paired window records are in [WINDOW_SUMMARY.csv](analysis/WINDOW_SUMMARY.csv).

| Event | Pass | Candidate | A prefix change | R prefix change | D prefix change |
|---:|---:|---|---:|---:|---:|
| 2 | 1 | R_FULL | +19 | −29 | −25 |
| 2 | 2 | R_FULL | +22 | +15 | +17 |
| 2 | 1 | R_HALF | −1 | −21 | −14 |
| 2 | 2 | R_HALF | +52 | −17 | +20 |
| 2 | 1 | D_FULL | +20 | −3 | +5 |
| 2 | 2 | D_FULL | +31 | +9 | +28 |
| 4 | 1 | R_FULL | −2 | +41 | −21 |
| 4 | 2 | R_FULL | −8 | +46 | +6 |
| 4 | 1 | R_HALF | −22 | +5 | −26 |
| 4 | 2 | R_HALF | +30 | +22 | +12 |
| 4 | 1 | D_FULL | −7 | +8 | −4 |
| 4 | 2 | D_FULL | +2 | −11 | −11 |

Event 2 and event 4 differ in parent weights and feedback. These differences cannot identify an isolated effect of training age. Earlier KRC1 already showed local non-monotonicity: its window-64 reference hinge and lost-prefix count increased. This run adds coverage, matched candidates, prediction checks and oracle accounting; it is not the first discovery of a local regression. See the [retained KRC1 report](../dflash_krc_event_20261027/REPORT.md). Historical event-4 serving results remain in their [separate report](../dflash_krc_events_3_4_20261028/REPORT.md); none are new serving measurements here.

## Correctness, work and cost

- All 20 physical K_ONLY witnesses reproduce the original post-K R decisions, margins, prefixes and hinge. Exact continuation restoration passes at every sampled window, including weights/buffers, Adam, gradients, counters, accumulation, RNG, modes and cache/sampler bindings.
- Both endpoints have **zero decoded scientific mismatches** against original KRC2/KRC4. Each records 258 timing/run-metadata differences separately; serialized container hashes are not claimed equal.
- All **1,720** sealed source/input files pass before/after verification. All **403** retrieved files pass transfer verification. All 80 candidates remain in analysis; none is excluded or replaced.
- The execution-focused CPU suite passes **45 tests locally and on A800**; **16 analysis tests** pass. Independent same-team review recomputes 114,772 candidate/block rows without reusing the report builder and rehashes 1,107 locally available sealed counterparts. The remaining 613 remote tensor/artifact entries were checked by the execution's before/after verification but were not downloaded for that separate local audit. This is not an external/blinded review or an independent GPU replication. See the [audit](INDEPENDENT_FINAL_AUDIT.md).

| Work | Actual | Bound |
|---|---:|---:|
| Ordinary Adam steps | 256 | 256 |
| Ordinary response presentations | 512 | 512 |
| Original-path response forwards | 1,536 | 1,536 |
| Diagnostic response forwards | 600 | 600 |
| Total core response forwards | 2,136 | 2,136 |
| Core backward equivalents | 996 | ≤1,064 |
| Disposable startup forwards / backwards / Adam | 4 / 4 / 0 | 4 / 4 / 0 |
| Serving / feedback generations | 0 / 0 | 0 / 0 |

The 996 backwards comprise 512 ordinary K, 446 original reference and 38 diagnostic D backwards. Zero-violation response skips remain recorded. Head/layer recomputation is separate internal work, not additional response presentations.

| Cost | Event 2 | Event 4 |
|---|---:|---:|
| Episode through endpoint comparison | 1,567.15 s | 1,505.07 s |
| Instrumented training loop | 1,417.30 s | 1,353.93 s |
| Live loading/resume | 26.61 s | 26.49 s |
| Save / export | 17.13 / 15.00 s | 17.15 / 14.91 s |
| Peak allocated / reserved GPU memory | 37.77 / 40.37 GB | 39.84 / 42.98 GB |

Campaign execution was **3,106.17 s (51.77 min)**. The enclosing supervisor, including preparation/postflight checks, ran **00:44:25–01:42:06 UTC on October 29**, 3,461 s, exit 0. CPU analysis and retrieval are outside these timers. Nested costs must not be added to episode totals. Snapshot/hash/restoration, candidate construction/scoring, D gradients and original correction costs are retained in [COSTS_WORK.json](analysis/COSTS_WORK.json) and the [numerical/cost supplement](analysis/NUMERICAL_COST_DETAIL.md); this instrumentation is not a deployment-cost estimate. D-gradient time is bundled with D candidate construction, not independently timed. Subphase host timers need not isolate asynchronous GPU work.

The two implementations retain the same scientific update. Diagnostic source adds wrappers and recorders; an event-4-only restoration wrapper preserves the original KRC2-only guard. At sampled no-op windows, temporary copies advance parameter version counters; the diagnostic uses exact restored-value checks instead of requiring unchanged counters, while preserving the other integrity checks. This avoids a false instrumentation failure without weakening scientific endpoint equality. No original author implementation was modified.

## Evidence, retention and interpretation limits

Raw private evidence remains under `/data/sources/minhao/runs/dflash_krc_mechanism_diagnostic_20261028` and the verified local `raw/` directory. It includes the sealed window/anchor plan, native predictions and canonical margins, block repair identities, branch decisions, costs, source identities and restoration receipts. Public tables contain scalar summaries, not prompts, response token arrays, teacher features or model/optimizer tensors.

After exact endpoint equality, the newly regenerated event-2 recovery container and export tensor were retired as scientific duplicates. The original complete KRC2 state/export remain, together with the replay's decoded metadata and exact removal receipt. The retired serialized container cannot be directly reloaded or hash-reproduced; it is not described as archived. The event-4 replay state/export and all original parents/endpoints remain retained.

Both GPUs are idle. Verified duplicate transport archives were removed after retrieval; A800 free space is 22.575 GB, above the preserved 8 GiB reserve. Exact paths, retained copies and before/after storage are in [POSTRUN_STORAGE.json](POSTRUN_STORAGE.json).

The summary retains every prescribed window and unfavorable result. No significance claim is made: overlapping anchors and repeated windows are not independent training replications. The 11 ARS statistical-fallacy checks are documented in the [descriptive analysis](analysis/REPORT.md), including phase reversals, selection, correlated observations and causal limits. That machine-generated page intentionally does not select an intervention; the final interpretation is this report and [INTERPRETATION.json](INTERPRETATION.json). Existing low losses or favorable local counts do not establish unseen-request benefit, training-seed robustness, a better reference policy, optimal correction strength/frequency, or training-inclusive speedup.

**Stopped after the agreed diagnostic and report.** Any next change remains a proposal for review.
