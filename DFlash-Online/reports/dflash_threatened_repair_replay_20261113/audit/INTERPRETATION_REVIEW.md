# Threatened-repair replay: result interpretation review

**Decision: unresolved local transfer; the prespecified advancement screen does not pass.** The intervention reaches the model and repairs the examples selected for replay. It does not demonstrate better independent prefix progress than ordinary K+ or matched random replay. Both primary point estimates are unfavorable and both intervals include zero. This is a completed negative advancement decision, not proof of zero effect or a failure to execute the mechanism.

## Scope and review provenance

Scientific execution `d51718a7fa993a1494e593d15a8539090227d8b8` completed the exact native KPLUS6→7 trajectory and 16 disposable same-state comparisons. This review independently reduces the retained scalar outputs without importing the production analysis, recomputes the original bootstrap only, and checks **2,251 published scalar values with zero discrepancies**. It introduces no model execution, new hypothesis test, seed, window or exclusion.

The reviewer previously authored analysis and parts of training/runner code and monitored execution. The reduction is separately implemented arithmetic, with same-team authorship overlap; this is **not** an external blinded review or independent GPU replication. Source material is the sealed handoff (retained with the private evidence), [published scalar results](../PRIMARY_RESULTS.json), [independent reduction](INTERPRETATION_REDUCTION.json) and separately written audit script (retained with the private review). Initial/final state evidence is in [state parity](../STATE_PARITY.json); execution ends at Adam step896 with exact historical native weights, moments, counters and RNG.

## What the comparison tests

P is the same pre-update model; N is its ordinary K+ update. T repeats selected historic repair positions that N is about to turn into a first rejection. U repeats a deterministic random selection from the same eligible memory, matched for proposal-position counts, stored weights and normalization. The two branches use the same padded full-response layouts and restore the native trajectory afterward. U is a competent control for adding replay at matched positions; it is not matched for gradient norm or clipping effect.

Only N advances. These local branch outcomes cannot be interpreted as final event-long T/U models, fresh-request latency, actual verifier-call reduction or online deployment improvement.

## Independent primary result

The unit is **accepted draft proposals per fixed block**. Each request averages its four sealed starts, then all16 fixed windows; all64 requests receive equal weight. The bootstrap resamples whole requests with their full arrays: 10,000 paired PCG64(seed1707) draws, shared indices for both contrasts, linear 2.5/97.5 percentiles.

| Contrast | Mean difference | 95% paired interval | Raw paired prefix-position difference / 4,096 block-windows |
|---|---:|---:|---:|
| T−N | −0.012451172 | [−0.026611328, +0.000494385] | −51 |
| T−U | −0.005371094 | [−0.017089844, +0.005615234] | −22 |

Absolute average prefix lengths: P5.375000, N5.385010, U5.377930, T5.372559. Neither lower bound is positive. The upper bound for T−N is near zero, but the retained interval still spans zero: do not turn the negative point estimate into established harm or the unresolved result into equivalence. These are conditional intervals on one native trajectory, one fixed random control and reused development requests; no training-seed uncertainty is measured. No p-value is invented and no generic standardized-effect benchmark is substituted for the actual units.

All16 windows were active, so the active-window secondary summary equals the primary. Each window has256 paired starts:

| Window | T−N net prefix positions | T−U net prefix positions |
|---|---:|---:|
| 8 | -4 | -3 |
| 16 | +4 | +10 |
| 24 | -1 | -9 |
| 32 | -1 | -1 |
| 40 | +4 | -9 |
| 48 | -15 | -13 |
| 56 | -4 | +3 |
| 64 | -10 | +7 |
| 72 | -5 | +1 |
| 80 | +9 | +1 |
| 88 | +2 | +0 |
| 96 | -20 | -12 |
| 104 | -2 | -2 |
| 112 | +4 | +0 |
| 120 | -10 | +5 |
| 128 | -2 | +0 |

T−N has5 favorable and11 unfavorable windows; T−U has6 favorable,7 unfavorable and3 tied. No single favorable window supplies an alternative primary. Across64 request means, T−N has15 wins/24 losses/25 ties; T−U has17/19/28. The largest negative request contributions are13/51 of T−N's observed net loss and13/22 of T−U's observed net loss, but the net-loss denominator mixes opposing signs. More interpretable gross request-net losses are93 and62, with the three largest contributing35.5% and45.2%, respectively. The respective top-three request gains contribute47.6% and40.0% of gross gains. These are descriptive concentration summaries, not extra significance tests or reasons to remove requests.

At the block-window level T−N gains89 prefix positions and loses140 (37 improved/49 worsened/4,010 tied). T−U gains74 and loses96 (34/34/4,028). These repeated outcomes are not4,096 independent requests.

## Support and actual control of the selected memory

The selection census contains13,040 tag-windows (815 fixed tags ×16):202 current-response exclusions,2,003 already not accepted under P, and10,835 eligible. Of eligible occurrences,84 are genuine first-rejection threats and26 have an earlier new blocker. In25/26 earlier-break cases the tagged token itself remains correct; in one it is wrong. These cases were reported, not silently treated as direct tag failures or selected for T.

The84 threats span75 distinct tags and52 requests. T selects54 occurrences, comprising50 distinct tags from41 requests across16 active windows. Fifteen windows have different U/T selections; the two arms overlap at only two tag-window occurrences, including the identical singleton at window128. All prespecified opportunity/placement thresholds pass. Threats are84/10,835=0.775% of eligible occurrences, and54/84=64.3% are selected under the fixed cap. Their rarity does not invalidate the passed support threshold, but a large conditional rescue rate should not be described as broad coverage.

On the **same54 selected T occurrences**:

| Endpoint | Accepted through tag | Exact rejection at tag | Earlier break | Direct tag correct | Wrong at tie |
|---|---:|---:|---:|---:|---:|
| P | 54 | 0 | 0 | 54 | 0 |
| N | 0 | 54 | 0 | 0 | 26 |
| U | 17 | 37 | 0 | 17 | 15 |
| T | 48 | 6 | 0 | 48 | 0 |

T rescues48/54 (88.9%) versus N. Compared with U on that same set, it gains32 cases and loses1. The minimal manipulation gate passes clearly. The106-occurrence U/T union also remains explicit: N accepts51, U62 and T100; T gains49/losses0 versus N and39/1 versus U. These are fitted, treatment-selected examples. They demonstrate native control, not independent transfer or a population success rate.

All16 physically executed T branches and15 U branches change BF16 weights versus N; the identical window128 U outcome is reused from T. T/N changed-entry counts range32,915,527–115,766,251; U/N range606,416–111,618,243. Native preclip norms range0.279–0.513. Added replay-gradient norms for T range0.243–3.380, and5/16 T branches activate clipping; U ranges0.00222–1.350 with2/15 clipped. N clips in0/16 instrumented windows. This weakens an explanation based on an absent or entirely rounded-away intervention. It also means the0.10 loss coefficient is not a statement of10% gradient influence; matching positions and weights does not equalize gradient strength. These diagnostics do not identify clipping or strength as the cause of the independent outcome.

## Acquisition versus preservation

All4,096 integer block decompositions satisfy the required identity before averaging. P has3,577 genuine first blockers and519 full/terminal prefixes; no blocker is invented for the latter.

| Branch relative to P | Gross acquired prefix positions | Gross lost prefix positions | Net | New earlier-error blocks | Old first blockers passed |
|---|---:|---:|---:|---:|---:|
| N | 177 | 136 | +41 | 43 | 52 |
| U | 127 | 115 | +12 | 46 | 43 |
| T | 143 | 153 | −10 | 51 | 49 |

T versus N has **34 fewer acquired positions and17 more lost**, giving−51. T versus U has **16 more acquired positions but38 more lost**, giving−22. Thus the observed independent difference is not simply “the selected repairs worked but were cancelled by another measured preservation gain”: selected-memory rescues occur in a different population, while the independent mean is unfavorable. Within independent probes, T has less acquisition and more loss than N; against U, its extra acquisition is outweighed by extra loss. These are exact descriptive accounting identities, not a proven parameter-level mechanism.

Of52 native repairs of a genuine P blocker, T displaces21 and U18. T still passes49 old blockers in total, so it retains31 of those native repairs and obtains18 other blocker repairs; U retains34 and obtains9. A correct later token behind a blocker is not counted as a useful repair. Prefix-position gains and numbers of repaired blocks remain separate quantities.

## Work, validity and cost

All planned scientific blocks are present and valid:16,384 conceptual branch blocks,16,128 physically executed and256 explicitly reused. No complete-response generation or timed-serving comparison occurred. Exact initial restoration, native duplicate checks, all six repaired qualification cases, window restoration, and final native parity pass. Execution001's adapter failure remains a failed qualification; it was not reclassified as a pass.

The completed retry uses167 physical Adam applications:128 native +31 disposable +6 qualification +2 duplicate. Adding the failed attempt's3 gives170, below the authorized171 cap. Retry work totals4,878 response forwards and545 response backwards; the prior attempt separately retains12 completed forwards, one failed forward,10 backwards. All unknown operations are zero. Logical response counts do not replace physical vocabulary/recomputation counts in the raw ledger.

The retry's response-forward categories are256 native,62 disposable source recomputations,203 replay,4,331 memory,20 qualification,4 duplicate checks and2 startup; these sum4,878. Memory includes4,128 P/N scans and203 selected-union branch observations. The expensive memory scan is real diagnostic work; do not price a deployable method using only the selected replay forwards. Conversely, this full experimental bill includes qualification, controls, state checks, exports and independent probes that are not all intrinsic serving operations.

Measured current campaign wall is7,164.792s (119.41min); cumulative occupied-card accounting including the failed attempt is7,280.388s (121.34min), below10,800s. Current occupied-card clock is7,094.401s. Main training peak allocated/reserved GPU memory is22.715/26.600GB. Phase walls are28.197s native qualification,4,446.440s training/qualification/native trajectory, and2,243.583s independent probes. Nested timers are not additive and are not a kernel-time breakdown. Ledger host-operation clocks for the4,331 memory forwards sum1,323.912s; physical T replay helper receipts sum142.600s across16 branches, and U140.710s across15. Those scoped clocks exclude some selection/restoration/state-I/O work and cannot alone support a deployment-overhead ratio. No serving speedup or training-inclusive gain is established.

## What this changes and what remains open

This result is more informative than an inactive preservation attempt: there are enough genuine threats, the BF16 intervention is substantial, and most selected native failures are repaired. The earlier PR45 frozen fitted-endpoint comparison had13rescues/8losses and an unresolved net interval, so it did not establish fitted benefit. The present same-state selected-threat check directly demonstrates its narrowly defined manipulation, while the independent advantage remains unestablished. The populations, conditioning and endpoints differ; it is not a causal attribution of the difference between methods.

Weakened explanations are: no actionable support, no applied gradient/BF16 change, inability to rescue the actual selected first rejection, and drift of the supposedly native continuation. Still unresolved are: why this replay influence fails to improve the independent average, whether gradient-strength/clipping differences matter, longer-trajectory effects, seed robustness, deployment cost and the prospects of other replay mechanisms. Neither rarity nor a low auxiliary coefficient alone proves insufficient treatment.

**One decision:** stop this fixed recipe at the requested report; do not advance it as an upgrade or start an event rollout. A separately authorized, prespecified result showing positive independent net progress over both N and U, with acquisition/loss accounting demonstrating that additional selected-memory control extends beyond fitted examples without offsetting losses, would change that decision. More selected-memory rescues alone would not.

## ARS statistical interpretation: 11/11 categories checked

Assessment: **CAUTION** because these are small conditional effects with unresolved signs, reused development data and one trajectory; the arithmetic and stated negative advancement decision are supported. No new inferential test is added.

| Category | Disposition | Assessment |
|---|---|---|
| Simpson reversal | CHECKED | Equal request/window/start weights are preserved. T−N is favorable in 5 windows, unfavorable in 11; T−U in 6/7 with 3 ties. The aggregate is negative amid mixed strata, not a reversal of uniformly positive within-stratum effects. No window is selected out. |
| Ecological fallacy | CHECKED | The estimand is the mean local accepted-prefix difference on 64 request clusters. It does not imply every request improves, and it is not a complete-request speedup or a future event effect. Request wins/losses/ties are reported. |
| Berkson/selection bias | CAUTION | Threat memory is conditioned on P accepting and N rejecting exactly at a historic repair. Its 48/54 rescue rate applies only to these selected tag-window occurrences. The independent panel was sealed without U/T outcomes but is a reused development panel, not a random new-domain confirmation. |
| Collider bias | CHECKED | No independent evaluation outcome enters selection or the primary denominator. The memory population deliberately depends on the native update; analyses conditional on that post-native event cannot be generalized to all model decisions or interpreted as a generic history/fragility causal relation. |
| Base rates | CHECKED | Report all 13,040 tag-windows: 10,835 eligible, 84 threats, 54 selected, and 26 earlier breaks excluded from targeting. Independent evidence has 4,096 paired block-windows but only 64 request clusters. The selected rescue percentage cannot replace these denominators. |
| Regression to the mean | CAUTION | Selection intentionally picks pending native failures. Same-state N and position-matched U controls plus independent probes prevent treating a return to correct on selected examples as sufficient success. The fixed selection and deterministic execution do not eliminate selection-conditioned optimism in memory summaries. |
| Survivorship | CHECKED | All 16 windows, 64 requests, 256 starts and 16,384 conceptual branch outcomes are retained; 256 identical-set aliases are explicit. No adverse request or window is removed. The failed first qualification and three consumed Adam applications remain in cumulative accounting. |
| Look-elsewhere | CAUTION | Only the two prespecified primary contrast intervals are recomputed, with shared bootstrap draws and a joint requirement that both lower bounds exceed zero. Window, tie and acquisition tables are descriptive; no new strata tests or best-window claim. Repeated historical mechanism screens and development reuse limit confirmatory interpretation. |
| Forking paths | CHECKED | Fixed coefficient, memory, windows, probe starts and PCG64 seed remain unchanged. The authorized retry repaired omitted teacher-tensor access and retained failed work; it did not retune thresholds or replace cases. No outcome-based sensitivity analysis or additional hypothesis test is introduced in this review. |
| Correlation versus causation | CAUTION | Restored same-P branches support attributing observed local differences to these complete update recipes on this fixture. They do not show selected-memory rescue mediates independent behavior, isolate direction from gradient strength/clipping, or establish deployment benefit, long-run learning or general superiority. |
| Reverse causality | CHECKED | P/N observations precede deterministic U/T selection; independent probes occur after exported branch states and never feed training. Thus independent outcomes cannot cause replay selection in this execution. Historical repair-risk associations remain separate from this treatment result. |

The audit artifacts retain every request/window contribution. No scientific files, settings or states were modified, and no additional execution is proposed by this review.
