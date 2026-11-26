# Point2A: C-referenced allocation scoring

**Completed, November25,2026.** [PR62](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/62). Actual execution commit: `1a1fb42d4fbf90d2143f0ada544713525e5d39be`. Three independently restored native updates; 28 token-gradient oracle backwards; no retry, feedback, full event or serving generation.

**Conclusion: changing the reference changes the allocation, but provides only a tiny, mixed local advantage over F_BETA and no discrete improvement on the disjoint TRAIN witnesses. Park this reference change under the agreed convergence rule.** F_C gains one net matching-prefix occurrence over F_BETA: one update response loses14 and the other gains15. It adds three useful repairs but damages one extra old-prefix position. All64 disjoint blocks tie on every discrete metric. Their assessment improves, but that does not establish useful drafting or transfer. The next decision should concern assessment/transfer rather than a reference/cap sweep; no follow-up run is authorized.

## Material Passport

- ARS experiment-agent run + validate; controlled one-window mechanism diagnostic.
- Native qualification, historical control reproduction, state restoration and independent scalar/receipt audit pass.
- Private prompts, tokens, anchors, feature paths and runtime tensors remain outside Git.
- All quantities are overlapping TRAIN occurrences, not serving calls or independent samples. No confidence intervals or serving-speed claim.
- Project stage and Graduate unchanged. Point2B and Point3 remain provisional.

## Single change and fixed contract

Only the sensitivity reference changes. F_BETA uses the unchanged beta gradient. F_C uses the **actual normalized native C gradient immediately before clipping**, captured during the one planned C update. After complete parent restoration, it computes `S_C=T32_parent(g_C)`, `Q_C=BF16(S_C)`, `r_C=assessment_cotangent(Q_C)`, and `w_C=J_Cᵀr_C`. Scores are `−dot(native unweighted token-FKL gradient,w_C)/(2Z_response)`. This is a declared local surrogate around C, not an assertion that finite BF16 accumulation is exactly linear in coefficients.

The allocation remains centered by positional beta, with budget `sum(beta)` and bounds `[0.75beta,1.25beta]`. It is **not centered or budgeted around native C coefficients**. Native C alpha0.5/cap0.25, positional gamma7, K+eta0.5, PAIR, LR3e-4, decay0.01, clipping1, accumulation2, five trainable layers, Flex attention, native BF16 head/FKL/Adam and eight-block head chunks remain unchanged. Native C uses its ordinary implementation; F_BETA/F_C use the same qualified cached candidate path.

The exact PAIR1 parent restores step128/presentations256/event1, warm optimizer, RNG, histories, runtime/modes and empty accumulation. Each temporary endpoint reaches129/258/event1 and restores fully afterward. The fixed two-response window has512+269=781 blocks,762 nonempty prefix pools and8,979 protected prefix occurrences. All coefficient arrays are sealed before their own update. The first complete eligible pool in each response is selected before scores:13+15 positions. No outcome selected data or settings.

Both unchanged controls reproduce PR61's **complete scientific identities**, including preclip gradients, parameters, Adam moments/options/counters, RNG and response observations. Only measured times and event labels are excluded. [Identity records](IDENTITIES.json), [state receipts](STATE_AUDIT.json), and the [pre-restoration scoring-state check](SCORING_STATE_PRESERVATION.json) preserve evidence. Source postflight verifies5,295 entries and15 protected files unchanged.

## Numerical qualification

All78 focused CPU checks pass on A800; independent source reviews found no blocker. The new C direction passes both complete native-token pools under the unchanged criteria: coefficient error at most `0.025beta+2e−7+2e−6abs(beta)` and alignment at least90% with the inherited allowance. Maximum observed coefficient error is **0.4084% of beta**, and the minimum reference-alignment ratio is **99.1426%**. The other pool has0.2160% maximum coefficient error and100.4857% alignment. Both reference pools are nonconstant; selected singleton coverage is zero and is not represented as tested native coverage.

Repeated fast scores and allocations are exactly identical. Mass, positivity, support, cap, repairs, normalization and parent-state checks pass. Native oracle contractions use represented FP32w_C with FP64 products/reduction and divide by2Z once after backward; effective BF16-direction evidence is separate. The earlier hidden-state2.047197% versus1.6% failure remains a failed historical diagnostic. See [numerical receipts](NUMERICAL_QUALIFICATION.json) and [28 oracle records](REFERENCE_TOKENS.csv).

## Allocation and prediction changes

| Contrast | Changed prefix coefficients /8,979 | Maximum abs movement / beta | Maximum half-L1 movement / pool budget |
|---|---:|---:|---:|
| F_C−F_BETA, primary |7,665|0.242993|0.086814|
| F_C−C, working method |8,936|0.500000|0.232173|
| F_BETA−C, context |8,937|0.500000|0.232173|

The two0.5 values describe differences **between** allowed allocations, not a larger cap for either allocation. No coefficients outside prefix support change. New scores change540 of55,744 within-pool pairwise order relations (429 strict reversals) and65 of8,979 centered signs. Thus this is not merely a common score rescaling. Coefficients, BF16 endpoints and optimizer states are not identical.

All three predictions below use **C-referenced scores**; the F_BETA−C prediction therefore differs from Point1's beta-referenced prediction even though its actual native endpoints reproduce Point1.

| Contrast | C-score prediction | Actual preclip-gradient contraction | Native window assessment benefit | Prefix change | Useful-repair change | Damaged-prefix change |
|---|---:|---:|---:|---:|---:|---:|
| F_C−F_BETA |+0.000207521|+0.000165736|+0.002019971|+1|+3|+1|
| F_C−C |+0.262155988|+0.262049141|+0.082713966|+22|+9|−1|
| F_BETA−C |+0.261948466|+0.261883405|+0.080693995|+21|+6|−2|

Positive assessment benefit means lower assessment; negative damaged-prefix change is favorable. Primary score contributions are+0.001188783 and−0.000981262:90.437% cancellation relative to summed absolute contributions. Its actual gradient contraction also remains positive, but the surrogate does not accurately calibrate the realized assessment difference. Every represented operand is promoted to FP32 before subtraction, with FP64 diagnostic products/reductions; exact-zero cases have no arbitrary floor. [All contrasts](CONTRASTS.csv), [allocation movement](ALLOCATION_CONTRASTS.json), [score ordering](SCORING_SUMMARY.json).

## Native agreement and disjoint TRAIN evidence

| Endpoint | Window R (FP64-reduced) | Matching prefix | Useful repairs /371 | Damaged positions /8,979 | Reachable followers /295 | Fixed-K shortfall | Disjoint R (FP64-reduced) | Disjoint prefix |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Parent |5.937279|8,979|0|0|0|666|6.844945|488|
| C |5.367780|9,071|57|2|38|583|6.919527|483|
| F_BETA |5.287086|9,092|63|0|40|563|6.921646|483|
| F_C |5.285066|9,093|66|1|47|564|6.892372|483|

Raw and useful repair totals coincide here, but are separately computed: useful repair requires survival of all original preceding matches. F_C versus F_BETA has7 block prefix gains,4 losses and770 ties; response changes are−14/+15. Useful-repair changes are5 gained,2 lost; one additional damaged prefix position/block. Surviving original usable followers are291 versus292. Fixed-K shortfall worsens by1 despite the net+1 prefix occurrence, because the fixed boundary and full matching-prefix metrics count different progress. F_C versus C improves both update responses (+8/+14), with12 block gains,2 losses and767 ties. No aggregate alone establishes universal preservation.

The original eight disjoint, repeatedly inspected TRAIN responses each contribute eight anchors:64 blocks,44 original first errors,488 old-prefix occurrences,60 usable followers and592 assessed occurrences. Every C/F_BETA/F_C discrete pair ties on all64 blocks. Each endpoint has483 matching-prefix occurrences, zero repairs and one damaged old-prefix position/block. All lose five prefix occurrences versus the parent.

F_C's disjoint assessment benefit is+0.029274653 versus F_BETA and+0.027155384 versus C; four of eight response assessments improve versus F_BETA and four worsen. Its assessment6.892372 remains worse than parent6.844945. Historical F_BETA−C disjoint assessment benefit is−0.002119269. These scalar differences are not demonstrated useful drafting. The tiny window-only gain with disjoint discrete ties meets the handoff's **inconclusive-transfer / park** branch, not its promotion branch. The small witness set may miss a one-step effect; ties do not prove universal non-transfer or equivalence.

Assessment uses fixed original K and triangular weights through K, native FP32 margin/softplus terms, per-response nonempty-block normalization and equal response averaging. Strongest wrong rivals are recomputed at endpoints. Original FP32 and matched-contribution FP64 values are both retained; original FP32-reduced window benefits are0.002019405 /0.082713604 /0.080694199 in table order. [Per-response data](ENDPOINT_READOUTS.csv), [per-block counters](ENDPOINT_BLOCKS.csv), [paired comparisons](READOUT_COMPARISONS.json), [denominators](DENOMINATORS.json).

## Cost, audit and stopping point

| Work/resource | Actual |
|---|---:|
| Attempts / retries |1 /0|
| Native Adam calls / response presentations |3 /6|
| Native token-gradient oracle backwards |28|
| Disposable startup |2 forwards/backwards;0 Adam;20.206s|
| Endpoint readout forwards |40:8 update-window +32 disjoint|
| Native diagnostic wall time |271.566s|
| Supervised phase / occupied GPU time |280.523s /280.073s (4m40s)|
| Historical +new occupied GPU time |5,510.858+280.073=5,790.931s|
| Post-startup Torch peak allocated /reserved |37.657 /48.499GB|
| Sampled device /child peak RSS |49.532 /25.741GB|
| Minimum sampled available host RAM |83.990GB|
| Charged source/raw/cache/local allowance |0.890GB, below2GB|
| Postflight A800 free space |407.182GB, above400GB|

The sealed limits were60minutes elapsed/one occupied card-hour, at most six Adam attempts and30 oracle backwards per attempt/60 total across one identified implementation retry,2GB allocated disk and400GB free reserve. Actual work used three Adam calls,28 oracle backwards and no retry. The estimate was10–15minutes after preparation; actual bounded GPU work finished sooner. Startup, head chunks, serial backward work and recomputation are separate in [WORK](WORK.json) and [resources](RESOURCES.json). Torch peaks reset after startup; device/RSS samples are30seconds apart, so exact whole-process peaks are not claimed. CPU implementation/publication was not continuously timed. No complete response generation, new checkpoint/recovery/export, full-vector file, inherited deletion or unrelated-job termination occurred. Both A800s are idle after completion.

Independent CPU audit passes **858,200 assertions**, verifying source/raw/input identities, all three native-derived contrasts, coefficient constraints, oracle gates, repeated scores, ledgers and retained control/restoration receipts. It reconstructs scalar/occurrence arithmetic, **not unsaved full-vector contractions**. All11 interpretation checks are covered. Private raw delivery contains61 files /51,985,836bytes; source and protected input authentication passes. The first audit attempt had a Git-archive working-directory error; the corrected CPU auditor and original error are preserved. No experiment was rerun. See [independent audit](INDEPENDENT_POINT2A_AUDIT.md), [postflight](POSTFLIGHT_AUDIT.json), [operational notes](OPERATIONS.json).

**Decision: do not adopt F_C or start a larger experiment from this result.** Keep the working recipe and park this reference change. Point1's local effect survives; magnitude calibration remains unresolved. Point2A changes priorities and endpoint weights but supplies no discrete disjoint support. Point2B (intervention geometry/size) and Point3 (assessment/verification progress) stay provisional. The evidence supports discussing whether assessment targets behavior that transfers, rather than assuming another numerical or cap adjustment will solve the problem. The paper's verification-role narrative is unchanged; this variant is neither a novelty claim nor evidence of speedup.
