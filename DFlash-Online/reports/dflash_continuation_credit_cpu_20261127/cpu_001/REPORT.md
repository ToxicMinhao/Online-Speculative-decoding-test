# Shared continuation credit and later-error severity

**Completed CPU diagnostic. Later-error severity adds predictive information, but improved repair allocation or serving is not established.** Shared credit distinguishes blocks after HIGH-pool centering, although its actual weight movement is small. The adjusted relationship between added credit and net prefix progress remains uncertain. Keep the deployed objective unchanged. If a further GPU experiment is approved, the strongest next question is **adaptive versus global-discount continuation credit**, with old TAIL retained as a baseline. Testing structural shared credit alone would leave the clearest new finding—severity information—untested.

This recommendation is a proposal, not an authorization or a method-selection conclusion. The running PR66 campaign and Graduate were not modified. No GPU follow-up was launched.

## What was measured

Execution source: `e76c5ec3b274737732f6b83e002178069c1145dc`, based on the specified PR65 publication. [INPUTS.json](INPUTS.json) records source/input identities and availability. We used every retained PR64 PAIR1 → native C+K+PAIR CONTROL observation: **128 training-buffer responses, 8,192 blocks**, comprising **2,114 HIGH, 4,215 LOW and 1,863 no-error** blocks. Optional PR65 TAIL observations cover the same complete cohort; they are descriptive, not an independent replication. The replay uses 64-anchor diagnostic pools per response, not the original up-to-512-anchor training pools.

Native predictions define correctness, including ties. Valid proposals include canonical EOS and exclude the seed, padding and post-EOS positions. Shared credit carries the minimum correct-token support across intervening errors and divides each correct follower's credit by the errors encountered. Wrong positions receive zero; the minimum never restarts at a new correct island. FP64 offline reductions consume saved FP32 supports. No full-vocabulary probabilities, model states or teacher features were loaded.

## Later opportunities exist, with retention losses

| Native-child outcome | HIGH | LOW | No-error | All blocks |
|---|---:|---:|---:|---:|
| Signed full-prefix progress | +10,239 | +10,721 | −435 | +20,525 |
| Progress beyond old cutoff K | 3,487 | 8,340 | 0 | 11,827 |
| Next barriers crossed / eligible barriers | 990/1,697 | 2,183/4,019 | 0/0 | 3,173/5,716 |
| Useful repairs / parent errors | 1,741/2,114 | 3,120/4,215 | 0/0 | 4,861/6,329 |
| Lost old-prefix occurrences | 290 | 739 | 435 | 1,464 |

A useful repair requires the old first error and all preceding matches to be correct in the child. Negative outcomes remain included; no-error blocks contribute retention losses but never repair-rate denominators. “Occurrences” counts overlapping sampled blocks, not distinct output tokens or verification calls.

Shared credit adds positive score to **819/2,114 HIGH blocks (38.74%)** and 1,704 LOW blocks. LOW weights remain unchanged by design. Among those 819 HIGH blocks, 514 cross the next barrier, 677 achieve useful repair, and 62 old-prefix occurrences are lost. The predictor's broader eligible set contains **2,234 later correct followers in 914 blocks across all 128 responses**; 991 followers are reached by the child. Its response/block/follower-weighted target prevalence is 42.60%. These observations establish that the newly considered continuation is sometimes reached, not that rewarding it causes repair.

The optional TAIL endpoint gains +20,505 full-prefix occurrences, versus +20,525 for native CONTROL, and loses 1,539 old-prefix occurrences versus 1,464. Its 30 additional beyond-K occurrences reduce the historical capped-outcome deficit from 50 to 20. HIGH outcomes improve while LOW/no-error retention worsens. This remains mixed endpoint evidence; redefining the outcome does not turn the previous pilot into a demonstrated success. [Outcomes](OUTCOMES.csv), [per-response outcomes](RESPONSE_OUTCOMES.csv).

## Added structural credit: uncertain adjusted association

The prespecified HIGH regression preserves response effects, joint geometry, immediate confidence, first-error difficulty and prefix fragility, then controls old score and later-error count before testing added credit. Each response has equal total weight, distributed by beta within its HIGH blocks.

Per **0.1 added score**, the estimated association with full signed progress is **+1.1099 positions**, with 95% whole-response bootstrap interval **[−0.3957, +2.4228]**. For beyond-K progress, the estimate is +1.0962 [−0.2194, +2.1966]. On the smaller observed scale, one adjusted-score SD corresponds to +0.1906 [−0.0680, +0.4161] positions. Both intervals cross zero.

All 2,000 fixed PCG64-seed-20261127 bootstrap draws are identified; none was replaced. The model retains 128 response effects and 1,822 residual degrees of freedom. Direct and accelerated calculations agree for the full fit and first five draws. This is a retrospective association under one historical update, not the effect of applying a proposed loss coefficient. [Association and all draws](ASSOCIATION.json).

## Severity improves prediction beyond structure

Four response-grouped folds keep each response's blocks/followers together. Training-fold standardization, equal response/block/follower weighting, natural class frequencies, fixed L2=0.001 and an unpenalized intercept are preserved. Probability clipping at 10⁻¹² is used only for log-loss evaluation. All fits complete; there are no unavailable OOF rows.

| Model | OOF log-loss | OOF Brier score |
|---|---:|---:|
| CONST | 0.683279 | 0.245074 |
| STRUCT | 0.594795 | 0.205817 |
| SEVERITY | **0.575367** | **0.198240** |

Adding later-error mean/max deficit improves log-loss over STRUCT by **0.019428**, 95% response-bootstrap CI **[0.002769, 0.034634]**, and Brier by **0.007577 [0.000147, 0.014541]**. Both improvements are positive in all four folds. STRUCT already includes error count/location, old score, carried confidence, original error difficulty and prefix fragility. Thus the result supports additional predictive information from later-error severity under this transition. It does not identify a causal correction mechanism.

The 2,000 bootstrap draws resample fixed OOF response losses without refitting; uncertainty is conditional on this checkpoint transition, folds and fitted models. It does not cover training-seed or event robustness. [OOF predictions](OOF_PREDICTIONS.csv), [model settings, coefficients and draws](PREDICTION.json), [fold metrics](PREDICTIVE_METRICS.csv).

## Actual allocation changes are small

Each response is centered independently. Every alternative changes all 128 pools and all 2,114 HIGH coefficients; all LOW coefficients and HIGH total masses remain unchanged. Maximum numerical HIGH-mass error is 7.11×10⁻¹⁵.

| Candidate versus old TAIL | Half-L1 coefficient movement | Fraction of native HIGH mass | Summed half-L1 effective movement |
|---|---:|---:|---:|
| SHARED | 1.735102 | 0.11701% | 0.006384 |
| OOF CONST-discounted | 0.740124 | 0.04991% | 0.002724 |
| OOF SEVERITY-adaptive | 1.026640 | 0.06923% | 0.003756 |

The HIGH-mass denominator is **1,482.849489**, summed across responses. SHARED movement is 0.04902% of all repair beta mass (3,539.579694). Effective coefficients divide each response's coefficients by its own selected prefix-plus-first-error weight sum plus 10⁻⁶. Pooling those effective changes does not create a training-event loss estimate.

Centering removes **19.85% of the beta-weighted squared added-score energy**; about 80.15% remains. This is an energy measure, not a percentage of credit mass. Therefore centering does not eliminate the new distinction, but the surviving coefficient change is modest. ADAPT versus CONST moves only 0.358838, or **0.02420% of native HIGH mass**. These allocations use only OOF probabilities and are hypothetical actions, never estimated policy values. [Allocation summary](ALLOCATION_SUMMARY.csv), [response pools](RESPONSE_ALLOCATION.csv), [coefficient records](ALLOCATION.csv.gz).

## Sensitivity, costs and decision

Gamma 5/7/10 gives mean response denominators 241.80/285.50/330.67. Relative to gamma7, old-TAIL repair coefficients change by half-L1 **274.34/281.06** at gamma5/10; effective changes are **0.82128/0.70147**. Eta0.25/0.75 changes component-only K+ coefficients by **143.25** and effective coefficients by **0.52561**, relative to eta0.5. These changes exceed SHARED's movement by roughly two orders of magnitude. Gamma can change total repair mass, so its half-L1 is a change magnitude, not purely conserved redistribution. No setting was selected or changed. These are coefficient sensitivities, not serving sensitivities. **C alpha/cap replay is deferred: canonical probabilities were not retained.** Margins cannot reconstruct them. [Sensitivity tables](SENSITIVITY_SUMMARY.csv).

The analysis took **38.411 seconds on A800's CPU**, peaking at **247.2 MiB RSS**, with one-thread numerical libraries, two-core affinity and GPUs hidden. Forty-eight focused CPU tests pass, plus a bounded independent synthetic boundary review. No model forwards, training steps or generations occurred. About12.4MB of new remote files leaves458.27GB free, above the400GB reserve. A missing optional thread-inspection package was removed from preparation dependencies; a wrong-directory local test invocation and unavailable remote rsync were corrected without repeating the scientific run. Existing libraries and the campaign were unchanged.

**Recommended next decision:** retain the current method operationally; if another controlled GPU test is worthwhile, include adaptive versus global discount rather than claiming structural shared credit already works. Fit any prospective predictor only from prior completed feedback, freeze it before evaluation, and retain a native/old-TAIL control. The small allocation contrast and mixed retention evidence require an explicit practical-benefit check. These CPU results justify that question; they do not answer it. Stop here.
