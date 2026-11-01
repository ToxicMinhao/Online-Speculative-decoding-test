# O_HALF matched-LR result: separate record and interpretation review

**The recorded experiment and reported decision pass this review.** The prepared
auditor passed on its first real execution without code changes: **643,740
checks**, all **1,072** new raw executions and **67,872** complete native blocks.
All paired counts, bootstrap intervals, gain/harm components, eight-job work
receipts and new cost sums reproduce. Execution commit:
`19ee22faf985e0576d287eaa8c8624b5a44b46f5`.

This is a separately implemented calculation and record review. The reviewer also
wrote production statistics/report code. It is **not an independent investigator
review, GPU reproduction or independent decoding of model-sized state tensors**.
Native checks read retained evidence; source/state and retirement audits are
complementary. The original archive was previously fully audited; this pass
cross-checks its sealed summary, canonical outputs and 256 paired rows and then
reconstructs every new execution. No model work was performed.

## What the result supports

| Event-4 comparison | Calls, candidate / comparator | Reduction | Paired nominal 95% interval |
|---|---:|---:|---:|
| K_HALF / O_HALF, primary |15,865 /16,042 |+1.1034% |[+0.4548,+1.8059]% |
| K / O, original LR |16,303 /16,379 |+0.4640% |[-0.2227,+1.1112]% |
| K_HALF / K |15,865 /16,303 |+2.6866% |[+1.9233,+3.5338]% |
| O_HALF / O |16,042 /16,379 |+2.0575% |[+1.4279,+2.7140]% |
| K_HALF / F |15,865 /16,300 |+2.6687% |[+1.8411,+3.5566]% |
| O_HALF / F |16,042 /16,300 |+1.5828% |[+0.8581,+2.3211]% |

K_HALF saves **177 calls** versus O_HALF:336 gross saved and 159 added, with 125 wins,
82 regressions and 49 ties among 256 seed/request outcomes. Seed 17 saves 86 calls
(+1.0703%); seed 29 saves 91(+1.1365%). These consistent point directions, a combined
point above 0.5% and an interval above zero satisfy **Decision A exactly**.

The primary interval's lower bound is **0.4548%, below 0.5%**. Therefore the result
supports the declared practical point-estimate rule, not a claim that the true
benefit exceeds 0.5%. Practical equivalence is not supported. These are 128 joint
request identities with both fitted seeds, not 256 independent samples; all methods
produce identical 88,266 output tokens across the 256 rows. The 128 distinct canonical
outputs range 37–2042 tokens, all EOS, zero cap hits.

Lower LR improves both implemented objectives in this matrix. It does not fully
explain away K_HALF's advantage. Conversely, the common-F interaction is only
+0.6196%[-0.1758,+1.5232]%, so these observations do not establish that lower LR
specifically amplifies K's relative benefit. K/O at the original LR remains
uncertain. Selection and normalization remain bundled in the implemented K rule.

## Gain and harm are request-level accounting

Relative to F, O_HALF has 418 gross gain calls and 160 gross harm calls; K_HALF has
560 and 125. On the same seed/request identities, K_HALF retains 309 of O_HALF's 418
gain calls (73.9234%), loses 109, and gains 251 elsewhere. It prevents 85 harm calls
while adding 50. Thus **251−109+85−50=177** net saved calls. The additional gain
balance contributes 142 and the harm balance 35. This shows that the aggregate
advantage contains both new gains and reduced harm; it does not prove a
preservation or gradient-conflict mechanism, and the 109 lost gain calls remain
visible. Positive parts were computed within seed before pooling.

## Work, costs and reporting boundaries

The new eight-event jobs completed 1,024 Adam steps and 2,048 ordinary response
presentations/forward-backward equivalents. All 48 bridge plus 1,024 final executions
are valid; the bridge does not enter performance statistics. No scientific work
was dropped or repeated. The earlier deployment admission failure had zero model
work and is preserved separately.

New O_HALF training takes 2,842.439 s in total; its summed event wall time is
3,155.609 s. Loading/initialization/resume validation totals 311.388 s. The recorded
campaign wall is 5,157.132 s. These timers have distinct and partly overlapping
scopes and must not be added indiscriminately. Peak updater allocation is
20,652,867,584 bytes (19.234 GiB); maximum reservation is 27,856,470,016 bytes
(25.943 GiB). They are per-process peaks, not simultaneous whole-machine residency.

Historical O/K/K_HALF training times are unmatched context. This review makes no
fresh relative training-cost or training-inclusive speedup claim. New serving
is checkpoint-batched with two repeats; archived methods have historical three
repeats. **No comparative serving latency or throughput result is justified.**
Raw timings are operational accounting only.

## ARS interpretation scan:11/11 checked

**1. Simpson's paradox — No primary reversal detected.** Primary pooled +1.1034%; seeds +1.0703% and +1.1365%. The report also shows K/F's mixed seed signs (+0.1227%, -0.1595%) rather than hiding them behind its -0.0184% pooled value.

**2. Ecological fallacy — Boundary correctly stated.** Bootstrap uses 128 joint request identities carrying both fitted seeds/all methods;256 seed/request outcomes are descriptive, not 256 independent draws. No population training-seed inference.

**3. Berkson's selection bias — Selection limitation retained.** The reserved short-prompt CodeAlpaca panel is conditional data and is reused after favoring K_HALF. This is a generalizability limitation, not evidence that a particular Berkson artifact occurred. No general-domain claim is made.

**4. Collider bias — No outcome-conditioned exclusion detected.** All 1,072 planned executions are valid and retained; all 128 panel identities remain. No generated-length/performance filtering or complete-case rescue. Existing panel/eligibility, canonical and checkpoint identities are unchanged.

**5. Base-rate neglect — Denominators accounted for.** Primary 125 wins/82 regressions/49 ties over 256 seed/request outcomes; gross 336 calls saved/159 added. All methods output 88,266 tokens across those rows;128 distinct canonical requests all terminate by EOS, zero cap hits. Gain retention uses 418 comparator gross-gain calls, not request count.

**6. Regression to the mean — Prior selection caution disclosed.** The report explicitly calls this reused-panel development/ablation evidence, not fresh confirmation. Ordinary O_HALF is a meaningful matched-LR control, but the nominal interval does not adjust prior selection of K_HALF.

**7. Survivorship bias — No attrition detected.** Eight training jobs and 1,072 new evaluations completed with zero invalid/unknown/unattempted scientific work. The earlier source-admission failure occurred before model work and remains separate; no unfavorable request was replaced.

**8. Look-elsewhere effect — Multiplicity caution disclosed.** All 8 prespecified comparisons, reverse primary, common-F interaction and both seeds are present. Secondary intervals remain descriptive/unadjusted; no new tests or selective significant subset were introduced.

**9. Garden of forking paths — Fixed analysis preserved.** Prespecified endpoint 4, seeds 17/29, LR1.5e-4, panel,10,000 PCG64seed 20261102 joint draws, linear percentiles and A/B/C/D/E rules reproduce. Operational tuple/list/source-deployment repairs are disclosed, with no outcome-driven scientific tuning or extra steps.

**10. Correlation versus causation — Mechanism limits correctly stated.** The comparison supports the implemented K recipe at matched LR, including selection and normalization together. It does not identify gradient conflict, memorization, preservation, optimal LR or a closed-loop deployment mechanism. Gain/harm is accounting, not proof of causation.

**11. Reverse causality — No leakage/temporal reversal detected in reviewed records.** The immutable evaluation panel and original O-generated feedback/schedules were sealed independently of these endpoint outcomes. Evaluation stays outside training; future performance is not used to select the observed learning trajectory.

No blocking interpretation error was found in the retained report. The important
cautions remain explicit: prior K_HALF selection on this reused development
panel, two fitted trajectories, conditional nominal intervals and unadjusted
secondary comparisons. No new significance test or multiplicity adjustment was
introduced after inspecting outcomes.

## One recommendation

**Keep K_HALF as the provisional coding base; seek agreement on one bounded
portability or online-setting check of this fixed base.** This follows the
prespecified Decision A. It does not authorize a follow-on run, controller change,
LR tuning, claim of general algorithm superiority or project-stage transition.

Evidence: [record reconstruction](RECORD_AUDIT.json), [structured review](REVIEW.json),
[retained report](../compact_evidence_001/final_report/REPORT.md),
[paired counts](../compact_evidence_001/final_report/PAIRED_REQUESTS.csv),
[all new evaluations](../compact_evidence_001/final_report/ALL_NEW_EVALUATIONS.csv),
[prespecified audit plan](../POSTRUN_AUDIT_PLAN.md).
