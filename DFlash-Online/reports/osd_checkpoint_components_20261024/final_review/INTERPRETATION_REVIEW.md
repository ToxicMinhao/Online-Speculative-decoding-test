## Material Passport

October 24, 2026 (local; completion October 25 UTC). ARS experiment-agent,
validate mode. Verification status: **ANALYZED**. Independent interpretation of
the completed four-arm report; no model rerun, remote access, new training or
production edits by this reviewer. Native validity and raw arithmetic are
separately audited by the coordinator and arithmetic reviewer. Current Stage 1 /
project None and all unrelated pauses remain unchanged.

Sources: [approved proposal](../../osd_next_experiment_review_20261024/PROPOSAL.md),
[review disposition](../../osd_next_experiment_review_20261024/REVIEW_CHANGES.md),
[implementation report](../IMPLEMENTATION_REPORT.md),
[completed summary](raw/campaign_001/evaluation/report/SUMMARY.json),
[pooled results](raw/campaign_001/evaluation/report/POOLED.csv),
[interactions](raw/campaign_001/evaluation/report/INTERACTION.csv), and
[cross-tabs](raw/campaign_001/evaluation/report/CROSS_TABS.csv).

## Finding and strongest control

**The retained body changes preserve substantial benefit, while retained
conditioning changes alone preserve little. Conditioning is more useful when
combined with the learned body.** This answers the endpoint-substitution question;
it does not locate a unique source of learning or establish a restricted-training
method.

The strongest control is the full four-cell comparison on identical requests:
F=(released C, released B), O=(learned C, learned B), C-only=(learned C, released B),
B-only=(released C, learned B). It exposes both conditional substitutions instead
of interpreting a single hybrid against F. All 64 requests qualify in the supplied
common four-arm/all-repeat report, with 19,547 identical output tokens per arm;
calls use designated repeat 0. Three repetitions are repeated executions of these
64 requests, not 192 independent questions or three training seeds.

| Arm | Calls | Saved versus F | Call reduction | Gross saved / harm | Improved / tied / regressed | Fraction of O savings |
|---|---:|---:|---:|---:|---:|---:|
| F | 3,140 | 0 | 0% | 0 / 0 | 0 / 64 / 0 | — |
| O | 2,795 | 345 | 10.99% | 359 / 14 | 56 / 3 / 5 | 100% |
| C-only | 3,108 | 32 | 1.02% | 78 / 46 | 29 / 11 / 24 | 9.28% |
| B-only | 2,896 | 244 | 7.77% | 266 / 22 | 55 / 1 / 8 | 70.72% |

B-only retains improvement on 51 of the 56 requests improved by O; C-only does so
on 27, while regressing on 21 of those 56. Both hybrids have less gross saving
and more gross harm than O. Neither supplies the proposed favorable trade-off of
similar gains with less harm. These post-outcome signs are descriptive; they do
not provide a prospective request router or matched-token negative-flip diagnosis.

## Interaction and uncertainty

| Endpoint substitution | Calls saved | 95% paired request bootstrap interval |
|---|---:|---:|
| Learned C with released B | 32 | [-7, 72] |
| Learned C with learned B | 101 | [56.98, 145] |
| Learned B with released C | 244 | [185, 303] |
| Learned B with learned C | 313 | [253, 373] |
| Interaction: 345 - 32 - 244 | 69 | [17, 118] |

Joint savings exceed the sum of isolated savings by 69 calls on this panel.
The observed conditioning effect grows from 32 to 101 calls when the trained
body is present. This is evidence of complementary endpoint behavior on the
verification-call scale. It is compatible with co-adaptation during joint
training, but does not identify that optimization mechanism: representation
compatibility and nonlinear proposal/verification trajectories remain possible
explanations. The interaction cannot uniquely be assigned to C or B. In
particular, 70.72% retained savings is not proof that 70.72% of learning occurred
in body parameters, and C-only's weak isolated effect does not make C irrelevant.

The 10,000 resamples use paired requests. Their intervals describe empirical
request-composition sensitivity conditional on this single endpoint and reused
development panel. Fixed-panel call totals are exact observations. These are
exploratory, pointwise intervals, not simultaneous guarantees, training-seed
uncertainty or fresh-domain confirmation. C-only's interval includes zero;
its positive 32-call panel total should not be presented as broadly reliable
standalone benefit. The positive interaction interval is suggestive within this
scope, not a universal architectural law.

Neither hybrid meets the predeclared **80% point-estimate screen**. That screen
was descriptive, not a noninferiority test or automatic selection rule. B-only's
retained-ratio interval is approximately [59.42%, 82.02%], spanning 80%; it would
be wrong to turn its 70.72% point estimate into inferential proof that 80%
retention is impossible. C-only's corresponding interval is [-2.12%, 19.73%].
All 10,000 resamples have positive O savings of at least 2% of F calls, so the
ratio denominator is stable in this analysis; denominator instability does not
explain the poor C-only retention.

## Timing and cost limitation

Recorded mean request totals are F 55.041 s, O 48.811 s, C-only 54.271 s and
B-only 50.987 s: observed reductions of 11.32%, 1.40% and 7.36%. **The coordinator
identified monitoring transfers during the active campaign, contrary to the
proposal's no-transfer timing rule.** These latencies are potentially affected
by concurrent I/O and are not strict isolated-timing confirmation. The magnitude
or direction of interference is not established here; do not silently correct
the values, drop requests or invalidate exact call/output evidence. The
bootstrap cannot repair systematic timing interference. Preserve the coordinator's
exact transfer/timestamp audit with the final report; no rerun follows automatically.

Request time includes prefill, drafting, verification and request cache maintenance;
it excludes publication and historical adaptation. Every hybrid inherits the
same jointly trained endpoint, with zero new training in this diagnostic.
C has 83.89 million parameters (about 8%); B has 964.74 million (about 92%).
These fractions do not determine backward compute, activation memory, publication
cost or total online benefit. No cheap-training or adaptation-inclusive speedup
claim is supported.

## Next decision, proposal only

Keep competent joint pure-FKL OSD as the working reference. This result weakens
the simple hypothesis that retaining its conditioning changes alone captures most
of its benefit; it does not reject independently trained conditioning-only models.
B-only is a viable inherited endpoint, but retains less benefit than O and keeps
about 92% of the draft parameters in its learned group. There is no automatic
compact-update conclusion or reason here to replace the objective.

If a further training question is selected, the most directly supported candidate
is whether body-only training from the same released initializer can obtain useful
benefit with conditioning frozen. It must be compared with competent full
training under matched feedback, anchors, normalization and exposure, with actual
costs measured. Its value should be a stated learning or cost question, since
freezing the other 8% is not inherently a useful saving. This is a candidate for
discussion, not a launch recommendation, selected algorithm or guarantee of success.
No result here proves body-only training works, full training is necessary, every
body weight matters, or a finer partition/new loss will succeed. Reopen these
questions only with a concrete restricted-training or compatibility hypothesis
and its own reviewed matched control. The experiment also does not establish
when online learning first helps or whether serving-time savings repay updates.

## Statistical fallacy scan: 11/11 checked

| ARS pattern | Applicability and disposition |
|---|---|
| 1. Simpson's paradox | Same paired cohort and output denominator across arms. No subgroup-wide reversal established; heterogeneous signs are retained. Pooled benefit is not universal per-request benefit. |
| 2. Ecological fallacy | Avoided by reporting request signs and gross harm; aggregate savings do not imply every request or token improves. |
| 3. Berkson's paradox | Panel selection is specified independently of gains; no validity exclusions. Prior development exposure still limits external confirmation. |
| 4. Collider bias | No post-outcome filtering in primary contrasts. Cross-tab signs must not become selected evaluation subsets. |
| 5. Base-rate neglect | Diagnostic sensitivity/specificity is not estimated. Explicit denominators are 64 requests, F calls for reduction, and O savings for retention. |
| 6. Regression to the mean | Requests were not selected as extreme improvers; paired F/O/hybrids control the panel comparison. Endpoint success motivated the test, so this remains conditional exploration. |
| 7. Survivorship bias | All 64 declared requests and 768 executions appear in the supplied complete report; no survivor-only efficacy cohort. |
| 8. Look-elsewhere effect | All four arms and predeclared contrasts retained. Multiple pointwise intervals cannot support simultaneous or confirmatory claims. |
| 9. Garden of forking paths | Fixed partition, cohort, repeat and analyses limit discretion. Preserve the timing deviation; do not invent exclusions, favorable repeats or new thresholds after outcomes. |
| 10. Correlation versus causation | Controlled endpoint substitutions support narrow effects on calls in this setting. They do not prove learning localization, training necessity or cheap trainability; timing has concurrent-I/O uncertainty. |
| 11. Reverse causality | Checkpoint substitutions precede generation; reverse direction is not an issue for this intervention. Causal claims about how joint training produced the endpoint remain untested. |

No additional scientific experiment is required to report the completed bounded
diagnostic honestly. Any follow-up or stage transition remains a separate decision.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/osd_checkpoint_components_20261024/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
