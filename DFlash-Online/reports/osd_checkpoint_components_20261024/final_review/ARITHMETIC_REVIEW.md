## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-10-25 UTC / October 24 Edmonton
- Verification Status: ANALYZED
- Version Label: component_arithmetic_v1

## Validation Report

Independent CPU recomputation from all 768 raw evaluation JSON records and the presealed 64-request panel reproduces the reported arithmetic. No missing measurement slots, historical F/O call mismatches, repeated-call or repeated-token-statistic mismatches, or summary/CSV numerical discrepancies were found. The source experiment was not rerun. Native publication/provenance and costs are audited separately by the parent task.

The raw checks cover exact saved prompt/output/stop/cap fields, F/O historical counts, verification trace counts and the sums of accepted-draft/committed-progress statistics. Report comparisons cover pooled metrics, conditional effects, interaction, cross-tabs, per-request records, all measurement CSV fields and the predeclared 10,000-sample paired bootstrap. The JSON sidecar contains hashes and complete recomputed values. Float tolerance is absolute 1e-10 / relative 1e-11.

## Pooled results

Every row uses the same 64 requests. Calls and token statistics use designated repeat 0. The identical canonical outputs contain 19,547 tokens; verification committed progress is 19,483 because the first output token of each request comes before the verification loop. All cap-hit counts are zero.

| Arm | Calls | Calls / 1,000 output tokens | Net saved vs F | Extra vs O | Gross gain / harm | Improved / tied / regressed | O gain retained |
|---|---:|---:|---:|---:|---:|---:|---:|
| F | 3,140 | 160.638 | 0 | 345 | 0 / 0 | 0 / 64 / 0 | 0.00% |
| O | 2,795 | 142.989 | 345 | 0 | 359 / 14 | 56 / 3 / 5 | 100.00% |
| C-only | 3,108 | 159.001 | 32 | 313 | 78 / 46 | 29 / 11 / 24 | 9.28% |
| B-only | 2,896 | 148.156 | 244 | 101 | 266 / 22 | 55 / 1 / 8 | 70.72% |

O/C-only/B-only call reductions are 10.987%, 1.019%, and 7.771%. Absolute net-savings bootstrap intervals are O [277, 411], C-only [-7, 72], and B-only [185, 303] calls. Both hybrids fall below the predeclared descriptive 80% point-estimate retention screen. B-only retains a substantial fraction (70.725%) without retaining the full benefit. Its empirical retention interval [59.420%, 82.019%] overlaps 80%; this is not a noninferiority test or evidence that it passes the point screen. C-only retains 9.275%, with interval [-2.115%, 19.733%].

The report label `retained_gain_status=positive_at_least_2pct` describes the full-O reference denominator, not an assertion that C-only itself improves by at least 2%. All 10,000 bootstrap reference denominators are positive and at least 2% of F calls; zero/negative/small-positive denominator counts are all zero. No resamples were removed.

## Conditional effects

| Addition | Calls saved | Pointwise 95% bootstrap interval |
|---|---:|---:|
| Learned conditioning on released body | 32 | [-7, 72] |
| Learned conditioning on learned body | 101 | [56.975, 145] |
| Learned body on released conditioning | 244 | [185, 303] |
| Learned body on learned conditioning | 313 | [253, 373] |
| Interaction | 69 | [17, 118] |

The accounting identity is 345 = 32 + 244 + 69. Interaction is positive on 37 requests, zero on 6, and negative on 21. The aggregate interaction concerns these two endpoint substitutions on this panel; it does not locate a universal training mechanism or establish that every body parameter is necessary.

## Descriptive cross-tabs

Rows are O improves/ties/regresses versus F. Columns are the hybrid improves/ties/regresses versus F. Each matrix totals 64.

| O status | C improves | C ties | C regresses | B improves | B ties | B regresses |
|---|---:|---:|---:|---:|---:|---:|
| Improves | 27 | 8 | 21 | 51 | 1 | 4 |
| Ties | 0 | 1 | 2 | 2 | 0 | 1 |
| Regresses | 2 | 2 | 1 | 2 | 0 | 3 |

B-only keeps positive call savings on 51 of the 56 requests where O improves. C-only does so on 27. These are post-outcome descriptions, not a validated request selector. Neither hybrid reduces total harm relative to O: O adds 14 calls on its regressions, compared with 46 for C-only and 22 for B-only.

## Timing and repeat stability

All four arms have identical per-request call counts, accepted-draft-token counts, and committed-progress counts across the three repeats. Timing repeats remain repeated measurements of 64 examples.

| Arm | Total seconds r0 / r1 / r2 | Mean total seconds | Pooled reduction vs F | Reduction by repeat |
|---|---:|---:|---:|---:|
| F | 55.096155 / 55.246886 / 54.779222 | 55.040754 | 0.000% | 0.000% / 0.000% / 0.000% |
| O | 48.974293 / 48.524871 / 48.932479 | 48.810547 | 11.319% | 11.111% / 12.167% / 10.673% |
| C-only | 53.740812 / 54.672218 / 54.400253 | 54.271094 | 1.398% | 2.460% / 1.040% / 0.692% |
| B-only | 51.577072 / 50.690000 / 50.694165 | 50.987079 | 7.365% | 6.387% / 8.248% / 7.457% |

The request timer includes cache maintenance. Its nested cache timer is reported in the JSON and is neither subtracted nor added a second time. Loading/publication lies outside this request timer.

**Timing qualification:** the parent audit reports that monitoring SCP/tar retrieval overlapped the running campaign despite the protocol prohibiting transfers during timing. These remain the observed measurements, with potential I/O interference; they are not a strict clean-timing estimate. Similar directions across repeats do not eliminate this confound. No records were excluded or rerun.

## Interpretation limits and fallacy scan

The bootstrap is exactly reproducible with NumPy PCG64 seed 20261024, 10,000 paired request resamples, common sample indices for every arm and all timing repeats, and linear percentile interpolation. It measures empirical request-composition sensitivity on a reused panel conditional on one trained endpoint. It is not uncertainty about the observed fixed-panel totals, a training-seed replication, fresh-domain confirmation, or a simultaneous guarantee. There are no p-value or power claims.

The endpoint substitutions do not measure the efficacy or cost of training only a subset of parameters; all inherited learned weights originate in full training. The experiment also does not establish net online acceleration or token-level negative flips.

Fallacy scan coverage: 11/11.
- Simpson paradox — checked: All repeat-specific pooled timing comparisons have the same direction as their aggregate; request heterogeneity is reported through gross gains/harm and cross-tabs, not hidden.
- Ecological fallacy — caution: Pooled effects do not imply every request benefits; O regresses on five requests and B-only on eight.
- Berkson selection — scope_limit: All 64 preselected requests retained; sample comes from reused evaluation pool, not arbitrary target population.
- Collider bias — not_detected: No observed-output subset selection or post-treatment covariate adjustment; all 64 are included.
- Base rate neglect — not_applicable: No diagnostic classification probabilities; sign-count denominator explicitly 64.
- Regression to mean — not_detected: Uniform presealed selection, no extremal-performance selection; exact frozen and O controls present.
- Survivorship bias — not_detected: All planned 768 records and all 64 requests included; no exclusions.
- Look-elsewhere effect — caution: All four arms and predefined interaction/cross-tabs retained; multiple intervals are pointwise descriptive, not simultaneous discovery evidence.
- Garden of forking paths — not_detected_in_arithmetic: Fixed repeat 0 calls, all repeats latency, fixed 10,000 resamples and seed20261024, no favorable-repeat replacement.
- Correlation versus causation — scope_limit: Interventions compare retained endpoint groups; findings do not identify a gradient mechanism or restricted-training success.
- Reverse causality — not_applicable: Checkpoint assignment precedes generation; no observational temporal-direction claim.

## Artifacts

- Raw evaluation records: `raw/campaign_001/evaluation/*_r[012]_*.json`
- Author reports: `raw/campaign_001/evaluation/report/`
- Independent numeric sidecar: `ARITHMETIC_REVIEW.json`

Overall confidence: the recorded arithmetic is consistent; scientific scope and timing qualifications above remain. Verification status is ANALYZED because this work replays existing artifacts and does not rerun the model experiment.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/osd_checkpoint_components_20261024/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
