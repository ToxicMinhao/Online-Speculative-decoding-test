## Material Passport

- Origin Skill: ARS academic-research-suite, experiment-agent validate.
- Verification Status: ANALYZED; local compact-evidence audit, no model rerun.
- Scope: fixed O/K checkpoints, 64 inputs new to the current study, all 64 previously executed in older unrelated project work.
- Review provenance: separate evaluator audit; this analyst also prepared the panel and earlier training code.

Audit status: **passed**. Issues: **0**.


Reconstructed 18,966 native blocks across 384 returned generations. Exact inputs, first-O canonical outputs, termination, within-arm repeat counts, alternating order and included 48-generation correctness gate were checked.

| Arm | Calls once/request | Sum of three-repeat mean times (s) | Accepted draft tokens/call | Committed progress/call |
|---|---:|---:|---:|---:|
| O | 3191 | 56.101814 | 5.441241 | 6.421498 |
| K | 3131 | 55.324202 | 5.563079 | 6.544554 |

K/O call reduction: +1.8803%; request-time reduction: +1.3861%. Requests improve/regress/tie in calls: 31/20/13. Gross calls saved/added: 93/33; net saved: 60.

Per-repeat request-time reductions: +2.9045%, +0.7005%, +0.5292%.

Counts use one repeat-0 trajectory per request; time uses the mean of three repeated measurements. Ratios are pooled sums. No bootstrap, p-value, population-generalization or training-seed robustness claim is made. This reuses fixed one-seed checkpoints; the new panel is independent of current-study inputs, not globally unseen project data.

Canonical output lengths range from 188 to 901 tokens (mean 321.172; total 20,555 once across 64 requests). Termination counts: {'eos': 64}; 0 requests hit the 2,048-token cap.

O/K accepted draft-token totals are 17,363/17,418; both commit 20,491 tokens beyond the 64 initial target-generated seeds. Accepted counts are clipped at actual stop boundaries; the full 0–15 accepted-length and 0–16 committed-progress histograms are retained in FINAL_AUDIT.json.

**Interpretation:** these fixed K weights require modestly fewer verification calls than O on this prespecified current-study-new panel. Every timing repeat has the same favorable aggregate direction, but its magnitude varies substantially; the three repeats do not provide independent inputs or training seeds. This supports a bounded checkpoint-level serving benefit on this panel. It does not establish a generally reliable algorithmic gain, isolate selection from selected-weight renormalization, or demonstrate adaptation-inclusive acceleration. Earlier development results remain a separate experiment and are not pooled here. The historical O full-training-startup numerical-equivalence limitation is unchanged.

Integrity: 168 sealed files covered by passed remote pre/postflight checks; 142 locally available source/evidence files independently rehashed. Checkpoint tensors are supported by remote immutability checks and native loaded-tensor equality, not a local tensor rerun. Runtime preflight and postflight exactly match all pinned version fields and seven native runtime source hashes.

No training, optimizer update, feedback collection or extra generation was recorded. Activation, weight checks, backend loading and full campaign wall time remain separate from available-checkpoint request latency.

### Statistical fallacy scan

**11/11 checked.** Overall interpretive confidence: CAUTION.

| Check | Assessment |
|---|---|
| Simpson's paradox | Repeat-level directions are reported explicitly; no pooled-workload generalization. |
| Ecological fallacy | Request improvements and regressions are reported; aggregate benefit is not assigned to every request. |
| Berkson's paradox | Current-study exclusions and historical project exposure restrict the sample; no specific induced-correlation claim. |
| Collider bias | All 64 completed paired requests are retained; no output-quality or performance filtering. |
| Base-rate neglect | Denominators are 64 requests, 384 timed generations and one trajectory/request for calls. |
| Regression to the mean | Fixed selected checkpoints are tested on new current-study inputs against O; one seed and earlier method screening still limit inference. |
| Survivorship bias | Attempted/returned/invalid/unknown/unattempted counts and all raw files are checked; failures cannot be silently dropped. |
| Look-elsewhere effect | One prespecified K/O comparison; earlier development screening and historical data exposure remain disclosed. |
| Forking paths | Seed, exclusions, order, checkpoint bytes and metric definitions were sealed before generation; no post-result reselection or metric change. |
| Correlation versus causation | Controlled fixed-checkpoint serving comparison supports a bounded observed difference, not a universal training-mechanism claim. |
| Reverse causality | Selection and checkpoint sealing precede this evaluation; previous unrelated-project exposure prevents a claim of globally untouched data. |

Stage 1/project None remains unchanged. The audit neither launches a follow-up nor authorizes retuning.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_first_error_fresh_confirmation_20261027/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
