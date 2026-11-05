# Separate final O/K events 5–8 analysis review

## Material Passport

- Origin skill/mode: ARS experiment-agent / validate.
- Origin date: 2026-10-29.
- Verification status: **ANALYZED**.
- Version label: final_separate_analysis_review_v1.
- Verdict: **passed with stated limits; no blocking finding**.

Scientific execution commit: `04fac901f7af5478b62ff730833c026637ddd2cd`. Operational recovery and analysis commit: `1cfb44fa3989e5ed2c5e13805007bb20f22189f8`.

This is a separate same-team, read-only analysis by a reviewer who did not author the experimental or report code. It shares team context and is neither blinded/external review nor independent GPU replication. No models, training, remote commands, new experiments, tuning, or literature search were run; only these two review artifacts were written.

Reviewed compact report: `REPORT.md`, 10,373 bytes, SHA-256 `f897bb19708188b345186e8183df2571456dd689e1026a85246bc6fb0c4bc3e3`. This verdict binds that report version.

## Direct reconstruction

Reviewer-authored standard-library calculations read all five raw `MEASUREMENTS.jsonl` files, all 1,920 corresponding record JSON files, canonical references, and the sealed plan. I checked unique complete six-slot request coverage, raw/measurement call and timing agreement, exact canonical tokens/termination and sealed prompts, repeat-consistent call counts, and the counter sums over 89,541 recorded native trace blocks. No discrepancy was found. This checks saved evidence, not fresh execution.

| Event | Panel | O / K calls | K call reduction | Call wins / regressions / ties | Complete-request reduction | Native reduction |
|---:|---|---:|---:|---:|---:|---:|
| 6 | development | 2997 / 2904 | 3.1031% | 36 / 18 / 10 | 3.2116% | 3.0213% |
| 6 | additional | 3086 / 3062 | 0.7777% | 31 / 21 / 12 | 0.5287% | 0.7093% |
| 8 | development | 2914 / 2880 | 1.1668% | 31 / 21 / 12 | 0.8542% | 1.1284% |
| 8 | additional | 3079 / 3031 | 1.5589% | 33 / 25 / 6 | 1.4392% | 1.4972% |
| 8 | fresh | 2997 / 2897 | 3.3367% | 37 / 13 / 14 | 3.2227% | 3.2092% |

The primary formula is 100 × (sum O calls − sum K calls) / sum O calls, using one validated trajectory per request. It is not the mean of request percentages. Every panel has 64 requests and 384 executions, with 96 O→K and 96 K→O pairs. All 15 within-panel repeat comparisons favor K in both complete-request and native time. The JSON retains each repeat's derived percentage.

Retained event-4 per-request counts in `raw/sealed/PLAN.json` match the earlier campaign's local raw measurement files exactly. Development is 2,994 / 2,942 calls, 1.7368%, with 34 / 20 / 10 wins/regressions/ties; additional is 3,130 / 3,058, 2.3003%, with 33 / 18 / 13. Each historical request set matches its current panel.

The development advantage is 1.7368% → 3.1031% → 1.1668%; additional is 2.3003% → 0.7777% → 1.5589%. Thus K's advantage persists but shrinks overall from event 4 to 8 by 0.5700 and 0.7414 percentage points. Neither path is monotonic. Fresh event 8 favors K by 3.3367%. Panels are not pooled, and historical timing is not a latency learning curve.

## Work, exposure and cost checks

Four feedback directories contain 128 responses each. Each of eight arm-event results records 128 optimizer steps and 256 presentations, starting at step 512 and ending at 1,024 per arm. This reconciles to 512 feedback responses, 1,024 added optimizer steps, 2,048 added presentations and 1,920 evaluations. The first eight development requests at event 6 contribute 8 × 2 × 3 = 48 executions within that main evaluation budget. The sealed plan and final status record zero extra optimizer steps/generations; the inherited horizon is 7,424. Feedback publication receipts point to O4/O5/O6/O7 respectively.

I separately summed K's emitted metrics only at completed optimizer boundaries, where each record covers two accumulated responses:

| Event | Windows | Selected / available occurrences | Selected percentage |
|---:|---:|---:|---:|
| 5 | 128 | 539665 / 918693 | 58.7427% |
| 6 | 128 | 521002 / 871671 | 59.7705% |
| 7 | 128 | 555274 / 941612 | 58.9706% |
| 8 | 128 | 530696 / 886796 | 59.8442% |

These yield the report's 58.74%, 59.77%, 58.97% and 59.84%. The denominators are supervised position occurrences, not distinct positions, response counts, weights or sampled anchors. This reaggregation does not recompute logits; the separate exposure audit additionally reconstructs available occurrences from retained anchors.

Raw cost totals agree with the compact report: O/K training 1,207.241818 / 1,249.154936 s, saves 75.289061 / 75.981432 s, exports 61.057157 / 61.252166 s, and arm-event walls 1,355.421008 / 1,399.827807 s. The component timers nest within wall time. K's observed training time is 3.47% higher; this does not isolate kernel overhead. Peak training allocation is 25.317550 GiB. Shared collection/preparation/teacher/feature totals agree with the rounded report.

The two process segments sum to 8,995.587782 s (149.926463 min). Raw `STARTED.utc` and `RECOVERY_FINISHED.utc` span 06:53:41→09:37:18 UTC: 9,817 s (163.616667 min), including the stopped interval and wrapper work. The report correctly distinguishes these overlapping time scopes and serving latency.

## Interpretation and limits

The compact report's conclusion is supported by the retained measurements, with these limits preserved:

- One continued O/K trajectory; timing repeats are repeated measurements, not training-seed replication or evidence of statistical significance.
- The seed-20261029 panel is current-study fresh. The sealed selection record discloses older unrelated-project exposure for 63/64 requests and `global_freshness=false`; the report states this explicitly. The remaining request's lack of recorded exposure does not establish global novelty.
- Fresh canonical output comes from the first O8 execution. All six O/K outputs agree, under the qualified native verifier; this is not an additional independent target-only reference.
- The source/state/retention audit validates compact witnesses separately. I did not download or independently decode remote model/optimizer tensors.
- No integrated asynchronous or training-inclusive speedup, mechanism explanation, general task superiority, stage advance, or further execution is established. Training and operational costs remain separate from serving comparisons.

ARS fallacy coverage: **11/11 checked**. Simpson's paradox: panels/endpoints separate. Ecological fallacy: per-request regressions retained. Berkson's paradox: selected task/population and study-fresh scope limit generalization. Collider bias: no performance filtering. Base-rate neglect: diagnostic setting inapplicable; O-call denominator explicit. Regression to the mean: fixed matched endpoints and no general duration inference. Survivorship bias: every planned execution present, operational stop preserved. Look-elsewhere effect: all fixed endpoints/repeats included. Garden of forking paths: sealed scope and formula, no tuning. Correlation/causation: descriptive one-trajectory result, no deployment claim. Reverse causality: no causal mechanism assigned to curve shape.

No blocking correction is required to the bound compact report. The machine-readable companion preserves the independently derived values, input measurement hashes, limits and review provenance.
