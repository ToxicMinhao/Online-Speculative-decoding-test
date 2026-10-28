# Startup limitation: evidence review

## Material Passport

- Date: October 26, 2026, America/Edmonton.
- ARS experiment-agent, validation mode. Status: **ANALYZED from retained evidence**, not a new training reproduction.
- Scope: archived source, original failure, numerical diagnostic, successful controlled qualification and completed U/P campaign receipts. Two delegated internal reviewers inspected the failure evidence and comparator startup separately.
- No training, GPU execution, generation, loss change, tolerance change or remote operation was performed. Original failure records and raw measurements are unchanged.

## Supported conclusion

**Keep retained O as the ordinary-OSD performance comparator. No implementation or scientific-settings defect was found that invalidates it, and this review does not justify retraining it.** The narrower question—whether the startup difference had absolutely no effect on its final weights or on the six-call P/O difference—remains unresolved. For that stronger causal claim, the reviewer's third category applies: evidence is insufficient. This is not evidence that a specific mismatch biased the comparison.

The previous caveat focused too broadly on the absence of a complete 128-step equivalence test. The actual residual difference is **first-use training versus a discarded process warm-up plus explicit full-RNG restoration**. A full-trajectory equivalence test is not a general requirement for a valid independently trained baseline. The completed result remains a valid descriptive finding: the tested P checkpoint provided no useful serving advantage over retained ordinary OSD on this panel. It does not prove every possible implementation of P must fail.

## 1. What failed, and what the later evidence established

The original qualification compared ordinary O and ranking-disabled U after two response presentations and one optimizer step. The first failed exact comparison was `draft.layers.0.self_attn.q_proj.weight`. Logged losses, accuracy, exposure, anchors and scalar gradient norm matched. The norm was 0.85546875. **It was a post-update weight-equality failure, not an output-token, loss-value or serving-speed failure.** The comparator stopped before checking the rest of Adam/RNG state. [Original traceback](/home/minhao2/EAGLE_SPEC/research/dflash_prefix_ranking_one_event_20261025/raw/campaign_001/STATUS.json:9), [presentation measurements](/home/minhao2/EAGLE_SPEC/research/dflash_prefix_ranking_one_event_20261025/QUALIFICATION_PRESENTATIONS.csv:1).

The original failing U0 tensor state was not serialized before the assertion. Its discrepancy magnitude is therefore unavailable. The retained O state, traceback, source and metrics remain intact; later numbers must not be described as recovered measurements of that first failure. [Original save/compare order](/home/minhao2/EAGLE_SPEC/research/dflash_prefix_ranking_one_event_20261025/worktree/experiments/prefix_ranking/qualify.py:377).

A subsequent diagnostic ran two ordinary OSD windows, O1 then O2, and ranking-disabled U0, each from fresh released weights/optimizer. **O1 differed from O2 as well as U0; O2 and U0 matched exactly.** The discrepancy therefore also occurred without selecting the new ranking implementation.

| Quantity in that later diagnostic | O1 versus O2; also O1 versus U0 |
|---|---:|
| Initial draft weights | Exactly equal |
| Relative gradient L2 difference after first backward | 0.257033% |
| Relative gradient L2 difference before optimizer | 0.250352% |
| Differing updated weight entries | 911,043 / 1,048,626,432 (0.086880%) |
| Maximum absolute updated-weight difference | 0.0009765625 |
| Relative updated-weight L2 difference | 0.004455% |

[Retained numerical summary](/home/minhao2/EAGLE_SPEC/research/dflash_prefix_ranking_gpu_diagnostic_20261026/NUMERICAL_SUMMARY.json:14), [raw pairwise comparison](/home/minhao2/EAGLE_SPEC/research/dflash_prefix_ranking_gpu_diagnostic_20261026/raw/main/trials/O1_vs_O2.json:1). The first retained tensor difference precedes AdamW; an optimizer-only explanation cannot explain that gradient difference. The small weight fraction/norm alone does not prove zero effect on near-tied draft decisions.

O2/U0 matched at initialization, both backward boundaries, pre-optimizer gradients and final weights/Adam/RNG. All three one-step checkpoints also produced identical outputs, accepted-token counts and verification calls on four saved requests: **39, 59, 26 and 64**. Thus no drafting effect was detected in that bounded check. These were diagnostic one-step checkpoints, not the retained 128-step O checkpoint. [Exact comparisons](/home/minhao2/EAGLE_SPEC/research/dflash_prefix_ranking_gpu_diagnostic_20261026/raw/main/trials/O2_vs_U0.json:1), [four-request records](/home/minhao2/EAGLE_SPEC/research/dflash_prefix_ranking_gpu_diagnostic_20261026/PER_REQUEST.csv:1).

### Cause: confirmed observations versus hypotheses

- **Confirmed:** the discrepancy was associated with the first versus later model/runtime constructions in the diagnostic process, and was not exclusive to U0. Initial weights, Adam and Torch RNG matched; Python RNG differed between O1 and O2, while actual sampling matched.
- **Confirmed:** with the same model and restored complete RNG, three repeated first-response forwards/backwards produced exactly matching gradients. That does not support a blanket claim of random variation on every backward. [Control result](/home/minhao2/EAGLE_SPEC/research/dflash_prefix_ranking_gpu_diagnostic_20261026/raw/followup/backward_control/RESULT.json:5263).
- **Not established:** the exact first differing operation. Forward activations/logits were not retained, so the gradient observation cannot separate forward numerics from backward/accumulation numerics. Python RNG consumption, compiler/kernel initialization and reducer setup remain possibilities, not identified causes. In particular, the evidence does not prove inherently nondeterministic FlexAttention backward.
- **Excluded as the active arithmetic explanation:** lambda zero bypasses ranking/reference computation entirely; it does not compute a ranking gradient and add zero. The ordinary loss path remains in use. [Branch and reference creation](/home/minhao2/EAGLE_SPEC/research/dflash_prefix_ranking_one_event_20261025/worktree/online_feedback/objective.py:94), [ordinary forward branch](/home/minhao2/EAGLE_SPEC/research/dflash_prefix_ranking_one_event_20261025/worktree/online_feedback/objective.py:254).

### What changed before qualification passed

The procedure added one fixed ordinary warm-up (two presentations/one temporary step), discarded its learned state, constructed fresh trainers and restored complete Python/NumPy/Torch CPU/CUDA RNG **after construction**. The pin was the retained diagnostic **O2 initial state**, not historical O's initial snapshot. Only RNG was restored from it; freshly loaded weights, optimizer, modes and counters were compared against it, not replaced by warmed weights. [Executed pin](/home/minhao2/EAGLE_SPEC/research/dflash_prefix_ranking_controlled_qualification_20261026/run_check.sh:29), [fresh-state gate](/home/minhao2/EAGLE_SPEC/research/dflash_prefix_ranking_controlled_qualification_20261026/runner.py:172).

Ordinary Oa/Ob and disabled U0/P0 then matched exactly in all checked states and gradient boundaries. All 27 exact receipts passed, followed by the active U1/P1 mask, gradient, reference/RNG and memory checks. No numerical tolerance or learning setting was loosened. Warm-up and full-RNG restoration were introduced together, so their separate contributions were not isolated. This resolved the **controlled qualification procedure**, not the exact internal cause of the older failure. [Passing U0 final-state receipt](/home/minhao2/EAGLE_SPEC/research/dflash_prefix_ranking_controlled_qualification_20261026/raw/qualification/U0/final_vs_Oa.json:1), [qualification report](/home/minhao2/EAGLE_SPEC/research/dflash_prefix_ranking_controlled_qualification_20261026/REPORT.md:12).

## 2. Retained O versus U/P startup

| Component | Retained O's original event | Qualified U/P event | Assessment |
|---|---|---|---|
| Draft weights | Released checkpoint, no resume | Same released checkpoint, fresh construction; exact initial gate | No identified learned-weight mismatch |
| Optimizer | Fresh DirectBF16AdamW; empty moments; no master weights | Identical optimizer code/options; empty moments checked | No optimizer-state contamination |
| Seed/RNG | `set_seed(17)` before construction; no explicit post-construction RNG pin | Same seed, then complete RNG restored from diagnostic O2 before sampling | Real procedural difference; original full initial RNG was not archived |
| Process warm-up | No training warm-up before its event | One discarded ordinary step; fresh constructors afterward | Runtime/cache state can persist; learned state does not |
| Training mode | Trainable model ends construction in `train()` | Same; independent reference is frozen/eval | No train/eval mode defect found |
| Reference work | None | Active U/P reference reads preserve RNG/current modes | Intended algorithm work; disabled arms create no reference |
| Feedback preparation | Generated/captured before training | Retained identical features reused | No identified extra training or parent RNG perturbation |

Original startup and zero counters are recorded in [O runner](/home/minhao2/EAGLE_SPEC/research/dflash_osd_one_event_20261025/worktree/experiments/osd_one_event/run.py:198) and [O initial status](/home/minhao2/EAGLE_SPEC/research/dflash_osd_one_event_20261025/final_review/raw/campaign_001/INITIAL_TRAINER.json:1). The original/current optimizer implementations are byte-identical, with direct BF16 parameters and no master copy: [optimizer](/home/minhao2/EAGLE_SPEC/research/dflash_osd_one_event_20261025/worktree/online_feedback/optimizer.py:6).

Both trainers seed before building the model and call `train()` afterward: [original trainer](/home/minhao2/EAGLE_SPEC/research/dflash_osd_one_event_20261025/worktree/online_feedback/training.py:204), [current trainer](/home/minhao2/EAGLE_SPEC/research/dflash_prefix_ranking_one_event_20261025/worktree/online_feedback/training.py:227). The original native worker was created before trainer seeding, and serving calls preserve parent RNG: [generation guard](/home/minhao2/EAGLE_SPEC/research/dflash_osd_one_event_20261025/worktree/online_feedback/sglang_backend.py:122). Source inspection found no random draw in parent feedback preparation that explains a learning-schedule difference. Full actual schedules already match.

The warm-up trainer/model/observers/bound methods are closed and released before fresh training; the initial gates check empty Adam, no gradients, zero counters, modes and weights. Both U/P main receipts passed. This rules out giving U/P an extra learned step. It does not erase process-global compiled runtime state, which is the purpose of the warm-up. [Cleanup](/home/minhao2/EAGLE_SPEC/research/dflash_prefix_ranking_one_event_20261025/worktree/experiments/prefix_ranking/controlled_startup.py:199), [main U initial equality](/home/minhao2/EAGLE_SPEC/research/dflash_prefix_ranking_execution_20261026/raw/campaign_001/U/initialization/initial_vs_pinned.json:1), [reference isolation](/home/minhao2/EAGLE_SPEC/research/dflash_prefix_ranking_one_event_20261025/worktree/online_feedback/objective.py:53).

## 3. Is another check needed?

**No additional check is required to retain the present descriptive result or to keep O as an ordinary-OSD baseline.** No O retraining follows from the absence of a full 128-step equivalence test. The comparison already uses valid immutable checkpoints and fresh common-input inference measurements; the substantive result is no useful P advantage.

If the separate goal is to isolate startup numerics, the smallest missing diagnostic is **one ordinary cold-process window with the exact O2 full RNG restored after fresh construction**, compared to the existing warmed Oa gradients and final state. It would use the original two presentations/anchors and one disposable optimizer step, with no new feedback, generation or active ranking. This adds the missing cold-with-identical-full-RNG condition: existing O1/O2 also differed in Python RNG, so merely repeating them would not isolate this question.

That check could test whether the cold/warm difference persists after controlling full RNG. It would not, by itself, determine the impact on a 128-step endpoint or explain six verification calls. It is a proposed discriminator only and **was not run**. Escalation to a full matched ordinary trajectory is neither necessary for reporting this null screen nor authorized here.

## Reporting and integrity

The report caveat should say the baseline remains valid for the measured performance comparison, identify the actual startup procedure difference, and avoid attributing the tiny P/O gap entirely to the ranking term. It should not suggest missing 128-step bitwise equality invalidates O, or claim the numerical cause is solved.

All 11 ARS fallacy categories were considered. This is a code/numerical evidence audit, not a new statistical significance analysis: no p-values, confidence intervals, population estimates or subgroup inference are asserted. Relevant safeguards are preserved: no causal attribution from process order alone; no extrapolation from four requests; no coordinate equality inferred from rounded scalar metrics; no hidden failed attempts; and no reinterpretation of a null practical result as a positive mechanism result. Raw failure records, historical reports and all measurements are retained.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_prefix_ranking_execution_20261026/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
