# Prefix-preservation guard: compiler repair and GPU requalification

**The compiler failure is repaired, and the bounded three-trial GPU qualification passed.** No main G training or serving evaluation was run in this repair attempt.

## What failed and what changed

The earlier enabled-G trial requested a retained computation graph so H and K could be differentiated separately from the same forward. Its compiled backward had already been built with donated buffers, whose storage could be reused. PyTorch rejected this combination before G completed H's backward.

The repair pins `torch._functorch.config.donated_buffer=False` **before startup and all model compilation**, for ordinary K, guard-disabled G and guard-enabled G in a fresh process. The receipt confirms the setting was initially true, was changed to false before startup, and remained false throughout qualification. The new run successfully executed both retained H backwards, resolving the observed integration failure.

There was no change to the K/H definitions, BF16 arithmetic, gradients' separation, optimizer, clipping, learning rate, feedback, sampled anchors, RNG initialization or exact qualification tolerances. Additional edits only improve failed-trial status, memory and backward-attempt accounting. Four source/test files changed; previous source and original failure records remain preserved. [Diff](IMPLEMENTATION.diff), [compiler receipt](raw/execution_001/qualification/RUNTIME_COMPATIBILITY.json), [original failure report](../REPORT.md).

## Native checks

| Check | Result |
|---|---|
| Released weights, fresh optimizer/counters, pinned RNG and modes | All three trials pass |
| Startup restoration, two backward evaluations with no Adam step | Pass |
| Guard-disabled G versus ordinary K | Exact gradients, clipping, native weights, Adam state, RNG and counters |
| Enabled G before correction versus ordinary K | Same exact checks pass after both accumulated response backwards and native AdamW |
| Enabled H gradients remain separate from K/Adam | Both H backwards complete; K comparisons remain exact |
| Final Adam state, RNG and counters | Exact match |
| Final parameters finite | Pass |
| New qualification budget | Exactly three temporary steps and six presentations |

All 489 A800 regression CPU tests and 34 focused CPU tests passed; the latter overlap with the regression suite. The focused tests include actual CPU Inductor graph reuse after an ordinary backward, and active/inactive projection, BF16 and failure-accounting fixtures. Local independent review also passed34 focused tests. CPU and GPU evidence have distinct scopes.

Evidence: [qualification result](raw/execution_001/qualification/RESULT.json), [enabled K gradients](raw/execution_001/qualification/G_enabled/after_k_backward_1.json), [native post-Adam equality](raw/execution_001/qualification/G_enabled/native_post_adam.json), [final nonweight equality](raw/execution_001/qualification/G_enabled/final_nonweights.json), [budget gate](raw/execution_001/qualification/BUDGET.json).

## The observed guard decision

The enabled trial had a nonzero H gradient, with norm **0.2748676244**, and **hᵀδK = −0.4361937154**. The prescribed rule therefore correctly left the native K displacement untouched. Native and realized update norms both equal **10.0516003904**; final G parameters exactly match K in this trial.

**This GPU step exercised H differentiation and the inactive guard branch. It did not exercise an active correction.** Active projection and BF16 branches have CPU coverage; no additional response or step was selected to force activation. The negative directional derivative is a local first-order measurement, not a measured post-update H improvement or native drafting gain.

## Work, runtime and memory

| Work | This attempt |
|---|---:|
| Disposable optimizer steps | 3 |
| Qualification response presentations | 6 |
| Separate startup response forward/backward evaluations | 2, zero Adam steps |
| Completed H backwards | 2 |
| Completed qualification K backwards | 6 |
| Draft response forwards including startup | 8 |
| Draft-layer forwards including recomputation | 40 |
| Vocabulary-head calls including recomputation | 1,764 |
| Main training / local probes / serving generations | 0 / 0 / 0 |

The previous failed attempt consumed two temporary optimizer steps; **five temporary steps have now completed across both attempts**. Its four completed plus one partial response presentation and two startup evaluations remain separately recorded. No feedback or teacher data was newly generated.

Qualification took **242.096 seconds**. The detached supervisor ran **08:26:22–08:32:45 UTC, October27**, or **6m23s**, including storage/integrity checks, CPU tests and postflight. These nested durations are not additive.

| Trial | Peak allocated GPU memory | Peak reserved GPU memory |
|---|---:|---:|
| K | 21.075 GB | 23.742 GB |
| G disabled | 21.076 GB | 23.683 GB |
| G enabled | 31.487 GB | 34.358 GB |

Memory peaks cover loading, initial-state checks and trial execution. Enabled G's measured H-backward timer totaled **2.495s** and the guard-decision timer **0.268s**. These are nested qualification measurements; exact comparison/copy overhead and compilation make this qualification unsuitable for estimating complete production update cost. Main-training cost and active-projection behavior remain unmeasured.

## Evidence and interpretation

All **483 sealed files** passed preflight and postflight integrity checks; all **72 retrieved compact evidence files** were size/hash verified. Both A800 GPUs were idle after completion, with **46.252GB** free disk space. No original result, checkpoint or feedback artifact was deleted or overwritten. The clean Fenrir DFlash-Research source remains unchanged.

The new controls establish equivalence within the repaired runtime. Full bitwise equivalence to historical K1 trained under the previous default compiler setting is not established by these controls: the old disposable K tensor references were RAM-only. The earlier and repaired K trial records match except elapsed training time, including response IDs, anchors, losses, denominators, gradient norm and counters; this is narrower evidence than full parameter equality. The retained K1 checkpoint is unchanged. Buffer donation is a compiler-memory policy change, not a new learning rule; its costs remain explicit.

**The observed blocker is resolved and the prespecified qualification passes. This does not yet show that the guard improves drafting.** The original128-step G event,18 local probes and384-generation K/G evaluation remain pending. No tuning or extra qualification cases were introduced, and this repair execution stops at this report.

[Independent audit](FINAL_AUDIT.md) reproduces the controls, accounting and limits.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_prefix_preservation_guard_20261027/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
