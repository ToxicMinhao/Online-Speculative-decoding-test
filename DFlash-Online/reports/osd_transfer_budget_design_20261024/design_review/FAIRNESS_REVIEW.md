# Independent fairness and analysis review

## Material Passport

- Origin: ARS experiment-agent, plan mode; 2026-10-24.
- Scope: review of the proposed new-versus-repeated-feedback experiment; no execution, remote access, training-code modification, or result claim.
- Reviewer: separate Codex agent with shared project context, same model family; not external or cross-model peer review.
- Verification status: source-read design audit. Proposed thresholds below are prospective design choices, not established findings.
- Sources: [proposal](../PROPOSAL.md), [working configuration](../../dflash_online_osd_rerun_20261023/sealed/config.yaml), [working protocol](../../dflash_online_osd_rerun_20261023/REPORT.md), [completed results](../../dflash_online_osd_rerun_20261023/RESULTS.md), current project workflow/experience records.

## Assessment

The proposed intervention is interpretable if it is described as **feedback allocation at matched optimizer steps and response presentations**. It is not an equal-token, equal-anchor, equal-FLOP, semantic-diversity, or temporal-freshness experiment. Holding hyperparameters fixed is necessary but does not make the computational exposure identical.

Keep the successful recipe unchanged. The intervention is which saved response buffer trains later events. No change to the loss, learning rate, anchor cap, per-pass resampling, response shuffling, effective response batch, optimizer implementation, model initialization, serving prompt, or timing procedure is warranted for this screen.

## Required clarifications before sealing

### 1. The primary policy contrast has 96 distinct scored requests

The 32 requests following event 1 see identical S1 and R1 checkpoints. They are useful for first-update versus released and marginal-update summaries, but contribute a deterministic zero to S-versus-R.

- Overall baseline/marginal panel: four windows × 32 = 128 distinct requests.
- Primary new-versus-repeated policy panel: windows after events 2–4 × 32 = **96 distinct requests**.
- Three repeats improve timing measurement; they do not make these 288 independent requests.
- At event 2 both training alternatives start from the same model, Adam moments, counters and RNG. This is the clean single-event allocation intervention.
- Events 3 and 4 compare cumulative policies whose learned states have diverged. Do not describe them as the isolated causal effect of that event's fresh buffer.

### 2. Preserve the actual serving prompts and disclose previous project use

The successful run uses raw GSM8K questions in the Qwen no-thinking template for its training stream. Its separate held-out prompts contain an additional reasoning instruction. The proposed next-buffer study must retain the training-stream prompt token IDs for canonical generation and all checkpoint replays; copying the old held-out prompt construction would change the workload being tested.

All 7,473 GSM8K training questions were already used by a previous project run. The new study may initialize from released weights and generate new answers, so earlier project use does not leak trained weights into this run. Nevertheless, it is not a previously untouched project dataset. Record that fact. Select the 640 distinct requests and the 32-per-window benchmark IDs using a fixed algorithm before looking at their new outcomes, and preserve ordering thereafter. Deduplicate by question content/IDs without using old model performance. No replacement after a cap hit, unfavorable response, or failed comparison.

The first 512 responses can supply training. The final 128 must not. Earlier benchmarked requests may legitimately become feedback only after their scoring window closes. Do not call all 128 benchmark requests permanently held out.

### 3. Define the repeated-feedback intervention precisely

R reuses the exact B1 generated token IDs and frozen-target replay information, rather than regenerating the first questions with its newer draft. Follow the qualified normal sampling and shuffling procedure on each later pass; do not freeze every training anchor merely to make R look more controlled.

The intervention therefore replaces old feedback with new feedback **including its responses and any qualified proposal-conditioned teacher inputs**. A resulting advantage does not isolate question novelty from response length, token exposure, proposal conditioning or feedback coverage.

This control is deliberately narrow: repeatedly reusing one initial buffer. A poor R result does not show that every replay scheme, cumulative replay buffer, or mixture of old/new feedback would fail.

### 4. Protect branch state and evaluation causality

At the fork, copy the complete model, optimizer moments, optimizer-step and event counters, scheduler state if present, Python/NumPy/Torch CPU/CUDA RNG and any sampler-specific generator state. Verify equality, not just model-weight equality. Use independent branch state paths.

Before each branch runs, restore that branch's own state. Persist its state after the event. Benchmarks and source-fit diagnostics must not advance either branch's training RNG, counters, optimizer state or response buffer. Do not reseed to 17 at every event. Later RNG values may legitimately differ because sampling uses buffers with different eligible-anchor counts; they must not differ because the other branch or a benchmark consumed random numbers.

Every evaluation window holds all its checkpoint weights fixed. Generate the next buffer's canonical responses under S, then complete all planned replays before either branch trains its next event. No concurrent/background training during timed evaluation. R's benchmark outputs never enter its buffer or alter the canonical stream.

### 5. Exposures and budgets

The shared first event plus three events on each branch gives:

| Quantity | Physical execution | Each logical trajectory at event 4 |
|---|---:|---:|
| Training events | 7 | 4 |
| Optimizer steps | 896 | 512 |
| Response presentations | 1,792 | 1,024 |
| Distinct training responses | 512 overall | S: 512; R: 128 |
| Canonical generations | 640 | One S-generated stream |
| Timed generations | 1,536 | 32 × 3 × (2 + 4 + 5 + 5) |

Report per-event and cumulative actual anchors, unique anchor positions, supervised-position occurrences, sums of loss weights, output-token counts and training time. Equal response presentations do not establish equal supervised exposure. Do not compensate after seeing outcomes by adding steps, trimming long responses, changing loss weights or resampling difficult questions.

No extra generation warmups or performance-dependent diagnostic generations. Existing runtime initialization remains required and separately timed. Any implementation qualification generation must be included in the declared generation budget rather than silently added.

## Proposed analysis contract

### Primary endpoint

For the 96 distinct requests after events 2–4, let C_S and C_R denote verified native-call counts, taken from one validated trajectory per checkpoint/request after confirming that all three repeats reproduce those counts. Define:

    allocation_call_reduction = 1 - sum(C_S) / sum(C_R).

Positive means fewer calls with additional feedback. Because canonical outputs must match, the relative calls-per-1,000-output-token reduction has the same value on the same cohort. Do not average per-request percentages. Also report absolute saved calls, wins/ties/regressions, and the three separate event contrasts; event 2 is the direct common-start comparison.

Main cumulative and marginal tables should use a common valid cohort within each window across F, S-current, S-previous, R-current and R-previous, after deduplicating identical checkpoints. Otherwise changing denominators can create apparent disagreement between tables. Pair-specific results can additionally be reported with explicit denominators.

### Materiality and uncertainty

A reasonable prospective screen threshold is **2% relative verification-call reduction** for S over R. This is a chosen minimum worth investigating in this bounded screen, not a power-derived detectable effect, a theorem, or a claim that smaller improvements have no value. Seal the threshold before new measurements.

Report a 95% stratified paired request-bootstrap interval using 10,000 draws and a fixed analysis RNG seed (for example 20261024). Resample requests within each of the three windows, carry all arms and timing repeats together, and recompute ratios from sums. For timing, first take the mean of the three latencies for each checkpoint/request; bootstrap request-level tuples, not individual repeats. Report all three raw-repeat aggregates as well.

These intervals describe sensitivity to the scored request sample conditional on the realized feedback/order/training trajectory. They do not include training-seed uncertainty, independently sampled feedback streams or hardware-session replication. The window stratification must remain visible; pooling across windows does not make the evolving models independent experimental runs.

Suggested decision labels:

- Point estimate at least +2% and lower interval endpoint above zero: a material positive allocation signal on this trajectory, subject to exposure/cost qualifications.
- Point estimate at most -2% and upper endpoint below zero: a material negative allocation signal.
- Interval wholly inside [-2%, +2%]: differences are small at the chosen screen scale on this sampled trajectory; this is not population equivalence or proof that feedback never matters.
- All other outcomes: inconclusive at the fixed budget. Stop without automatic extension.

The primary full-panel decision requires completion and validity of every planned primary comparison. If one is invalid or absent, keep available paired estimates and their intervals as explicitly subset-based descriptive results; do not label the sealed 96-request decision passed. This avoids rewarding a method for dropping unfavorable or numerically divergent cases. There is no reason to replace missing cases.

### Validity gates and timing order

Validate exact input IDs, output IDs, termination/EOS and output cap, checkpoint identity, capture-disabled status, positive timing, and internally consistent accepted/committed counts. Persist validation status and errors before aggregating. Output preservation means acceleration of the same generation; mismatching outputs are neither wins nor silently discarded successful runs.

With five methods and three repeats, each request cannot see every order position equally often. Rotate order across request index as well as repeat, using a sealed schedule independent of outcomes, so aggregate order positions differ by at most one where feasible. Always restore valid caches under the qualified policy after checkpoint switching. Do not infer per-kernel costs from complete-request timers.

### Source-fit diagnostic budget

The proposal currently leaves the source-fit diagnostic size unspecified. Pin a finite diagnostic set before execution so this does not become an unbudgeted second campaign. One sufficient option is eight source responses per distinct buffer, selected before observing outputs, and up to 64 eligible diagnostic anchors per response selected once without changing training RNG. Use those same anchors, teacher targets, masks, weights and denominators for before/after eval/no_grad forwards; R uses the same B1 diagnostic set throughout. Keep normal training anchors at the qualified cap of 512.

For seven physical events, before/after analysis of eight responses permits 112 response-level diagnostic forwards and at most 7,168 anchor presentations. These are diagnostics, not optimizer steps or generations, and their runtime must be recorded separately. The result is a fixed sampled source objective, not an assertion about every training token. Zero-supervision cases remain visible without resampling. Cross-branch source losses on different response sets are not directly comparable.

## Cost interpretation

Measure complete blocking updates and their teacher preparation, training, persistence, export and activation components without summing nested spans twice. Canonical serving already includes feedback capture if enabled. Benchmark replay latency excludes capture/training/activation and represents an available checkpoint. Keep benchmark-only costs out of deployment episode totals.

R has cheaper access to old feedback than S has to fresh feedback; that is an intended practical distinction. Report actual R reloading/preparation cost, rather than charging fictional new teacher computation or concealing repeated preparation. Count actual optimizer and objective work for both arms.

Only S generates all 640 canonical requests. Fresh checkpoint-ready benchmark timings on 32 of each next 128 requests cannot establish a fully measured frozen or R deployment episode. Any extrapolated full-window savings must be labeled an estimate and carry sampling uncertainty. Report S's observed online serving-plus-update duration and the experimental total separately. Do not claim net online speedup by subtracting 640-request costs from a 128-request timing saving.

## Conclusion

This is a fair, bounded allocation test under a competent fixed recipe once its estimand, fork state, benchmark cohort, diagnostic budget and exposure limits are pinned. It can distinguish useful new-feedback allocation from repeatedly optimizing the first buffer within this run. It cannot by itself establish temporal specialization, forgetting, a minimum successful buffer size, equal-compute superiority, seed robustness, or a novel mechanism. Continue through all declared windows on valid null/harmful results; pause only for correctness, resource or execution failures, without changing scientific settings.

## Final review of prepared execution design

Reviewed `EXPERIMENT_PLAN.md`, `REQUEST_MANIFEST.json`, `CONFIG_DIFF.json` and `DESIGN_VALIDATION.json` after preparation. The plan incorporates the substantive fairness findings above. No further scientific or budgeting blocker was found beyond the rotation correction already identified by the coordinator.

- The request manifest contains five 128-request blocks, with 32 benchmark requests in each of B2–B5 and eight source-probe requests in each of B1–B4. B5 has no feedback/probe assignment. No answer generation was performed for this review.
- The common checkpoint lists yield 1,536 timed executions, and the shared event accounting yields 896 physical steps and 1,792 response presentations. The primary S/R contrast correctly excludes the identical first window and has 96 requests.
- The final plan selects eight source responses and the unchanged 512-anchor sampler cap. This supersedes my optional 64-anchor diagnostic suggestion above. Its budget is exactly 112 response diagnostic forwards and at most 57,344 diagnostic anchor presentations. It does not change training exposure.
- The four recorded configuration differences are run identity/output location and the bounded event/step horizons. Constant-LR behavior and common learning settings remain explicit.
- For the corrected rotation `(3*i + r) % N`, direct integer enumeration over 32 requests × three repeats gives offset counts 48/48 for two arms, 24 each for four arms, and 20/19/19/19/19 for five arms. This establishes aggregate position balance within one without claiming within-request full balance.
- The threshold is prospective and finite-panel; conditional bootstrap uncertainty, the common valid cohort, source/transfer separation, incomplete-result visibility, and absence of automatic expansion are stated correctly.
- Historical source matches and static preparation are not presented as current A800 qualification or storage admission. Failed SSH authentication leaves those execution gates pending. No launch authorization or GPU readiness is inferred from this design review.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/osd_transfer_budget_design_20261024/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
