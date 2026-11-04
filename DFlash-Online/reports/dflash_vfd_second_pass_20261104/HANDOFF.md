# Executor handoff: Does first-pass progress identify which GSM8K responses to revisit?

**Run one four-event screen.** Keep K+ (Verification Frontier Distillation, VFD) unchanged. Test whether repeating the 64 responses whose verification errors moved toward repair produces lower complete-request latency on unseen GSM8K questions than full two-pass VFD. Compare against a static selector, random selection, and no second pass to explain the result. No cross-dataset evaluation, hyperparameter search, additional training arms, or full 58-event run.

## 1. Start point and shared workload

- Use the [audited PR #25 method and settings](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/23f3afc8620fcf8bd48048166b281d939c92150a/DFlash-Online/reports/dflash_kplus_20261103/REPORT.md); scientific execution pin: `e0a9d3c235e1efd15404c549282d21c99243a21c`. Restore its authenticated **event-zero** Qwen3-8B/DFlash-b16 state, including seed-17 training RNG and initial optimizer. Do not start at K+58. The original intermediate packets were retired.
- Use the first **four consecutive 128-response events** in the original ordered GSM8K training stream (512 requests). In each event, generate feedback once using the **preceding FULL checkpoint** and frozen Qwen3-8B target. Prepare each response once and give identical packets to all policies. This tests policies on common feedback; it does not measure five independent live deployments.
- In each arm learn all 128 responses once with native K+: BF16 AdamW, constant LR 3e-4, microbatch 1, accumulation 2, native loss/normalization, position decay, training anchors, clipping and context limit. At event 1, do this first pass once and fork the **complete** post-pass state into five copies. From event 2 onward, each arm must carry its own weights, optimizer, RNG and counters through both passes. Never reset to FULL between events.

## 2. The only intervention: which responses get a second pass?

| Arm | Pass 2, per event | Purpose |
|---|---:|---|
| FULL | All 128, native K+ | Serving baseline |
| ONCE | None | Value of any second pass |
| RANDOM64 | 64 uniformly without replacement | Value of choosing at the same update count |
| STATIC64 | Top 64 by initial repair opportunity | Does initial verification feedback already suffice? |
| PROGRESS64 | Top 64 by the progress rule below | Proposed addition |

Use FULL's native response permutations as shared event-order controls: record them and run each subset in its induced second-pass order, **never in score order**. Keep native training anchor sampling; score anchors are separate. Do not pad skipped responses with dummy optimizer steps. Use a named, independent hash/seed for RANDOM64 and for score ties: for example, lowest SHA-256 of UTF-8 `vfd-s2-v1|random|event_id|response_id` or `vfd-s2-v1|tie|event_id|response_id`. Use the same tie order for both selectors. Record actual selected lengths, valid blocks and worker time: equal response counts do not guarantee equal compute.

## 3. Fixed scoring rule — no fitting or tuning

For each response, select **up to 16 distinct native eligible block anchors** without replacement, by lowest SHA-256 of `vfd-s2-v1|anchor|event_id|response_id|anchor_id`. Use all if fewer are available. For scoring only, run a no-gradient evaluation pass on these anchors at **event entry** (STATIC64 and PROGRESS64) and again **immediately after pass 1** (PROGRESS64 only). Use the same anchors, context, masks, canonical target IDs, evaluation mode and precision before/after. Do not compare a training-mode logit with an evaluation-mode logit. Scoring must not change trainable state, RNG or subsequent native sampling.

For each sampled block, locate the **first mismatch before training**, comparing native draft top-1 with the canonical target token. A block is an **initial opportunity** if a mismatch exists and its immediate next position is valid and already matches. Count all sampled valid blocks, including fully matching ones, in the response denominator.

- **STATIC64 score:** fraction of sampled valid blocks that are initial opportunities.
- **PROGRESS64 score:** start with those same opportunities. A block earns zero unless, after pass 1, its original matching prefix remains intact, the **same** first mismatch remains, and its immediate successor still matches. For a qualifying block, compute the draft margin at that original mismatch, before and after: `m = logit(canonical token) − max logit(other token)`. Let `d = max(0, m_after − m_before)` and `g = max(0, −m_after)`. Give the block credit `2 × sigmoid(m_after) × d/(d+g)`, or zero if the denominator is zero. Average credit over **all** sampled valid blocks. A response with no eligible block gets score zero; use the fixed tie rule to select exactly 64.

The ratio measures how much of the initial logit gap closed when there was positive movement; the sigmoid discounts cases still far from a top-1 switch. Use native argmax to decide token match, then cast logits to FP32 for margin arithmetic. Count repaired errors and broken prefixes separately even though neither earns credit. This score is a hypothesis about remaining repair opportunities, not a probability of future improvement or a per-response causal attribution. Rank using only the current event's completed feedback; never use evaluation prompts or later events.

## 4. One sealed endpoint on GSM8K

- Before the run, seal **200 unique GSM8K test questions**, disjoint from all adaptation IDs and normalized prompt texts. Prefer historically unused test items: partition by prior project use, order each group by SHA-256 of `vfd-s2-v1|eval|stable_id`, and fill unused-first. Record old-panel overlap; if historical use cannot be established, call the panel *train-disjoint*, not historically fresh. Hash the manifest, policy version and analysis rules before examining outcomes.
- Evaluate **only after event 4**, on exactly the same prompts in all arms. Use PR #25's complete-request timer, C=1, greedy BF16, block 16, FA3, context 4,096, 2,048 new tokens or EOS, native cache flushing, capture off, no concurrent training or checkpoint transfer during timed requests. Use two timed sweeps. For the ten consecutive 20-request blocks, cycle the five-arm order by block; reverse the corresponding order in sweep two. Record every measurement; exclude model loading separately. Check identical output IDs and termination against the canonical reference on every measured request; reuse verified reference IDs or generate them once if missing.
- **Primary:** percentage reduction in summed complete-request time, PROGRESS64 versus FULL. Average the two timings for each arm/request first, then compute `100 × (1 − sum(PROGRESS64)/sum(comparator))`; positive favors PROGRESS64. Report the same paired estimate versus STATIC64, RANDOM64 and ONCE, with 95% intervals from 10,000 paired **request** bootstrap draws (fixed seed 20261104). Repeats do not increase sample size. Report target verification calls and accepted draft tokens/call to check whether timing agrees with decoding behavior. One training seed/four events is a screen, not seed-level replication.

## 5. Cost cap, validity and decision

**Science budget:** 512 shared feedback generations and 512 teacher preparations; 1,664 physically committed Adam steps and 3,328 response training presentations (event-1 first pass shared); at most 1,408 additional single-response scoring forwards with the 16-anchor cap (event-1 pre-score shared); **2,000 timed evaluations**, plus at most 200 untimed canonical-reference generations if missing. A scoring forward still counts at 16 anchors. Track logical costs per policy as if deployed alone: FULL / each half arm / ONCE take 128 / 96 / 64 Adam steps per event. Report preparation, scoring, training and export wall time separately. Never add asynchronous worker time to measured request latency.

Before production, check event-zero restoration, eligible anchors, available memory/storage and at least one before/after score example. On disposable state only, allow up to 8 extra scoring forwards and 4 extra training presentations/2 optimizer steps for qualification; no tuning. Check score bounds, prefix gates, native RNG isolation, selection identities and event-1 fork equality. If integrity or token parity fails, flag the affected comparison and stop; do not silently change precision, score, panel or budget.

Deliver a short report, implementation/config pin and compact machine-readable tables: per-event selections and scores (including zero/tie rates, sampled opportunities, repaired/persistent/broken counts and selected lengths), state/count checks, every paired request/repeat, and worker costs. At event 1, report STATIC64/PROGRESS64 selection overlap directly; later they rank under different model states. End with **one decision**:

1. **Advance to a longer in-domain run** if PROGRESS64 is faster than FULL with a positive lower paired 95% limit, also beats STATIC64 and RANDOM64 on point estimates, and no simpler arm clearly dominates. Show control intervals too; crossing zero means the added scoring mechanism is not yet established.
2. **Efficiency only:** if PROGRESS64 saves measured updater time and the lower 95% bound on its latency reduction versus FULL exceeds −0.5% (the predeclared acceptable slowdown), report that limited claim. If RANDOM64 or ONCE achieves the same trade-off, prefer the simpler policy. No serving-speedup claim from cheaper training.
3. **Otherwise:** state whether the measured effect argues against the rule or is inconclusive at this budget. Do not tune the score or extend the run automatically.
