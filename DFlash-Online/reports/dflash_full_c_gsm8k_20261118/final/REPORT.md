# C58 final results: fixed 58-event GSM8K training and MathQA transfer

## Material Passport

- Scope: final results from the completed fixed C58 experiment; this publication adds no model execution.
- Execution commit: `b4459820b3822c3d760bb3fd880e4d139fa2bbc2`.
- Qualification commit: `a32f259d87697f09653c2e2eab786bb17603de1b`.
- Source: [EXECUTED_FINAL_AUDIT.json](EXECUTED_FINAL_AUDIT.json) and the two dataset summaries, scalar measurements and status records.
- Verification: executed raw-record audit passed on both datasets. Independent local reduction passed for both datasets: all four primary C/K point estimates and 95% interval endpoints reproduce exactly, including request-order and draw-matrix identities. See [independent bootstrap audit](INDEPENDENT_PRIMARY_BOOTSTRAP.json) and [interpretation review](INTERPRETATION_REVIEW.md). This is not a second model or raw-token replay.

## Completion

All 58 events and 7,424 production Adam steps completed, including the final 49 untrained responses. Both 1,024-request panels finished: 18,432 scored executions and 576 warmups in total. Zero invalid, unknown or unattempted final executions remain. Both A800 GPUs were idle at the completion check.

## C58 versus native K+58

Positive values favor C58. Brackets are the prespecified paired-request, marginal 95% bootstrap confidence intervals.

| Dataset | Verification-call reduction | Native accepted-tokens/call improvement | Complete-request time reduction |
|---|---:|---:|---:|
| GSM8K | 0.317% [0.050%, 0.583%] | 0.368% [0.058%, 0.681%] | 0.284% [-0.253%, 0.824%] |
| MathQA | 0.295% [0.100%, 0.492%] | 0.345% [0.116%, 0.576%] | 0.298% [0.098%, 0.499%] |

C58 has a small supported drafting improvement on both panels under these prespecified intervals. The MathQA serving-time improvement is also supported. GSM8K serving-time improvement remains unresolved because its interval crosses zero; this does not establish equivalence or a speed regression.

GSM8K calls fall from 43,267 to 43,130: 137 net saved, 416 requests improve, 366 regress and 242 tie; 780 gross saved versus 643 added. MathQA calls fall from 73,132 to 72,916: 216 net saved, 430 improve, 370 regress and 224 tie; 974 gross saved versus 758 added.

These effects are small (about 0.3%). Counts use one repeat-consistent trajectory per request. Time uses each request's three-repeat mean, then summed ratios. Native acceptance and call improvements are coupled accounting views of identical outputs, not two independent confirmations.

The bootstrap uses 10,000 PCG64 whole-paired-request draws, seeds 20261118/20261119, with lexicographic request-ID ordering and common draws across metrics. These are conditional marginal intervals for one seed-17 trained trajectory on the reused original GSM8K/MathQA panels; they do not establish seed robustness, family-wise significance, fresh-panel confirmation, or training-inclusive speedup. MathQA measures transfer of GSM8K-trained checkpoints. Native K+ remains the established primary method; no automatic tuning, new experiment or stage advance follows.

## Complete-method comparisons against ordinary OSD

The incremental C comparison above remains primary. Relative to O58, C58 reduces calls by 3.577% on GSM8K (95% CI 3.228–3.924%) and 2.500% on MathQA (2.256–2.751%). Its request-time reduction is 3.094% (2.416–3.770%) and 2.394% (2.148–2.645%), respectively. Most of this complete-recipe advantage already belongs to native K+: native K+/O58 call reductions are 3.271% and 2.212%. Beating OSD does not establish the incremental value of C; C/native K+ is the relevant contrast for that question.

All three arms, their raw summed measurements and all prespecified contrasts are retained in the two dataset summaries and [confidence-interval table](CONFIDENCE_INTERVALS.csv).

## Training and validity contract

C starts from the released drafter and original seed-17 initial-state witness, with a fresh native BF16 AdamW optimizer. It changes only the existing correct-prefix KL coefficients, using unchanged smoothing 0.5 and relative cap 0.25. Native K+ selection, first-error redistribution, response normalization, LR 3e-4, clipping, two passes and natural anchor sampling remain unchanged. The 58 feedback windows contain 128 own-generated responses each; the final 49 stream responses do not train the model. This is a complete own-feedback trajectory, not shared-feature replay with the historical controls.

Retained O58 and KPLUS58 were not retrained. All three are freshly measured with greedy BF16 block-16 serving, 2,048 tokens or EOS, context 4,096, concurrency one, disabled capture and the qualified cache policy. Counts use one validated repeat-consistent trajectory; timing uses three rotating repeats. Exact canonical output/termination and retained O58/KPLUS58 call counts pass for both panels. Native accepted proposals and actual stop-clipped accepted tokens remain separate in the numeric tables. See [configuration and identities](PROVENANCE.json).

The original initial-state qualification failure remains preserved. Its confirmed cause was one Rich terminal-style Python RNG draw after seeding; preloading that module before the unchanged seed call repaired exact initialization without changing scientific settings or relaxing comparisons. The resolved [startup report](../STARTUP_RNG_REPORT.md), original [qualification report](../QUALIFICATION_REPORT.md), and original failure evidence remain in this PR. The elapsed cap alone was explicitly extended from 24 to 30 hours; the original clock and 40 GPU-card-hour cap were preserved.

## Execution cost and evidence limits

The production wrapper took 15.885 hours, including training and endpoint evaluation. The original clock through the automated audit reports 21.200 elapsed hours and 25.943 occupied GPU-card hours, within the authorized 30/40-hour caps. The standalone final raw-record/statistical audit took 63.410 seconds and added zero model work. These times are not matched training-overhead estimates versus historical K+.

The public package contains numeric results and audit evidence; private responses, tokens, native traces, model/recovery state and runtime artifacts remain outside Git. The executed raw-record auditor passes both datasets. A separate local agent independently recomputed all four primary C/K metrics and their intervals exactly from the authenticated private measurement receipts. This is independent implementation of the scalar reduction, not independent training/serving replication or a second token-level audit. The public table verifier passes 215,366 checks with zero failures, confirming that sanitization preserves the reviewable numeric results. All original failure and qualification records remain preserved.

This publication does not assert a completed independent whole-campaign archive/storage-closeout audit beyond the linked result-audit scope. Existing pre-run storage retirement remains documented in [STORAGE_BEFORE_AFTER.json](../STORAGE_BEFORE_AFTER.json). Final C58 recovery/export and retained controls stay preserved for review; no further training or cleanup is introduced by this publication.

## Reviewer files

- [All confidence intervals](CONFIDENCE_INTERVALS.csv), expressed as fractions except the absolute acceptance difference.
- [GSM8K summary](gsm8k/SUMMARY.json), [per-request table](gsm8k/PER_REQUEST.csv), [all scored repeats](gsm8k/ALL_REPEATS.csv), [repeat totals](gsm8k/REPEAT_TOTALS.csv).
- [MathQA summary](mathqa/SUMMARY.json), [per-request table](mathqa/PER_REQUEST.csv), [all scored repeats](mathqa/ALL_REPEATS.csv), [repeat totals](mathqa/REPEAT_TOTALS.csv).
- [Executed final audit](EXECUTED_FINAL_AUDIT.json), [independent primary bootstrap audit](INDEPENDENT_PRIMARY_BOOTSTRAP.json), [11-category interpretation review](INTERPRETATION_REVIEW.md).
- [Public table verifier](verify_published_tables.py), [public reproduction result](PUBLIC_TABLE_REPRODUCTION.json), [publication review](PUBLICATION_REVIEW.json).
- [Work accounting](WORK.json), [cost accounting](COSTS.json), [58 event costs](EVENT_COSTS.csv), and per-dataset evaluation-cost records.
- [Execution/configuration provenance](PROVENANCE.json) and [public file manifest](PUBLICATION_MANIFEST.json).

From this directory, run `python verify_published_tables.py --output local-verification.json` (choose a new output filename) with NumPy to reproduce the primary confidence intervals from the published scalar tables. No model, private response or GPU is required.
