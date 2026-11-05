# Single-step K-plus-correction diagnostic

**The diagnostic meets its prespecified local advancement criterion.** The original ordinary K step reproduced every retained per-response prefix count exactly. One post-K correction reduced lost old-prefix tokens from **153 to 113** and increased net matching-prefix gain from **+1,740 to +1,807**. All **527** useful K repairs survived; **6** additional original first errors were repaired.

This is a local, training-conditioned comparison of the same two responses and anchors. It does **not** establish better held-out drafting or serving speed. The result supports considering the subsequent one-event test stated in the handoff; no such test, tuning or extra correction was run.

## Local comparison

| Metric | Ordinary K | K + one correction | Denominator / meaning |
|---|---:|---:|---|
| Newly earlier errors | 47 | 41 | 774 sampled blocks; new prefix shorter than its pre-update prefix |
| Lost old-prefix tokens | 153 | 113 | 5,236 pre-update matching-prefix occurrences |
| Useful original first-error repairs | 527 | 533 | 632 original first-error blocks; all earlier correct positions must remain correct |
| Total matching-prefix tokens | 6976 | 7043 | Sum over the same 774 blocks |
| Net matching-prefix gain versus pre-update | +1740 | +1807 | Pre-update total 5,236 |
| Strict negative margins on protected positions | 50 | 40 | 5,236 fixed protected occurrences |
| Wrong-at-tie protected positions | 1 | 3 | Native BF16 argmax is wrong at zero margin |
| All protected argmax errors | 51 | 43 | Negative margins plus wrong ties |
| Global hinge V | 0.003567589 | 0.002353957 | Weighted numerators 109.5 / 72.25; common denominator 30693 |

Repair identity accounting: **527 retained, 0 lost, 6 newly obtained**. The correction removes 40 lost old-prefix occurrences and adds 67 net prefix occurrences beyond K. These are overlapping sampled-block occurrences, not independent tokens or requests. Primary improvement is established by the prefix counts, not by the lower hinge alone. Three wrong ties remain after correction; this surrogate gives zero direct derivative at zero margin and does not guarantee complete preservation.

Relative to post-K, **17 blocks improve, 8 regress and 749 tie** in matching-prefix length. They gain 77 and lose 10 occurrences, giving the net +67. Six K-damaged old prefixes are fully restored, and the correction introduces no damage to previously intact original prefixes. The 8 regressions concern post-K prefix progress; retaining every original first-error repair does not mean that every block keeps its entire post-K prefix.

### Per response

| Response | Blocks | Useful repairs K → C | Earlier errors K → C | Lost prefix K → C | Net prefix K → C |
|---|---:|---:|---:|---:|---:|
| train_00072 | 363 | 232 → 237 | 36 → 30 | 137 → 97 | +702 → +767 |
| train_00121 | 411 | 295 → 296 | 11 → 11 | 16 → 16 | +1038 → +1040 |

## What ran and what stayed unchanged

The same pinned Qwen3-8B target and Qwen3-8B-DFlash-b16 draft, qualified BF16 K/AdamW recipe, Flex attention, constant LR 3e-4, weight decay 0.01, clipping 1, response accumulation 2, original seed-17 RNG and startup were reused. Released weights, empty Adam, original modes/counters and RNG passed the exact initial-state check. The original first presentations are `train_00072` and `train_00121`; all anchors and conditioning are retained. K's original response normalization and objective remain unchanged.

Original correct-prefix membership and triangular weights were captured from K's existing pre-update forward. The correction uses one global denominator across both responses. Its gradients are accumulated separately in FP32; the global squared norm, coefficient and correction vector use FP32. Parameters are corrected from actual post-K BF16 weights and rounded once back to BF16. No FP32 master parameters, extra Adam step, epsilon, clipping, line search or resampling was introduced.

Global gradient squared norm: **0.003807620844**; coefficient `-V / ||g||²`: **-0.936960101128**. K's update norm is **10.051600390**. Proposed correction norm is **0.057815990** (0.5752% of K); BF16-realized correction norm is **0.036644860** (0.3646% of K). Norm reporting uses FP64; decision arithmetic remains FP32. The correction follows the full K step rather than shrinking or replacing it. Exact Adam moments, counters, RNG, modes, caches and existing gradients remain unchanged around correction/diagnostic replay.

The complete ordinary-K first-step reproduction matches 527 useful repairs, 47 earlier errors, 153 lost tokens and +1,740 net prefix tokens, including all eight metric checks separately for each response. K's update norm also agrees with the retained reference. No discrepancy, nonfinite failure or retry occurred in this GPU run.

## Work and costs

| Work | Actual |
|---|---:|
| Core training presentations / Adam steps | 2 / 1 |
| Direct corrections | 1 |
| Core response forwards / backward equivalents | 6 / 4 |
| Separate startup response forwards / backwards / Adam | 2 / 2 / 0 |
| Complete-response generations / exports / automatic retries | 0 / 0 / 0 |

All 774 sampled blocks were reused. There are 98 initial projection chunks across the two response forwards. K evaluates its chunk objective 196 times including recomputation; its 392 head calls include current and frozen-teacher projections. Post-K uses 196 objective/head calls including recomputation; post-correction uses 98 without backward. Separate startup uses 392 head calls including both projections and recomputation. Draft/layer counts and attempted/completed work remain in [SUMMARY.json](SUMMARY.json) and the raw work log. No new target-model forward or feedback generation occurred.

| Cost | Seconds | Scope |
|---|---:|---|
| Separate disposable startup | 37.682 | Includes loading; zero Adam, exact restoration |
| Fresh K loading and initial-state verification | 19.857 | Before the measured step |
| Ordinary K step with prefix recording | 5.584 | Two training presentations and one Adam step |
| Post-K violation forwards and gradients | 2.492 | New correction work |
| Direct parameter correction | 0.157 | No optimizer update |
| Post-correction scoring | 2.450 | Diagnostic-only recheck |
| Diagnostic process wall time | 97.812 | Includes state copies/checks and reporting |
| Supervisor elapsed | 163.000 | Includes preflight, remote CPU tests and postflight |

Post-K violation work plus applying the correction cost **2.650s** here. The 2.450s recheck is separate diagnostic work. These phase timers do not cover every CPU snapshot/check overhead; they do not establish an optimized online update cost or any serving speedup. The 0.059s capture timer covers only per-chunk observation, not all input copying and snapshots, and is nested inside K's timer. Do not sum nested timers again.

Peak GPU allocation/reservation: ordinary K phase **25.274/28.427 GB**; correction/recheck phase **29.469/35.830 GB**. The latter includes the resident K model. Peak allocated memory rose by **4.195 GB**. No separate fresh uninstrumented timing baseline was run.

## Correctness and retained evidence

Five additive source/test files were installed in A800's `/data/sources/minhao/DFlash-Online`. Historical objectives and author training/inference files were unchanged. 48 local CPU checks and 21 remote new-module checks passed; these suites overlap. The initial CPU recorder failure was fixed by making observation explicitly no-grad; it was an instrumentation error, not a change to K or the correction formula. Its failure record remains [preserved](CPU_FOCUSED_INITIAL.log). The existing qualified `donated_buffer=False` pin was retained before compilation.

The diagnostic completed on A800 at 10:59:44 UTC on October 27, exit 0. All 565 sealed source/input artifacts passed postflight, and all 53 retrieved evidence files passed size/hash verification. Both GPUs are idle. No prior artifacts were removed to admit this run. After verified retrieval, only the upload/evidence transfer containers were retired on both hosts; extracted evidence and source remain, with no replay capability lost. About 37.79GB remains on A800 and 82.42GB on Fenrir. [Storage record](STORAGE_MAINTENANCE.json).

- [Prespecified handoff](HANDOFF.md), [numerical contract](NUMERICS.json), [implementation review](REVIEW.md).
- [Complete result](raw/execution_001/RESULT.json), [exact reproduction gate](raw/execution_001/REPRODUCTION.json), [budget receipt](raw/execution_001/BUDGET.json).
- [Per-response table](PER_RESPONSE.csv), [per-block audit table](analysis/PER_BLOCK.csv), [independent audit](FINAL_AUDIT.md).
- [Post-correction optimizer/state equality](raw/execution_001/CORRECTION_STATE_UNCHANGED.json), [postflight](raw/POSTFLIGHT.json).

**Stop at this local result.** It demonstrates a useful correction on this one fitted batch under BF16. It does not establish transfer across requests, robustness across batches, a full-event gain or deployment acceleration.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_prefix_correction_step_20261027/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
