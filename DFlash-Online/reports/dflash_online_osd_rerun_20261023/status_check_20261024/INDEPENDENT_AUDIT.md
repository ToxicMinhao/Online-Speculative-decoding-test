# Independent metric audit — persistent Qwen/GSM8K OSD rerun

## Material Passport

ARS experiment-agent; local read-only metric and code audit on October 24, 2026.
Sources: copied `STATUS.json`, `evaluation/SUMMARY.json`, `evaluation/PER_REQUEST.csv`, and the sealed-run driver/core source snapshot. No remote operation, model execution, scientific configuration change or additional experiment. No source code modified.

## Verified totals

The 6,144 CSV rows contain exactly 1,024 distinct held-out requests × three repeats × two methods. All 3,072 request-repeat pairs are marked valid; there are no missing method partners or invalid pairs. Independent sums reproduce every total and both reduction ratios in `SUMMARY.json` (checked at relative tolerance 1e-12). The status records 7,473 served training-stream requests, 58 completed events and 7,424 optimizer steps; completion is reported, not re-executed in this audit.

| Metric | Released F | Final OSD O | Reduction |
|---|---:|---:|---:|
| Total capture-disabled complete-request time, all three repeats | 2580.635854 s | 2288.265065 s | 11.3294% |
| Verification calls, one repeat of 1,024 questions | 50,584 | 44,730 | 11.5728% |
| Calls per 1,000 output tokens | 160.662163 | 142.069005 | 11.5728% |
| Committed output progress per call | 6.203997 | 7.015940 | — |

Each repeat has 314,847 output tokens, with exactly the same per-request output lengths and call counts across repeats for each checkpoint. Across the 1,024 requests, OSD requires fewer calls on **952**, more calls on **44**, and the same number on **28**. Three requests hit the output cap in both arms. Mean latency over the three repeats is lower for OSD on 963 requests and higher on 61; these descriptive timing signs do not independently establish significance.

| Timing repeat | F complete-request time (s) | O complete-request time (s) | Reduction |
|---|---:|---:|---:|
| 1 | 860.338911 | 763.035571 | 11.3099% |
| 2 | 860.272022 | 762.472867 | 11.3684% |
| 3 | 860.024921 | 762.756627 | 11.3099% |

All numeric fields checked are finite. Recorded accepted draft tokens are nonnegative and no greater than committed progress; committed progress equals output length minus one (the initial prefill-generated token is excluded). All request and native generation latencies are positive. CSV prompt/output lengths and comparisons are consistent, but **CSV output counts alone do not establish token-ID equality**: that conclusion relies on the retained canonical-comparison records and execution checks, whose full raw collection is being handled separately by the parent audit.

## Interpretation and accounting

- The valid-pair gate in `experiments/osd_persistent/run.py:evaluate` requires the exact sealed prompt IDs, the first fresh F response IDs for that request (all F/O repeats), and capture disabled. Invalid pairs remain in raw tables and are excluded from ratio calculations. Lower-level generation validation checks EOS/cap and reconstructs the committed answer from verification blocks.
- The effect estimate is **performance with the final checkpoint already available**, measured without training. It is not an adaptation-inclusive speedup, and does not establish every intermediate update was beneficial. Repeats are repeated measurements of 1,024 requests, not 3,072 independent questions.
- Ratios are `1 − sum(O)/sum(F)`, not averages of per-question percentage changes. The fresh paired comparison should not combine historical timing measurements.
- `request_latency_s` is the complete backend request around the native RPC, including prefill, drafting/verification, required cold-cache setup and trace/capture bookkeeping appropriate to its mode. In heldout evaluation feedback capture is off. `native_generate_s` is a narrower nested native generation timer; do not sum them or infer kernel-specific costs from either.
- Production episode cost is `controller_serving_s + blocking_update_s` = **7490.206963 + 20421.374651 = 27911.581614 s**. Capture is already inside serving. Preparation, training, optimizer/checkpoint persistence, export and activation are nested update components and must not be charged a second time. One-time loading, qualification, publication checks, retirement/orchestration and final benchmark replay/activation overhead remain separate.
- `online_episode_wall_s` is 28209.721487 s; final benchmark wall is 10356.953300 s; overall recorded campaign wall is 38567.300474 s. The benchmark wall includes many checkpoint switches and diagnostic persistence, so it is not the serving-only runtime.
- During partial collection the STATUS reconstructed subtotal can be stale; the final completed subtotal now agrees with serving plus updates. Exposure metrics in trainer `RESULT.records` must sum optimizer-stepped rows only because those rows contain the accumulated two-response window, while intermediate rows contain one response.

## Source locations

- `/home/minhao2/EAGLE_SPEC/research/dflash_online_osd_rerun_20261023/worktree/experiments/osd_persistent/run.py`: paired evaluation, pooling, progress and top-level costs.
- `/home/minhao2/EAGLE_SPEC/research/dflash_online_osd_rerun_20261023/worktree/online_feedback/sglang_backend.py`: complete request/activation timers and metric definitions.
- `/home/minhao2/EAGLE_SPEC/research/dflash_online_osd_rerun_20261023/worktree/online_feedback/controller.py`: nonoverlapping serving and complete blocking-update spans.
- `/home/minhao2/EAGLE_SPEC/research/dflash_online_osd_rerun_20261023/worktree/online_feedback/native.py`: canonical generation/EOS/cap validation.

**Conclusion:** the copied final CSV independently supports an 11.33% complete-request-time reduction and 11.57% verification-call reduction for the final OSD checkpoint on the paired held-out workload, subject to the retained exact-output validity records. The gain is reproducible across these three timing repeats; adaptation costs remain a separate, much larger incurred cost.

Parent raw-evidence verification subsequently passed for all 6,144 timed records,
3,072 canonical pairs and 285,942 reconstructed verification blocks. See
[RAW_AUDIT.json](RAW_AUDIT.json). This resolves the transfer-pending boundary above.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_online_osd_rerun_20261023/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
