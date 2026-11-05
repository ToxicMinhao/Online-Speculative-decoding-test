# G prefix-preservation guard: implementation and qualification report

**Implemented; GPU qualification failed, so the main event was not started.** No native drafting or serving-time result exists for G. This is an integration failure, not evidence for or against the guard.

## Implementation

The new guard uses the existing DFlash-Online K training path. It computes H on each current correct prefix, accumulates separate FP32 H gradients, and globally projects the actual native BF16 AdamW displacement. K's objective, clipping, Adam moments/counters and qualified settings remain unchanged. Inactive/disabled projection leaves the native parameters untouched. Local probes and the genuine two-arm K/G evaluator are implemented within the declared budgets.

The source changes are in `worktree/online_feedback/prefix_guard.py`, with small integrations in `objective.py` and `training.py`; qualification/evaluation live in `worktree/experiments/prefix_guard/`. Installed A800 root: `/data/sources/minhao/DFlash-Online`. Fenrir's clean DFlash-Research folder and historical checkpoints remain unchanged. See [implementation diff](IMPLEMENTATION.diff) and [implementation notes](IMPLEMENTATION.md).

## Checks completed

- 49 focused local CPU tests passed. A800 passed 483 regression CPU tests plus 31 focused tests; these sets overlap and are not 563 distinct tests.
- Independent review covered normalization, gradient isolation, projection, probes, cache/state restoration, cost accounting and failure handling. A CPU DDP/checkpoint fixture also passed; it did not exercise the failing GPU compiled backward.
- All 438 sealed source/dependency artifacts passed both preflight and postflight integrity checks.
- Ordinary K completed its disposable one-step control.
- Guard-disabled G completed its disposable one-step control and matched K **exactly** after both backward calls, before/after clipping, after native AdamW and in the final state. Checked tensors include parameters, gradients and Adam state; counters, RNG and modes also matched. Presentation/anchor records agree.
- Enabled G matched the released initial parameters, empty optimizer, counters, modes and pinned RNG. Its first H backward failed before completing; enabled pre-correction equivalence and projection therefore remain unqualified.

Evidence: [qualification result](raw/campaign_001/qualification/RESULT.json), [disabled final equality](raw/campaign_001/qualification/G_disabled/final.json), [work log](raw/campaign_001/qualification/WORK.jsonl), [launch traceback](raw/launch.log).

## Confirmed failure and bounded repair

At `online_feedback/prefix_guard.py:258`, H uses `torch.autograd.grad(..., retain_graph=True)` because the same current forward must subsequently provide K's gradient. The installed PyTorch compiled backward rejected this retained-graph call: its compiled function contains **donated buffers**, whose storage may be reused after backward and is incompatible with this graph-retention requirement.

This confirms a missing compiler integration setting. It is not a measured gradient-sum discrepancy, a BF16 rounding result, a speculative-verifier error or a negative algorithm result. The traceback does not independently identify which compiled subgraph owns the donated buffers; the training path includes compiled FlexAttention.

The matching Torch 2.13 source confirms this guard in `torch/_functorch/_aot_autograd/runtime_wrappers.py:3535–3539`; `torch/_functorch/config.py:345–347` documents the requirement. Earlier ordinary backwards can compile and cache a donated-buffer backward before G requests graph retention. The exact compiled node remains unidentified.

The smallest proposed repair is to pin `torch._functorch.config.donated_buffer=False` **before startup/training compilation**, equally for the ordinary K, disabled G and enabled G qualification trials and subsequent G training. Keep the scientific objective, BF16 arithmetic, optimizer and exact gates unchanged. This changes compiler buffer reuse and may affect memory/cost, so successful enabled qualification and exact K control checks are still required; the repair has not been executed or proven by this report. Add a focused compiled-retained-graph regression check and preserve failed-trial counters/peak memory on exception paths. Do not reuse warmed compiled functions from this failed process. Any later qualification attempt must count its work in addition to the two temporary steps already consumed here; it must not overwrite this attempt or describe the cumulative work as only three steps.

The campaign remains stopped under the handoff's correctness-failure rule. No tolerance was relaxed, no source/runtime repair was installed after the failure, and no automatic retry occurred.

## Actual work and cost

| Work | Planned | Completed / attempted |
|---|---:|---|
| Disposable qualification AdamW steps | 3 | 2 completed; enabled G never reached its step |
| Qualification response presentations | 6 | 4 completed, 1 partial, 1 unattempted |
| Separate startup response forward/backward evaluations | 2 | 2 completed; zero AdamW steps |
| Main G AdamW steps | 128 | 0 |
| Main G response presentations | 256 | 0 |
| Local draft-only response probes | 18 | 0 |
| Fresh K/G complete-response generations | 384 | 0 |
| New feedback/teacher preparation | 0 | 0 |

Across startup and qualification, hooks recorded 7 draft response forwards, 35 draft-layer forwards and 1,360 vocabulary-projection calls including checkpoint recomputation. Four measured K backwards completed; one H backward was attempted and failed. The log's fifth `qualification_k_backward_attempted` is an enclosing wrapper entry, **not** a fifth completed K backward.

Qualification took **195.462 seconds**; the campaign function took 196.163 seconds. The detached supervisor ran **07:58:02–08:02:53 UTC on October 27, 2026**, or **4 minutes 51 seconds**, including preflight and CPU tests. Ordinary K's recorded peak allocated/reserved memory was 21,075,324,416 / 23,741,857,792 bytes; disabled G's was 21,075,979,776 / 23,683,137,536 bytes. Enabled G's failed-trial peak was not persisted and remains unavailable.

Two reporting limitations are preserved explicitly: the original enabled-trial status is stale `running` although the global result and traceback record failure; its guard-only summaries/peak memory were not written before the exception. Raw records are unchanged. [Derived accounting](analysis/QUALIFICATION_ACCOUNTING.json) disambiguates these fields without rewriting evidence.

## Retention and conclusion

All 48 compact evidence files (780,772 bytes) were verified after retrieval; postflight was separately retrieved. No experiment artifact was deleted. Both A800 GPUs are idle; 46,407,061,504 bytes were free at the final check. Original feedback/features, K1 and historical O/U/P/R artifacts remain intact.

**What this establishes:** the disabled guard preserves K in the measured one-step GPU control. **What remains unknown:** enabled-gradient isolation, projection behavior, learning and native serving benefit. The original 128-step event and 384-generation evaluation remain unexecuted. No efficacy claim, algorithm tuning, later event or project-stage change follows from this failed qualification.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_prefix_preservation_guard_20261027/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
