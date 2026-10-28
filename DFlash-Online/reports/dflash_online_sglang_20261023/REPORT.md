# DFlash-Online SGLang/FA3 integration — qualified

The A800 implementation passed its bounded GPU qualification. SGLang/FA3 now
implements the existing online controller's serving, feedback preparation and
publication interface. SpecForge remains the training implementation with the
selected OSD recipe unchanged. This qualifies an infrastructure option; it does
not show that a runtime difference caused earlier learning failures or establish
an adaptation benefit.

Maintained project: `/data/sources/minhao/DFlash-Online` on A800.
Fenrir `/home/minhao2/DFlash-Research` was not modified. This directory's
[worktree](worktree/) is the review/transfer snapshot. No full campaign or GitHub
upload was performed. Stage 1 and previously paused research remain unchanged.

## What changed

- [sglang_backend.py](worktree/online_feedback/sglang_backend.py): the same
  `render → generate → prepare_feedback → publish` interface used by the controller.
- [sglang_worker.py](worktree/online_feedback/sglang_worker.py): a serialized worker
  using the existing SGLang environment, isolated from newer SpecForge dependencies.
- [sglang_hooks.py](worktree/online_feedback/sglang_hooks.py): observes unchanged
  native prefill/verification and handles idle cache flushing and publication.
- [training.py](worktree/online_feedback/training.py): admits explicitly validated
  SGLang capture provenance. Its training procedure was not replaced.
- [qualify_sglang.py](worktree/online_feedback/qualify_sglang.py): bounded GPU checks
  and an explicit publication-only recovery path.

Select `SGLangDFlashBackend` explicitly for future runs. The inherited
`model.target_backend` training field does not select online serving. Historical
experiment scripts were not silently rewritten. The Transformers backend remains
available; see [usage](worktree/README.md).

The released draft backbone, native proposal/verification implementation, upstream
training model and optimizer machinery were not patched by this integration.

## Feedback, publication and costs

Serving captures frozen-target auxiliary features and normalized head features
from actual native prefill and committed verification rows. Rejected-history rows
never become canonical training features. Exact token IDs are retained without
retokenization; only the new response receives loss.

Native capture covers `prompt + response[:-1]`. To satisfy the existing trainer's
complete-sequence tensor contract, a separate idle target forward processes the
exact complete sequence and retains **only its final feature row**. That row
supplies neither a teacher label nor visible draft training context under the
existing alignment and masks. All supervised teacher rows remain actual native
capture. The extra forward is charged as preparation; this is not a claim that
feedback preparation is free.

Each request starts with empty logical target/draft caches. Publication validates
checkpoint structure, finite values and semantic configuration, loads the draft,
refreshes SGLang's copied KV helper weights, checks their consistency and clears
caches. Failure blocks further serving. Full exported/native tensor equality is
an explicit qualification check, outside normal request timing.

Feature capture is disabled in evaluation. Complete request latency includes the
RPC boundary, cache maintenance, prefill, drafting, verification and observation;
serving additionally includes feature capture and persistence. Observer timers
include transfers/synchronization and can wait for outstanding GPU work: they are
not isolated kernel times or an independently measured causal capture slowdown.
Do not add capture time again to a request latency that already includes it.

Native stop-clipped acceptance counters are retained separately from actual
committed draft proposals. The latter use
`min(raw accepted proposals, actual committed progress)`.

## Qualified settings

| Item | Value |
|---|---|
| Target/draft | Released Qwen3-8B / Qwen3-8B-DFlash-b16 |
| DFlash source | `07ebd93db9f472af339b644bb70221ad8428328a` |
| SpecForge source | `1839aae30003e3573d00979eb1edd57347b3578c` |
| Native runtime | Existing qualified SGLang/sgl-kernel FA3 fork; actual file identities in runtime receipt |
| Native environment | Torch 2.9.1+cu130, Transformers 4.57.1, Python 3.12.14 |
| Training environment | Torch 2.13.0+cu130, Transformers 5.12.1, Python 3.12.15 |
| GPUs | A800 80 GB: serving GPU 0, training GPU 1 |
| Decoding | Greedy, block size 16, concurrency 1, 128 output tokens or EOS for this check |
| Context | 2,048 canonical tokens; internal native capacity adds 16 speculative slots |
| Prefill chunk setting | 8,192; long/chunked behavior has CPU fixtures, not a >2K GPU qualification |
| Training | OSD forward KL, constant 3e-4, BF16 AdamW, weight decay 0.01, BF16 KL, Flex attention |
| Sampling | Existing eligibility/preparation; up to 512 anchors; responses shuffled each pass |
| Event override | Two responses, two passes, accumulation two: **two optimizer steps, four presentations** |

The selected production-sized recipe remains 128 responses/event, two passes,
58 events and 7,424 optimizer steps. This check did not execute that campaign.
[Installed configuration](evidence/qualification_004/config.yaml),
[native runtime receipt](evidence/runtime_004.json),
[training runtime receipt](evidence/runtime_training.json),
[final source manifest](SOURCE_FINAL.json).

## Verification results

**239 CPU tests passed on Fenrir and A800.**
[Local log](CPU_TESTS_FINAL.log), [A800 log](evidence/CPU_TESTS_A800_FINAL.log).

The GPU check completed across the original attempt and publication-only recovery,
with eight total generations and no repeated training:

| Check | Observed result |
|---|---|
| First captured response vs capture-disabled replay | Identical 128 output IDs; 17 verification calls in both |
| Second captured response vs capture-disabled replay | Identical 128 output IDs; 27 verification calls in both |
| Prefill-only one-token request | Correct first token; zero verification calls |
| Teacher feature/mask coverage | Full lengths 161 and 178; auxiliary width 20,480; native labels bound to exact responses |
| Update accounting | Two steps, four presentations; two optimizer windows × 3,600 supervised-position exposures = 7,200 |
| Actual parameter change | 45 of 58 tensors changed, including 695,103,457 BF16 coordinates |
| Trained publication | All 58 exported tensors match native packed/unpacked values |
| Frozen reset | All 58 released tensors match native values |
| Trained reload | All 58 exported tensors match again |
| Same fixed later input | Updated, frozen and reloaded produce identical 128-token outputs; 18 verification calls each |
| Evaluation isolation | Trainer weights, Adam state, RNG and counters match the saved state exactly before/after replay |

The final three executions show consistency, **not an OSD gain**. No gain is
required for this correctness qualification. The supervised-position count sums
over overlapping sampled examples and is not a count of distinct answer tokens.

[Original generation/training evidence](evidence/qualification_003/RESULT.json),
[completed training event](evidence/qualification_003/trainer/events/event_1/RESULT.json),
[successful recovery](evidence/qualification_004/RESULT.json),
[weight-change check](evidence/WEIGHT_CHANGE.json).

## Preserved failures and narrow repairs

1. First startup failed before model execution: the legacy sgl-kernel needed the
   already-installed CUDA 12 compatibility-library path.
2. Second startup failed because the worker PATH omitted its existing `ninja`.
   The worker now receives its own environment's executable/library paths and
   compiler settings. No packages were installed or scientific settings changed.
3. Generation, capture and the two-step update passed, but publication stopped at
   semantic configuration comparison. Transformers 5 had added
   `rope_parameters={rope_theta:1000000, rope_type:default}` to the equivalent
   legacy `rope_theta=1000000`, `rope_scaling=None` configuration. The narrow
   normalization accepts only that redundant default spelling. Tests still reject
   changed theta/type/scaling/additional parameters and nonfinite tensors.
4. Publication resumed from the exact saved export and trainer state. Recovery
   performed **zero additional optimizer steps**, no new training responses or
   exports, and only the three remaining checkpoint replays.

The old failed buffer remains untouched; its failure is not relabelled as a
successful uninterrupted event. Recovery establishes the remaining publication
and replay checks. No automatic resume of an old research campaign was attempted.

## Measured qualification costs and retention

Training took 21.04 s; checkpoint persistence 17.28 s; export 11.46 s. The training
event boundary before publication took 49.81 s. Feedback preparation took 0.055 s
in total, with feature persistence separately recorded. Successful whole activation
calls took 1.56 s (trained), 1.51 s (frozen reset), and 1.33 s (trained reload).
These are operation measurements from this small check, not serving speedups.

Original generation/training attempt: 94.53 s wall time. Publication-only recovery:
59.91 s. The two earlier startup failures took 8.03 and 21.02 s. Their sum is
183.49 s of attempt wall time; implementation/debugging time between attempts is
not included. Peak training allocation in the actual update was 12.56 GiB.

A800 free space was 28,113,125,376 bytes before launch and approximately
19,658,371,072 bytes afterward, above the 8 GiB reserve. Fenrir retained about
85.23 GB free. Both A800 GPUs were idle after completion. No research artifacts
were deleted during this task. Compact evidence is copied to Fenrir; the actual
trainer state, exported checkpoint and captured tensors remain on A800 under:

`/data/sources/minhao/runs/dflash_online_sglang_20261023/qualification_003/`

The final source matches the runtime source identities used for successful
recovery. The retained unsuccessful attempts, configs, timing boundaries and
native traces make the integration reviewable without treating a CPU test pass
as GPU qualification.

## Interpretation and next authorized boundary

Matching the native serving backend is useful for a controlled comparison with
the successful reference, but SGLang is not a mathematical prerequisite for OSD.
This integration removes one infrastructure difference for future comparisons.
It does not prove equivalence with earlier Transformers outputs or explain earlier
negative results. Longer contexts, 128-response storage/throughput and the full
58-event learning trajectory are outside this qualification. No further run was
started.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_online_sglang_20261023/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
