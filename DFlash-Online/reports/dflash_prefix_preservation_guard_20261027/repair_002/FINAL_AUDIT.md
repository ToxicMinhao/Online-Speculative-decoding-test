# Independent audit: repaired guard qualification

**The declared three-trial GPU qualification passed.** Disabling compiled-backward buffer donation before startup allowed both retained-graph H backwards to complete, while the measured K gradients and native AdamW results remained exactly equal to ordinary K. This qualification ran no main training event, mechanism probes or complete-response generations. The GPU step did **not** activate projection, so it does not establish native correctness of the active correction branch or any drafting benefit.

The audit independently checked equality receipts, trial records, work accounting, runtime pin, failure history, CPU logs and source integrity. All **72 retrieved files, 1,074,831 bytes**, match their recorded sizes and SHA-256 values. Postflight verifies **483 sealed files**. No source changes, GPU execution or retries were performed during this audit.

## Exact controls and compiler compatibility

- Startup restored the pinned released state and full RNG, with fresh optimizer/counters and zero optimizer steps. All three trials then passed their initial-state and fresh-core checks.
- Disabled G matched K exactly after both accumulated backwards, before/after clipping, after native AdamW and at its final state.
- Enabled G completed two H-gradient evaluations with a nonzero accumulated H-gradient. Both K-gradient receipts, pre/post-clipping receipts and native post-AdamW state matched K exactly. Final Adam state, counters, RNG and mode metadata also matched. Final weights were finite and exactly equal to K because projection was inactive.
- Response identities, anchor positions/masks, ordinary K loss and selection/exposure metrics agree across the three trials. Enabled G adds H diagnostics; elapsed times differ.

The runtime receipt records `torch._functorch.config.donated_buffer: True → False` before startup and all trainer constructions, remaining false at completion/exit. The training environment is Torch **2.13.0+cu130**. The earlier retained-graph exception did not recur in this native run.

Comparing the old failed attempt's completed ordinary-K trial with repaired ordinary K, both response records agree exactly in every retained field except `training_s`, including anchors, losses, denominators, exposure, reported gradient norm and counters. **This supports compatibility in the observable one-step record; it is not a full old/new weight or Adam-state comparison.** The old comparison tensors were RAM-only. Numerical equivalence to historical K1's complete event is not established by these records.

Evidence: [runtime pin](raw/execution_001/qualification/RUNTIME_COMPATIBILITY.json), [qualification result](raw/execution_001/qualification/RESULT.json), [enabled native AdamW equality](raw/execution_001/qualification/G_enabled/native_post_adam.json), [enabled final nonweights](raw/execution_001/qualification/G_enabled/final_nonweights.json), [original K record](../raw/campaign_001/qualification/K/TRIAL.json).

## What the enabled step exercised

The accumulated H-gradient norm was **0.2748676244**. Its inner product with the ordinary K displacement was **−0.4361937154**, so the specified guard correctly left the native optimizer-produced parameters untouched. Proposed, projected and realized update norms were all **10.0516003904**, with zero correction norm and no rounded half-space violation.

This tests nonzero H computation, separate backward execution, FP32 accumulation, the global sign calculation and inactive-path preservation. The native active-projection/BF16-rounding branch was **not exercised** by this step; its existing CPU tests are separate evidence. No additional GPU cases were added to manufacture activation. Lower H, prefix preservation, repair progress and serving gains were not evaluated after this update.

## Work and costs

| Work in this repaired attempt | Observed |
|---|---:|
| Disposable optimizer steps | 3 completed |
| Qualification response presentations | 6 completed |
| Startup response forward/backward evaluations | 2 completed; 0 optimizer steps |
| Qualification K backwards | 6 completed |
| Enabled H backwards | 2 completed; 0 failed |
| Main training steps / presentations | 0 / 0 |
| Local mechanism probes / complete-response generations | 0 / 0 |

All budget receipts pass. Instrumentation records **8 response/draft forwards, 40 layer forwards and 1,764 vocabulary-head calls**, including recomputation. These head calls are not additional training presentations. The original failed attempt remains separate and unchanged: it completed two temporary steps, so the two attempts together used **five temporary optimizer steps**.

Qualification lasted **242.096 s**; its enclosing entry point lasted **242.120 s**. The supervisor ran **08:26:22–08:32:45 UTC**, **383 s**, including preflight and CPU checks. Nested times must not be added. Remote CPU logs report **489 regression tests and 34 focused tests passed**; these overlapping suites are not distinct totals.

Enabled G's peak allocated/reserved CUDA memory was **31,487,066,112 / 34,357,641,216 bytes**, with scope explicitly covering loading, initial checks and trial execution. K's peaks were **21,075,324,416 / 23,741,857,792 bytes**. Enabled H-term construction, H-backward and projection/accounting timers were **0.0614 / 2.4949 / 0.2678 s**. These are qualification measurements with inspection overhead; they do not establish steady-state adaptation overhead or serving performance.

**Conclusion:** the compiler repair resolves the observed retained-graph failure in the prescribed native control, and measured H-gradient extraction does not contaminate K's update. Active GPU projection and the main experiment's scientific outcome remain untested. The qualification-only entry point stopped successfully with exit code 0; no main event was launched.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_prefix_preservation_guard_20261027/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
