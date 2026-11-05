# ShareGPT mismatch: frozen-target numerical-path check

ARS bounded diagnostic, October 22, 2026. User requested a recheck of whether
draft-dependent execution shapes can explain the output difference without a
speculative-verification defect. The historical campaign remains paused.

**The same unchanged target reproduces both historical outcomes.** At the
disputed final prediction, the F schedule gives `tags` an advantage of one BF16
representable step (0.125); the P schedule makes `tags` and `users` tie. Native argmax then selects
the lower-ID `users` token. Execution-path numerical sensitivity and the BF16
output tie are now directly observed; the particular kernel or earlier operation
responsible for the perturbation is not isolated.

## What was run

One freshly loaded pinned Qwen3-8B target, using the original BF16/SDPA runtime.
No draft model was loaded. No optimizer, training, checkpoint publication, new
answer generation or campaign continuation occurred. Original F/P inputs,
proposals, positions and cache cropping were reconstructed through the existing
`online_feedback.native_replay.replay_frozen_target` helper. All recorded target
argmax rows had to match; no output or proposal was replaced.

After each path, the final verification call was repeated once from an exact copy
of its pre-block cache. No precision, tolerance or scientific setting changed.
FP32 conversion was used only to subtract and summarize the already computed
BF16 logits, not to recompute model predictions.

| Saved execution path | ` tags` (ID 9492) | ` users` (ID 3847) | Native choice | Final-call repeat |
|---|---:|---:|---|---|
| Frozen draft F | 25.125 | 25.000 | ` tags` | Entire logit vector bitwise equal |
| Projected draft P | 25.125 | 25.125 | ` users` | Entire logit vector bitwise equal |

The first 511 answer tokens and the prompt are identical. F's last verification
block starts at output position 508 with width four; P's starts at 510 with width
two. Both predict the disputed token from full input position 539. One BF16
spacing at this logit magnitude is 0.125. P's two maximum logits tie exactly;
`torch.argmax` selects ID 3847. All final logits are finite. This is reproducible
within the checked paths, not evidence of random corruption.

F reconstruction matches all **3,602** historical verification argmax rows; P
matches all **3,292**. Both regenerate the saved canonical labels exactly. Target
parameter identities, version counters and disabled-gradient flags are unchanged
through both paths. These guards are not a repeated bitwise whole-weight scan,
but the diagnostic uses one target object and never loads or updates a learner.

Independent source/trace audit also reconstructs all 628 original F/C/P blocks:
accepted prefixes, correction tokens, cap handling and logical cache contents
are consistent. F/C reject proposed `users` and commit their target's `tags`;
P accepts `users` because its own target prediction agrees. No rejected-token
commit or offset error was found.

## Interpretation and limits

The evidence supports this chain:

**Updated draft proposals → different verification boundaries/cache computation
history → different finite-precision target logits → BF16 tie → different greedy
token.**

This distinction is recognized in primary implementation documentation:
[PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.9/notes/numerical_accuracy.html#batched-computations-or-slice-computations)
explains that mathematically identical batched/sliced computations need not be
bitwise identical, while [vLLM speculative decoding guarantees](https://docs.vllm.ai/en/v0.21.0/features/speculative_decoding/#lossless-guarantees-of-speculative-decoding)
separate algorithmic correctness from numerical output stability. Those sources
describe the general phenomenon; the experiment above supplies the local evidence.

The earlier strict output-equality gate genuinely failed, but that failure alone
should not have been treated as evidence that speculative decoding was broken.
The present diagnostic supplies a stronger classification: **reproduced
execution-path sensitivity with a measured BF16 tie, without an identified
verification/commit defect.** It also shows target-weight modification is not
required to reproduce the historical discrepancy.

This does not isolate final block width from accumulated numerical cache
differences, identify a particular low-level kernel, establish exact-real or
higher-precision logits, or certify all possible inputs. No precision sweep or
cache-crossing experiment was added. The previous negative drafting results on
the two qualified groups remain unchanged. The ShareGPT pair still differs under
the original exact-output definition; historical reports are not retroactively
rewritten and the remaining eight requests were not executed. Any future change
in mismatch handling must be explicit rather than silently reporting this pair
as identical-output acceleration.

## Work, cost and retained evidence

Exactly **443 target forwards**: 437 saved verification blocks, two prefills,
two final-feature calls required by the existing replay helper and two identical
final-block repeats. Zero new generated answers and zero training steps. Exit 0;
wall time **27.464 s**, including **7.886 s** loading; peak allocated GPU memory
**16,846,258,176 bytes**. These are diagnostic costs, not serving benchmarks.

A800 storage admission reserved 128 MiB plus 8 GiB; free space before work was
13,475,131,392 bytes. No artifact deletion or model download was required.

- [Exact result and logits summary](RESULT.json)
- [Diagnostic implementation](check.py)
- [Plan and storage admission](PLAN.json)
- [Both final native logit vectors](final_native_logits.pt)
- [Runtime log](run.log), [exit code](EXIT_CODE)
- [Original mismatch](../MISMATCH.json)

Canonical diagnostic directory on A800:
`/data/sources/minhao/runs/projected_target_feasibility_20261022/numerical_check_001/`.
Production DFlash-Online code, learned checkpoints and the original campaign
artifacts were not modified.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/projected_target_feasibility_20261022/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
