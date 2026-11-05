# Independent final audit: prefix-preservation G screen

**Passed with no missing evidence or accounting discrepancy. The guard made no correction in any of the 128 steps, and the final G export is byte-identical to retained K1. This run therefore demonstrates no improvement from the guard.**

The independent, CPU-only [auditor](audit.py) reconstructed the retained measurements and verified all 1,357 retrieved files. Its complete checks and recomputed values are in [FINAL_AUDIT.json](FINAL_AUDIT.json). No training, model forward, generation or source modification was performed during this audit.

## Execution and validity

- The original 128 feedback responses supplied two passes, 256 presentations and 128 optimizer steps. All 65,096 sampled anchors match the sealed ordinary-event schedule. The available proposal exposure is 949,560 occurrences; selection remains the unchanged K objective.
- All 256 H backwards and 256 K backwards completed. The three prescribed measurement steps produced exactly 18 response-level draft-only probes, with 729 probe vocabulary-projection chunks. These probes were not training examples or serving generations.
- All 384 K/G evaluations completed, with zero invalid, unknown or unattempted executions. Every output and termination matched the retained canonical reference. The audit reconstructed 18,162 native verification blocks across all repeats and checked repeat-consistent counts, request ordering, disabled capture and publication accounting.
- The previous successful qualification was reused. The main run added no temporary optimizer steps: its separate startup used two forward/backward response evaluations with zero Adam updates and verified restoration. Five historical temporary steps remain accounted for across the original failed and repaired qualification attempts.
- Preflight and postflight confirm the 542-file seal. The original failed qualification and the compiler repair evidence remain preserved. The active projection branch was not exercised on GPU; CPU branch tests remain its direct implementation evidence.

## Native result and checkpoint identity

| Measurement, one validated trajectory per request | K1 | G |
| --- | ---: | ---: |
| Verification calls | 3,027 | 3,027 |
| Accepted draft tokens | 16,513 | 16,513 |
| Committed progress | 19,483 | 19,483 |
| Output tokens | 19,547 | 19,547 |
| Sum of request means over three repeats | 52.179745 s | 52.149549 s |

There are 64 call-count ties, zero improvements and zero regressions. Pooled call reduction is **0%**. The **0.0579%** request-time difference is descriptive measurement variation, not evidence of a guard speedup.

The [final export comparison](raw/FINAL_WEIGHT_COMPARISON.json) records identical 2,097,259,104-byte safetensors files with SHA-256 `e4746e0b41b6a9201036cb65a85de6b4c6331958e470a9c736c8e62976fc1086`. This establishes equal final exported weights, including under the repaired compiler policy. It does not establish equality of every intermediate gradient or optimizer state.

## What the guard and local probes show

Every native directional derivative was negative, ranging from −0.4361937154 to −0.0038118967. Thus all 128 ordinary K updates passed the specified first-order condition unchanged. All correction norms are zero, and there are no realized BF16 half-space violations. This is **no applied intervention**, not evidence that the projection successfully protected prefixes.

The ordinary, projected and norm-scaled candidates have identical recorded outcomes at each probe step; the norm-scaled control has scale 1 and matching realized norms. The following counts use each step's sampled blocks once, not three times:

| Step | Mean H before → after | Earlier-error blocks / sampled blocks | First errors repaired with preceding prefix intact | Net matching-prefix change |
| --- | --- | ---: | ---: | ---: |
| 1 | 0.063069 → 0.021695 | 47 / 774 | 527 / 632 old first errors | +1,740 |
| 64 | 0.066645 → 0.056188 | 2 / 748 | 70 / 650 old first errors | +201 |
| 128 | 0.043910 → 0.036832 | 1 / 396 | 66 / 292 old first errors | +178 |

These [local probe records](analysis/local_probe_summary.csv) show that aggregate H can improve while some previously correct prefixes lose tokens. A negative aggregate first-order derivative did not trigger protection against those local errors. The probes use training conditioning and overlapping sampled anchors; they are neither independent requests nor evidence of subsequent-request transfer. Because all candidates coincide, this run provides no directional-versus-shrinkage comparison.

## Cost and interpretation limits

The complete event took 610.048 s, including 577.763 s training, 17.561 s checkpoint saving and 12.079 s export. Nested within training are 197.631 s H backward work, 34.386 s guard decision/projection work and 10.280 s probes. These costs must not be added to the event total again. Peak allocated/reserved GPU memory was 38.435/41.970 GB. No freshly matched K training run was executed, so these component costs do not establish a total training-overhead percentage versus K.

Evaluation activation took 264.623 s across 193 publications, separate from request latency; model loading and historical qualification are also separately recorded. Campaign orchestration reports 1,280.990 s, while the enclosing supervisor including checks ran 22 min 21 s. Reusing feedback means zero new collection/preparation work, not zero historical feedback cost.

The original 64-request development panel is held out from training but has been used in development. The supported conclusion is narrow: **this aggregate guard did not alter the learning trajectory's final weights or improve native drafting in this screen, despite observable local prefix damage in the probes.** No retuning, additional event or mechanism revision is implied or executed.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_prefix_preservation_guard_20261027/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
