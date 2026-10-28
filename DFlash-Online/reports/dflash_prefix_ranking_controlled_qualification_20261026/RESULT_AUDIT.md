# Independent qualification-result audit

**Result: passed.** The saved evidence supports completion of the controlled initialization and active-ranking correctness checks within the authorized budget. The audit found no failed, missing or contradictory receipt among the checked boundaries. It does not establish a performance benefit or authorize main U/P training.

This is an independent local audit of the retrieved compact records, original ordinary-event receipt and sealed implementation. The model-sized tensors remain on A800: their exact comparisons were executed there and retained as detailed comparison receipts. This audit does not claim to have loaded and recomputed those tensor comparisons locally.

## Budget and completion

Recounting all 142 `WORK.jsonl` entries gives:

| Work | Started | Completed |
|---|---:|---:|
| Current training forward | 14 | 14 |
| Training backward | 14 | 14 |
| Temporary optimizer step | 7 | 7 |
| Reference training forward | 4 | 4 |
| Detached comparison forward | 4 | 4 |

There are fourteen completed presentation records and twenty-one gradient snapshots, in the exact order warmup, Oa, Ob, U0, P0, U1, P1. Each trial performs two presentations and one temporary step. The warmup is marked discarded. The run records zero generations, teacher-model forwards and main optimizer steps; no main campaign started. Checkpointed loss/head recomputation is additional internal work, separately counted in the receipts rather than mislabeled as new response presentations.

Exit code is zero. Qualification wall time is **586.165 seconds**; supervisor start-to-finish time, including preflight and CPU checks, is **643 seconds**. The remote CPU suite passed all five tests.

## Equality and sampling

All **27 exact-comparison receipts** have zero missing keys, metadata differences or tensor differences. Seven initial-state receipts match the pinned released/fresh state. Ob, U0 and P0 additionally match Oa at initialization, after each backward, immediately before the optimizer, after the optimizer, and in numerical presentation records. Both active arms' final full RNG states match Oa.

All seven logged full 128-response permutations exactly match the original ordinary event. The two executed responses are `train_00072` and `train_00121`; every trial's saved anchors, keep masks, pass identity and step counters match the original records. The local original-event file's SHA-256 matches the qualification's pinned receipt digest. Oa, Ob, U0 and P0 also reproduce the original first-window scalar metrics exactly. No reduced two-item shuffle was substituted.

Every trial records unchanged frozen target-table versions/layouts. All reviewed diagnostic source files match their execution seal. The final remote retention audit reports unchanged hashes for **105 production files and 132 retained O artifacts**, with no qualification/dependency mutation.

## Active ranking checks

U1 and P1 produce the same fixed-reference masks and weighted coefficients:

| Presentation | Sampled blocks | Supervised proposal positions | Protected positions | Weighted rho |
|---|---:|---:|---:|---:|
| 1 | 363 | 5,340 | 2,710 | 0.6379002333 |
| 2 | 411 | 6,060 | 2,526 | 0.5540060401 |

Rho is weighted mass, so it is not the unweighted protected-position fraction. These are the two qualification responses, not a full-event exposure estimate.

All four independent prefix-oracle comparisons have zero mismatches. Both active arms complete two prefix checks and two rival-gradient checks; together they check **32 detached logit rows** across the full 151,936-token vocabulary. Native BF16 projection followed by FP32 differences gives a maximum gradient discrepancy of zero on these bounded samples, under the existing declared component tolerances. The two matched-mode full-block comparisons also report exact hidden-state equality. These checks verify implementation properties; they are not evidence that the additional ranking loss will improve subsequent drafting.

## Runtime, memory and retained evidence

The actual training process records **Torch 2.13.0+cu130 on NVIDIA A800 80GB PCIe**, unchanged BF16 AdamW settings, LR `3e-4`, and weight decay `0.01`. The preflight's Torch 2.9.1 record belongs to the separately pinned native serving runtime, which was not used for generation in this check.

Maximum CUDA allocation across trials is **23,250,440,192 bytes (23.250 GB)**; maximum reserved memory is **26,520,584,192 bytes (26.521 GB)**. Loading and failure-aware peak accounting are included. Qualification snapshots synchronize and copy substantial state, so these trial durations are not estimates of ordinary training throughput or serving latency.

The final retention receipt records **102,768,626,838 bytes** of diagnostic artifacts and **216,475,389,952 bytes** free, above the required reserve. Both A800 GPUs are idle with no listed GPU process. No raw checkpoint deletion or additional execution was performed by this audit.

## What this resolves—and what remains open

The controlled warm initialization reproduces ordinary behavior exactly for this one-step window, and U/P at lambda zero preserve it. Active lambda-one mask, gradient, reference-isolation and memory checks pass. The earlier failed equality gate remains part of the evidence; it has not been relabeled as passed.

This result does **not** prove equivalence to the retained O checkpoint's full 128-step trajectory, identify a particular kernel as the earlier discrepancy's cause, or measure U/P output preservation or serving gains. A future comparison that reuses that historical O artifact must retain this limitation. Main U/P training and evaluation remain unstarted.

Machine-readable audit: [RESULT_AUDIT.json](RESULT_AUDIT.json), containing 78 passing checks, recounted work, per-trial costs and explicit interpretation flags.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_prefix_ranking_controlled_qualification_20261026/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
