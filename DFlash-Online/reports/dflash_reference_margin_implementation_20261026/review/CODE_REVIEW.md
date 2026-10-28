# R integration source review and reproducible compatibility check

## Material Passport

ARS experiment-agent; bounded implementation review and CPU reproducibility check.
Completed October 26, 2026. Current research state remains Stage 1 / project None.
This is a delegated, same-model-family integration review. The reviewer authored
the ranking helper and its focused tests; the objective/trainer integration was
authored separately. This is not an external review or an independent audit of
the reviewer's own helper implementation.

**Verdict: no confirmed defect found in the reviewed objective/trainer integration
for the pinned BF16 configuration.** The source review and tiny CPU results do not
qualify GPU/Flex behavior or establish learning benefit.

The comparison source is
`/home/minhao2/EAGLE_SPEC/research/dflash_prefix_ranking_one_event_20261025/worktree`.
The reviewed source is
`/home/minhao2/EAGLE_SPEC/research/dflash_reference_margin_implementation_20261026/worktree`.
The machine-readable receipt records SHA-256 hashes for both objective, trainer,
ranking and support modules, the CPU fixtures, and the upstream chunk/controller
dependencies. Paths in that receipt resolve the workspace symlink to its physical
`EAGLE_SPEC_ARCHIVE/20261021_dflash_foundation` location.

## Source findings

| Check | Source location | Finding |
| --- | --- | --- |
| Current/reference rival selection and precision | [current penalty](/home/minhao2/EAGLE_SPEC/research/dflash_reference_margin_implementation_20261026/worktree/online_feedback/objective.py:160), [reference penalty](/home/minhao2/EAGLE_SPEC/research/dflash_reference_margin_implementation_20261026/worktree/online_feedback/objective.py:226), [rival arithmetic](/home/minhao2/EAGLE_SPEC/research/dflash_reference_margin_implementation_20261026/worktree/online_feedback/ranking.py:65) | Each model independently selects its strongest non-label rival. In the pinned BF16 setup, gathered BF16 projection values become FP32 before subtraction and softplus. R applies detached-reference subtraction and pointwise ReLU before weights/reduction. |
| Reference reuse and chunking | [reference work](/home/minhao2/EAGLE_SPEC/research/dflash_reference_margin_implementation_20261026/worktree/online_feedback/objective.py:198), [aligned chunk inputs](/home/minhao2/EAGLE_SPEC/research/dflash_reference_margin_implementation_20261026/worktree/online_feedback/objective.py:264) | One existing full-block reference forward precedes the current forward. Existing chunked reference head projections supply the extra scalar values and correctness flags; R adds no draft or teacher forward for logging. The no-grad call retains per-position scalars/flags, not reference hidden activations or vocabulary logits. |
| Prefix allocation and ordinary KL | [allocation](/home/minhao2/EAGLE_SPEC/research/dflash_reference_margin_implementation_20261026/worktree/online_feedback/ranking.py:54), [KL](/home/minhao2/EAGLE_SPEC/research/dflash_reference_margin_implementation_20261026/worktree/online_feedback/objective.py:123), [position weights](/home/minhao2/EAGLE_SPEC/research/dflash_reference_margin_implementation_20261026/worktree/online_feedback/objective.py:179) | R uses P's unchanged reference-prefix coefficients. Teacher forward KL, same-position labels, supervision masks and decay weights remain unchanged. |
| Denominator and response averaging | [response denominator](/home/minhao2/EAGLE_SPEC/research/dflash_reference_margin_implementation_20261026/worktree/online_feedback/objective.py:271), [reduction adapter](/home/minhao2/EAGLE_SPEC/research/dflash_reference_margin_implementation_20261026/worktree/online_feedback/training.py:137) | The weighted denominator is summed over chunks and receives epsilon once per response. No protected-count or `1 + lambda` normalization is introduced. The existing response averaging adapter is unchanged. |
| Checkpoint-safe diagnostic counts | [additive chunk terms](/home/minhao2/EAGLE_SPEC/research/dflash_reference_margin_implementation_20261026/worktree/online_feedback/objective.py:165), [R metrics](/home/minhao2/EAGLE_SPEC/research/dflash_reference_margin_implementation_20261026/worktree/online_feedback/objective.py:290), [trainer aggregation](/home/minhao2/DFlash-Research/training/specforge/training/controller.py:427) | Counts and component numerators are returned as additive terms, not mutated counters. The controller accumulates them once before backward. Activation uses protected eligible positions; correctness regression uses all eligible reference-correct positions. Shared loss denominator and count denominators are retained. |
| Lambda-zero path and O/U/P isolation | [ranking enable flag](/home/minhao2/EAGLE_SPEC/research/dflash_reference_margin_implementation_20261026/worktree/online_feedback/objective.py:94), [forward branch](/home/minhao2/EAGLE_SPEC/research/dflash_reference_margin_implementation_20261026/worktree/online_feedback/objective.py:318), [U/P reference interface](/home/minhao2/EAGLE_SPEC/research/dflash_reference_margin_implementation_20261026/worktree/online_feedback/objective.py:192) | Zero lambda bypasses reference creation/ranking. U/P keep the prefix-only interface and original auxiliary formula; ordinary O's arithmetic remains unchanged. The exact CPU comparison below supplies bounded old/new compatibility evidence. |
| Registered parameters and optimizer target | [shared current draft](/home/minhao2/EAGLE_SPEC/research/dflash_reference_margin_implementation_20261026/worktree/online_feedback/objective.py:90), [frozen reference](/home/minhao2/EAGLE_SPEC/research/dflash_reference_margin_implementation_20261026/worktree/online_feedback/objective.py:53), [optimizer target](/home/minhao2/EAGLE_SPEC/research/dflash_reference_margin_implementation_20261026/worktree/online_feedback/training.py:264) | The wrapper retains the author's current draft parameters. The frozen reference remains outside the registered module tree. The optimizer still targets `bundle.model.draft_model`; R adds no trainable parameter or optimizer. |
| Resume identity | [R contract](/home/minhao2/EAGLE_SPEC/research/dflash_reference_margin_implementation_20261026/worktree/online_feedback/training.py:281), [restore gate](/home/minhao2/EAGLE_SPEC/research/dflash_reference_margin_implementation_20261026/worktree/online_feedback/training.py:376) | R has a separate objective name and records lambda, precision, fixed-reference/prefix rules, rival rule, pointwise ReLU and equality derivative. Full contract equality is checked before weights, optimizer or RNG restore. Existing O/U/P contract content remains unchanged. |

Metric interpretation retains the existing accumulation convention: intermediate
rows describe one presentation, while optimizer-boundary rows contain the whole
window. Sum complete boundary rows when aggregating exposure/counts; summing every
row would duplicate intermediate presentations. The component ratio metrics use
the recorded weighted response denominators; the training reduction remains
response mean. The existing teacher-head cost timer deliberately includes
checkpoint recomputation, whereas scientific counts do not.

## Saved CPU evidence

[Script](/home/minhao2/EAGLE_SPEC/research/dflash_reference_margin_implementation_20261026/review/compare_old_new.py),
[JSON receipt](/home/minhao2/EAGLE_SPEC/research/dflash_reference_margin_implementation_20261026/review/OLD_NEW_COMPATIBILITY.json),
[complete log](/home/minhao2/EAGLE_SPEC/research/dflash_reference_margin_implementation_20261026/review/OLD_NEW_COMPATIBILITY.log).

Reproduction command, from any directory on this workspace:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/minhao2/.venvs/dflash-specforge/bin/python \
  /home/minhao2/EAGLE_SPEC/research/dflash_reference_margin_implementation_20261026/review/compare_old_new.py
```

The saved run exited **0** with Torch `2.13.0+cu130` and CUDA uninitialized.
It imported the entire previous helper namespace separately, including previous
ranking/RNG helpers, then used identical tiny author weights and initial RNG for
each old/new pair. The three real-reference O/U/P pairs and two injected-prefix
U/P pairs passed exact tensor/tree equality for loss, accuracy, all numerical
model metrics, 14 parameter-gradient tensors, and final RNG. Both implementations
preserved registered parameter identities across `begin_event`, excluded
reference state from `state_dict`, and released the event reference. The script
asserts CUDA remains uninitialized throughout. All **41 recorded checks** passed.

| Fixture | Arm | Old/new identical loss | Protected positions | Active auxiliary evidence |
| --- | --- | ---: | ---: | --- |
| Real reference | O | 0.28875353932380676 | 0 | Ordinary control |
| Real reference | U | 0.28875353932380676 | 0 | Zero auxiliary |
| Real reference | P | 0.28875353932380676 | 0 | Zero auxiliary |
| Injected prefix | U | 0.9981051087379456 | 6 | Positive auxiliary numerator; gradients differ from O |
| Injected prefix | P | 0.9618932008743286 | 6 | Positive auxiliary numerator; gradients differ from O |

The injected case replaces only `_reference_prefix` with a clearly labeled
synthetic, seed-excluding mask on alternating blocks. It establishes active U/P
arithmetic-path compatibility, including nonzero auxiliary gradients. It does
not assert that the frozen reference naturally selected those six positions.
The real-reference case remains alongside it to exercise actual reference work.

This run used an eager CPU model with BF16 weights/logits and BF16 KL arithmetic,
block size 4, four anchors and projection chunks of two blocks. It performed ten
tiny backwards, **zero optimizer steps and zero generations**. Its natural
reference-prefix count was **zero in U and P**, so this receipt does not
independently establish nonempty natural reference selection. Active U/P
arithmetic is checked with the injected-mask cases; R's new arithmetic has its
separate focused tests. No optimizer-resume execution,
full event, native GPU/Flex qualification, throughput measurement or efficacy
claim is supplied by this script.

The earlier real-reference-only JSON and log are preserved as
`OLD_NEW_REAL_REFERENCE_ONLY.json` and `OLD_NEW_REAL_REFERENCE_ONLY.log`. The
expanded receipt supersedes that narrower coverage claim without relabeling its
zero-prefix fixtures as active ranking evidence.

Only files under this `review/` directory were written for this reproducibility
step. Production code, tests, historical sources and remote state were unchanged.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_reference_margin_implementation_20261026/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
