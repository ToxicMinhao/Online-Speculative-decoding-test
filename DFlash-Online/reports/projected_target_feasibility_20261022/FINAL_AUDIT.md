# Final stopped-run audit

**The campaign stopped correctly at a strict output mismatch; the full screen is incomplete.** This independent ARS audit reads [the final compact evidence](retrieved/final_recovery/) locally. It performs no model forward, generation, training, remote operation or source change. The [GSM8K](FIRST_GROUP_AUDIT.md) and [CodeAlpaca](SECOND_GROUP_AUDIT.md) audits remain valid; their 165 previously retrieved files are byte-identical in the final snapshot.

| Group | F completed | C completed | P completed | Group qualification |
|---|---:|---:|---:|---|
| GSM8K | 16 | 16 | 16 | Passed |
| CodeAlpaca | 16 | 16 | 16 | Passed |
| ShareGPT | 16 | 16 | 8 | Failed exact-output gate; incomplete |

The retained records confirm **136 completed generations = 16 reused original F + 120 new recovery generations**, including the completed mismatching output. Exactly eight planned ShareGPT P-B requests were not executed. All six training events completed, totaling **96 presentations and 48 optimizer steps**; the original failed preparation attempt performed zero training. No retry or enlarged budget is present. Exit status is 1. Recovery runner wall is **1,175.149 s**; adding the original attempt's **37.984 s** gives **1,213.133 s** consumed runner time. Original F serving is already inside the original wall and must not be added again. This excludes repair/development, external preflight and retrieval time.

Independent reconstruction validates all **11,730 recorded verification blocks** against each record's own output: widths, anchor positions, target-posterior commits, accepted draft counts, total progress and EOS/cap termination agree. One record fails equality with F: `sharegpt_A_07` under P. All other present records pass the corresponding prompt/output/termination checks. Thus the reporter's **135 record-valid, 96 group-qualified** distinction is correct: all 40 present ShareGPT records remain outside qualified comparisons because its group did not finish. All full-planned pooled percentages are unavailable; no ShareGPT gain is admitted by the report. The first two groups' null/harmful results remain explicit.

[MISMATCH.json](MISMATCH.json) preserves the exact saved-trace diagnosis. The ShareGPT prompt has 29 tokens; F, C and P each return 512 tokens with `finish_reason=length`. Their first **511 response tokens are identical**. Only response index **511** (zero-based; full-sequence index **540**) differs: F/C emit token **9492, ` tags`**; P emits **3847, ` users`**. The ending context is “This is particularly evident in platforms like YouTube, where …”.

| Arm | Final block start in response | Verification width | Posterior row predicting the differing token | Saved target argmax IDs |
|---|---:|---:|---:|---|
| F | 508 | 4 | 2 | `[11, 1380, 9492, 646]` |
| C | 508 | 4 | 2 | `[11, 1380, 9492, 646]` |
| P | 510 | 2 | 0 | `[3847, 646]` |

F/C reject the proposed ` users` and commit the saved target's ` tags`; P accepts ` users` because its own saved target posterior selects it. Both predictions consume the same mathematical canonical prefix through full-sequence input index539. **Commit indexing is consistent.** Different block widths/positions and cache histories can produce different numerical computations even with the same canonical token prefix. That is a plausible explanation, not a demonstrated cause: no new forwards, logits, probability margins or cache-tensor comparisons are available. BF16 ties are not established. The mismatch remains a strict failure; no precision change or output-gate tolerance was applied.

The remaining ShareGPT training evidence passes its independently checkable boundaries. Its eight A payload extracts have exact prompt/response concatenation, response-only masks, expected shapes/dtypes and recorded finite tensors. Saved native teacher replay checks **22,694 posterior rows** in **1,456 target forwards** (eight prefills, 1,440 verification blocks, eight final feature forwards); all labels match F and final feature logits do not supply labels. Only A trains C/P. Both arms use identical ordered presentations, anchors and keep masks, **14,948 supervised position exposures**, **6,665.177930334529 total decay weight**, initial/final Python/NumPy/Torch/CUDA RNG and the pinned eight-step schedule. Runtime receipts confirm released initialization, fresh optimizers, 58-tensor publication and P reference equality before/after training. These checks do not override the failed serving-output gate.

Across all groups, teacher preparation used **2,300 target forwards** and checked **35,626 posterior rows**; these are preparation costs, not additional generated answers. The reporting and target-guard limitations in the earlier audits and [cost audit](COST_AUDIT.md) still apply. ShareGPT's final group-wide native-target identity/version check occurs after both arms finish evaluation and was not reached; local frozen embedding/head checks and optimizer exclusions passed, but post-failure whole-target equality is not established. No target mutation is demonstrated. ShareGPT P has `outputs_equal=false`, no successful group qualification and no optimizer-state retirement receipt; its full restart state remains a recovery dependency. This audit does not inspect the large remote state. All 263 retrieved inventory entries match their recorded byte sizes. Stage 1 remains unchanged; no automatic continuation, new generation or silent omission of the failed case is justified by this audit.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/projected_target_feasibility_20261022/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
