## Material Passport

Task: O/C/K/A implementation and correctness review. ARS workflow: experiment-agent, code implementation and CPU validation. Date: October 26, 2026. Status: CPU implementation complete; GPU qualification and scientific execution have not started. This document reports implementation evidence, not algorithm efficacy. Project stage remains unchanged.

## What was implemented

| Arm | Objective name | Loss | Supervised positions |
|---|---|---|---|
| O, retained | `osd` | Qualified forward KL | All existing valid proposals |
| C, new | `osd_ce` | Canonical-token CE | All existing valid proposals |
| K, new | `osd_first_error` | Qualified forward KL | Current matching prefix and its first mismatch |
| A, new | `osd_ce_first_error` | Canonical-token CE | Current matching prefix and its first mismatch |

All arms use the existing `online_feedback` trainer and SpecForge draft forward. The new mechanism changes the token loss and/or its selection mask. It adds no reference model, auxiliary penalty, optimizer, attention mask, or tunable coefficient. Existing O/U/P/R branches and artifacts remain intact.

For K/A, each training chunk obtains argmax predictions from its current logits. A detached mask selects valid proposal positions before and including the first canonical-token mismatch independently in each block. The known seed is excluded. Canonical EOS is included when eligible, then the block ends; padding, sequence boundaries and invalid gaps cannot reopen a prefix. An incorrect first proposal still receives loss. Selection is recomputed for each presentation; no feedback from evaluation is used.

The full block is forwarded with the original attention. Only direct prediction-loss terms are selected; excluded positions can still participate in context and shared hidden-state computation. Thus, zero direct logit gradient is not a claim that all internal gradients involving those positions vanish.

CE is `-log_softmax(current_logits)[canonical_token]`, using the same qualified BF16 projection and probability arithmetic as ordinary OSD. It does not call the legacy float32 `hard_ce` path. K retains the original frozen-teacher distribution and predecessor-feature alignment. C/A still receive the same target features for draft conditioning, but do not need the teacher vocabulary projection for their CE loss.

For each response, all sampled-block weighted numerators and selected-weight sums are accumulated before division by `selected_weight_sum + 1e-6`. Decay weights, response averaging, gradient accumulation and optimizer behavior remain unchanged. With all proposals selected, K recovers O and A recovers C on the tested loss/gradient fixtures.

## Review and CPU evidence

- **455 CPU tests passed on both Fenrir and A800**, including **32 new tests**. No failures or skips; CUDA remained uninitialized on both. Full evidence: `CPU_TESTS.json`, `CPU_TESTS.log` and `a800_cpu/CPU_TESTS.json`.
- **72 archived/current compatibility checks passed** across O/U/P/R and the earlier hard-CE/projected objectives, including nonempty auxiliary-loss fixtures. Losses, accuracies, metrics, gradients, RNG and registered parameter identities matched. These are tiny CPU fixtures, not a claim of a new 128-step GPU equivalence test. Evidence: `review/OLD_NEW_COMPATIBILITY.json`.
- New tests cover first/last/no error, EOS, padding, changing predictions, detached masks, exact BF16 CE/KL oracles, all-selected equivalence, zero excluded-position direct gradients, full-block attention, fresh optimizer/state, retained schedules, response averaging and checkpoint-recomputation accounting.
- Campaign fixtures exercise exactly 768 mock backend executions, rotating order, the four requested comparisons, exact saved token/termination checks, retained O count reproduction, failure retention and historical-only F accounting. Mock executions are CPU unit-test work, not scientific generations.
- Independent review found one report-wrapper defect: a rejected reuse of an existing output directory could still rewrite its report in `finally`. The wrapper now rejects an existing directory before entering that block. A regression test checks that every retained artifact remains byte-identical.

The initial broad suite passed 442 tests before the campaign tests were added. The final 455-test suite supersedes that intermediate count. A first reviewer-side test command used an incomplete local environment and failed three imports; the qualified environment subsequently passed all tests. No dependencies or scientific settings were changed to obtain these results.

CPU eager fixtures cannot establish GPU/Flex equivalence, memory feasibility, exact native outputs, or useful adaptation. Those remain qualification and experiment questions.

## Code and logging

The review diff is `IMPLEMENTATION.diff`; `CHANGED_FILES.json` lists its files and source identities. Core changes are confined to:

- `online_feedback/first_error.py`: detached current-prefix selection and additive diagnostics.
- `online_feedback/objective.py`: BF16 canonical CE and selected forward-KL/CE reduction.
- `online_feedback/training.py`: new objective contracts and exposure accounting through the existing trainer.

The small `experiments/first_error/campaign.py` module prepares the C/K/A training and O/C/K/A evaluation components by reusing the existing startup, trainer and evaluator. Two backward-compatible evaluator parameters allow retained F to remain outside fresh timing arms. Existing evaluator defaults still require F/O as before. The diff contains seven production files (330 added / 15 removed lines) and four test files (824 added lines); the core objective/trainer portion adds 168 lines and removes 11. No upstream training or inference code was changed.

The implementation is installed at `/data/sources/minhao/DFlash-Online` on A800. The 110-file preceding source seal was checked before installation; 106 unchanged entries still match and the four modified existing files have verified backups under `/data/sources/minhao/runs/dflash_first_error_implementation_20261026/previous_source`. New files are listed in `CHANGED_FILES.json`. This task did not edit experiment artifact directories or Fenrir's clean `/home/minhao2/DFlash-Research` source. Installation evidence is in `a800_cpu/INSTALL.json`. No cleanup was needed.

Per-presentation records contain available and selected occurrence counts, their weighted sums, first-error histograms at proposal offsets 1–15, no-error/empty block counts, loss numerators and denominators. The event sums **only completed accumulation-window rows**, which already include both response presentations. Checkpoint recomputation returns the same metrics without mutating exposure counters. Its actual additional computation remains in cost timers. Selection timers, training, export/persistence, initialization/loading and peak memory are recorded separately; nested timers must not be added twice.

## Prepared campaign and interpretation

`CAMPAIGN_PLAN.json` pins the retained input files and checks their accounting: 128 feedback features, the original 256 presentations/anchor draws across two passes, and the existing 64-request panel. O is reused. Each of C/K/A starts from released DFlash with fresh optimizer and seed 17, taking 128 steps: **384 new main optimizer steps**. No new feedback is collected.

Preserved settings include constant LR `3e-4`, BF16 AdamW and probability arithmetic, weight decay `0.01`, response batch 1 with accumulation 2, shuffled passes, up to 512 sampled anchors per response, decay gamma 7, eight-block loss chunks, Flex attention and maximum training length 2,272. Serving remains the qualified Qwen3-8B/DFlash-b16 pair, native SGLang/FA3, greedy block size 16, 2,048 tokens or EOS, context limit 4,096 and the original cache policy.

Fresh evaluation is **O/C/K/A × 64 requests × 3 repeats = 768 generations**, capture disabled, after every trainer has closed. The first eight requests' native correctness gate is inside that total. All outputs and termination must equal retained F; fresh O must reproduce its retained verification counts. Invalid measurements remain visible and cannot enter gain claims. The four comparisons are:

- **K versus O; A versus C:** effect of current first-error position selection under each loss.
- **C versus O; A versus K:** effect of canonical CE versus forward KL under each position rule.

Retained F supplies canonical outputs and historical verification counts only. There is no fresh F latency comparison. Serving results describe performance with a checkpoint available; training and activation costs remain separate.

If selection improves both K/O and A/C, that supports its usefulness across these two losses in this one-event screen. Improvement in only one pair indicates an interaction with loss choice. C/O improvement without A/C improvement would favor the simpler full-position CE explanation. If none improves ordinary OSD, this screen provides no positive evidence for the proposed direction. No outcome is known yet, and no mathematical guarantee or broad online-serving conclusion follows from the implementation.

## Proposed next qualification, not executed

A bounded proposal is seven temporary optimizer windows: one discarded ordinary warm-up, ordinary O, C, all-selected K, all-selected A, natural K and natural A. At accumulation 2 this is 14 response presentations and seven discarded optimizer steps, with zero reference forwards, feedback generations or serving generations. Tensor-level mask and gradient assertions use existing forwards or small fixtures; any additional model work must be counted explicitly.

Use identical pinned initial state, RNG and original anchors for matched checks. Check ordinary compatibility; all-selected K against O and all-selected A against C; active masks and BF16 CE arithmetic; gradients, Adam state and counters; full attention, fresh-state isolation and memory. Keep qualification separate from efficacy and discard its states. This is a proposed qualification budget for review, not a passed gate or execution authorization.

The delivery contains callable campaign components, not an automatically launching native qualification driver. Before GPU work, bind that driver to the reviewed source, model/runtime/configuration and retained feature/schedule identities, enforce the sealed 128/64 scope, and verify actual storage admission. An old U/P qualification cannot qualify C/K/A. No main launch occurs in this CPU-only delivery.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_first_error_implementation_20261026/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
