# Consolidation protocol audit — projected-target screen

## Material Passport

- Origin Skill / Mode: ARS experiment-agent / validate.
- Origin Date: 2026-10-22.
- Verification Status: ANALYZED; source and saved-artifact reanalysis, without a model rerun.
- Version Label: consolidation_protocol_audit_v1.
- Scope: independent delegated review of the executed F/C/P protocol. No remote connection, model execution, scientific redesign, or source changes.

**The saved evidence supports the two qualified negative comparisons. I found no additional material implementation discrepancy in label alignment, reference freezing, matched C/P training, B isolation, or reported counts. The full three-group screen did not pass: ShareGPT has a real exact-output mismatch and eight missing P-B generations.** This audit does not certify universal native output invariance or establish the numerical cause of that mismatch.

## Evidence and source identity

Reviewed the original [handoff](../projected_target_component_20261022/HANDOFF.md), the actual [runner](runner.py), and `repair_001/online_feedback/{training,objective,projected_target,native,native_replay}.py`. Each of these six files is byte-identical to its member in [the executed source archive](repair_001/executed_source.zip), with matching SHA-256 entries in [the recovery binding receipt](retrieved/final_recovery/BINDINGS.json). Also read the archived SpecForge sampler/block mask/chunk reducer and native DFlash generation code directly. Prior audit conclusions were not used as validation inputs.

Recorded upstream revisions are SpecForge `1839aae30003e3573d00979eb1edd57347b3578c` and DFlash `07ebd93db9f472af339b644bb70221ad8428328a`; the exact executed files include the recorded compatibility/instrumentation changes. Runtime receipt: Python 3.12.15, Torch 2.13.0+cu130, Transformers 5.12.1. The executed manifest equals the sealed approved manifest after removing only the deterministic per-group `index=1..16` added by runner lines 327–330.

Locally reanalyzed all 136 saved generation JSON records, all six `TRAINING.json` receipts, 24 feedback receipts, the 24 compact feature/mask extracts, and all C/P initial/final RNG files under [final recovery](retrieved/final_recovery/). The local compact retrieval contains no feature `.pt` payloads or learned weight files; numerical model/feature equality checks are therefore supported by the executed checks and receipts, not independently rerun here.

## Protocol checks

| Item | Source and raw evidence | Finding |
|---|---|---|
| Target/label alignment | `native_replay.py:46–59, 78–106` checks prefill and every saved verification argmax, accepts only canonical committed input rows, and requires reconstructed labels to equal the exact response. `objective.py:138–148` gathers the same-position token at `anchor+offset`. | All 24 label vectors equal their F response tokens. Saved replay receipts account for 35,626 verification argmax rows. No off-by-one mismatch found. |
| Conditioning and masks | Archived `dflash_mask.py:53` permits target-feature keys strictly before the anchor; the only clean token in each draft block is its anchor. Slot zero, prompt tokens and out-of-range labels are excluded. | No future canonical label enters through target conditioning. All 24 compact raw input/mask extracts equal prompt+response and zero-prompt/one-response masks. Author sampling intentionally omits the first response token as a prediction label because the anchor and its successor must both be supervised. |
| Projection and precision | `projected_target.py:20–67` uses detached FP32 softmax, the stated water-filling formula and `c=1.1`; correct argmax rows retain their distribution. `objective.py:97–119` applies FP32 soft/hard CE with common weights. | Matches the handoff. The frozen head multiplication remains BF16 before casting, consistent with the specified BF16 model and FP32 softmax/loss. Native tie rule is first/lowest-ID logit argmax. Underflow is left at zero; no probability floor or post-normalization was introduced. |
| Immutable reference | `FrozenReference` deep-copies the pre-event draft, disables gradients and runs in eval mode outside the registered module tree. `begin_event` runs once before both passes; `end_event` runs after all backwards. RNG is preserved. | All three P `CHECKS.json` receipts record exact released-state equality at creation and after training, optimizer exclusion and disabled reference gradients. Export config has attention dropout 0.0. Targets are recomputed from the same frozen reference and current sampled anchor, rather than from the updated learner. |
| Matched learning | `run_arm` builds a fresh trainer per arm from released weights; seed is reset to 17 by `_build_runtime`. Both arms receive the same eight feature paths, same two-pass order, response mean before accumulation, and eight scheduled optimizer steps. | All six raw training receipts have 16 presentations / 8 steps and stepping pattern `[false,true] × 8`. C/P features, anchor draws, exposure and eight actual learning rates match exactly in every group, including stopped ShareGPT. Paired initial/final RNG archive members are byte-identical in all six comparisons. |
| B isolation | Runner prepares feedback only from the first eight A requests; `prepare_feedback` rejects evaluation-purpose records. Training event features and all presentation IDs are A-only. | No B feedback or benchmark assistant answers enter training. All 48 prompts are distinct; prompt messages contain only the selected user text. Selected B source rows are absent from the sealed historical exclusion list. This last statement is limited to that retained-history inventory, not a fresh scan of inaccessible history, semantic duplicates or pretraining. The eight GSM8K A reuse exceptions remain explicit. |
| Native publication | Export/reload validates keys, shapes, dtypes and semantic configuration, followed by exact equality against live trained weights; generation creates fresh target/draft caches per request. | All six activation/check receipts pass publication. Source shows no cross-request cache reuse. The target parameter guard is identity/version/gradient/optimizer based, not a repeated bitwise whole-target scan; the final group-wide guard was never reached for ShareGPT. |

Recomputed sampled supervised positions **per arm** are GSM8K 15,014, CodeAlpaca 14,914 and ShareGPT 14,948; corresponding weighted denominators sum to 6,690.117535430221, 6,661.177241408448 and 6,665.177930334529. These are presentation exposures, with repeated positions allowed, not counts of distinct response tokens. The exact schedule starts at `6.237396779379481e-4` and ends at `2.3739677937948036e-5`; it is the preserved eight-step scheduler behavior, rather than an unreported constant `6e-4` rate.

## Native outputs and reported quantities

Independently reconstructed every recorded verification block from saved anchor/proposal/posterior/commit fields. All 136 records have consistent geometry, accepted-prefix length, committed response tokens, EOS/cap status, trace length and metadata counts. `verification_calls` counts verification blocks only: it excludes the one target prefill per request and the separately charged feedback-replay forwards. `accepted_draft_tokens` counts committed accepted proposals, so it can differ from all computed proposals near EOS/cap.

| Qualified subset | F calls | C calls | P calls | P versus F | P versus C |
|---|---:|---:|---:|---:|---:|
| GSM8K A | 356 | 351 | 394 | +10.67% | +12.25% |
| CodeAlpaca A | 456 | 424 | 496 | +8.77% | +16.98% |
| Both A, 16 requests | 812 | 775 | 890 | +9.61% | +14.84% |
| GSM8K B | 303 | 367 | 350 | +15.51% | −4.63% |
| CodeAlpaca B | 598 | 656 | 637 | +6.52% | −2.90% |
| Both B, 16 requests | 901 | 1,023 | 987 | +9.54% | −3.52% |

Percentages are `100 × (arm/base − 1)`, calculated from paired sums. P versus C gains/regressions/ties are A **1/15/0**, B **11/2/3**. P versus F are A **4/11/1**, B **2/12/2**. Qualified accepted-draft totals F/C/P are A **3,436/3,474/3,357** and B **4,229/4,103/4,138**. Output lengths are identical across arms: A **4,251** tokens with **14/16 EOS**, B **5,131** tokens with **13/16 EOS**. Thus the reduction versus C is reduced held-out harm, while P remains worse than F and does not retain C's source fitting.

Counts reconcile to **136 unique executions, 96 presentations, 48 optimizer steps**: 16 F records are copied from the original failed-before-training attempt and 120 are new recovery executions. GSM8K and CodeAlpaca contribute 96 qualified records. ShareGPT contributes 40 present records, with only eight P-A records and no P-B records. The raw wall receipts sum to `37.98424434475601 + 1175.1491422411054 = 1213.1333865858614 s`; copied F records must not be charged as additional executions.

## Material limitations and wording precision

1. **The ShareGPT equality failure is preserved and genuine.** Only `sharegpt_A_07`, P, differs from F, at zero-based response index 511. F/C commit token 9492 from the target posterior of the block at output 508, width 4; P commits 3847 from the posterior of the block at output 510, width 2. All three traces follow their saved target argmax correctly. There is no trace evidence that a rejected proposal was committed. Without logits, margins or a controlled model rerun, attributing the difference to BF16, a kernel or a specific cache operation remains unproved. Whole-group exclusion from gain reporting is appropriate.
2. **`full_posterior_match` means all-row argmax-ID agreement.** `native_replay.py:90–94` compares `output.logits.argmax(-1)` to saved IDs, not complete logits or probability distributions. It is adequate for the strict greedy-label binding being checked; calling it bitwise/full-distribution posterior equality would overstate the evidence. The replay validates the saved native block/cache path, not numerical equivalence to arbitrary full-sequence teacher execution.
3. **A batch adaptation screen is the actual intervention.** Each group first produces all F A/B answers, then each learner receives one eight-response, two-pass event. This evaluates a bounded online-feedback component but does not measure a long-lived per-response deployment stream or prove preservation under repeated events.
4. **Small, incomplete and instrumented measurements.** Two groups and 16 B requests support only descriptive paired results under this recipe. Different arm losses are not comparable. Single instrumented timings, including nested projection work, do not establish speedup; do not add nested objective subcosts to total training time again. Exact optimizer continuation of five completed arms was intentionally retired; the audit does not claim their full historical training can be resumed from the compact local evidence.

## Fallacy scan and verdict

All **11/11** ARS statistical fallacy types checked. Simpson reversal: absent for the displayed A/B directions across the two qualified groups. Ecological inference: avoided by retaining paired request results. Berkson/selection and survivorship: caution because historical freshness filters and ShareGPT invalidity limit the target population. Collider: no fitted adjustment used. Base-rate neglect: not applicable to a diagnostic classifier. Regression to mean: no selection on measured F extremes; matched F/C retained. Look-elsewhere: no sweep or best-arm selection in the executed fixed screen. Forking paths: the replay compatibility repair is explicit, with original failure retained and settings unchanged. Causal overreach and reverse causality: training precedes adapted evaluation and B never supplies feedback, but this does not identify a generalizable mechanism or the cause of the ShareGPT failure.

**Verdict: the qualified negative results and execution accounting are source-consistent; the complete campaign remains failed/incomplete.** This is an independent source-and-evidence audit, not a fresh native reproduction. No model run, scientific setting change or automatic continuation is warranted by the audit itself.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/projected_target_feasibility_20261022/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
