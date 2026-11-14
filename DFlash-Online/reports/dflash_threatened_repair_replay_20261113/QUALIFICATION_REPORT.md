# Threatened-repair replay: first native qualification and repair

## Material Passport

- Status: **incomplete; scientific experiment not started**.
- Executed source: `fdbf319b2e68e166c53a8304ad26d0c4fa500e29`.
- Repaired source: `98d9a59f9bc63cabdb830ff9a36cbec3456c3f73` (CPU-tested; not rerun on GPU).
- Token-free staging envelope: `7b33b70` (metadata only).
- Scope: one fixed native KPLUS6→7 trajectory with sixteen disposable comparisons; no new complete-response generation.
- User approved a scoped MIR reproduction exception for this pilot. Source review is not an author-supported MIR reproduction.

The first native attempt stopped because the new replay adapter omitted a required teacher tensor. This is an implementation integration error, not a failed research mechanism. No native scientific update, U/T comparison, independent probe result or efficacy estimate exists.

## What passed and what stopped

| Check | Outcome |
|---|---|
| Source, retained inputs, storage and runtime admission | Passed |
| Native fixed-prefix reconstruction | Passed, including all fifteen retained proposals at four selected starts |
| Independent memory observation using identical original layouts | Passed on two response layouts |
| Ordinary native qualification step | Passed |
| Disabled replay | Exact native parity passed |
| Coefficient-zero direct path | Exact native parity passed |
| Padded-zero replay | Failed before Adam on missing teacher field |
| Active U and T qualification | Unattempted |
| Restoration of original KPLUS6 state | Exact weights, Adam, RNG, pending gradients and runtime restoration passed at step 768 / presentation 1536 |
| Main trajectory and sixteen comparisons | Unattempted |

The failing nested case still says `running` in its original record because the exception preceded its case-finalization line. The outer qualification, failed physical operation, runner and campaign correctly record failure. The publication labels the case failed on that evidence and preserves the original record unchanged.

## Cause and narrow repair

Saved feature packets contain `target_last_hidden_states`. The shared read-only `prepare_batch` helper uses the upstream DFlash normalizer, which returns `input_ids`, `hidden_states` and `loss_mask`; it deliberately does not carry the teacher's final hidden states. Ordinary online OSD training explicitly carries the raw teacher tensor into its model inputs.

The replay implementation incorrectly tried to read `batch["target_last_hidden_states"]` after normalization. N, disabled and coefficient-zero paths did not need that field, which explains why they passed. The first padded replay encountered the missing field before teacher projection or its optimizer application.

The repair uses the existing saved teacher tensor, detached and moved to the draft device, while retaining the native predecessor-row indices. It changes neither teacher distributions nor token alignment, precision, KL direction, normalization, anchors, coefficient, clipping or optimizer semantics. The original read-only helper and ordinary K+ path remain unchanged.

A regression fixture must exercise the real feature normalizer rather than mocking a batch that already includes the teacher tensor. All **121 focused CPU tests pass**, including two new tests using the real normalizer, teacher-row alignment and active/padded-zero replay. The separate reviewer independently ran 22 replay/training tests, reconciled 12 work-count comparisons and verified the incomplete report contains no efficacy claim. Reviews are from separate agents of the same model family, with implementation authorship overlap disclosed; they are not cross-model reviews or another GPU reproduction. Passing CPU fixtures does not establish GPU qualification.

## Actual work and retained evidence

| Training-side category | Completed response forwards | Completed response backwards | Adam steps |
|---|---:|---:|---:|
| Disposable startup | 2 | 2 | 0 |
| Memory-observation qualification | 2 | 0 | 0 |
| N / disabled / coefficient-zero | 6 | 6 | 3 |
| Ordinary source pair preceding the failed padded replay | 2 | 2 | 0 |
| Total completed | **12** | **10** | **3** |

An additional replay forward was attempted and failed after draft computation: **13 attempted forwards, 12 completed, one failed; zero unknown work**. Physical hooks recorded13 draft forwards,65 layer forwards,504 objective chunks and1,059 vocabulary-head calls. These are different units and include checkpoint-related execution; they must not be summed as independent responses. The qualification summary's six native presentations counts completed pair receipts only; eight ordinary qualification response presentations actually executed.

The separate native gate recorded 119 draft blocks,117 target calls,178 context forwards and two model loads, with no complete-response generation. Total occupied GPU time was 185.986740 s (3.100min); campaign wall time 255.610137 s (4.260min). Phase timings are nested, not additive. Maximum recorded allocated/reserved training memory among completed cases was 15.855 / 18.581 GB; no claim is made that this covers the unrecorded peak of the failed case or the full scientific workload.

Both cards were idle afterward; A800 had 503.217 GB disk free. All 100 compact run files (4,732,312bytes) were retrieved and hash-verified on Fenrir. Original failure, qualification, raw ledgers, plan, source identities and private native traces remain intact. No checkpoints or historical evidence were deleted. The automatic independent reducer verified that the incomplete report contains no efficacy estimate; it did not audit a completed experiment.

## Retry boundary and estimate

Three of the handoff's six disposable qualification optimizer steps have been consumed. The native reference endpoint and full gradient tensors were temporary in-memory objects and were not persisted. Recreating a reference plus the three remaining cases requires at least four new steps: seven qualification steps total and a worst-case 169 physical steps including all 162 scientific/duplicate applications. An unchanged fresh six-case qualification requires 171 total physical applications including this failed attempt, exceeding the original 168 ceiling by three.

At this report update, no retry has been launched and no step budget has been silently reset. The authorized next action is the unchanged six-case qualification after this integration fix, with the failed three steps counted separately and an explicit total ceiling of 171. The three occupied-GPU-hour cap would continue to include the 185.986740 seconds already consumed. This is a qualification-budget amendment, not an algorithm or experimental-setting revision. The user subsequently approved “Rerun qualification and proceed if it passes.” The revised controller must authenticate the old receipts and enforce the remaining cumulative budget before restarting in a separate destination. The original failed attempt and this report remain retained.

If qualification passes, the original complete scientific panel would follow unchanged. The provisional remaining execution estimate is 1–2 GPU hours plus evidence retrieval/review, with no guarantee from the short fixture. No scientific conclusion about threatened replay, selection quality or transfer is currently supported.

## Decision

The adapter bug is concrete and repairable. Disabled/zero neutrality and exact restoration held; active sparse replay and the scientific comparisons remain unqualified. This attempt weakens no previous scientific explanation. The next decisive observation is a passing active replay qualification under the unchanged mechanism; results about transfer require the full prescribed comparison.
