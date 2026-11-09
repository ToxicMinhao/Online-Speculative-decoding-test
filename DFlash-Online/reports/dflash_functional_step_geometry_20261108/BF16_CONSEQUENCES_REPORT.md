# Does native BF16 gradient sensitivity change the FSG update?

**On this retained complete update window, the numerical differences changed the raw FSG proposals and a few local prefix predictions, but did not change the guarded endpoint. Both layouts fell back to exactly the same ordinary K+ weights.** This is evidence about one applied decision, not evidence that BF16 sensitivity is harmless throughout training.

The original native BF16 path was used throughout the head and draft backward. The FP32 auxiliary-head helper from the preceding diagnostic was not used. No new Adam step, feedback, teacher preparation, serving generation or checkpoint was produced. The old failed numerical gate remains recorded; this diagnostic does not declare qualification complete.

Execution source: [`8c2dcdaa1cc6a91774811ccd36090dd1fdc7db71`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/8c2dcdaa1cc6a91774811ccd36090dd1fdc7db71), [PR #33](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/33). [Compact results](BF16_CONSEQUENCES_SUMMARY.json), [per-response prefix results](PREFIX_RESULTS.csv), [per-parameter gradient comparisons](GRADIENT_PARAMETERS.csv). ARS experiment-agent execution/reproducibility checks and same-team source, numerical and work reviews were used; these are not independent GPU replications.

## The controlled comparison

The test loaded the retained KPLUS4 pre-state and the already saved first ordinary K+ Adam endpoint. It used both original qualification responses and all **126 + 247 = 373** retained anchors. The normal projection processes eight blocks per head chunk; the alternative processes four. Full-block draft attention, saved tokens/features, anchors, model state and the FSG formula were unchanged.

Both alternatives use **the same production-eight protected-prefix membership and normalized weights**. Each layout computes its own native BF16 head logits, auxiliary gradient and output energy. The original Gram solve and BF16 candidate construction are reused. Native and both candidate endpoints are scored using one common production-eight reference/risk calculation. The existing FP32 proxy arithmetic and FP64 reductions/small solve remain unchanged; “native BF16” does not mean these established scalar reductions were converted to BF16.

Before interpreting the alternate layout, production-eight reproduced **all available retained scientific receipt fields exactly**: native endpoint statistics, both response summaries, sketch/Gram/coefficients, native risk, raw proposal statistics and fallback decision. Historical rejected raw-candidate tensors were not retained, so exact historical raw-weight identity is not claimed. Source/input identities and the loaded native endpoint are authenticated separately.

This is a full-window, eight-versus-four consequence test. It is not a rerun of the earlier two-anchor, sixteen-row-versus-thirty-two-row gradient oracle, and its smaller failure counts must not be presented as improvement of that old gate.

## There was a real numerical perturbation

| Response | Original anchors | Gradient relative L2 difference | Coordinates outside the existing tolerance |
|---|---:|---:|---:|
| First | 126 | 0.210549% | 1,021 / 1,048,626,432 |
| Second | 247 | 0.161339% | 63 / 1,048,626,432 |

The unchanged coordinate criterion still fails for both responses. These discrepancies are retained as evidence, not silently relabeled passes. They demonstrate that the two layouts did perturb the auxiliary computation; agreement of their final decisions is therefore informative for this window.

The continuous proposal difference is **0.006662% of the ordinary update norm**. After BF16 realization, **4,722 of 1,048,626,432 weight coordinates differ** between candidates, and their difference norm is **0.047098% of the ordinary update norm**. Neither proposal is identical to ordinary K+. Exact represented-weight comparisons are performed directly; they are not inferred solely from a high cosine similarity or rounded norm.

## Raw proposals versus the update that would be kept

| Measurement | Ordinary K+ endpoint | Raw FSG, eight-block chunks | Raw FSG, four-block chunks |
|---|---:|---:|---:|
| Continuous displacement norm | 1.4583258594 | 1.4559686184 | 1.4559686454 |
| BF16-realized displacement norm | **1.4583258594** | 1.4584726072 | 1.4584724864 |
| Pooled protected KL | 0.07269526344 | 0.05238979991 | 0.05256885969 |
| Net matching-prefix occurrences gained | +298 | +301 | +297 |
| Original first errors usefully repaired | 93 | 92 | 91 |
| New earlier errors | 1 | 0 | 2 |
| Old matching-prefix occurrences lost | 1 | 0 | 2 |
| Existing guard decision | Native endpoint | **Reject: norm exceeds native** | **Reject: norm exceeds native** |

The proposals reduced the continuous step norm, but BF16 rounding made each realized norm about **0.0101% larger than the ordinary update**. The unchanged norm guard therefore rejected both. Their lower measured protected KL does not override that guard. Both rejected proposals were nevertheless scored diagnostically, so the shared fallback does not conceal their differences.

Across the two responses, the raw candidates differ at **39 / 5,385 valid proposal occurrences** and by **four total matching-prefix occurrences**. There are 373 sampled blocks, 2,613 old matching-prefix occurrences and 305 original first-error opportunities. Occurrences can overlap across sampled blocks; these are not independent tokens, requests or examples of complete serving.

The eight-block candidate retained 88 of ordinary K+'s 93 useful repairs, lost five and added four. The four-block candidate retained 87, lost six and added four. Thus lower protected KL alone does not mean preserving every useful repair or improving every prefix metric.

After applying the original guard, both committed endpoints are **exactly the native K+ endpoint**, with zero differing weights and zero differing local proposals. Both inherit the native +298 prefix gain and 93 useful repairs. This committed-prefix result is obtained by selecting the already scored native state; no redundant extra forward or generation is needed.

## Supported interpretation

The answer to “does this sensitivity matter?” has two parts:

- **It affects the proposal:** gradients, rounded weights, protected KL and some prefix predictions differ. It is not purely an invisible reporting discrepancy.
- **It did not affect the kept update in this window:** the existing guard rejected both alternatives and selected the same native endpoint.

We can retain BF16 while studying these practical consequences. This test does not establish a kernel defect, certify the whole implementation, or show that future accepted FSG updates will be insensitive. In particular, identical fallbacks test agreement of this conservative decision; they do not test the stability of a useful accepted correction.

A separate issue remains visible: **the continuous FSG proposal had a smaller displacement norm than ordinary K+, but its BF16 realization failed the norm guard and was not selected in this window.** If this later happens frequently, it must be distinguished from an applied FSG mechanism failing to improve drafting. This one case cannot establish that frequency. No guard, tolerance, learning rate, strength or precision policy was changed in response to the outcome.

## Work, costs and integrity

| New work | Executed / ceiling |
|---|---:|
| Response-level forwards | **10 / 10** |
| Auxiliary full-draft backwards | **4 / 4** |
| Adam applications | **0 / 0** |
| Feedback / teacher preparation / serving generations | **0 / 0 / 0** |
| Published training/serving checkpoints | **0** |

The ten forwards comprise two retained-native forwards, four reference forwards and four raw-candidate forwards. There were **893 physical head calls**: 47 prefix-scan, 188 production proxy/recomputation, 376 alternate proxy/recomputation and 282 common-scoring calls. The **282 proxy calls** are included within those head calls, not additional work. The full retained anchors also exercise partial final chunks: six/seven blocks for production-eight and two/three for alternate-four. These are not maximum-512-anchor or maximum-sequence-length resource checks.

The supervisor ran **116 seconds** (23:53:43–23:55:39 UTC, November 8), exit 0. The runner's 99.364-second clock excludes its final dependency postcheck; the consequence callback took 39.079 seconds. Peak GPU allocation/reservation was **28.549 / 33.720 GB**. These are diagnostic costs, not measured online training overhead or serving speedup.

All **94 CPU tests passed locally and on A800**. All **1,520** source/input dependencies matched before and after execution. Scientific-state restoration is exact, including weights, Adam, counters, RNG, ordinary gradients and runtime; optimizer objects/versions are unchanged. All **20 compact files / 1,925,795 bytes** were copied to Fenrir and independently rehashed. Original failures, model states and feature packets remain preserved.

The bounded diagnostic is complete. **The main pilot remains paused and broader qualification remains incomplete.** No conclusion about held-out drafting, training-seed robustness or serving improvement follows from these training-conditioned blocks.

## Review and retention

Same-team numerical, source/narrative and work audits checked the compact results against the retained receipts. The work audit passed 4,666 assertions and independently rehashed every compact raw file. Source-archive integrity relies on the sealed/native receipts because the redundant transfer archive had already been removed. Reviewers disclose implementation authorship overlap; no independent GPU replication was performed. All 11 ARS interpretation checks were considered, with one-window, overlapping-occurrence, fallback-selection and no-serving limits retained.

At the recorded A800 postflight (23:59:11 UTC), both GPUs were idle and 58.119 GB disk space was free. Only the verified redundant 21,138,601-byte source-transfer archive was removed on each host; source remains in the published commit and authenticated A800 mirror. Exact cleanup receipts and private raw diagnostic records remain in the study directory. No scientific replay or recovery capability was removed.

## Material Passport

- Workflow: ARS experiment-agent run/validate; the bounded execution completed, and its interpretation is **ANALYZED**.
- Scope: one retained two-response window, native BF16 head chunks eight versus four, fixed production-eight protection and unchanged FSG guard.
- Evidence: retained pre/native states and anchors, original qualification receipt, new raw result/work/restoration records, compact tables and same-team audits.
- Source: execution commit `8c2dcdaa1cc6a91774811ccd36090dd1fdc7db71`; report publication is a separate documentation commit.
- Restrictions: original numerical criterion remains failed; no pilot, training campaign, new precision policy or tolerance change follows from this result.
