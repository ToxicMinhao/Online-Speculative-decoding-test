# AdaFlash clipping-location diagnostic

## Material Passport

- Status: completed bounded execution; descriptive analysis, not an endpoint replication.
- Scientific execution commit: `64cb56509d1c4ed3bd2ceb8365a346603edb2957`.
- Six fixed saved responses in three pairs; two independent one-step branches per pair.
- Six disposable AdamW steps; no complete-response generation or serving checkpoint.

## Main finding

**Clipping location materially changes the learned update in these three fixed batches. It is not just a small numerical rearrangement.** The two rules have identical pre-update teacher CE, raw reverse KL, sampled inputs and predictions. Nevertheless, the mixed-gradient difference is 58.25–105.93% of the paper-arm gradient norm, and the actual BF16 update difference is 73.56–79.48% of the paper-arm update norm.

The two update norms are nearly equal (10.084–10.102), while their cosine similarity is only 0.685–0.730. After one step, **10.74–12.18% of trainable parameter coordinates differ between the branches**. The relative update-difference percentage is a vector-distance measure; it does not mean that 74–79% of all model weights changed.

**Local usefulness is mixed.** Repository-rule clipping gives more matching-prefix gain in pair 1 and less in pairs 2 and 3. This diagnostic neither favors the official rule consistently nor establishes that its full-training endpoint would match the existing paper-loss arm.

## Paired changes

| Pair | Gradient cosine | Gradient difference / paper norm | BF16 displacement cosine | Displacement difference / paper norm | Prefix gain: paper / repo |
|---|---:|---:|---:|---:|---:|
| 1 | 0.837657 | 105.93% | 0.684579 | 79.48% | +2298 / +2452 |
| 2 | 0.888400 | 78.13% | 0.716601 | 75.29% | +1581 / +1475 |
| 3 | 0.909673 | 58.25% | 0.729608 | 73.56% | +1200 / +1005 |

The displacement comparison uses the actual rounded BF16 weights after native AdamW. Prefix counts are occurrences across overlapping sampled blocks, not independent output tokens or native verification calls.

## Scope and arithmetic

Both arms share BF16 projection, log-softmax/exp, teacher-argmax CE and mixture arithmetic. The paper arm caps individual vocabulary contributions. The repository-rule arm caps whole-position reverse KL after normalization by the full response denominator, using FP32 cap/rescaling and the common BF16 mixture. This tests clipping placement under common arithmetic; it is not a bitwise reproduction of the official kernels.

The teacher features were reconstructed from exact saved native verification inputs. Full posterior IDs and committed tokens were checked. Retired original feature tensors cannot be compared byte for byte.

These are source-fitting and update-sensitivity measurements. They do not establish which loss wins after 58 events, held-out transfer, native latency, or the cause of the prior A58 result.

## Costs and stop

Total wrapper time: 281.863 s. Preparation: 30.337 s. Diagnostic: 224.354 s. Startup, reconstruction, forward/chunk/recomputation and step counts are retained in SUMMARY.json and raw receipts.
No additional event, tuning or serving run was started.

## What the clipping changes

All values below use the same six pre-update response presentations, each once. Means are response means, matching the training reduction; position counts are overlapping sampled occurrences.

| Pre-update quantity | Paper-entry rule | Repository-position rule |
|---|---:|---:|
| Mean teacher-argmax CE | 1.355057 | 1.355057 |
| Mean effective clipped RKL | 0.308809 | 6.277644 |
| Mean complete mixed loss | 1.145758 | 2.339189 |
| Positions affected by the respective cap | 21,420 / 25,155 (85.15%) | 6,333 / 25,155 (25.18%) |

Paper clipping caps 690,155 / 3,821,950,080 vocabulary-entry occurrences (0.01806%). The two cap rates above describe different operations: any capped entry within a position versus a capped whole-position sum. They are not interchangeable clipping percentages. Whole-response denominators range from 765.715 to 2,695.223.

Clipping therefore changes the effective RKL contribution despite both arms retaining the nominal 0.8 CE / 0.2 RKL mixture. Comparing their own numerical loss values as a measure of which draft is better would be invalid.

## Same-response fitting

| Pooled source-fit quantity | Paper-entry rule | Repository-position rule |
|---|---:|---:|
| Blocks / valid supervised occurrences | 1,719 / 25,155 | 1,719 / 25,155 |
| Pre-update matching-prefix occurrences | 10,634 | 10,634 |
| Post-update matching-prefix occurrences | 15,713 | 15,566 |
| Net matching-prefix gain | +5,079 | +4,932 |
| First errors repaired with preceding prefix intact | 1,131 / 1,482 | 1,071 / 1,482 |
| Blocks developing an earlier error | 163 / 1,719 | 136 / 1,719 |
| Old-prefix occurrences lost | 589 / 10,634 | 455 / 10,634 |

Repository-rule clipping loses fewer old-prefix occurrences overall but also repairs fewer old first errors and gains 147 fewer prefix occurrences in total. The six individual responses remain visible in SOURCE_PREFIXES.csv. The fixed first-six selection, one seed, overlapping blocks and source-fit conditioning do not support a population-level significance claim.

## Validity, accounting and cost

- All six target reconstructions passed: six prefills, 297 verification forwards and 4,752 checked posterior positions, plus six terminal-feature forwards. No new answer was generated.
- The first pair's paper-arm losses, component numerators/denominators, exposure and native pre-clip gradient norm exactly reproduce the corresponding common logged A58 first-update metrics. This is additional initialization/reconstruction evidence, not bytewise equality of retired features or a full 58-event equivalence test.
- All six branches exactly restored the retained released/fresh-Adam state. Paired pre-update block predictions/labels/weights and final RNG states matched. The retained initial artifact was unchanged afterward. Full historical feature bytes were unavailable and are not claimed recovered identically.
- Exactly six Adam steps and twelve training backwards completed. Core draft forwards were 24: twelve training and twelve detached post-step evaluations. Disposable startup added two forwards/backwards and no Adam steps. Observed totals were 26 draft forwards, 130 layer calls and 3,008 vocabulary-head calls including objective-checkpoint recomputation. There were no invalid, unknown or unattempted cases and no serving exports.
- 56 local CPU tests and 29 A800 focused tests passed. The independent tiny-model integration also exercises the exact six-step accounting and report tables.
- Total execution was 281.863 s (4 min 42 s), including 30.337 s preparation and 224.354 s diagnostic execution. Native backend loading was 21.008 s; target reconstruction 5.190 s. The six branch-runtime constructions took 105.558 s; disposable startup work 8.665 s. Instrumented training/restoration was 13.351 s for the three paper branches and 15.366 s for the three repository-rule branches. These include observer/state-copy work and do not establish production training overhead.
- Maximum allocated GPU memory was 23,176,500,224 bytes (21.585 GiB). A800 ended with about 128.62 GB free and idle GPUs. No old artifact cleanup was needed. The compact archive was retrieved and SHA-256 verified (`f192779aa99fdeac1436940122cc237a44283d68242210c3a3fbef7ca0d5f7e6`); reconstructed tensors remain on A800, outside Git.

## Consequence for reviewer reporting

The earlier A58 result should continue to be labeled **paper-defined AdaFlash OPD loss under our matched protocol**. This diagnostic shows that substituting the official repository clipping rule is a substantive algorithmic change. It does not show that this change explains A58's endpoint deficit, reverses K+'s measured advantage, or leaves that advantage unchanged.

No full event was rerun. No additional run is started by this result. The next decision belongs to the user/reviewer; the current evidence supports accurate labeling and an explicit official-clipping comparison limitation.

## Evidence

- [Code and review PR](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/30), execution `64cb56509d1c4ed3bd2ceb8365a346603edb2957`.
- [Official loss definition](https://github.com/ZinYY/AdaFlash/blob/478bf0ff7a3a5232b13d0f14e752d5fe1da0b718/pipeline/training/rkl_model.py#L168-L207).
- PAIRED_BATCHES.csv: gradient/displacement statistics and paired prefix totals. SOURCE_PREFIXES.csv: all six responses in both branches. LOSSES.csv: all pre/post components and clipping counts. SUMMARY.json: exact accounting and costs.
- Private raw source tokens, anchors, predictions, restoration receipts and runtime traces are retained separately; no private response text or model-sized gradient tensor is published.

## Review status

A separate same-team raw-evidence audit passed **247/247 checks**, including independent sequential prefix counting, per-parameter-to-global statistics, shared pre-forward components, historical first-pair metrics, and complete work accounting. All11 ARS interpretation-risk categories were assessed. The loss implementer also performed this audit, so authorship overlap is disclosed; there was no independent GPU/backward replication. See AUDIT.md and AUDIT.json. One audit-only status-label expectation was corrected before the final audit; production artifacts were unchanged.
