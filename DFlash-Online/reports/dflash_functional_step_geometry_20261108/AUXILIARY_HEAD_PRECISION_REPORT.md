# FSG auxiliary-head precision diagnostic

**The more accurate head backward improves operator accuracy, but does not pass the complete auxiliary-gradient qualification. The pilot remains paused.** No new Adam step, feedback collection, teacher preparation or serving generation was performed. The original failures and acceptance tolerances remain unchanged.

Execution source: [`1c6dec2cd7feac62926ca7ed0abb7948918f5da6`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/1c6dec2cd7feac62926ca7ed0abb7948918f5da6), [PR #33](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/33). [Compact measurements](AUXILIARY_HEAD_PRECISION_SUMMARY.json) and [all per-parameter comparisons](PARAMETER_COMPARISONS.csv). ARS experiment-agent execution/reproducibility validation was used; source, numerical and work reviews are same-team reviews, not independent GPU replications.

## What changed, and what stayed fixed

Only the auxiliary vocabulary-head backward multiplication was revised for this diagnostic. It receives the actual BF16 incoming gradient and the frozen BF16 head matrix, promotes those represented operands to IEEE FP32, performs their matrix multiplication, and rounds the resulting hidden gradient back to BF16. The local precision/autocast settings are restored before the unchanged draft backward. No FP32 master model parameters or persistent FP32 head copy were introduced.

Native forward logits, prefix decisions, weighting, sensitivity formula, model/Adam state, ordinary K+ training and serving were unchanged. The helper is opt-in and used by this diagnostic; it has not been substituted into the production FSG updater. This is an explicitly authorized auxiliary numerical-policy test, not a claim that the old BF16-only gate passed.

The runner reused the exact retained GSM8K feature packet and saved pre/native-post-update endpoints. The fixture has 56 prompt tokens, 127 response tokens, and the first two original anchors, 56 and 57, from its retained 126-anchor presentation. Both layouts have four protected occurrences and normalize over this same two-anchor fixture. No anchors or answers were resampled.

Two comparisons were kept separate:

1. **Complete gradient:** two 16-row chunks and one 32-row projection each independently compute their own native logits, protected weights and sensitivity inputs. Their returned hidden gradients traverse the unchanged pre-state draft graph.
2. **Fixed-input operator accuracy:** both layouts receive the same actual BF16 incoming gradient and frozen BF16 weights. An independent FP64 product promotes those original operands directly. Its output is also rounded to BF16 before the separate downstream control; no FP64 draft backward is used.

## Operator accuracy improves substantially

The table reports relative L2 error against the independent FP64 product. “Before final cast” observes the revised FP32 product; the native row observes its BF16 output, not its inaccessible internal accumulator.

| Head calculation | Before final cast / native output versus FP64 | BF16 output versus BF16-rounded FP64 | Differing BF16 coordinates |
|---|---:|---:|---:|
| Native BF16, two 16-row products | 0.235104% | 0.295406% | 4,982 / 131,072 |
| Native BF16, one 32-row product | 0.171144% | 0.121423% | 3,889 / 131,072 |
| Revised IEEE FP32, two 16-row products | 0.000110297% | 0.00104851% | 10 / 131,072 |
| Revised IEEE FP32, one 32-row product | 0.0000765464% | 0.0000168561% | 10 / 131,072 |

Both revised products pass the predeclared operator check: their BF16 outputs satisfy the unchanged tolerance against BF16-rounded FP64, and their pre-cast discrepancy norms are lower than those of the corresponding native products. Only four of the 32 rows carry nonzero protected sensitivity; coordinate counts are not independent observations or token-accuracy measurements. The raw FP32 products are not exact FP64 products. Small remaining differences can cross a BF16 rounding boundary.

The fixed-input native controls are direct BF16 matrix products, not a rerun of ordinary K+ or a new full-model baseline. This operator result establishes improved accuracy for this multiplication on this fixture only.

## The complete gradient still fails

The unchanged gate requires every coordinate to satisfy `abs(error) <= 1e-5 + 0.016 * abs(reference)`, **and** global relative L2 to be at most 1.6%. Passing the latter cannot override coordinate failures.

| Complete-gradient comparison | Previous native-head diagnosis | Revised auxiliary head |
|---|---:|---:|
| Coordinates outside tolerance | 356,522 / 1,048,626,432 | **192,305 / 1,048,626,432** |
| Affected parameter tensors | 55 / 58 | **50 / 58** |
| Global relative L2 difference | 0.639685% | **0.571439%** |
| Maximum absolute coordinate difference | 0.0078125 | 0.0078125 |
| Gate | Failed | **Failed** |

The revised own-branch hidden gradients differ by only 0.00104837% relative L2, compared with approximately 0.300651% previously. Prefix masks, proposals and weights remain exact between layouts. Native forward logits still differ because their BF16 projection layouts remain unchanged; each branch's actual incoming BF16 gradient is preserved rather than replaced by a common value.

The remaining failure is not uniformly distributed. For example, the first layer's key-normalization weight has **39 / 128** failing coordinates, despite only 0.01834% of all parameter coordinates failing globally. A smaller global error is not a pass or a reason to discard these failures.

The previous figures come from the retained [zero-Adam diagnosis](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/d5b07c130a914e9ff8f1aeab158e725e80362f4b/DFlash-Online/reports/dflash_functional_step_geometry_20261108/GRADIENT_DIAGNOSTIC_REPORT.md); they are historical numerical controls, not matched timing measurements.

## What the fixed-input downstream control adds

With incoming gradient and head weights held fixed, each revised head output differs from BF16-rounded FP64 at just ten hidden-gradient coordinates. Nevertheless, propagating these outputs separately through the unchanged draft backward gives:

| Hidden gradient supplied to draft backward, compared with BF16-rounded FP64 control | Hidden relative L2 difference | Parameter-gradient relative L2 difference | Failing parameter coordinates |
|---|---:|---:|---:|
| Revised two-chunk result | 0.00104851% | 0.584202% | 191,648 |
| Revised full result | 0.0000168561% | 0.495186% | 114,532 |

This shows that **very small remaining BF16 hidden-gradient differences are sufficient to produce a failed parameter-gradient comparison in this native chain**. The residual failure does not require each branch's different forward-derived sensitivity: it is also present in the fixed-input control. The earlier exact-repeat control showed that identical hidden gradients produced exactly identical draft gradients on this retained case.

These observations localize the next unresolved numerical question to how small adjoint differences propagate through the native BF16 draft backward. They do not identify a specific layer or kernel defect, establish that the draft backward is mathematically inaccurate, or prove that a production implementation bug exists. The FP64 reference covers only the head multiplication; it is not a high-precision reference for the entire draft backward. No further precision change or downstream experiment was performed.

## Work, costs and integrity

| New work | Executed / ceiling |
|---|---:|
| Response-level draft forwards | 2 / 2 |
| Head-only backwards | 2 / 2 |
| Complete-draft parameter VJPs | 5 / 5 |
| Direct operator-control matrix products | 7 / 7 |
| Adam applications | **0 / 0** |
| Feedback / teacher preparation / serving generations | **0 / 0 / 0** |

The two head-only backwards contain three revised head-backward matrix products, in addition to the seven direct operator controls. Thirteen physical vocabulary-head forward calls include three reference scans and checkpoint recomputation. These counters are not interchangeable with whole-response training backwards. No startup warmup backward or extra optimizer qualification step was added; previous qualification work remains separately charged.

The supervisor completed in **84 seconds** (23:20:59–23:22:23 UTC, November 8), exit 0. Exit 0 means the diagnostic completed, not that the scientific gate passed. The runner clock was 67.193 seconds and excludes its final dependency postcheck. Peak GPU memory was **16.096 GB allocated / 21.032 GB reserved**. The additional transient head copies are 2.489 GB in FP32 and 4.979 GB in FP64, created sequentially; the latter is diagnostic-only. Maximum production shapes remain unqualified.

Single diagnostic timings for the fixed-input head calculations were 2.611 / 1.261 ms for native chunked/full products and 13.282 / 6.593 ms for revised chunked/full products. The independent FP64 product took 76.264 ms. Revised timings include conversion, finiteness checks and associated synchronization. These are neither kernel-only measurements nor estimates of full-event training overhead. The complete head-branch timings also include checkpoint and first-use effects; no timing-repeat campaign was run.

All **1,515** source/input dependencies matched before and after execution. Model values, Adam state, counters, ordinary gradients, RNG and runtime restored exactly; optimizer objects/versions and precision settings were unchanged afterward. Neither TF32 environment override was set. No generated checkpoint was published. All **19 compact files / 2,144,890 bytes** were copied to Fenrir and rehashed without differences. A missing remote `rsync` executable required a read-only tar-stream transfer; it did not cause a scientific retry.

All **34 focused CPU tests** passed locally and on A800; **43 existing geometry regression tests** also passed locally. The separate work/state audit passed **3,123 assertions**; the numerical review checked all 11 ARS interpretation categories, and narrative review found no unsupported causal or efficacy conclusion. These reviews disclose authorship overlap. Both A800 GPUs were idle after completion. Existing inputs, endpoints, earlier failed records and historical reports remain preserved. Only redundant 21.112 MB source-transfer archives were removed on each host after source verification; exact paths and before/after capacity are recorded in cleanup receipts. No scientific replay capability was retired.

## Decision

The reviewer's second outcome applies: **head accuracy improved, but the complete gradient comparison still fails.** Keep the pilot paused and return this evidence for review before choosing another numerical change. The targeted diagnostic is complete; production tile/partial-chunk shapes, maximum-size memory, SCALAR, native MathQA, serving preservation and the remaining integration checks are not qualified.

This result neither establishes an FSG learning benefit nor a mechanism failure. It establishes that the proposed head-only repair is insufficient under the unchanged gate. The BF16 realized-update norm guard is untouched; no new filtered proposal was applied, so this run supplies no new evidence about how often FSG would fall back in training.
