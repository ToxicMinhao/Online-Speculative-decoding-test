# AdaFlash clipping diagnostic: retained-evidence audit

**Verification status: ANALYZED.** All 247 independent scalar/record checks passed. This review recomputed summaries from compact retained GPU evidence; it did not rerun any model or independently reproduce model-sized gradients. Scientific execution: `64cb56509d1c4ed3bd2ceb8365a346603edb2957`.

The reviewer is a separate same-team agent who authored the isolated loss module and its CPU tests. Source authorship overlaps; this is not external or blinded review. Prefix reconstruction below uses a separately written sequential Python scan and does not import the production metric function.

## Checks and findings

- All six branches restored released/fresh-Adam state, cleared accumulated gradients, used their exact retained anchors and matched paired pre-update predictions, labels and weights. Final training RNG states matched between each pair's clipping branches. The retained initial artifact remained unchanged.
- Across all six pre-update responses, teacher CE, raw RKL, clipping-activity source counters and denominators match exactly between branches. This supports interpreting their update difference as clipping-rule sensitivity under the common arithmetic.
- The paper branch's first two presentation records match **all 18 and 19 common historical A58 metrics**, respectively, including loss, numerators, denominators, supervised exposure and the first accumulated gradient norm. Later diagnostic pairs reset to released weights, so their historical A58 metrics are not valid reproduction targets. Identical logged scalars do not establish retired feature-tensor byte equality.
- Work accounting matches the bound: six Adam steps; twelve training and twelve post-step response forwards; twelve core backwards; two disposable startup forwards/backwards and zero startup Adam steps. Native feature reconstruction used six prefills, 297 verification forwards, 4,752 checked posterior positions and six terminal-feature forwards. No new answer generation, checkpoint export, unknown or unattempted case is reported.
- Each global gradient/displacement sum, norm, cosine and relative difference was independently recomputed from retained per-parameter statistics. These agree with the published tables. Full gradient tensors were deliberately not retained, so this verifies aggregation of the measured statistics rather than independently recomputing backward results.

## Clipping and local fitting

The six responses contain **1,719 sampled blocks and 25,155 valid proposal occurrences**. Denominators range from 765.7146606 to 2695.2233887 and represent whole responses, not individual projection chunks.

Pre-update paper-entry clipping affects 690,155 / 3,821,950,080 vocabulary entries (0.0180577%) and at least one entry in 21,420 / 25,155 positions (85.1521%). Repository-position clipping affects 6,333 / 25,155 whole positions (25.1759%). These cap frequencies measure different operations and must not be treated as the same percentage.

Mean response CE is identical, 1.355056707. Mean effective RKL is 0.308809387 for paper-entry and 6.277644246 for repository-position. Mean mixed loss is 1.145757885 and 2.339188995. Their numerical losses are different objectives and do not rank model quality directly.

| Independently recomputed quantity | Paper entry | Repository position |
|---|---:|---:|
| Old matching-prefix occurrences | 10,634 | 10,634 |
| New matching-prefix occurrences | 15,713 | 15,566 |
| Net matching-prefix gain | +5,079 | +4,932 |
| Useful repairs / old first errors | 1,131 / 1,482 | 1,071 / 1,482 |
| Blocks with new earlier error / all blocks | 163 / 1,719 | 136 / 1,719 |
| Old-prefix occurrences lost / old prefix | 589 / 10,634 | 455 / 10,634 |

The repository rule improves net prefix gain in pair 1 (+2,452 versus +2,298), but reduces it in pairs 2 (+1,475 versus +1,581) and 3 (+1,005 versus +1,200). It loses fewer old-prefix occurrences overall but repairs fewer original first errors. Results are mixed. Every count is an occurrence in overlapping sampled training blocks, not a native verification-call change.

The actual BF16 displacement differs by 73.56–79.48% of the paper displacement norm, with cosine 0.685–0.730. This is vector distance, not the percentage of altered parameters. The proportion of parameter coordinates differing between arms is 10.74–12.18%. The difference survives native BF16 AdamW rounding and is not merely a numerical rearrangement. It does not establish a better 58-event endpoint for either rule.

## Cost and limitations

Wrapper time is 281.863 s; preparation is 30.337 s; diagnostic execution is 224.354 s. Recomputed branch construction costs sum to 105.558 s; instrumented training/restoration totals are 13.351 s (paper) and 15.366 s (repository). Maximum allocated memory is 23,176,500,224 bytes. Observer copies, state restoration and runtime construction are included, so this is not a production adaptation-overhead comparison.

No GPU rerun, native-serving comparison, held-out transfer check or 58-event loss comparison was performed in this audit. Reconstructed target posteriors/commits satisfy the native contract, but retired historical feature bytes remain unavailable. The report's conclusion is supported: A58 remains a paper-defined AdaFlash OPD arm under the common protocol, and official-repository clipping equivalence must not be claimed. These three local batches do not show that clipping explains the prior endpoint deficit or changes K+'s final advantage.

## ARS fallacy scan: 11/11 checked

1. **Simpson's paradox:** pair directions are reported; mixed local results are not collapsed into universal superiority.
2. **Ecological fallacy:** block occurrences are not interpreted as independent requests, verification calls or population transfer.
3. **Berkson/selection bias:** fixed first-six convenience selection was determined before results; representativeness is limited.
4. **Collider bias:** no inclusion/adjustment uses either arm's outcomes.
5. **Base-rate neglect:** cap, repair and loss denominators are explicit and refer to distinct units.
6. **Regression to the mean:** cases were not selected for extreme outcomes; paired starts are identical.
7. **Survivorship bias:** all declared branches completed; no unfavorable case was excluded.
8. **Look-elsewhere effect:** both branches and all pair/response results are retained; no significance search.
9. **Garden of forking paths:** scope and clipping arithmetic were fixed before execution; no efficacy-based adjustment.
10. **Correlation versus causation:** local controlled clipping sensitivity is supported; a cause of A58's 58-event deficit is not established.
11. **Reverse causality:** the intervention precedes the update; no unsupported cross-request temporal claim is made.

## Evidence

- `../retrieved_001/production_001/diagnostic/batch_*/RESULT.json`: predictions, labels, weights, clipping/loss components, state receipts and costs.
- `../retrieved_001/production_001/diagnostic/BATCH_*_COMPARISON.json`: per-parameter and global gradient/displacement statistics; RNG equality.
- `../retrieved_001/production_001/preparation/`: source selection, reconstruction and terminal-preparation records.
- `../source_receipts/EVENT1.json`: original first-event historical metrics.
- `AUDIT.json`: all 247 checks, per-response prefix recomputation, clipping denominators, historical comparison, costs and fallacy assessments.
- `audit_raw.py`: audit-only scalar reconstruction with no production metric imports. The initial audit expected a generic preparation-status label; this was corrected to the actual `completed` schema. No scientific data were changed.
