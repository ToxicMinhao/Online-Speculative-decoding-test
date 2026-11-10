# Native reconstruction gate review

**Passed for both datasets and both checkpoints.** Actual execution: `d7b1311e7b2f910e935bf81b4e40282767d7540c`.

The first hash-selected GSM8K and MathQA requests were checked at the prescribed nonterminal boundaries for KPLUS4 and KPLUS6. Four fresh original walks reproduce their authenticated historical traces. **All eight neutral suffixes match exactly across 267 verification rounds and 4,005 proposal-token occurrences.**

The CPU audit independently checked every proposal, target argmax, native position, known seed, actual committed progress, canonical output, stop clipping, EOS/cap termination and remaining call count. Original 2,048-token caps and each source successor’s remaining output budget are preserved. All 19 retrieved snapshot files were authenticated; 8,431 assertions passed.

Native receipts report bitwise-identical source target-prefix KV and rebuilt neutral draft-prefix KV, with destination-native context ingestion and no source draft-cache transplant. The CPU review verifies those receipts and the exact downstream traces; it does not independently compare the large KV payloads, which were not transferred.

| Dataset | Checkpoint | Neutral source rounds | Remaining suffix calls |
|---|---|---|---|
| GSM8K | KPLUS4 | 1, 22 | 42, 21 |
| GSM8K | KPLUS6 | 1, 22 | 42, 21 |
| MathQA | KPLUS4 | 1, 25 | 48, 24 |
| MathQA | KPLUS6 | 1, 24 | 46, 23 |

One initial **CPU-audit schema mistake** was corrected: the native trace stores 15 proposal tokens separately from its known seed and has 16 verifier/position rows. The first auditor assumed equal lengths. Its failure record remains preserved; the corrected audit passes. This was not a native failure, and no runtime, scientific source or tolerance changed.

This is a same-team review with implementation-author overlap, using separate CPU checks of retained evidence. It added zero model work. It qualifies the constructor at the required boundaries; it does not establish full-campaign completion, a restart bottleneck, serving speedup or a training mechanism benefit. Final work and postflight accounting remain separate.

For the eventual explanation, **an actual advance means greater committed progress at the first differing round**. It remains included whether or not it qualifies as a verified first-error repair. The latter requires correcting KPLUS4’s first rejected canonical token while preserving all preceding matches. An advance lacking that check must not be called a demonstrated repair. No partial scientific outcome is inferred here.
