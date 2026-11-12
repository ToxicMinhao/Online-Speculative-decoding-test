# Independent retained-evidence audit

**Status: passed, 455,467 checks, zero discrepancies.** This was a CPU-only reduction of the retrieved native records, with no production analysis imports and no new GPU execution. The author also prepared the panel, so this is a separately coded audit within the same executor team, not an external replication.

The audit covers all 192 requests, 8,647 K+5 starts, 51,005 accepted-slot margin records, 1,397 recent repairs, 235 matched pairs, all structural pools, all 1,343 K+6 replay starts, and the fixed bottom-32 full-vocabulary margin samples. It reproduces exact matching, tie breaking, control reuse, SMD, outcome classes, direct token agreement, paired effects, and both 10,000-draw request-cluster bootstrap intervals.

| Quantity | Independently reproduced result |
|---|---:|
| Recent-repair exact-slot rejection | 68 / 235 = 28.9362% |
| Established-control exact-slot rejection | 37 / 235 = 15.7447% |
| Paired difference | +13.1915 percentage points |
| Request-cluster 95% interval | [+5.7612, +20.7207] percentage points |
| Earlier rejection, repaired / established | 13 / 235 versus 14 / 235 |
| Direct token agreement at j, repaired / established | 162 / 235 versus 189 / 235 |
| Structural-pool difference on the same matched repairs | +23.0921 percentage points |
| Structural-pool 95% interval | [+17.2029, +29.0210] percentage points |
| Matched requests | 107 |
| Distinct controls / maximum reuse | 223 / 2 |
| Absolute margin SMD | 0.0238996 |

The lower interval exceeds the prespecified +5-point spending threshold and the fixed support/balance checks pass. The excess is not accompanied by more earlier breaks; direct token agreement also favors established controls. This supports recent repair history as an additional risk marker in the **matched population** at this fixed checkpoint transition. It does not establish a causal forgetting mechanism, a successful retention intervention, or native serving improvement.

Only 235 of 1,397 repairs (16.82%) could be matched under the fixed rule. The other 1,162 repairs remain outside this estimate: 207 lack structural controls and 955 fail the margin/zero-tie rule. The decrease from the structural contrast to the margin-adjusted contrast is consistent with current confidence accounting for part of the association, but does not provide a causal decomposition.

## Original report-validator failure

The original native analysis succeeded in memory. The later report validator encountered a serialization issue: the v1 proposal-depth histogram was originally hashed with integer keys, while JSON round-trip storage turns those keys into strings. Depths 10–15 then sort differently from integer depths under sorted-key serialization.

The independent audit confirms:

- Original payload fingerprint: `bd9629be436d0016979ab8cae2f736e7479c3a3374db5ac5a091f3a7b2648cf8`.
- Ordinary string-key rehash: `e11d3fd09d4dc2dcb744be793c9d2d83eadc729a9bf4af43000abee8856dde0a`; it fails the original fingerprint.
- Restoring only the declared v1 `coverage.unmatched_depth_counts` integer-key type reproduces the original fingerprint exactly.
- The separate file-byte seal also passes; original retained payload bytes and fingerprint were not changed.

This is a report serialization/checker issue, not a changed match list, failed native experiment, or retrospective reselection. The failed report attempt remains part of the evidence.

Machine-readable audit: `INDEPENDENT_AUDIT.json`. No training or follow-up experiment was performed or authorized by this audit.
