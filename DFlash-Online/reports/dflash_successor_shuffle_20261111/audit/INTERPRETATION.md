## Material Passport

- Workflow: ARS-Codex 3.24.2, experiment-agent validate.
- Verification status: **ANALYZED**. Raw-record arithmetic and production reconciliation passed; no native model run was repeated.
- Actual execution: `4d977e9ffd2cab9d58ae8806c29df11d843353da`, clean execution source, PR #39.
- Inputs: finalized local `delivery/retained`, admitted after `delivery/TRANSFER_VERIFICATION.json` reported a passed comparison of all 5,013 files / 454,125,542 bytes with the remote inventory.
- Review scope: all 64 prespecified common starts, their 64 parent records and 384 child records (three arms, two repeats), with native sidecars, canonical continuations and sealed input/checkpoint records. The audit hashed and rechecked 903 consumed files.
- Authorship disclosure: this reviewer also authored production `matched_start.py`. The audit uses separately written direct loops and imports no production comparison helper. It is independent arithmetic within the same team, not blinded independent authorship or a second experiment.

The audit exactly reproduces all **2,796 production row and group fields** in `final_report/MATCHED_START.json`. It verifies the deterministic 64-request selection, common prompt and target seed, checkpoint identity, canonical output and termination for both child repeats, native sidecar equality, and complete parent attempt/block accounting. Both child repeats agree on every first-block quantity used here. No request or arm was filtered.

## Parent strata and denominators

| Parent stratum | Requests | Old-error denominator per arm | Nonempty earlier-prefix denominator per arm |
|---|---:|---:|---:|
| Successor high | 10 | 10 | 10 |
| Successor low | 53 | 53 | 53 |
| Successor ineligible | 0 | 0 | 0 |
| All correct | 1 | 0 | 1 |
| Seed EOS | 0 | 0 | 0 |
| Total | 64 | 63 | 64 |

The parent bit is evaluated at its own first erroneous proposal using the canonical continuation. A canonical EOS successor is eligible; slots after the first canonical EOS are excluded. All-correct and seed-EOS parents have no old error and therefore cannot count as repairs. The single all-correct parent is a tie for every child. Zero-request strata remain explicit; no rate is inferred from a zero denominator. All 64 parent attempts returned and validated, with 64 verification blocks and no unknown or unattempted requests.

## Prespecified canonical-proposal counts

A repair means the child's proposal at the parent's old first-error slot is correct and every earlier correct slot remains intact. Prefix gains and losses compare the contiguous canonical-matching proposal prefix from that same request start.

| Child | Repairs / 63 old errors | Earlier prefixes broken / 64 | Prefix tokens gained | Prefix tokens lost | Wins / losses / ties |
|---|---:|---:|---:|---:|---:|
| K | 11 / 63 | 6 / 64 | 62 | 7 | 11 / 6 / 47 |
| TRUE | 13 / 63 | 6 / 64 | 62 | 7 | 13 / 6 / 45 |
| SHUFFLED | 12 / 63 | 6 / 64 | 62 | 7 | 12 / 6 / 46 |

| Parent stratum | Child | Repairs / old errors | Prefix breaks / nonempty prefix | Tokens gained / lost | Wins / losses / ties |
|---|---|---:|---:|---:|---:|
| Successor high | K | 6 / 10 | 1 / 10 | 32 / 1 | 6 / 1 / 3 |
| Successor high | TRUE | 5 / 10 | 1 / 10 | 27 / 1 | 5 / 1 / 4 |
| Successor high | SHUFFLED | 6 / 10 | 1 / 10 | 27 / 1 | 6 / 1 / 3 |
| Successor low | K | 5 / 53 | 5 / 53 | 30 / 6 | 5 / 5 / 43 |
| Successor low | TRUE | 8 / 53 | 5 / 53 | 35 / 6 | 8 / 5 / 40 |
| Successor low | SHUFFLED | 6 / 53 | 5 / 53 | 35 / 6 | 6 / 5 / 42 |
| All correct | Each arm | 0 / 0 | 0 / 1 | 0 / 0 | 0 / 0 / 1 |

TRUE repairs one more old error than SHUFFLED and two more than K on this subset, but all three gain 62 and lose 7 prefix tokens: the same net gain of 55. Repair counts measure how often repair occurs; gained-token totals also depend on its extent. The successor-high and successor-low groups differ in size and show different arm orderings. These descriptive counts do not establish a preferential repair effect for the high-bit group.

## Supplemental native-verification check

Canonical-prefix advance and native accepted-prefix advance are kept as separate quantities. The supplemental **verified repair** requires canonical repair with the earlier prefix intact and the repaired old-error slot lying inside the child's native accepted prefix. An advance alone does not satisfy that definition.

For these 64 starts, canonical and native prefix lengths agree for every parent and child. All 11 K, 13 TRUE and 12 SHUFFLED canonical repairs are also verified repairs. Native advances have the same counts and the same 62 gained / 7 lost tokens. Every arm has zero canonical repairs without native acceptance and zero native advances without verified repair. This agreement is an observed property of these records; the definitions remain distinct.

The diagnostic concerns the shared first conditioning prefix only. It does not decompose complete-request verifier calls, identify the cause of a full-request difference, or establish serving latency or training-inclusive speedup. No inferential test was added to these counts. The full-request endpoints and their prespecified uncertainty remain separate, and the experiment uses one continuing training seed and one permutation.

## Audit execution and provenance

The first local audit invocation stopped at a reader-schema check before computing comparisons: matched subset rows carry two additional sealed fields, `matched_start_ordinal` and `matched_selection_sha256`, beyond the base-panel fields. All 64 selected identities and their order already matched. The audit reader was corrected to require every base field unchanged and to independently reconstruct both added fields. The successful invocation then processed the complete unchanged panel. This was an audit-only compatibility correction; production code, execution records and scientific definitions were unchanged. No model work or remote access occurred.

The machine-readable result is [MATCHED_START_AUDIT.json](MATCHED_START_AUDIT.json); per-request and complete five-stratum tables are [MATCHED_ROWS.csv](MATCHED_ROWS.csv) and [MATCHED_GROUPS.csv](MATCHED_GROUPS.csv). [PROVENANCE.json](PROVENANCE.json) binds the executed audit script, preparation receipt, transfer receipt and output files. The preparation receipt preserves the original script hash; the final provenance records the reader correction's new hash.

## ARS interpretation checks

All 11 categories were checked at this diagnostic's scope.

| Category | Disposition |
|---|---|
| Simpson's paradox | Both parent-bit strata and pooled counts are shown. The observed arm ordering varies by stratum; no pooled mechanism claim is made. |
| Ecological fallacy | Request-level rows are retained; aggregate counts are not attributed to every request. |
| Berkson's paradox | Selection is fixed by input hash within the already eligible panel, not by child outcomes. The restricted pool and older project exposure limit generalization. |
| Collider bias | Descriptive grouping uses the common parent state before the arm fork; no conditioning on child success or regression is added. |
| Base-rate neglect | All five stratum sizes and both repair/prefix denominators are reported, including zeros. |
| Regression to the mean | Parent-error subsets are descriptive and all three children are retained. Parent-to-child improvement alone is not attributed to the successor rule. |
| Survivorship bias | All 64 selected starts, all three arms and both repeats are present; no failure or tie was omitted. |
| Look-elsewhere effect | Every prespecified count and stratum is reported. Supplemental verification counts are labeled; no favorable subset or new significance test is selected. |
| Garden of forking paths | The sealed hash subset and production definitions are unchanged. The audit reader correction is disclosed and preserves exact selection. |
| Correlation versus causation | These first-block counts are not treated as evidence of complete-request causal mediation or speedup. |
| Reverse causality | Temporal order is common parent followed by child training/evaluation. Descriptive parent-bit associations are not given an additional directional causal interpretation. |
