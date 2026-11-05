# KP v1 native qualification: compact-record review

Execution commit: `d10521a1aa7d256bb75b37a6aae07586ae7cc8b3`. All four cases passed. The published report auditor returned no errors; an independent reconstruction of the per-case ledgers matches the fixed budget.

**Review scope:** read-only review of retrieved compact records by a separate same-team agent. This is not an independent GPU run or a fresh comparison of the large model, optimizer or gradient tensors. Those tensors were not copied locally; exact decoded equality is supported by execution-time receipts.

| Case | Purpose | Current forwards/backwards | Reference forwards | Adam steps | Window seconds | Peak allocated GB |
|---|---|---:|---:|---:|---:|---:|
| Q1 | Native K | 2/2 | 0 | 1 | 30.703 | 20.655 |
| Q2 | Neutral wrapper with reference enabled | 2/2 | 2 | 1 | 37.528 | 22.861 |
| Q3 | Production KP | 2/2 | 2 | 1 | 36.443 | 24.958 |
| Q4 | Forced-active fixture | 2/2 | 2 | 1 | 37.096 | 22.861 |

Total: 8 current response forwards/backwards, 6 reference forwards, 4 disposable Adam steps, **zero generations**, zero unknown actions and zero extra warmups/steps. Suite wall time: 546.859 seconds (9.11 minutes).

Every case used a distinct process with six separately empty compiler/kernel caches on the same A800. Restore receipts start at 512 steps/1,024 presentations and end at 513/1,026 while retaining four completed events. No completed event 5 or production continuation state was fabricated. The sealed 746-token prepared sequence was evaluated before the 2,069-token longest sequence, using identical natural anchors across cases.

**Neutral equivalence (Q2 versus Q1):** ordinary response losses/metrics, sampled anchors, normalized pre-clip gradients, BF16 updated parameters, Adam state and RNG/counters/history match exactly in the recorded comparisons. Gradient and parameter discrepancy counts and maximum errors are zero. Reference work executed in Q2.

**Production path (Q3):** the true released-F rule is active at 51/4,136 and 55/5,410 protected occurrences. Normalized protection losses are 0.00908293 and 0.01286132; ordinary K losses remain 0.12850811 and 0.11691575. Scientific values are recorded finite. The added objective changes the update while preserving anchors/RNG/counters. These observations do not establish drafting benefit.

**Forced-active plumbing (Q4):** the qualification-only fixture produces the required added pre-clip gradient and changed Adam moments. Recorded differing coordinates: 1,046,762,296 gradient entries, 1,095,401,476 moment entries and 35,065,175 BF16 parameter entries. This artificial fixture is not the production rule or an efficacy result.

**Frozen state and restoration:** all four cases record unchanged frozen reference/head/embedding state and exact decoded serialization/restoration. The reviewed source checks exclusion of reference/head/embedding parameters from the optimizer. No receipt reports a frozen-state or restoration discrepancy.

Maximum allocation is 24.958 GB (23.244 GiB); maximum reservation is 26.139 GB (24.344 GiB). Initialization includes reference setup, and windows include cold compilation plus qualification gradient observation/persistence. These are overlapping scopes, not matched production-overhead estimates.

The records support proceeding with the prespecified campaign after actual storage re-admission. They do not establish lower verification counts or serving latency, and do not authorize tuning or extra experiments. The companion JSON contains exact ledger totals, comparison summaries, cost scopes and hashes of reviewed compact artifacts.
