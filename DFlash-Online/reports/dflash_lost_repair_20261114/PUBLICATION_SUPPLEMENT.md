## Retained-evidence publication supplement

Execution `29b6a15c5f5f16135663ea61e3d31f21b3f1d465`. Execution status: **completed**; census complete: True; read-only exposure complete: True.
This supplement changes no checkpoint, schedule, labels in the original report, or scientific setting. It adds sanitized receipts and independently coded historical-category expansion; it is not a new native replication.

Observed starts: 4877. Full census denominator: 4877. Category counts: `{"H": 252, "M": 2147, "all_correct_or_terminal": 625, "other_rejection_history": 1853}`.
Neutral parity: `{"passed": 4877}`. See PARITY_RECEIPTS.csv for per-arm cache/output checks; null values mean unavailable or inapplicable, not success.

Original row range: `[768, 896]`; batch digest: `90d84cdfa142e8be0eda2ef03778c190e51e74e899aac277e2e80fce32c6fda3`. Source, plan, configuration, recovery and schedule identities appear in IDENTITIES.json.
Independent audit receipt: `research/dflash_lost_repair_20261114/delivery/retained/delivery/RAW_AUDIT.json` (VERIFIED)

| Measured clock | Seconds |
|---|---:|
| Whole campaign | 2853.545327266678 |
| Aggregate occupied GPU | 2853.5035617239773 |
| census phase | 2673.612681567669 |
| training phase | 150.0979630202055 |
| Read-only training initialization_s | 33.40894734300673 |
| Read-only training schedule_s | 1.6513815373182297 |
| Read-only training wall_s | 109.79217995516956 |

These clocks are nested; do not sum them. Work-ticket timings are diagnostic occupied-GPU-clock intervals and do not establish individual kernel costs. All measured operation counts, native startup counts and training peak memory are in COST_ACCOUNTING.json.
Incomplete evidence retains observed counts but cannot decide the full-event gate. The original GATE.json remains authoritative; publication does not turn a partial run into a pass.
Raw prompts, tokens, tensors and connection credentials are excluded from this supplement.
