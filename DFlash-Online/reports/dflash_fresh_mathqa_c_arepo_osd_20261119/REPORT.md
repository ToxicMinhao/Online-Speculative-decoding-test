# Fresh MathQA: C58 versus released AdaFlash loss and ordinary OSD

## Material Passport

ARS experiment-agent run + validate; completed 2026-11-20. Native execution and retained-evidence reproduction are verified. This is one frozen-endpoint study, not an independent GPU replication. Research Stage 1 / project None and native K+ as the primary baseline remain unchanged.

Actual execution commit: `96f0a13d189b8f5a1d3cfcf103a0ff1097f18cd8`. Report-only commits are separate in PR #55. No scientific source changed after launch.

## Result

**C58 improves complete-request time and drafting efficiency over AREPO58 on this fresh 1,024-request MathQA panel.** The primary time reduction is **1.6103% (paired 95% CI 1.3375–1.8839%)**; verification calls fall **1.6803% (1.4089–1.9516%)**. This supports the complete fitted method against the literal released AdaFlash loss under this matched serving protocol. It does not establish which component caused the benefit.

| Comparison | Complete-request time reduction, 95% CI | Verification-call reduction, 95% CI | Call wins / regressions / ties |
|---|---:|---:|---:|
| C58 / AREPO58 | 1.6103% [1.3375, 1.8839] | 1.6803% [1.4089, 1.9516] | 599 / 295 / 130 |
| C58 / O58 | 2.3436% [2.0960, 2.5992] | 2.4252% [2.1804, 2.6739] | 672 / 233 / 119 |
| AREPO58 / O58 | 0.7454% [0.5000, 0.9948] | 0.7577% [0.5207, 0.9999] | 516 / 335 / 173 |

Positive reductions favor the first arm. C/AREPO is the prespecified primary time comparison; C/O and AREPO/O intervals are descriptive and unadjusted. All three repeats favor the same direction. There is no arbitrary minimum-gain threshold.

| Comparison | Gross calls saved | Gross calls added | Net saved | Faster / slower / tied request means |
|---|---:|---:|---:|---:|
| C58 / AREPO58 | 2,178 | 798 | 1,380 | 662 / 362 / 0 |
| C58 / O58 | 2,476 | 469 | 2,007 | 718 / 306 / 0 |
| AREPO58 / O58 | 1,512 | 885 | 627 | 590 / 434 / 0 |

## Absolute endpoint measurements

Counts use one validated trajectory per request after all three repeats agree. Times are sums of per-request three-repeat means, not averages of request-level speedup ratios.

| Arm | Calls | Native accepted tokens/call | Stop-clipped matching tokens/call | Committed progress/call | Summed mean request time (s) |
|---|---:|---:|---:|---:|---:|
| O58 | 82,755 | 5.912416 | 5.922857 | 6.912416 | 1454.700303 |
| AREPO58 | 82,128 | 5.965188 | 5.975027 | 6.965188 | 1443.857261 |
| C58 | 80,748 | 6.084225 | 6.094615 | 7.084225 | 1420.607286 |

Mean complete-request latency is O58 **1420.6058 ms**, AREPO58 **1410.0169 ms**, and C58 **1387.3118 ms**; per-request values remain in the tables.

Each arm returns the identical **573,061 output tokens**: 951 EOS and 73 cap-terminated requests, lengths 66–2,048, mean 559.6299 and median 387.5. Committed progress is 572,037 tokens, excluding one initial target-prefill token per request. Native accepted totals are O 489,282; AREPO 489,909; C 491,289.

The native counter is `commit_rows−1` per verification round. The separate matching counter is `min(accepted_end−1, commit_rows)` and can exceed the native counter by one on a stop-clipped round; it is not a raw untruncated count. See [counter definitions](COUNTER_DEFINITIONS.json). Acceptance and verification calls under identical outputs are coupled accounting views, not independent replications.

| Repeat (1-based) | Arm | Complete-request total (s) | Native-generation total (s) | Cache-maintenance total (s) |
|---|---|---:|---:|---:|
| 1 | O58 | 1451.410901 | 1424.793304 | 0.496291 |
| 1 | AREPO58 | 1440.685799 | 1414.095983 | 0.496266 |
| 1 | C58 | 1417.141519 | 1391.177526 | 0.496418 |
| 2 | O58 | 1453.647616 | 1427.061267 | 0.552387 |
| 2 | AREPO58 | 1442.662008 | 1416.078006 | 0.559050 |
| 2 | C58 | 1419.099287 | 1392.912078 | 0.549572 |
| 3 | O58 | 1459.042392 | 1424.980353 | 8.439847 |
| 3 | AREPO58 | 1448.223975 | 1413.329611 | 9.442939 |
| 3 | C58 | 1425.581051 | 1390.919173 | 9.805304 |

Cache-maintenance time is higher in repeat 3 for every arm; those observations remain in the primary complete-request result. Native-generation and cache timers are separate measured intervals, not a detailed kernel breakdown. Wrapper overhead means they need not sum to complete-request time. No slow row was excluded or replaced.

## Validity and experiment contract

All **10,552 planned generations** completed: 24 bounded qualification, 1,024 fresh O58 canonicals, 288 scheduled warmups and 9,216 scored completions. There were **zero invalid, unknown or unattempted executions**, zero retries, zero optimizer steps, zero feedback collection and zero new trained checkpoints. All three input export hashes are unchanged after evaluation. [EXECUTION_MANIFEST.json](EXECUTION_MANIFEST.json) is the immutable launch plan: its historical pending-reauthentication fields are not current status. [COMPLETED_PREFLIGHT.json](COMPLETED_PREFLIGHT.json) records the passed native admission checks, and [POSTFLIGHT_EXPORT_IDENTITIES.json](POSTFLIGHT_EXPORT_IDENTITIES.json) records unchanged final weights.

The independent auditor checked all 10,552 raw generations, 864,732 verification blocks, 778,509 scored/warmup trace-table rows, complete scheduling/accounting and the prescribed bootstrap. It passed on its first attempt in 128.424 seconds. Its author previously wrote the panel audit and repaired the runtime; the raw/statistical reducer imports no production reduction helper. This is same-team independent reduction, not independent GPU replication. See [audit](FINAL_EVIDENCE_AUDIT.json), [auditor](verify_final.py) and [interpretation review](INTERPRETATION_AUDIT.md).

The panel was sealed before generation from pinned MathQA test revision `4f958013c164a40745a5b771dfb005df308deb4a`: 2,985 source rows, 1,799 known project exclusions, 1,186 eligible and 1,024 selected with seed 20261119. IDs, normalized prompts and rendered tokens were checked. Historical exclusion receipts were authenticated and extended; unknown histories, semantic near-duplicates and pretraining exposure are outside this freshness claim. No outputs, generated lengths or performance selected requests.

Inference remains BF16/FA3, greedy, five draft layers, block 16, 2,048 tokens or EOS, context 4,096 and concurrency 1. Feedback feature capture is disabled; qualified CUDA graph initialization and native traces are retained. Fresh canonical outputs come from O58 with the native greedy target verifier, not standalone target-only generation. Every scored result matches exact canonical tokens and termination.

Both cards qualified against the same fixed old witnesses and prepared disjoint untimed canonical shards. All timed runs used fixed GPU `GPU-9986735a-3e68-d1bd-5998-9634ef24341e`, while GPU B was idle. The 48 scheduling groups cycle six arm permutations, each eight times; 144 activations each include two warmups. This is aggregate balance, not per-request position balance. No bulk transfer, compression, hashing or CPU-reference work ran during scored timing.

### What happened to the earlier matching issue?

PR54’s Python-RNG startup discrepancy was traced to a terminal-formatting import and repaired before C58 was trained. This study reuses that completed checkpoint and constructs no optimizer, so it does not repeat that training-startup check. Historical verification counts from different requests are inapplicable and were removed only in the new fresh-panel adapter. Exact checkpoint identity, canonical tokens/termination, native accounting and within-arm repeat checks were preserved and passed. No general matching waiver was used.

## Statistics

For each request, average its three times within each arm; estimate `100 × (1 − sum(C means)/sum(reference means))`. Use 10,000 paired whole-request bootstrap draws from PCG64 seed 20261119, lexicographic request-ID order and linear 2.5/97.5 percentiles. All arms and repeats stay together within each sampled request. The intervals are conditional marginal 95% intervals for these fixed endpoints and panel; repeats are not independent requests or training seeds.

Request/source-order SHA-256: `5c8410e626ebefcde2f95be710ce7c237fe4094bb049cb3e83e4c1888d2b97d1`. Draw-matrix SHA-256: `8fa17a82dc7a152450744e6c2c56acf2282ffdbcd500c839f2e9ec5c124a4f3c`. Full witnesses: [BOOTSTRAP_WITNESS.json](BOOTSTRAP_WITNESS.json).

## Work, timing and resources

| Phase | Wall time (s) | Backend loading included (s) |
|---|---:|---:|
| qualification_A | 87.228 | 21.509 |
| qualification_B | 87.618 | 21.258 |
| canonical_A | 1059.799 | 20.508 |
| canonical_B | 1001.562 | 20.757 |
| scored_A | 14146.687 | 20.007 |

Launcher wall time was **4.2724 hours**; charged execution elapsed time was **4.2636 hours**, and occupied-card time **4.5508 hours**, within the 8-hour and 12-card-hour caps. The admitted forecast was 4.1 hours raw / 5.6 hours with contingency. The two qualification and two canonical phases overlap across cards; do not add their wall times as elapsed time.

Scored complete-request intervals total **12957.495s** across all repeats. Warmups total **743.578s**. Checkpoint activation costs **144.804s**, loaded-weight validation **9.356s**, postflight export hashing **6.047s**, production raw/statistical analysis **43.956s**, and independent audit **128.424s**. These nested/component timers are not additive. Loading, canonical preparation, warmups and activation stay outside scored request latency.

Five native engine initializations performed **255 separately counted graph-startup forwards** (85 captures), with zero startup failures. These are not additional complete-response generations. Maximum allocated/reserved GPU memory was **57.323/57.634 GB**. Raw phase receipts are in [PHYSICAL_WORK_ALL.json](PHYSICAL_WORK_ALL.json).

Post-execution storage charge was 3.673 GB, compiler growth zero, and shared free space 493.976 GB. This is below the 30 GB ceiling and above the 400 GB reserve. After delivery and retirement of redundant packaging, final A800 study allocation is 3.835 GB and shared free space is 493.813 GB. The verified private archive contains 31,792 members / 2.706 GB logical evidence in a 156.290 MB compressed file. Final delivery/storage accounting is recorded separately in [STORAGE_DELIVERY.json](STORAGE_DELIVERY.json); packaging and transfers occurred after timed serving. The original gzip transfer was stopped in favor of the smaller identical-payload zstd encoding; its time and removed partial copy remain recorded. This administrative transfer was not a scientific retry. PR54 closeout already verified 117,867 compact members and 165,436 keeper/retirement checks before this run. No old checkpoint or unique historical result was deleted.

## Interpretation and limits

The prespecified result is supported lower C latency with consistent drafting benefit over literal repository AdaFlash loss on this fresh panel. AREPO also improves over ordinary OSD here. These are modest endpoint benefits, not proof of training-seed robustness, a new task-family result or training-inclusive speedup. Each recipe has its own historical feedback trajectory; this is not a common-feedback loss-only causal comparison. It does not identify which allocation component caused C’s benefit. AREPO implements the released loss within our matched fixed-length protocol, not AdaFlash’s complete concurrent-serving system.

No parameter was tuned, panel expanded, algorithm changed or follow-up launched. The separate [direct support-plus-score reference](CPU_REFERENCE.md) remains inactive and CPU-only. All 11 ARS interpretation categories are covered in [INTERPRETATION_AUDIT.md](INTERPRETATION_AUDIT.md).

## Reproducible evidence

The [final publication review](FINAL_PUBLICATION_REVIEW.md) passes 82 checks and resolves both items in the preserved earlier review.

- [Readiness and source identities](READINESS.md), [execution manifest](EXECUTION_MANIFEST.json), [schedule](SCHEDULE.json), and [unchanged exports](POSTFLIGHT_EXPORT_IDENTITIES.json).
- [All 9,216 timing rows](ALL_REPEATS.csv), [paired-request table](PER_REQUEST.csv), [repeat totals](REPEAT_TOTALS.csv), [request outcomes](REQUEST_OUTCOMES.json), and [complete numeric summary](SUMMARY.json).
- [Work ledger](WORK.json), [cost summary](COST_SUMMARY.json), [raw audit](RAW_AUDIT.json), [independent audit](FINAL_EVIDENCE_AUDIT.json), [CPU auditor tests](test_verify_final.py), and [storage/delivery receipt](STORAGE_DELIVERY.json).

Archive construction took 31.232s, encoding conversion 6.121s, the completed transfer 688.485s, and stream verification 5.238s. The discontinued gzip transfer took 391.262s. Launch to verified delivery was 4.6589 hours; this interval includes post-run work and waiting, not just serving. Prelaunch interactive development and final publication are not fully timed.

Private prompts, generated responses, token traces and complete raw runtime records remain outside Git. The complete private evidence archive and member manifest are retained on A800 and verified on Fenrir; see the delivery receipt for exact paths, identities and verification scope.
