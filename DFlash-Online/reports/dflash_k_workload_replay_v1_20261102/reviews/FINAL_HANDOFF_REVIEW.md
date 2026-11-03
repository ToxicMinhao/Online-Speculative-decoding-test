## Material Passport

- ARS-Codex 3.24.0, experiment-agent `validate`; **ANALYZED**, not a GPU reproduction.
- Reviewer: `/root/track2_resource_preflight`, separate same-team review with shared context and earlier source/resource-review involvement. Independent reconstruction and conclusion preceded curated-draft review and its attributed peer-result summary; no peer audit artifact was used as analytical input.
- Execution `8a4cbcc8b78adfee845a0a1409db4a0d29c54058`; production_001. Local-only review began after `retrieved_001/RETRIEVAL_VERIFIED.json` confirmed19,639 compact files at2026-11-03T02:02:59Z. Bulk tensors were intentionally not retrieved.
- **Disposition: no discrepancy in the evidence checked; no demonstrated overall advantage for R. One next proposal: park this replay allocation and retain K.** No additional experiment is authorized.
- Review scope: 51679 explicit checks across 9594 read/hash-recorded files; exact inputs and reconstructed numbers are in [FINAL_HANDOFF_REVIEW.json](FINAL_HANDOFF_REVIEW.json). This check count is coverage, not a statistical confidence measure.

# Independent final handoff review

## Scientific conclusion

The pooled R/U gain is12 calls out of13,037: **+0.0920% [−0.4775,+0.6432]%**. R/K adds21 calls: **−0.1615% [−0.8027,+0.4487]%**. U/K is **−0.2538% [−0.9200,+0.3989]%**. Neither replay arm establishes balanced-final superiority over K, and R does not establish superiority over U. The intervals also do not establish equivalence. Raw tokens, stops, counts, repeat stability and all nine call intervals were reconstructed from the saved rows; no final result was inferred from a peer review.

The domain tradeoff is material: R/U saves86 coding calls (**+1.1534% [0.4136,1.8845]%**) but adds74 math calls (**−1.3259% [−2.1867,−0.5011]%**). These secondary intervals are descriptive/unadjusted. R/K domain intervals both include zero. Pooled R/U and R/K time reductions are−0.6691% [−1.7590,+0.3440]% and−0.6441% [−1.7911,+0.4262]%; no common-mode serving-time gain is established.

## Future trajectory and mechanism evidence

All 3,072 pre-update rows were checked against saved prompts/calls and complete cross-arm output/termination agreement. Events1and2 have equal calls. Future events 3–8 contain768 unique requests, including256 math and512 coding. No iid trajectory bootstrap was applied.

| Event | Domain | K calls | U calls | R calls | R/U reduction | R/K reduction |
|---|---|---:|---:|---:|---:|---:|
| 3 | coding | 6492 | 6470 | 6487 | -0.2628% | +0.0770% |
| 4 | coding | 7547 | 7519 | 7529 | -0.1330% | +0.2385% |
| 5 | math | 5299 | 5213 | 5231 | -0.3453% | +1.2833% |
| 6 | math | 5337 | 5333 | 5363 | -0.5625% | -0.4872% |
| 7 | coding | 8380 | 8368 | 8370 | -0.0239% | +0.1193% |
| 8 | coding | 7317 | 7277 | 7268 | +0.1237% | +0.6697% |

Future K/U/R totals are **40,372/40,180/40,248**. R/U is−0.1692%, R/K+0.3071%, U/K+0.4756%. R trails U in five of six events and in both future domains. Both replay arms show a descriptive benefit at the first math return (event 5); R loses its advantage over K on event 6. This supports reporting recurrence observations, not established selection superiority or a well-supported smaller-dose intervention.

I independently reproduced all seven greedy32 selections/scores and common-fresh/uniform subsets, and all eight AlgorithmR membership/RNG transitions. R coverage exceeds U in every replay event. R selects29/32 current-domain old packets at event 4 and32/32at events 5–8; U remains mixed. At event 8 R selects0 math despite71 math candidates, while U selects21. This is compatible with specialization versus retention and shows workload discrimination. It does not isolate that mechanism causally, prove a balanced loss versus K, or show that a hybrid/dose change would help.

## Prespecified outcomes and one proposal

| Outcome | Assessment |
|---|---|
| A | Not met: neither required call interval has positive lower bound; future R/U and pooled timing are unfavorable. |
| B | Not met: U and R do not establish final improvement over K. |
| C | Only point-sign pattern, not established R-over-U improvement. Raw report’s automatic C branch must not be read as a superiority conclusion. Conditional dose proposal is not justified by these mixed future records. |
| D | Descriptive domain specialization/retention pattern relative to U is compatible with the observations. Balanced benefit or causal retention loss versus K is not established; a16+16hybrid is unvalidated. |
| E | Higher coverage did not establish aggregate serving benefit. Do not claim all replay fails or descriptors are uninformative: coding R/U and future replay/K point effects remain visible. |
| F | No robust pooled call gain or time gain. The call-efficiency-to-speedup caution applies; tiny selector overhead does not explain away lack of drafting benefit. |
| G | Dominant disposition: valid but inconclusive overall effect, with opposing domains and inconsistent R/U future behavior. The useful-consistent-both-domains premise for an unchanged replication is not met. |

**One proposed next action: keep K and park this fixed coverage-selected replay allocation.** No smaller-dose, hybrid, second-seed, longer-budget or diagnostic experiment follows from this result. Reopening needs a separate user decision and a concrete new mechanism question/evidence. Stage 1/projectNone remains unchanged.

## Cost and resource findings

| Arm | Presentations / Adam steps | Replay | Prompt / response tokens | Active anchors | Native training seconds | Worker seconds | Export/verify seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| K | 2048 / 1024 | 0 | 114374 / 588760 | 542308 | 2806.719 | 3355.996 | 148.328 |
| U | 2048 / 1024 | 224 | 116786 / 585134 | 543435 | 2764.578 | 3323.331 | 143.833 |
| R | 2048 / 1024 | 224 | 115765 / 587487 | 545614 | 2775.998 | 3337.304 | 141.843 |

All declared work reconciles:5,380 complete generations (3,072 stream+2,304 final+4 parity),1,024 terminal preparations,6,144 forward/backward presentations and3,072 Adamsteps; replay448included. Started/completed work counters agree;24 commit receipts, four raw capture controls and the production event 1neutral qualification agree. Per-arm native RNG equality is a production scientific-state qualification, not a new decoding of those tensors by this reviewer.

Shared descriptor time is0.453000s; selection total0.682773s includes scoring0.547674s and greedy0.082408s. Maximum counted persistent selection arrays are8,520,704 bytes, excluding transients/Python/allocator overhead. Adaptation-ready wall is15,327.051s, final-evaluation wall2,191.939s, outer campaign17,525.487s. Final publication/weight-validation times52.432s/7.057s lie outside request timing. Teacher preparation84.036s includes terminal forward plus loading/binding/concatenation; prepared-packet save48.736s is not all capture I/O. Different timers nest and must not be added to the outer total. Equal presentations are not equal token FLOPs or training-inclusive speedup.

Current campaign admission was76,723,990,528 free bytes against75,023,917,685 required:1,700,072,843 headroom after the full declared plan. Its4GiB additional compiler,3GiBcompact,2GiBworkspace,8GiBreserve and5%tensor margin remain explicit. Across3,150 recorded checks, minimum free was43,447,312,384 bytes; maximum compiler growth100,327,424 bytes; maximum compact allocation1,953,501,184 bytes. Postflight compact allocation was1,967,943,680 bytes and free space45,090,390,016 bytes. These are observed samples, not continuous peaks. The old8GiBadditional-compiler plan did not fit at preflight; no history deletion or transfer credit was used to admit this run.

Per-worker CUDA allocator maxima (after construction/restore, through train_event/checkpoint commit) were20,723,064,832/27,577,548,800 allocated/reserved bytes forK, and20,788,358,144/27,573,354,496forU andR. They exclude later validation and are not whole-job/device/host peaks. Final timing has37 quiescence receipts, with no foreign GPU/owned transfer process observed at those boundaries; this is not continuous process monitoring.

## Retention and provenance

All **2,986** retired paths are unique and new-run-owned:1,024 native,1,024 terminal,896 prepared packets,21 preceding states and21 preceding exports. Retirements report completed status, archived=false, retained compact dependencies present and nlink1. Their208,569,153,843 logical bytes (208,574,205,952 allocated-entry bytes) are nominal removed-path totals, not measured net reclaimed capacity. Direct intermediate replay/resume is lost; files were retired, not archived.

The separately supplied fresh stat receipt at2026-11-03T02:03:44Z exactly covers **138 paths/135 unique(device,inode)pairs**: one shared initial, three event 8 full states plus three latest hardlink aliases, three final exports and128reservoir packets. All final state/packet sizes match preserved bindings; there is no retired-path overlap. Three latest aliases share their event 8 state inode with nlink2. Unique allocated bulk is29,558,915,072 bytes, versus48,456,331,687 logical bytes counted by paths; free space at that stat was44,769,894,400 bytes. This upgrades evidence to fresh presence/size/hardlink verification at that timestamp. It is **not** a fresh content hash, tensor decode or independent scientific reproduction. U/R initial references and complete restore metadata remain compact receipts.

Retained preflight verifies3,453sealed identities and complete canonical/input checks. I independently checked1280 unique source IDs and normalized rendered/user hashes; the tokenizer decoding/normalization procedure itself remains the prior preflight evidence. An old direct byte-hash receipt for coding canonical/wrapper bytes was unavailable; current exact bytes are sealed, native/logical output and termination checks match retained data, and old hash-verified scalar/validity/termination receipts provide additional support. That does not verify a missing historical byte hash.

## Handoff coverage

Paths below are relative to `retrieved_001/production_001`. The full evidence list and input hashes are in the JSON companion. A receipt-level check is explicitly distinguished from independent record reconstruction.

| Handoff | Area | Disposition / evidence scope |
|---|---|---|
| 1,12 | Question and immutable authority | Execution pin and complete inventories checked; 3,453-file source/model/runtime seal validation is a retained production receipt. Fixed eight-event K/U/R question; no K58 or natural-traffic inference. |
| 2,10 | Frozen recipe and states | Prior pinned-source review plus all initial/event receipts; full states not newly decoded in this review. Independent released equality and neutral-state checks are production observations. |
| 3 | Exact data and canonical outputs | 1280 identity/rendered/user hash uniqueness checked; exact final prompts/tokens/termination reconstructed against retained canonicals. Normalization decoding itself and missing old coding canonical byte-hash limit rely on retained preflight evidence. |
| 4 | Exposure and RNG | 24 commit/exposure totals reconstructed; independent common-fresh/uniform RNG choices; per-arm native training RNG and exact anchor equality rely on pinned source and production scientific-state receipts. |
| 5 | Shared reservoir | Algorithm R and full PCG64 states before/after all eight insertions matched exactly; final128 slots bound to fresh-stat file sizes. |
| 6 | Descriptor and coverage selection | All seven greedy32 selections and final scores matched fixed float64/einsum arithmetic;1024 packet identity/norm/row qualification receipts checked. No new descriptor extraction from retired tensors. |
| 7 | Shared feedback and future outcomes | 3072 raw calls/prompts and complete cross-arm outputs/stop semantics checked; events 1/2neutral; events 3–8 totals reconstructed. Shared K history and capture-mode timing limitations retained. |
| 8 | Final endpoint and statistics | 2304 raw final rows/canonicals, repeat stability, per-request position balance, all nine point estimates/call CIs reconstructed; time CIs recomputed;37 timing barriers checked. Native sidecar presence checked, not every native block revalidated here. |
| 9 | Work and included qualifications | Started/completed generation/preparation/forward/backward/step totals reconcile; four parity controls compared directly; neutral/mixed-pass gates and59-test launch record retained. No model rerun or independent fixture rerun. |
| 10 | Storage, costs and retention | 3150 discrete samples,2986 retirement paths, expected138bulk paths/135 inodes,24 event cost receipts reconciled. Freshstat is not a fresh tensor hash. Nested timers and allocated/logical distinctions retained. |
| 10 | Delivery and separate review | Raw report retains all arms/signs and detailed machine-readable evidence. Curated draft later read; three cost/review/retention placeholders remained at that snapshot. Parent owns assembly/publication and final artifact-link checks. |
| 11 | Outcome A–G and one next action | G dominant; D-like domain specialization descriptive; no established C superiority, dose/hybrid validation or consistent replication premise. One proposal: park allocation and retain K. |

## Statistical fallacy assessment

All 11 categories assessed; this is not11 clean certificates.

| Category | Assessment | Evidence/limit |
|---|---|---|
| Simpson paradox | CAUTION | Final R/U pooled+0.0920% masks math−1.3259% and coding+1.1534%; harms shown. |
| Ecological fallacy | CAUTION | R/U has101 wins,107 regressions,48 ties; pooled ratio is not every-request benefit. |
| Berkson paradox | CAUTION | Historically selected and reused math/coding development inventories do not represent deployment traffic. |
| Collider bias | NO_DISCREPANCY_IN_CHECKED_RECORDS | No invalid/failure filtering in reconstructed complete rows; selector inputs are past prompt descriptors under pinned source, not outcomes. |
| Base rate neglect | CAUTION | Final128/128 differs from future256 math/512 coding; pooled call ratio weights actual call totals, not traffic prevalence. |
| Regression to mean | CAUTION | Contemporaneous K/U controls and fixed inputs; no selected retry/checkpoint. No K58 gain addition. |
| Survivorship bias | NO_DISCREPANCY_IN_CHECKED_RECORDS | All planned2304 final and3072 stream rows plus4 parity complete; started/completed work reconciles; no discarded failures found. |
| Look elsewhere | CAUTION | R/U primary and R/K required remain inconclusive; secondary domain/time intervals are unadjusted and not simultaneous coverage. |
| Garden of forking paths | NO_DISCREPANCY_IN_CHECKED_RECORDS | Fixed seeds/dose/memory/selector/schedule/8 eventbudget; no scientific rerun. Reused panel still limits confirmation. |
| Correlation versus causation | CAUTION | Current-domain selections and final split are compatible with specialization, not isolated causal proof; higher coverage is not draft utility or net throughput. |
| Reverse causality | NO_DISCREPANCY_IN_CHECKED_RECORDS | Strict past reservoir reconstruction, prompt-row qualifications and neutral firsttwoevents hold; future trajectory remains adaptive, not iid. |

## Later curated-draft review

After independently recording the conclusion above, I read `final_delivery_001/REPORT.md` at SHA256`3229805eebcf6db6348aff49cf52068bee736509ff55688e984b395f808a52c4`. Its numerical/domain/future/selection tables and A–G interpretation agree with this reconstruction. Cost/review/retention placeholders were still awaiting assembly at that snapshot. I requested one wording qualification: tie the no-contention statement to exited training workers and37 before-batch/after-final barriers, rather than imply continuous transfer monitoring. This later draft check did not determine the independent scientific conclusion. Parent owns final assembly, link verification and publication.

No remote access, GPU work, source mutation, new experiment, endpoint retuning or additional statistical family was performed. Native sidecar presence and production checks were inspected; every native trace block and full tensor was not independently revalidated here.

## Closing draft check

The filled curated report was rechecked at SHA256 `1dd5d730dafc0fa5e2498c86eb21328b859a0aa7255ce707846b5f6918d337f1`. The three placeholders are filled and the timing statement now explicitly describes 37 sampled barriers and prior training-worker exit. Costs and retention agree with the reviewed receipts; absent isolated forward/backward/Adam timing, capture-overhead timing and host/full-process GPU peaks remain disclosed. **No remaining blocker in these reviewed sections.** Other agents’ check totals in the curated report remain attributed to their reviews; they did not determine this review’s independent conclusion. Final publication/link assembly remains the parent task.
