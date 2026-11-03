# Final training, cost, resource and retention review

Verification status: **ANALYZED**. No unresolved cost/retention discrepancy was found in this scope. Scientific execution is `8a4cbcc8b78adfee845a0a1409db4a0d29c54058`; this review performs local CPU receipt accounting only.

The reviewer authored the additive trainer and seal. This is not an independent review of that implementation or a new GPU reproduction. The separate reconstruction audit owns endpoint/count/bootstrap verification. The extractor received a separate same-team source/schema review before use; its eight synthetic fixtures passed. No repository source was changed, no model was loaded, and no remote operation was performed by this review.

## Evidence and scope

The parent [retrieval receipt](../retrieved_001/RETRIEVAL_VERIFIED.json) verifies 19,639 compact files (1,919,536,120 bytes), archive SHA256 `4a3281b7f83db9f1b4d982231254c406a5d98d24475e5bfbc541487648b426e9`. The archive excludes `.pt` and `.safetensors`. Extraction began only after the verified gate at 2026-11-03 02:02:59 UTC. The [full extraction](cost_retention_final_001/COST_RESOURCE_RETENTION.json) hashes 3,293 consumed receipts and reports zero missing required files and zero issues. Its generic inventory caveat is supplemented here by the parent transfer verification; this reviewer did not repeat that entire transfer-hash audit.

Companion files retain [event rows](cost_retention_final_001/TRAINING_COST_ROWS.csv), [retired paths](cost_retention_final_001/RETIREMENT_PATHS.csv), [retained artifacts](cost_retention_final_001/RETAINED_ARTIFACTS.csv), [scope notes](cost_retention_final_001/COST_RESOURCE_RETENTION.md), and the [fresh bulk-presence join](cost_retention_final_001/BULK_PRESENCE_JOIN.json).

## Exposure and per-arm cost

Each arm has eight completed event receipts, 2,048 response presentations and 1,024 Adam steps. K uses 2,048 fresh presentations; U and R each use 1,824 fresh plus 224 replay presentations. These are equal presentation/step budgets, not equal token or active-anchor budgets.

| Arm | Prompt-token presentations | Response-token presentations | Active anchors | Replay presentations |
|---|---:|---:|---:|---:|
| K | 114,374 | 588,760 | 542,308 | 0 |
| U | 116,786 | 585,134 | 543,435 | 224 |
| R | 115,765 | 587,487 | 545,614 | 224 |

Relative to K, U has 2.109% more prompt-token presentations and 0.616% fewer response-token presentations; R has 1.216% more prompt-token presentations and 0.216% fewer response-token presentations. Active anchors are 0.208% and 0.610% higher for U/R. These totals count repeated exposure, not unique tokens, actual FLOPs or token-matched training. Event 1 has identical recorded exposures across arms (18,950 prompt tokens, 65,358 response tokens, 65,096 active anchors), consistent with the producer neutral-state gate.

All times below are sums over disjoint events for the same arm. Rows have different, often nested scopes and must not be added together.

| Recorded scope, seconds | K | U | R |
|---|---:|---:|---:|
| Trainer construction | 177.169 | 177.743 | 176.785 |
| Own-state restoration | 66.832 | 68.835 | 69.706 |
| Feature admission/validation | 28.813 | 34.728 | 34.857 |
| Native response training | 2,806.719 | 2,764.578 | 2,775.998 |
| Complete native pass loops, including input loading | 2,816.819 | 2,774.571 | 2,786.059 |
| Checkpoint save | 143.924 | 144.292 | 146.842 |
| Complete training workers, including validation/cleanup | 3,355.996 | 3,323.331 | 3,337.304 |
| Training subprocess wall, including Python startup | 3,378.498 | 3,345.528 | 3,359.606 |
| Deferred export and exact-weight verification | 148.328 | 143.833 | 141.843 |

Initialization worker times are K 29.186 s, U 25.256 s and R 26.848 s. Adding only disjoint attributable scopes—initialization worker + eight complete training workers + later exports—gives K **3,533.509 s**, U **3,492.420 s**, R **3,505.994 s**. Replacing worker time with subprocess wall, including startup, gives 3,558.028 / 3,516.909 / 3,530.446 s. Both subtotals exclude shared serving, terminal preparation, descriptors, selection, reservoir maintenance and campaign resource scans; they are not end-to-end adaptation totals for three independent workloads.

Native response-training totals are U 1.501% and R 1.095% below K in this execution. This is descriptive cost evidence, not a controlled method-speedup result: token/anchor exposure differs, worker order is K/U/R, and global qualified compiler caches are reused. The named `_compile.compile_inner` sums are 63.898 / 32.808 / 32.821 s for K/U/R; compiler names overlap and those times are already included in training. No isolated cold-cache or compilation-free cost was measured.

## Campaign and shared costs

The outer campaign wall is **17,525.487 s** (4.868 h), including cleanup but excluding separate preflight/postflight and the subsequent report generation. Adaptation-ready wall is **15,327.051 s**; final capture-OFF evaluation wall is **2,191.939 s**. The eight event-wall scopes sum to 15,216.045 s and are nested within adaptation-ready wall. These outer scopes are not added to their components.

Shared recorded components are serving backend load 20.507 s, pre-update checkpoint activation/validation 91.322 s, teacher preparation 84.036 s, prepared-packet persistence 48.736 s, descriptors 0.453 s, selection 0.683 s, reservoir maintenance 23.960 s and resource scans 1,384.528 s. Teacher preparation includes native/terminal loading, binding checks, concatenation and terminal forward; it is not isolated GPU teacher compute. Packet persistence excludes other capture I/O. Final evaluation activation 52.432 s and weight validation 7.057 s are within its wall scope. These shared costs cannot be assigned three times to the arms.

The selector records 0.548 s scoring and 0.082 s greedy selection, nested within 0.683 s total selection. Maximum counted persistent numeric arrays are 8,520,704 bytes; this excludes transient arrays, Python metadata and allocator overhead. The descriptor uses existing terminal-preparation outputs, and the receipts do not report extra model work for selection. Worker, initialization, export, selection and resource-scan sums reproduce the campaign component totals to floating-point precision.

## Resource observations

Campaign storage admission passed at **76,723,990,528 free bytes**, versus **75,023,917,685 required**, leaving 1,700,072,843 bytes above the full declared initial requirement. The plan preserves an 8 GiB free reserve, 4 GiB additional compiler-growth allowance, 3 GiB compact allowance, 2 GiB workspace and 5% tensor headroom, including pending/old-state and export overlap.

Across 3,150 discrete resource checks, minimum free space is **43,447,312,384 bytes** (40.463 GiB), maximum compiler growth is **100,327,424 bytes**, and maximum compact allocation is **1,953,501,184 bytes**. Postflight passed with 45,090,390,016 free bytes, the same compiler growth and 1,967,943,680 compact bytes. The later compact figure includes later writes; it is not a contradiction of the earlier sampled maximum. These are sampled filesystem values, not continuous high-water marks or measured reclaimed space. `pending_write_bytes=0` is the sampled default gate argument; separate write-space checks do not log actual pending allocations here.

| Arm | Maximum CUDA allocated bytes | Maximum CUDA reserved bytes |
|---|---:|---:|
| K | 20,723,064,832 | 27,577,548,800 |
| U | 20,788,358,144 | 27,573,354,496 |
| R | 20,788,358,144 | 27,573,354,496 |

These are maxima of the eight per-worker allocator peaks reset after construction/restore. Their window includes native training and checkpoint persistence/commit, but excludes later boundary validation, state hashing and cleanup. They are not whole-job device usage or host-memory peaks. Preflight host availability was 137,295,110,144 bytes, a single availability observation rather than measured peak consumption.

## Final scientific states and retention

Producer final-boundary receipts pass for all arms at event 8, global step 1,024 and 2,048 presentations, with 58 Adam state entries and one CUDA RNG state. Their live comparisons record exact own-state continuation, no RNG reset, no mismatches and zero extra model/optimizer steps. Initial U/R receipts independently compare released weights, empty Adam and RNG to K; the event-1 decoded neutral comparison passes for weights, Adam, RNG, counters and anchors while allowing only arm/run/path/timing metadata differences. This review reads those checks; it does not re-decode omitted tensor files.

| Arm | Final complete-state bytes | Producer SHA256 |
|---|---:|---|
| K | 6,299,184,859 | `7714e20bd31414c27560ac3797aabae267f4274d60f93b1e1e53ec167aa405fa` |
| U | 6,299,261,083 | `2cf8746163be622413d68e268b95d9b8f304917f8478143c5d4bb8553300ec04` |
| R | 6,299,277,275 | `317ea8d72b8f71b0036aa1b0184dca87e5ec8f5b79c36e30f365bdbbb48dbc6c` |

Each final native export separately passes exact-weight verification at step 1,024, with zero extra model/optimizer steps. The earlier event RESULT files correctly retain `export_status=deferred`; the later EXPORT_CHECK and updated boundary receipts establish materialization. Deferred zero export timers are not treated as the completed export cost.

The parent [fresh filesystem inventory](../retrieved_001/FINAL_BULK_PRESENCE.json), timestamped 2026-11-03 02:03:44 UTC, confirms **138 bulk paths on 135 unique device/inode pairs**: one shared initial state, three complete final states, three final exports and 128 reservoir packets, plus three latest-state hardlink aliases. The [local join](cost_retention_final_001/BULK_PRESENCE_JOIN.json) finds every expected bulk artifact, matching recorded byte sizes where identities are present, matching latest/event-8 inodes for every arm, and no retired path still listed. Unique retained bulk allocation is 29,558,915,072 bytes (29,558,608,470 logical bytes). Fresh stat establishes existence, size and linkage at that time; file contents still rely on producer hashes and validation, not fresh tensor hashing or loading.

All **2,986** retirement receipts are completed, distinct paths: 1,024 native capture tensors, 1,024 terminal tensors, 896 evicted prepared packets, 21 preceding serving exports and 21 predecessor complete states. No prepared-but-unfinished retirement receipt or duplicate retired path is present. Their nominal logical bytes total 208,569,153,843 and nominal allocated bytes 208,574,205,952; all recorded link counts at unlink are 1. These are cumulative removed-file sizes, not a measured causal disk-space gain. The exact paths/reasons/retained evidence are in RETIREMENT_PATHS.csv.

The retired tensors were **deleted, not archived**. Direct replay of the 896 evicted prepared packets and raw reconstruction from the removed native/terminal tensors is lost. Direct exact resume/serving from intermediate events 1–7 is lost after their states/exports are removed; compact scientific/validation receipts remain. Final complete states support their recorded exact boundary state, and final 128 prepared packets remain. This does not demonstrate interrupted-campaign recovery or authorize another event.

## Data and canonical provenance limits

This is the fixed controlled recurrence stream M1/M2/C1/C2/M3/M4/C3/C4: 512 reused PR23 GSM8K training requests and 512 reused sealed PR20 CodeAlpaca requests, with original IDs and prompt tokens preserved. The preflight source receipt matches four retained source files to old receipts and current remote bytes. Within the selected 1,024 training plus 256 evaluation requests, exact token digests, normalized rendered prompts and normalized user text are all unique under the recorded NFC/whitespace normalization. That current-panel disjointness does not make the already used evaluation panel unseen or establish freedom from model pretraining or earlier development exposure.

The [canonical source check](../retrieved_001/production_001/sealed/preflight/CANONICAL_SOURCE_CHECK.json) explicitly records a coding provenance limitation: no earlier direct canonical/wrapper byte-hash receipt was located. All 128 original coding wrappers/native traces currently validate; exact canonical tokens, termination and counts match them, and counts/termination/validity additionally match the hash-verified historical ALL_REPEATS.csv. Fresh hashes bind those original bytes for this seal. This is weaker historical byte provenance than an original canonical-byte receipt and must remain disclosed. It is not evidence of a detected output mismatch.

This accounting establishes the costs and retained capabilities of the completed one-seed run. It does not provide training-inclusive deployment speedup, multiple-training-seed robustness, naturally occurring traffic evidence, or an improvement over the prior trained K58 state. Current gains cannot be added to the historical full-K result. No efficacy recommendation or follow-on experiment is inferred here.

## Reproduction

```sh
python audit_tools/summarize_cost_retention.py --run-root retrieved_001/production_001 --output reviews/cost_retention_final_001
```

The output directory was new. Extractor SHA256: `cee82d4185083d8a037884ae23b950ce78cafc72bdff7b363201fc6d9e4a9170`. The supplementary stat join and component reconciliation are recorded in the accompanying receipts.
