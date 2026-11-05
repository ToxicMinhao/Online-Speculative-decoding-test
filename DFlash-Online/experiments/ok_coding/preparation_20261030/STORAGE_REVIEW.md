# Read-only storage admission review

Review date: 2026-10-30. Status: **ANALYZED — storage not admitted**. No files were deleted, no model tensors were loaded, and no training or serving work was started. This review follows the October 14 completed-experiment retention authorization and the current coding handoff; it does not reopen an older experiment or authorize removal of selected checkpoints.

The selected-input budget is **366,710,105,681 bytes (366.710 GB)**. A800 has **150,397,759,488 bytes (150.398 GB)** free, leaving a **216,312,346,193-byte (216.312 GB) shortfall**. This bounded audit has not established a positively identified disposable set large enough to close that gap while preserving selected checkpoints, archive dependencies, raw scientific evidence, and paused/incomplete recovery data. The fixed campaign should not launch on the present admission evidence.

## Evidence and scope

The [budget](STORAGE_BUDGET.json) includes all eight independent feedback pools, capture plus prepared copies, four genuine initialization states, eight retained endpoint state/export pairs, rolling predecessor overlap, 16 GiB compiler-cache allowance, 10% slack, 8 GiB reserve and 4 GiB compact/workspace allowance. Peak phase is seed 29 / event 4 / K pending save. It uses the selected prompts and full output cap, rather than assuming favorable output lengths.

The initial [large-file inventory](REMOTE_LARGE_FILES.json) was supplemented with [fresh inode/link metadata](REMOTE_STAT_REVIEW.json) and [a recent-run feature-tensor metadata inventory](REMOTE_FEATURE_METADATA.json). The authoritative and remote A800 rules matched SHA-256 `bf7e2b0a9451c7de19602be0ae394eb832c7d1a142cae380c7c956015cfe4c01` before remote inspection. The approved helper was sourced; only metadata/disk-usage and process queries ran. A first metadata attempt found no generic `python` on the shell PATH and performed no Python or scientific work; the successful attempt used the qualified absolute interpreter. The record preserves that operational failure. No GPU compute processes were reported.

[STORAGE_REVIEW.json](STORAGE_REVIEW.json) binds the input hashes and records exact paths, aliases, sizes, classifications and limits. The review did not hash candidate tensor contents, extract their complete compact metadata, or perform a deletion-time open-file/dependency audit. Consequently **approved reclaimable bytes are zero in this report**: the possible candidates below are not a deletion manifest.

## Why the apparent large totals do not establish sufficient disposable space

| Observation | Allocated bytes | Decimal GB | Interpretation |
|---|---:|---:|---|
| Sum of 509 listed paths | 691,702,288,384 | 691.702 | Double-counts linked checkpoint names |
| The same list, 495 unique inodes | 640,034,893,824 | 640.035 | Unique listed allocation; not disposable capacity |
| Hard-link overcount | 51,667,394,560 | 51.667 | Removing one alias while another remains frees none of that inode |
| All cache directories | 4,900,900,864 | 4.901 | Includes useful compiled/runtime caches; no blanket purge proposed |
| Temporary directory tree | 565,579,776 | 0.566 | Even retiring every byte cannot materially close the gap |
| All listed training-state-style containers, unique | 155,262,771,200 | 155.263 | Includes selected endpoints and paused/incomplete recovery |
| Selected/incomplete groups excluded from retirement here | 79,733,157,888 | 79.733 | Mathematical endpoint/restart retention and bank recovery |
| Other training-state containers, upper bound | 75,529,613,312 | 75.530 | Still requires individual nonessential-state and compact-evidence audit |
| Feature/capture `.pt` files under recent `runs/` | 44,191,854,592 | 44.192 | Metadata classification only; not a blanket eligibility decision |

The existing frozen EAGLE reference copies and several `latest`/event checkpoint names already use hard links. Repeating the earlier deduplication would not produce the naive sum of their path sizes. The names `nonoptimizer_archive` and `lossless_delta_v1.zip` also do not imply disposable optimizer data: these are retained representations of historical serving weights and may depend on preserved base checkpoints.

The large-file listing uses a size threshold, so it misses many small feature files. The additional recent-run traversal explicitly checks that concern. Its largest remaining feature groups are KW 16.491 GB, O/K events 2–4 13.042 GB, O/K events 5–8 9.944 GB and the earlier O one-event experiment 4.148 GB. The 58-event OSD rerun has no remaining matching feature `.pt` payload in this inventory; its rolling tensor retirement already occurred. These figures do not establish that all such features may now be removed: the latest KW retention requirements and original parent/qualification bindings must remain respected.

Large files under correction-bank campaigns include replay arrays and recovery state. The retained [v0.1 review hold](../correction_bank_v01_20261018/campaign_001/REVIEW_HOLD.md) records an incomplete campaign and explicitly preserves its recovery/protection/solver evidence. They are not admitted as disposable merely because no GPU job is running. Raw per-request JSON/JSONL records, reports, traces, source catalogs, failure evidence and lossless selected-weight archives are likewise outside a blanket cleanup.

## Small candidates requiring further verification

Three exact paths are identifiable as bounded diagnostic candidates, totaling **10,486,419,456 bytes (10.486 GB)**:

1. `/data/sources/minhao/runs/dflash_prefix_ranking_one_event_20261025/campaign_001/qualification/disposable_O_state.pt` — a disposable ordinary state from the failed qualification, whose failure remains documented in the [source report](../dflash_prefix_ranking_one_event_20261025/REPORT.md).
2. `/data/sources/minhao/runs/dflash_prefix_ranking_execution_20261026/campaign_001/process_warmup/initial.pt` — an old process-startup artifact; the later main initialization and execution have separate [source evidence](../dflash_prefix_ranking_execution_20261026/REPORT.md).
3. `/data/sources/minhao/runs/dflash_online_gpu_check_20261022/retained_old_uncommitted_weights/model.safetensors` — an explicitly retained old uncommitted export from the superseded foundation qualification; [qualification evidence](../dflash_online_gpu_check_20261022/REPORT.md) is retained.

Their names and completion context justify inspection, not immediate deletion. Before retiring any one, verify its exact content hash/inode aliases, absence of an active or archive dependency, retained compact results, and required small RNG/configuration/counter metadata. Record the lost tensor-level replay or optimizer continuation capability. Even if all three and every cache/temp byte proved disposable, their roughly 15.953 GB combined capacity is far below the 216.312 GB deficit.

The other 75.530 GB of state containers might contain nonessential Adam history alongside already exported serving weights, but they cannot be called safely reclaimable without checking each retained checkpoint obligation and preserving the rest of its scientific state. This review does not propose destroying selected exports or all historical restart states to force the run to fit.

## Local storage and decision limit

Fenrir had **75,552,542,720 bytes (75.553 GB)** free at review. Even the unrealistic assumption that every free local byte could be transferred into A800 capacity would leave approximately **140.760 GB** of the new budget uncovered, before local reserve, staging and required retained artifacts. The current budget is explicitly `retain_local` with no assumed offload; an unqualified transfer plan is not admission.

The finding is a concrete failure of current storage admission, not proof that no future storage arrangement could work. Reopening admission requires additional available capacity or a separately verified sufficient retention/placement plan that preserves the handoff's data and checkpoint requirements. It does not justify reducing the output cap, changing the dataset/panel, dropping a seed, skipping retained pools/endpoints, lowering reserve, or starting only a favorable-looking prefix of the campaign.

All observations are descriptive capacity/accounting checks; no efficacy or mechanism conclusion is drawn. Statistical fallacy coverage is not inferential here: aggregation/hard-link duplication, incomplete inventory and selection of disposable candidates were explicitly examined; no significance, population, causal, or performance claim is made.


Implementation and CPU admission source: `5cf1c7e2e7c4a4d08d2225b58c56488578112daf`. No GPU scientific execution occurred.
