## Material Passport

- Origin: ARS experiment-agent, run and validation workflow; October 30, 2026.
- Status: **CPU preparation complete; GPU campaign blocked by storage admission**.
- Implementation and actual A800 CPU preflight source: `5cf1c7e2e7c4a4d08d2225b58c56488578112daf` (clean, published).
- Reviewed baseline publication: `8d5d4185976fc8d78b7d55a19928d86c7cb86e95`.
- Verification scope: source/input seals, CPU correctness and preparation reviews. No GPU experiment or efficacy result.

The handoff is implemented without changing O or K. It starts four fresh coding trajectories—O/K under training seeds 17 and 29—from released DFlash, runs four 128-response events per trajectory, and measures F/O/K after events 2 and 4. This tests coding-task transfer of the existing K component. It does not add a new loss or continue the mathematical checkpoints.

**Execution stopped before model work because the complete-retention storage gate failed.** No training, feedback generation, evaluation, full-model qualification, checkpoint initialization or automatic retry occurred. All 5,632 planned complete-response generations remain unattempted. There is no result about whether K helps coding tasks yet.

### Implemented and checked

The new `experiments/ok_coding` directory adds data preparation, genuine empty-Adam initialization, seed-specific complete-state continuation, fresh-process/compiler-cache parity, natural event-1 anchor comparison, balanced F/O/K evaluation, conservative storage admission and raw-record reporting. Existing `online_feedback`, upstream training, inference and historical experiment files are unchanged.

Forty CPU tests passed on Fenrir (20.789 s) and the same 40 passed on A800 with CUDA hidden (19.614 s). These include tiny CPU model fixtures and a complete synthetic 4,608-record evaluator/report fixture; they are not scientific model work or GPU qualification. Focused final reporting tests and independent scalar bootstrap reconstruction also pass. Five invalid supervisor bounds are rejected before any start marker. A review finding about recovery bypassing source admission was fixed: explicit complete-boundary recovery rechecks source/model/runtime/configuration and resources before creating a backend. Partial scientific work or incomplete retirement remains fail-closed.

Before transfer, all 2,615 files in the previous live scientific source seal matched. The A800 CPU preflight then verified **2,687 new source/input files**, the sealed scope and configurations, and idle hardware: two NVIDIA A800 80GB PCIe devices, 81,920 MiB each. Available host memory was 137,787,781,120 bytes, above the 85,899,345,920-byte admission requirement. The storage gate stopped before later model/runtime-byte checks and native initialization, so those remain pending for this campaign. Prior qualification does not establish coding-specific peak memory or output preservation.

### Sealed data

Source: `sahil2801/CodeAlpaca-20k` at `3bd2de8099ee8a140653ad9b9df783554d5efabb`, `default/train/0000.parquet`, 3,339,837 bytes, SHA-256 `5a2fa0c510d42843292c583f357086abb59f95796e0f55727b97e449aa93bed9`. The original 20,022 rows retain their order and the project's raw-row partition, split at 18,998. Only instruction/input columns were read; source solutions are unused.

| Reservoir | Raw rows | Historical exclusions | Eligible | Selected | Largest selected prompt |
|---|---:|---:|---:|---:|---:|
| Training | 18,998 | 5,007 | 13,991 | 512 | 170 tokens |
| Evaluation | 1,024 | 690 | 334 | 128 | 95 tokens |

The 5,697 excluded rows comprise 5,681 retained prior exclusions and 16 later allocations. Source catalogs are lookup-only; the separately sealed historical candidate pool is a genuine reservation and remains excluded. No duplicate keys or other eligibility failures remain in these exact source bytes. Selection seeds are 20261030 and 20261031; training seeds do not affect data selection. The same panel and ordered training prompts are used for both seeds, while all eight feedback pools must be generated separately.

All 14,325 eligible-source prompts were independently retokenized against the pinned no-thinking author template. Root also separately reproduced the 640 selected source texts/token arrays, ordering, disjointness and all 4,608 evaluation slots. Original whitespace is preserved. Data binding: `6def4832d12698fc2733e91c427db77558f3fe50e7f51063e9c2a11f9bc961af`. The full inventory scan is preparation-author evidence; reviewers verified its retained receipts and matching allocations but did not repeat the entire 36,973-path scan. Freshness is bounded by inventoried retained history, not pretraining or inaccessible records.

Private exact request inventories, exclusions, tokens, tokenizer/source receipts and the preparation seal remain under `preparation_001/`; the actual A800 path-bound plan is under `production_001/sealed/`. No generation outcome entered selection.

### Concrete storage blocker

| Admission quantity | Bytes | Decimal GB |
|---|---:|---:|
| Required peak free capacity | 366,710,105,681 | 366.710 |
| Available at actual A800 admission | 150,373,031,936 | 150.373 |
| Shortfall | 216,337,073,745 | 216.337 |

The bound retains both capture and prepared-feature copies for eight maximum-length feedback pools, eight required endpoint state/export pairs, four initial states, rolling predecessors, atomic state/export overlap, 16 GiB of isolated compiler-cache allowance, 10% payload slack, 4 GiB compact/workspace allowance and the unchanged 8 GiB reserve. It assumes no favorable generated lengths or unverified offload.

A separate metadata audit found no verified sufficient disposable set. Large-file sums double-count hard-linked aliases; removing those aliases would not reclaim the apparent bytes. All cache/tmp trees together are only about 5.47 GB. Fenrir's approximately 75.55 GB free space is also insufficient to bridge the gap while preserving the retention contract. No historical artifact was deleted and no retention, output-cap, seed, panel, event or reserve requirement was relaxed. The shortfall differs slightly from the earlier 216.312 GB audit because source/input transfer occupied about 25 MB before the actual admission.

This establishes a current resource blocker, not proof that no future placement could work. Launch needs at least 216.337 GB more admitted A800 free capacity, or a verified sufficient retention/placement arrangement. The current implementation's admission mode is local retention; an assumed transfer is not capacity.

### Work accounting and pending result

| Scientific work | Planned | Completed | Unattempted |
|---|---:|---:|---:|
| Feedback generations | 1,024 | 0 | 1,024 |
| Terminal-teacher preparation forwards | 1,024 | 0 | 1,024 |
| Ordinary response presentations | 4,096 | 0 | 4,096 |
| Native Adam updates | 2,048 | 0 | 2,048 |
| Evaluation generations | 4,608 | 0 | 4,608 |
| Extra full-model qualification/warmup work | 0 | 0 | 0 |

There are no invalid scientific rows, unknown scientific attempts or new trained checkpoints: execution never reached those stages. The actual CPU admission exit code is 1 with a preserved resource error. CPU checks, preparation, reviews and transfer are separately recorded operational work. No serving-time, call-count, updater-cost or effectiveness table can be populated from this blocked preparation.

Once resource admission is resolved, the unchanged endpoint remains event 4 K/O separately for each seed, with contemporaneous K/F and O/F and event 2 as secondary. Calls use one validated trajectory per request; times use three-repeat request means. Conditional paired-request intervals share the same PCG64 draws across seeds. No convergence, population-level seed robustness, asynchronous or training-inclusive speedup claim is supported by this preparation.

### Evidence and review limits

- `PREFLIGHT.json`, `PREFLIGHT.log`, `STORAGE_ADMISSION.json`: actual failed A800 admission, source `5cf1c7e2e7c4a4d08d2225b58c56488578112daf`.
- `ROOT_DATA_REVIEW.json`, `SEPARATE_DATA_REVIEW.md`: separate selection/token/schedule checks.
- `SEPARATE_CODE_REVIEW.md`: hash-bound same-team review and repaired recovery finding.
- `STORAGE_REVIEW.md`: inode-aware read-only capacity/retention review.
- `CPU_FINAL.log`, `CPU_REMOTE.log`: 40 passing tests in each environment.

Reviews are separate same-team checks with authorship disclosed, not an external review or an independent GPU rerun. One reviewer initially mishandled the tokenizer's BatchEncoding in an audit script; that failed audit attempt and its correction are preserved, and did not change the prepared inputs. All scientific scope and Stage 1 / project None remain unchanged.
