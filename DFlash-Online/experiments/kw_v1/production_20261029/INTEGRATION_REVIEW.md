# Reuse of the passed KW qualification for production

## Material Passport

ARS-Codex 3.24.0 experiment-agent implementation/integration review. Reviewer: `kw_campaign`, same-team author of campaign plumbing and the repaired qualification harness; this is not an independent code-author review. Local read-only source/record review plus local staging only. No remote operation, model execution, training or serving is performed here. The parent records the current user authorization for unchanged O/K/KW events 9–12; Stage 1 / project None remains unchanged.

## Finding

The existing campaign accepts the passed qualification without a code change or another qualification execution. Use a distinct production root `/data/sources/minhao/runs/dflash_kw_v1_20261029/production_001`, retain executed scientific source `708e974e06f5d9f4b186c32607c16b37cbd4b14f`, and copy the successful receipt byte-for-byte into that root. Preserve the failed original root receipt and all successful qualification compact records.

The local repository HEAD `8c81b06` adds result documentation only relative to 708. The remote execution source is to remain the exact already-qualified 708 files. Do not deploy changed README/report files and then claim the old whole-source seal still matches. In particular, do not rewrite tracked `GITHUB_SOURCE.json`: the earlier zero-work admission failure demonstrated that this invalidates the source seal.

## Required files and operational changes

`prepare_production.py` stages local files only, refuses existing generated outputs, and changes exactly two fields of the passed `neutral_repair_001/PLAN.json`: `config` becomes `production_001/sealed/config.yaml`, and `qualification` becomes `production_001/sealed/QUALIFICATION_INPUTS.json`. All scientific fields, source708, parents, windows, panels, schedules, retention rules and storage budget remain identical. The inherited `qualification_repair.scope` describes the historical qualification; current production authorization is the separate `AUTHORIZATION.json`.

Copy unchanged into the new root:

- `sealed/config.yaml`, `MODEL_PIN.json`, `RUNTIME_PIN.json`, `TRAINING_PIN.json`, `REMOTE_INPUT_IDENTITIES.json`, and `QUALIFICATION_INPUTS.json` from the original study's sealed directory.
- All eleven original `data_preparation/*.json` files, including all four windows and `EVALUATION_SCHEDULE.json`. No selection, shuffle or outcome-based revision is needed.
- Successful `neutral_repair_001/IMPLEMENTATION_QUALIFICATION.json` as `production_001/IMPLEMENTATION_QUALIFICATION.json`. SHA-256: `d731b3e5ca2bdf5ee1a9a53f80eae724ba9a3964ed508314981529b766e65beb`.

The new `sealed/PLAN.json`, `PLAN_CHECK.json`, `SEAL.json`, `QUALIFICATION_REUSE.json`, and `PREPARATION_CHECK.json` record the operational path change and receipt reuse. The seal carries all successful708 source/input identities forward, adds the new required paths and operational records, and derives the exact top-level `experiments/kw_v1` file set from that source inventory. It includes no disposable comparison tensor. Original source seals and receipts are preserved unchanged.

## Binding and gate behavior

`campaign.require_qualification(root)` requires a passed receipt, integer work counters within the fixed 8/8/16 ceilings, true memory/neutral/isolation flags, and exact equality of these bindings:

- K8 SHA-256: `1b1a646eb1dac0b61629a0eb3586b424dea847f487a849f63ec800489de6e9ed`.
- Canonical whole qualification-input JSON SHA-256: `403896b58e38146f15087ffeb3a8b68cbaf519cb6ec6e52968b9c5bbfc62fc7b`.
- Execution commit: `708e974e06f5d9f4b186c32607c16b37cbd4b14f`.

The receipt itself is additionally hashed by the new production seal. Its actual audited work remains 8 forwards, 8 backwards and 14 disposable applications, all 30 attempts complete, 31 exact checks passed, zero serving and persistent updates. Reusing that receipt creates zero new qualification work. Its nested paths to retired comparison tensors are provenance, not files read by the campaign qualification gate.

`preflight.verify` requires the new root's sealed PLAN/config/pins/qualification/original identities, exact current source files, original O8/K8 states and exports, retained event8 qualification features, all evaluation order records, qualified environments and idle hardware. Its descriptive `full_model_preview_memory_qualification_pending` field is a legacy preparation annotation; `campaign.require_qualification` is the actual passed-qualification launch gate. No additional model qualification follows this field.

## Storage and launch

The full campaign still requires **215,574,560,848 bytes** available at initial admission, including 8 GiB reserve, 10% slack, complete captured/prepared feature copies, pending state/export overlap, required event10/12 retention, and rolling event9/11 retirement only after verified successors. Host availability must be at least 80 GiB. No offload throughput is assumed. The parent's newly observed post-retirement space is to be verified again by unchanged preflight; this local review does not claim a fresh server measurement.

Production uses the original immutable O8/K8 endpoints under `/data/sources/minhao/runs/dflash_ok_events_5_8_20261029/campaign_001/{O,K}/events/event_008/`; KW migrates that exact K8. None of the seven retired qualification comparison tensors is a dependency. Original O8/K8, the four longest event8 teacher features, and all new campaign feedback must remain available.

After transfer verification and current idle-GPU/storage admission, the parent's operational wrapper invokes the existing supervisor once and then generates the CPU-only report from retained records:

`bash /data/sources/minhao/runs/dflash_kw_v1_20261029/production_001/launch_campaign_and_report.sh`

The wrapper calls `supervise.sh` with the fixed 28,800-second bound, preserves execution and reporting exit codes separately, and never retries. The supervisor refuses an already-started root. `run_a800.sh` sources the required helper/environment, runs read-only preflight and CPU tests, invokes the campaign once, and performs postflight. It contains no qualifier invocation or automatic retry. The campaign starts a fresh `campaign_001`, collects exactly 512 shared feedback responses from preceding O checkpoints, runs 1,536 live steps / 3,072 presentations across O/K/KW, and includes 1,024 KW previews / 504 credits and 4,032 evaluations. The native event10 gate is already part of those evaluations. Original failed qualification work remains separate historical cost; reused qualification costs must not be counted as newly executed production work.

## Limits

This review verifies compatibility of the source and recorded bindings. Remote source hashes, actual free storage, original state contents and hardware admission remain the launcher's responsibility. The qualified four-response neutral comparison establishes implementation behavior under the repaired fresh-process control; it does not establish efficacy or equivalence across arbitrary compiler histories. Production orchestration remains unchanged as authorized, and valid null or harmful scientific results do not authorize tuning or another event.
