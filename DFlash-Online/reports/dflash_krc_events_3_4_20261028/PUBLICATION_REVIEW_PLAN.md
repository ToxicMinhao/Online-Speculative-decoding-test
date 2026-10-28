# KRC events 3–4 compact-publication checklist

Prepared October 28, 2026, using the ARS experiment-agent validation workflow. This is a local schema and publication-scope review while native evaluation continues. No remote operation, artifact transfer, scientific-code change or experiment was performed. Final `REPORT.md`, `SUMMARY.json`, `TRAINING.csv`, `FINAL_AUDIT.json` and `analysis/` were not yet present locally at inspection; this document does not certify their future bytes or assert an experimental outcome.

## Source identities

- **Actual execution:** `11b7718a04416a89025a6df69c0c33e558e5fe1b`, as recorded by run-local `EXECUTION_SOURCE.json`. Preserve its complete identity and `dirty_files: []`; do not substitute the eventual report-publication commit.
- **Analysis implementation:** `ea4f289c6e18383b7560c8c4103a9d91ca3943bf`, containing `provenance/analysis_tools/dflash_krc_events_3_4_20261028/{build_report.py,tests_report.py}`. Pass this full commit as `--postprocess-commit` when building the report. The run-local files exactly match those copies: builder SHA-256 `3c8e8462a15b7490bc074732a69b13216816d156977f5116d7c62b5e3d20a454`; tests SHA-256 `f3e6103e4d8c4590ff80484a6cbbde369f28e29d1daa1077978d05adad9c520c`.
- The diff between these commits is empty under `experiments/`, `online_feedback/`, `training/` and `inference/`. Analysis provenance is separate from the source actually executed. Any subsequent analysis change needs its own revision and exact script hash.

## Explicit compact-file list

Copy only the following generated result files from the run root into the matching paths under `reports/dflash_krc_events_3_4_20261028/`, after checking their actual bytes. These are the 13 compact report outputs plus one execution-identity file; a missing file remains missing and must not be fabricated.

```text
REPORT.md
SUMMARY.json
TRAINING.csv
EXECUTION_SOURCE.json
analysis/development/COMPARISONS.csv
analysis/development/EVENT_COUNT_CURVE.csv
analysis/development/PER_REQUEST.csv
analysis/development/ALL_REPEATS.csv
analysis/development/REPEAT_TOTALS.csv
analysis/additional/COMPARISONS.csv
analysis/additional/EVENT_COUNT_CURVE.csv
analysis/additional/PER_REQUEST.csv
analysis/additional/ALL_REPEATS.csv
analysis/additional/REPEAT_TOTALS.csv
```

The existing PR16 report directory contains exactly these 11 reviewed support files: `AUDIT_PLAN.md`, `CPU_FINAL.log`, `CPU_INDEPENDENT_PRELAUNCH.log`, `CPU_INDEPENDENT_PREPARATION_AUDIT.log`, `EVALUATOR_REVIEW.md`, `LAUNCH.json`, `PRELAUNCH_REVIEW.md`, `PRELAUNCH_REVIEW_IDENTITIES.json`, `REPORT_REVIEW.md`, `STORAGE_MAINTENANCE.json`, and `TRAINER_REVIEW.md`. Retain them without a blanket recopy of the run directory. A bounded pattern scan found no private-key/provider-token/authenticated-URL matches or raw prompt/response token-array fields in those 11 files; it is not exhaustive secret detection.

Completion, audit and publication-review receipts may be useful additional compact evidence, but their eventual contents need an explicit addition to the file list. For an audit extract, select scalar status/issues, training counters/exposure/branches, panel counts/comparisons/repeat totals, native-block counts, integrity counts and source identities. Name it clearly as an extract, bind it to the original audit SHA-256, and do not describe it as a byte-identical raw audit. `FINAL_AUDIT.json` embeds broader preflight/postflight/status/cost structures and is not automatically covered by the 14-file list above.

## Fields requiring attention

| File or fields | Treatment |
|---|---|
| `COMPARISONS.csv`, `EVENT_COUNT_CURVE.csv`, `REPEAT_TOTALS.csv` | Scalar results and labels; preserve all directions, adverse results, absent historical values and separate panels. |
| `PER_REQUEST.csv` | Stable request IDs, counts, termination labels and validity/errors. `canonical_output_tokens` is a length, not a token sequence. Review freeform error fields. |
| `ALL_REPEATS.csv` | `prompt_tokens`, `output_tokens`, `accepted_draft_tokens` and `committed_progress` are counts. Stable request IDs and `original_index` identify the evaluation cohort. `checkpoint_id` and `record_path` are absolute artifact locators; review `errors`, `structural_errors` and `report_errors` for unexpected embedded payloads. |
| `TRAINING.csv` | Scalar exposure/work/costs, correction-branch JSON and `full_state_path`/`export_path`; no model or optimizer tensors. |
| `SUMMARY.json` | Carries whole campaign status and selected nested costs, execution source, resume and retention receipts. Inspect all freeform error/traceback/runtime strings and nested structures, including `panels.*.execution`, `status`, `costs`, `audit.issues`, `resume`, and `retention`. Schema expectations alone do not exclude a payload embedded in a failure message. |
| `REPORT.md`, `EXECUTION_SOURCE.json`, existing launch/storage/review receipts | Hashes, source paths, checkpoint paths, usernames/server aliases, PIDs and filesystem metadata are operational provenance. They are not credentials or artifact downloads. Preserve this distinction in the publication note. |

The prior publication review treats recorded usernames, absolute paths and server aliases as intentional operational provenance, not automatic privacy violations (`research/dflash_online_publication_20261028/PRIVACY_REVIEW.md`). Apply that existing treatment consistently. If a public copy normalizes paths or removes a sensitive error payload, preserve the original sealed/local evidence and record each transformation plus original/public hashes. Never silently rewrite the sealed source, plan, receipt or audit to match its public copy.

Keep raw `PLAN.json`/panel definitions, `REQUESTS.json`, `CANONICAL.json`, per-generation JSON, native request captures, feedback responses, correction/window records and raw logs outside this publication list. They can contain full questions, `prompt_ids`, `response_ids`, `input_ids`, proposed/committed token sequences, reference responses or token-aligned labels. Keep pools/reference schedules, tensors, replay features, checkpoints, complete states, model files, transport archives, staging `code/` and caches outside the compact package as well. Source/configuration identities can be published without copying their underlying data. `.gitignore` excludes common tensors and caches but does not exclude all JSON/JSONL/log/archive files; use an explicit staging list.

## Final package checks

- [ ] Generate from retrieved, independently audited evidence using the pinned analysis script. Preserve incomplete/invalid/unknown/unattempted outcomes. A completed endpoint requires the actual 1,152 valid executions, both 64-request panels, three repeats per arm and a passed audit; do not fill absent results.
- [ ] Verify execution `11b7718…`, analysis `ea4f289…`, and the eventual publication revision remain distinct. Verify the script hash and execution additions manifest against the recorded versions.
- [ ] Review actual JSON/CSV/text recursively for prompt/response text, raw token arrays, secrets, authenticated URLs and freeform failure payloads. Lengths/counts and stable request IDs are intended compact evidence; do not remove unfavorable rows or relabel missing historical O1/O2 additional-panel values as zero.
- [ ] Preserve the main KRC4/O4 comparison, K4/O4 and KRC4/K4 controls, event-1/2/4 count context, cost nesting and timing-repeat limits. Native/cache timings are nested; historical latency is not a learning curve; repeated serving is not independent training replication.
- [ ] Add a clear availability note: raw data/tokens, model and optimizer states, and referenced host paths are retained evidence outside this public package. Generated report links to `raw/sealed/PLAN.json`, `raw/PREFLIGHT.json`, `raw/POSTFLIGHT.json`, raw campaign costs and the full audit will otherwise be unresolved. Mark those links as unpublished evidence or adapt the public copy to available compact/source links, recording any transformation; do not expose raw inputs merely to resolve links. From the GitHub repository root, the published launch script is under `DFlash-Online/provenance/run_tools/dflash_krc_events_3_4_20261028/run_campaign.sh`.
- [ ] Record every final published path, size and SHA-256 in a compact publication manifest; inspect the staged file list and diff for unexpected raw artifacts. Keep the original generated report and raw audit recoverable. This checklist does not authorize deletion or claim retired tensors are archived.

## Local staging helper

Run-local `publish_results.py` implements the exact 14-file list above. Its command is `python publish_results.py SOURCE_RUN_ROOT NEW_STAGING_DIRECTORY`; it does not upload, commit or modify the source reports. It requires the pinned execution/analysis identities, rejects suspicious payload fields/credential patterns and symlink inputs, and leaves missing partial-result files absent. An existing destination is never overwritten. Only six known report links and an appended availability note may change; `PUBLICATION_MANIFEST.json` records original/public hashes and exact replacements. All other copied bytes are unchanged. Existing support files are not copied or modified by the helper.

Five synthetic CPU fixtures pass. An additional check using the actual report builder with synthetic receipts staged all 14 files and preserved all 60 table rows. Independent review corrected the launch-script repository prefix; its execution-commit target was verified locally. No actual campaign result package was staged. These checks do not replace final review of generated result bytes and freeform errors.

This plan supports the already authorized report publication. It adds no experiment, remote action, new research direction or stage change.
