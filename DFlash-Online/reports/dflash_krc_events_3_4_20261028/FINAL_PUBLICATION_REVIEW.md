# Final compact-publication review

ARS-Codex 3.24.0 experiment-agent validation; status **ANALYZED**. This local same-team review checks exact publication bytes, privacy boundaries and provenance. It does not rerun models or replace the independent execution audit. The reviewer authored the staging helper; a separate same-team reviewer checked that helper and its fixtures. No remote operation, Git commit/push, model work or raw-data publication occurred during this review.

**No publication blocker found in the reviewed package and support files identified below, including the final scalar correction CSV and extraction receipt.**

## Exact package and transformations

The 14 files in `public_package/PUBLICATION_MANIFEST.json` are exactly the list in `PUBLICATION_REVIEW_PLAN.md`: `REPORT.md`, `SUMMARY.json`, `TRAINING.csv`, `EXECUTION_SOURCE.json`, and each panel's `COMPARISONS.csv`, `EVENT_COUNT_CURVE.csv`, `PER_REQUEST.csv`, `ALL_REPEATS.csv` and `REPEAT_TOTALS.csv`. They total **534,743 bytes**. No file is missing, no extra data file or symlink is present, and the only additional package file is the 8,484-byte manifest.

Manifest SHA-256: `4020e8fe18102889029f8abd4f7efdf54f43067b40d6a2282ca273e9b2a49812`. Every original and public size/hash in that manifest was recomputed and matched. All **13 non-report files are byte-identical** to their retained originals. The original report remains 10,565 bytes, SHA-256 `39bb0fc6d94aad629b5c3ed39cc727cf2e41757e41665d82baabef465461da7c`; its public copy is 11,769 bytes, SHA-256 `6506db37b576086815957ffa94738183c4ac8b751c5558044d13ec5143e7f761`.

Applying only the manifest's six literal link replacements and one availability-note append reconstructs the complete public report exactly. All **60 Markdown table rows** and **364 numeric tokens** after excluding Markdown link URLs are unchanged. Favorable and unfavorable results, empty historical values, error accounting and separate panels remain intact. The publication process introduced no numerical interpretation or source-data transformation.

All six link occurrences in the public report have an inspected destination: compact summary/training/manifest files, the existing PR16 `LAUNCH.json`, or the exact execution-commit launch-script object. The GitHub source path includes the required `DFlash-Online/` repository prefix. Raw plan, preflight/postflight, full audit and complete-cost references are explicitly labeled retained private evidence. No dead raw-data link remains in the final compact audit Markdown either.

## Privacy and artifact paths

Whole-file pattern scans and structured/manual inspection covered the 15 package files, including the manifest, and the six supplemental result/review files listed below: **21 files, zero credential/private-key/authenticated-URL/contact-address/IP-address/raw-token-array pattern findings**. JSON dictionaries and arrays were inspected recursively; CSV headers, row counts, string categories, count fields and path fields were checked. This is a bounded review of these bytes, not universal secret-detection certification.

- `SUMMARY.json` contains no error or traceback payload. Audit issues and resume mismatches are empty. Its strings and arrays are statuses, source identities, result rows, accounting explanations and artifact metadata.
- Each `ALL_REPEATS.csv` has 576 executions. All 3,456 structural/error/report-error fields across both files are empty arrays. Each `PER_REQUEST.csv` has 64 requests, with all 128 error fields empty. Stable request IDs and execution labels follow the declared numeric cohort/run naming schemes.
- Prompt/output/accepted/progress fields are scalar counts, not token sequences. No complete question, response, token trajectory, correction-reference response, tensor or model/optimizer payload is present. Additional-panel historical O1/O2 values remain absent.
- Absolute paths identify known source, state/export, retained checkpoint, evaluation-record and transport locations. The public note correctly treats usernames/host paths as operational provenance, not public downloads. The existing decision to retain this metadata is preserved; no original or public path was silently redacted.
- `COMPLETION.json`'s result comparisons, training rows, retained-state identities and audit counts exactly match the reviewed summary. `POSTRUN_STORAGE.json` records one redundant transport archive removal and retained locations, with no production artifacts retired. It contains scalar storage/status/path/hash metadata only.

The full `FINAL_AUDIT.json`, raw logs, sealed input plans, canonical outputs, generation/correction records and tensors remain outside the publication set. The compact audit now explicitly states this boundary. Its stale runtime-label disclosure also remains visible; preserved metadata is not misrepresented as extra training.

The separately extracted `CORRECTIONS.csv` contains **256 rows × 76 scalar columns** for native-Adam steps 257–512. All **19,456 fields** were matched directly to retained window summaries or explicit fixed labels; all 256 ordered input hashes match the extraction receipt. Numerical/boolean fields contain only finite scalar values or preserved blanks. The four string fields use fixed completion/correction statuses, the pre-correction phase label and a precision descriptor. No response identifiers, token IDs, arrays or detailed response records are exported. All four null coefficients and 252 absent optional parameter-version flags remain blank. Reference-prefix fields explicitly precede direct correction and do not claim a final post-correction benefit. The receipt binds the exact CSV, extraction-script and original-audit hashes; `report_analysis_git_commit` identifies the existing report analysis, while the new extraction script has its own hash identity. The core package and report remain unchanged.

## Provenance

**Execution:** `11b7718a04416a89025a6df69c0c33e558e5fe1b`. **Analysis:** `ea4f289c6e18383b7560c8c4103a9d91ca3943bf`. These identities are consistent across the report, manifest, execution-source record, completion record and all three execution-source copies in the summary. Dirty execution files are `[]`.

All **18 execution additions** were independently matched to their exact Git blobs at the execution revision. The report builder's object at the analysis revision matches SHA-256 `3c8e8462a15b7490bc074732a69b13216816d156977f5116d7c62b5e3d20a454`, as recorded by the report and summary. The publication helper has a separate exact identity below; publishing it later does not change either execution or analysis provenance. The compact audit explicitly discloses the analysis auditor's historical O1-source-pointer difference from the execution archive.

## Reviewed supplemental identities

These exact local bytes are covered in addition to the package manifest. The parent's supplemental publication manifest should bind their destination copies to the same identities.

| File | Bytes | SHA-256 |
|---|---:|---|
| COMPLETION.json | 11073 | `b03298ba699a2b43c1bade6777407023bc26fb04039a72847d4268aa41f99456` |
| POSTRUN_STORAGE.json | 797 | `8b68192daf69c5ddfcfa7581ab9e61958bc483858188726cfd990bcde299a8c1` |
| FINAL_AUDIT.md | 8388 | `b08d024299a81b783633f14d2b3131494800b27318c86fe94b18248714cc15be` |
| FINAL_REPORT_REVIEW.md | 9212 | `02411f3fda992df0fe78f8e84d26b2ace0b8f62fcfed109422cf222c7e8a5fb1` |
| CORRECTIONS.csv | 153199 | `614bcf947f787dfa871740e4c8b0994c390cb716bbe5d857134ab93d9224e6b2` |
| CORRECTIONS_EXTRACTION.json | 5415 | `a7be63991461a5b2774f31b2d4f1b47c99954ce07b4d3a53d01f7242d497f758` |
| publish_results.py | 10084 | `7316582bdd7f03f9391b2ba10315eb9b7f8ad0d5b64a5a63dd662476e77dec3b` |
| tests_publish_results.py | 7249 | `e51805014228ffaaf364deb5cc4b6da17c79c161babb8446c42d18981bd28d54` |
| export_correction_records.py | 9829 | `da6c5e43f05ffe38db9a7308b38a3ccd4e9385b88e848f721498bce36d535715` |

The helper, tests and scalar extractor were separately read as source, rather than treated as experimental data. Their credential-shaped strings and private-payload markers are explicit synthetic rejection fixtures or detection patterns, not actual credentials or private responses. Five focused CPU fixtures passed, and the actual report-builder synthetic integration preserved all table rows. The separate helper review found no remaining blocker after correcting the launch-script URL. The extractor performs local scalar projection and consistency checks only; no optimizer update or model pass was added.

This review covers the listed exact bytes and manifest-bound paths. It does not approve arbitrary later edits, linked private evidence, or a directory-wide publication. Stage 1 / project None remains unchanged.
