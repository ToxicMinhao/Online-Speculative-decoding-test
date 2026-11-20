# Final public-report review

## Material Passport

- Scope: local text, numeric, link and privacy review against completed evidence. No model, remote or network work.
- Reviewer: the separate interpretation reviewer; no authorship of the endpoint implementation or statistical reducer. No independent GPU replication claim.
- Scientific execution: `96f0a13d189b8f5a1d3cfcf103a0ff1097f18cd8`.
- Result: **101 checks passed; no scientific-number or interpretation blocker found.** Two documentation/closeout items remain.

## Targeted findings

1. **P2 — delivery assertion is ahead of its receipt.** `REPORT.md:93,105,107` links `STORAGE_DELIVERY.json` and states that the private evidence archive was verified on Fenrir. The receipt is absent at review time. This is the known pending transfer/member-verification dependency, not a newly discovered experiment failure. Complete and check the receipt before publishing the assertion, or mark delivery pending. Both missing links point to this file.
2. **P3 — identify the immutable plan as historical input.** `EXECUTION_MANIFEST.json` retains 22 `current_presence_and_identity="pending_remote_reauthentication"` fields from readiness. Current authentication passed. Label this document as the immutable launch/preflight input plan and link or include the completed preflight receipt, so a reviewer does not mistake sealed planned values for current status. Preserve original plan bytes and binding.

## Verified content

- The public summary exactly matches the completed retained summary. Per-request/repeat tables, work/clock/cost, output-audit and export-identity receipts match retained evidence. The public independent audit deliberately omits duplicate statistics and explicitly points to the summary; all other audit fields are unchanged.
- Primary C/AREPO time reduction 1.6103% [1.3375, 1.8839] and call reduction 1.6803% [1.4089, 1.9516] are correctly rounded. Both secondary contrasts, call and latency wins/regressions/ties, gross calls, lengths, cap/EOS counts, native/stop-clipped/progress measures and all nine repeat rows agree.
- Counter definitions distinguish `commit_rows−1`, `min(accepted_end−1,commit_rows)` and raw pre-stop matching. No incorrect inequality between native and stop-clipped matching is asserted.
- Loading is labeled as included in phase wall time. Overlapping card phases and nested/component timers are not added. Audit 128.424s denotes the inner verifier timer; outer execution is 128.592s. These are compatible different scopes.
- The report identifies one prespecified primary contrast, marginal/descriptive secondary intervals, fixed checkpoints, distinct own-feedback histories, within-math transfer and absence of training-inclusive or full-AdaFlash-system claims. All 11 ARS categories are present. No erroneous equivalence or isolated-component causal claim was found.
- 42 local Markdown links were checked: 40 resolve, and the 2 pending links name the delivery receipt. No nonempty native prompt/response/token/feature payload keys or model/tensor/archive files were found in the public JSON/package scan. IDs, hashes and provenance paths remain metadata. External links were not fetched in this local-only review.

The scientific report can proceed once the documentation/closeout items are resolved. Neither requires training, generation, changed tolerances or a new experiment. This review does not itself verify the not-yet-delivered private archive. Source-bound details and checks are in `PUBLIC_REPORT_REVIEW.json`.
