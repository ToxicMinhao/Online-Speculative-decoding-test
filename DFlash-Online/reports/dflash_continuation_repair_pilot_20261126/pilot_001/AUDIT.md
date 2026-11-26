# Independent evidence audit

All four final CPU audits pass. They operate on retained evidence and add no model forward, backward, optimizer step or generation.

| Scope | Checks | Receipt |
|---|---:|---|
| Qualification, warm state, schedule, actual work and source chain | 6,957 | [AUDIT_NATIVE.json](AUDIT_NATIVE.json) |
| All 792 outputs, termination, native traces, repeats, primary arithmetic, bootstrap and decision | 7,787,452 | [AUDIT_SERVING.json](AUDIT_SERVING.json) |
| Matched parent/BASE/TAIL prefix metrics | 1,856,640 | [AUDIT_MECHANISM.json](AUDIT_MECHANISM.json) |
| Phase costs, failures, work and resource limits | 337 | [COST_AUDIT.json](COST_AUDIT.json) |

Three separate agents independently recomputed these quantities with implementations separate from production reducers. They share task history, source and some implementation authorship; this is not blinded independent replication. No second GPU campaign was run. Small allocation mathematics and native integration are supported by prior CPU fixtures and the preserved GPU qualification, not independently regenerated billion-parameter tensors.

The native work audit checks saved scientific loss/gradient/parameter/Adam/RNG fingerprints and runtime parity assertions. It cannot reconstruct full tensors or repeat the FP64 oracle from compact coefficient summaries. The mechanism audit independently rebuilds prefix outcomes from canonical/proposal tokens; it cannot establish strongest-rival optimality across the full vocabulary or repeat the native model forward. The serving audit reconstructs 47,676 verification blocks, confirms all 792 exact output/termination checks and recomputes the 10,000 paired-request bootstrap. Postflight additionally rehashes 492 live input/endpoint files and four source trees.

Initial final native/serving audits stopped because compact retrieval omitted six extensionless prior process receipts. Those failed audit receipts remain private and unchanged. All six files (30 bytes) were fetched and authenticated against original REPAIR hashes; native_002 and serving_002 then passed. This was a delivery omission, with zero new GPU work. Successful public receipts omit private source IDs, tokens, anchors and full locators.

All 54 final focused implementation tests pass. Independent audit fixtures pass 14 native and 13 serving tests; mechanism fixtures enumerate 1,365 cases. Audit code is in [audit_code](audit_code/); exact file hashes are in AUDIT_CODE_IDENTITIES.json. All 11 statistical fallacy categories are addressed in [INTERPRETATION_REVIEW.md](INTERPRETATION_REVIEW.md). The intervals concern the fixed checkpoints and reused panel, not training-seed replication. Nonsignificance is not equivalence, and descriptive TRAIN gains are not serving gains.
