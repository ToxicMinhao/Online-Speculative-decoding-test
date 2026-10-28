# Independent review of the continuation auditor

## Material Passport

- ARS academic-research-suite 3.23.1, experiment-agent validate; local source and fixture review.
- Reviewer: independent evaluator/test agent; auditor authored by the separate data/audit agent. Shared workspace and study context were available; this is not a blind or cross-model review.
- Verification status: **ANALYZED**. No current-run results, model tensors or live remote state were inspected. No scientific generation, training, GPU or remote operation occurred.
- Scope: `audit_results.py`, `test_audit_results.py`, `AUDIT_PREPARATION.md`, and the sealed producer schemas needed to assess their assumptions.
- Result: **No remaining material blocker found within the documented compact-evidence scope.** Two concrete findings were corrected by the auditor's author and independently rechecked.

## Findings and disposition

| Finding | Consequence | Resolution |
|---|---|---|
| The original final gate required root campaign completion and valid aggregate inventories, but did not require each evaluation child to have completed. `evaluation()` could return `incomplete` with all 384 rows when its own status was `failed`, without adding an audit issue. | A contradictory root-completed/child-failed receipt set could pass the final audit. This was a material completeness gap. | The final gate now requires complete feedback pools, passed own-arm resume receipts, completed training receipts, and complete event-2/event-4 evaluation summaries whose own execution status is `completed` and returned count is 384. A focused regression supplies all 384 valid measurement rows with failed child status and proves the aggregate audit fails; the corresponding completed child passes. |
| Evaluation `cap_hit` was checked using output length equal to 2,048, while the qualified scorer defines it using `finish_reason == 'length'`. | EOS exactly at the output cap could falsely fail an otherwise valid record. This is a schema issue; it is not evidence of a problem in the running campaign. | The check now follows the qualified termination definition. The regression accepts EOS at length 2,048 with `cap_hit=False` and rejects an inconsistent `True`. |

Only local postrun auditor/test files changed for these fixes. Sealed campaign, training, inference and evaluator source were unchanged.

## Independent checks

The final **16 CPU tests passed in 1.162 seconds**, with CUDA hidden and bytecode writes disabled:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /home/minhao2/.venvs/dflash-specforge/bin/python -m unittest test_audit_results -v
```

The tests use synthetic receipts and immutable historical artifacts. I independently reran them after reviewing the fixes. The original 14-test set had also passed independently before the findings were repaired; the two added regressions cover the previously untested cases.

An additional read-only check used retained capture-enabled `train_00002.json` from the original one-event run. The auditor reconstructed 36 verification calls, 213 accepted draft tokens, 249 committed tokens and a 250-token response, with zero issues. This complements the capture-disabled native fixture in the suite. It confirms compatibility with that retained producer schema, not correctness of new-run outputs.

The review traced the following checks through the auditor and their producers:

- Missing required files are recorded globally; malformed sections become explicit issues. Missing `CLOSED.json`, failed or incomplete children, and nonzero supervisor exit cannot yield a passed audit.
- Event-2 parent state identities are bound to the initial seal. Each event binds its own arm's parent paths, objective, saved/live counters, Adam/RNG equality receipt, horizon, accumulation boundary, completed event history and closure. Returned exports and rolling state paths must belong to the correct new arm directory.
- Feedback uses the pre-update O export; its serial native record, canonical commits, capture headers, terminal-feature receipt, prepared feature metadata and both arms' feature lists are checked. Dataset windows remain bound to the independent preparation evidence.
- Native traces independently reconstruct the first draft/target mismatch, accepted prefix, committed token including mismatch, EOS/cap clipping and scalar counters. The wrapper/native capture copies and exact serial inventory are bound; 384 feedback plus 768 evaluation records exhaust the authorized 1,152 generation slots.
- Training checks each response pass, optimizer boundary, cumulative counter, learning rate and normal sampled anchors. K's selected counts, first-error histogram partition, inclusion bounds, exponential weights and per-response epsilon are checked. Additive selection diagnostics are summed at optimizer boundaries only, preventing accumulation-row double counting.
- Evaluation checks exact retained-F inputs, outputs and termination, two-arm ordering, three-repeat count consistency, publication and tensor-equality receipts, the included 48-generation gate, activation accounting and reported totals.
- Qualified feedback/header/terminal and loading-memory cost fields were checked against the current sealed producers. Storage receipts recompute required bytes including checkpoint overlap and reserve. Nested costs are not added into a fabricated deployment total.

## Limits retained in the auditor

The auditor does not reload bulky tensors. Exact weight/Adam/RNG restoration is supported by sealed runtime checks and their saved/live receipts; it is not a fresh independent tensor comparison on this machine. Later rolling-state versions are associated through own-arm paths, exports and runtime receipts, rather than an independently retained local copy of every overwritten optimizer state.

Shared feature byte immutability is enforced by the sealed driver's before/after checks. Local compact headers and identity declarations do not recreate deleted or unretrieved tensor contents. Main-training logits were not retained, so aggregate mask diagnostics do not reproduce the earlier native current-logit qualification.

Partial optimizer counters are described as observed completed calls, with explicit interrupted-call uncertainty. Native file inventories and sequential source support the bounded work claim; they cannot detect an unrecorded external process. Trainer closure receipts support intended phase separation but are not an independent overlapping-activity trace.

The whole actual continuation corpus has not been audited by this reviewer. Retrieval must include the files listed in `AUDIT_PREPARATION.md`; any later audit discrepancy should first be traced to raw evidence and the producer schema. This review does not establish campaign completion, efficacy, seed robustness or adaptation-inclusive acceleration.

## Reviewed source identities

- `audit_results.py`: SHA-256 `1a7243988d3b3bc904c3b33223ce0350ef8849ada971d54061e487197b0dd3a6`.
- `test_audit_results.py`: SHA-256 `5ac631db83a79f5bbd2381cd6685e0cdffc461499b2282d3e4ed1b37ee359559`.

The methodological review covers all 11 ARS categories through the auditor's explicit scan: events/repeats remain separate; aggregate results do not imply per-request gains; fixed-panel selection and older exposure are disclosed; correctness exclusions and partial work remain visible; denominators and all prespecified comparisons are retained; no significance, unique-mechanism or deployment-causality claim is made. This coverage is an interpretation check, not a universal validity certificate.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_ok_continuation_20261027/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.

Historical-source gap: this run archive contains 2 unresolved source-pin occurrences; the missing versions remain explicitly identified.
