# Independent compact-evidence and work audit

**Result: passed within the stated evidence scope.** The first execution-evidence
and work-ledger audits found no invalid or missing records. Scientific source:
`d51718a7fa993a1494e593d15a8539090227d8b8`. This review performed no GPU/model work,
changed no scientific code and made no alteration to experiment data.

Evidence is the verified execution002 delivery,50,302files/447,860,866bytes. Its
inventory SHA256 independently matches
`7bba748079052cee3dc9125861ee27dd8381ecb04b3fd2bdc31fdc71aaac310a`.
The original failed attempt remains preserved and is included in cumulative work.

## Correctness evidence

`EXECUTION_EVIDENCE_AUDIT_001.json` passes318,996 checks across raw memory,
selection, layout, restoration, update and probe receipts:

- The sealed population contains815immutable tags across128training requests and
 129original full layouts. All16windows retain all815tag observations from both
 P and N, including exclusions and nonselected records.
- Independent reduction from raw P/N records reproduces eligibility, exact-tag
 threats, deterministic T hashes, position-matched U hashes, both selections,
 overlaps, wrong ties and the complete selection rows. Current-request exclusion
 matches the actual two native training presentations, not an inferred pair.
- There are54selected-T occurrences. Every window is active. Window128 has the
 same immutable U/T set; U correctly aliases T. This explains31physical
 disposable updates and63physical exports for64conceptual P/N/U/T endpoints.
 No opportunity window or regression was discarded.
- Both branches retain the same ordered union of original full response layouts,
 anchors and keep masks. Sparse support, zero-weight padding, selected weights,
 teacher/head/recomputation counts and the single shared denominator over all selected replay rows
 reconcile. These are independent checks of records and scalar arithmetic; they
 are not new teacher-gradient or tensor computations.
- All branch P-restoration and N-restoration receipts pass; the two native
 duplicate checks pass. Active branches reproduce native preclip gradients, add replay before the sole
 clipping/Adam operation, and preserve the one-step counter progression. The final native
 state exactly matches historical NATIVE7 under the retained in-process checker:
 weights, optimizer, counters and RNG; global step896/presentations1792. All256
 native presentations retain the intended two-pass/one-step-per-pair accounting.
-64parent reconstructions and256neutral starts retain all15-proposal/posterior/
 commit/output/termination parity. All16,384conceptual endpoint rows link to
16,128physical raw blocks plus256exact alias rows. There are no invalid or
 unattempted endpoint rows. Native qualification uses the original four bounded
 neutral starts and reports zero Adam steps/new generations.

A separate raw-token scorer by the preparation reviewer independently checks
canonical prefix, EOS/cap scoring and publication rows. This review inspected that
scorer and found no concrete issue in those semantics; its own final audit passed.
The in-process cache-state claims remain execution receipts rather than a new CPU
proof of GPU KV contents.

## Physical work and budget

`WORK_AUDIT_002.json` passes54 cross-checks against both raw operation ledgers and
published accounting. Completed execution002 work:

| Category | Response forwards | Response backwards | Adam applications |
|---|---:|---:|---:|
| Disposable startup |2|2|0|
| Qualification, including memory fixture |20|18|6|
| Native trajectory |256|256|128|
| Memory observations |4,331|0|0|
| Native duplicates |4|4|2|
| Disposable U/T source updates |62|62|31|
| Added replay |203|203|0|
| **Execution002 total** |**4,878**|**545**|**167**|

All execution002 operations completed with zero failures or unknown work. Prior
execution001 adds3qualification Adam applications,12completed forwards, one failed
forward, and10completed backwards. Cumulative work is170Adam applications,
4,890completed forwards/one failed forward and555backwards. No failed work is
relabelled unattempted or removed.

Execution002 occupied-card time is7,094.400874333456seconds. Adding the retained
185.9867403432727seconds gives7,280.387614676729seconds, within the unchanged10,800
ceiling. Local168-Adam and cumulative171-Adam limits both hold. Current campaign
wall time7,164.792083965614seconds is separate; nested phase durations are not
additive kernel time.

The native-work ledger records19,329draft-block operations,19,388target calls,
359,497context-forward operations and67model loads/publications. These include
startup, reconstruction, neutral checks and endpoint diagnostic forwards; they
are not new complete-response generations or serving verification-call gains.
The campaign reports zero new complete-response generations.

## Audit provenance and limits

The audit scripts import only the Python standard library and no production
selection/aggregation/model helpers. Ten focused synthetic tests pass. Audit
source identity should be recorded with publication.

`WORK_AUDIT_001.json` is retained unchanged. Its raw numbers/checks were correct,
but its field `current_scientific_adam_attempts=161` included two duplicate checks.
Revision002 replaces that ambiguous label with159native-plus-disposable Adam
applications and2duplicate applications. This is an audit-label clarification,
not an experiment or measurement correction. There was no failed experiment-data
audit to overwrite.

This agent previously contributed native-probe/campaign code and CPU audit
infrastructure. The calculation is separate, but authorship is not blinded and
this is not an external reproduction. Billion-parameter states and saved teacher
tensors were not reloaded; their exact/numerical comparisons are supported by
retained native qualification/restoration receipts. This review establishes
internal execution/accounting consistency, not efficacy, deployment speedup or a
causal explanation of the final outcome.
