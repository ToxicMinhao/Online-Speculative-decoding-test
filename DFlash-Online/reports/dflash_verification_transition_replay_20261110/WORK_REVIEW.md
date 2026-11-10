# Independent work/provenance audit

## Material Passport

Verification status: ANALYZED. CPU-only; no new model work. Same-team review with prior authorship overlap in admission fixtures and input preparation.

Execution: d7b1311e7b2f910e935bf81b4e40282767d7540c. Core ledger/accounting checks passed.

| Work | Attempted | Completed | Failed | Unknown | Remaining allowance |
|---|---:|---:|---:|---:|---:|
| context_forward | 29342 | 29342 | 0 | 0 | not separately capped |
| crossed_suffix | 256 | 256 | 0 | 0 | 0 |
| draft_block | 27800 | 27800 | 0 | 0 | 4968 |
| model_load | 521 | 521 | 0 | 0 | not separately capped |
| neutral_suffix | 8 | 8 | 0 | 0 | 0 |
| original_completion | 256 | 256 | 0 | 0 | 0 |
| target_call | 28053 | 28053 | 0 | 0 | not separately capped |

Occupied GPU time: **1301.658s**; limit 14,400s.
Neutral checks: 8; originals: 256.
Unused crossed allowance: 0; no-divergence requests: 0; terminal crossed cells: 0.
Source files authenticated: 1139; four draft-weight postflight checks pass.
Full source/target/runtime postflight review: passed.
All selected requests completed; zero failed, unresolved or unattempted required scientific operations. Unspent block allowance is unused capacity, not missing planned work.

Only existing ledger intervals are used for work costs. Native initialization/publication, original walks, suffixes and context operations overlap; never add all categories as kernel time.
Context-token totals include repeated ingestion. No peak-memory estimate is invented from endpoint idle-memory receipts.

Scientific transition outcomes, bootstrap findings and recommendations are outside this audit.

## Raw-work reconciliation

Independently read all 520 native records. They contain 27,749 proposal-plus-verification rounds. Startup contributes three additional draft blocks and 48 standalone verifier blocks, giving 27,800 charged blocks. Target calls add 256 request prefills and 48 startup calls to the native rounds, giving 28,053.
Context work comprises 256 target prefills / 23,888 token occurrences, 28,005 ordinary/continuation draft appends / 221,306 token occurrences, and 1,081 constructor-only ingestion segments / 28,922 token occurrences. These reconcile to 29,342 context operations and 274,116 processed token occurrences across target and draft paths.
The stored rebuild=True label also covers continuation-time appends. Constructor-only counts above come from the separate saved reconstruction receipts; 14,739 rebuild-labeled operations must not be described as constructor rebuilds alone.

## Recorded cost and resources

| Existing ledger interval | Count | Seconds |
|---|---:|---:|
| Native draft publication | 520 | 726.949 |
| Initial native-engine startup | 1 | 21.980 |

Wrapper wall time: 1338.0s; native-runner wall time: 1334.351s; final read-only identity audit: 23.871s, performed after occupied GPU accounting ended.
Admitted storage: 75,657,842,688 free bytes, with 8,589,934,592 reserve plus 2,147,483,648 artifact allowance. No dedicated peak-memory measurement is available.

All 1,421 full postflight identities were compared against the exact preflight expected paths and hashes: 282 artifacts plus 1,139 source files. Both actual commit and clean-tree receipt agree. The 57 A800 launch CPU tests passed. Seven focused private auditor fixtures pass; an initial helper lookup failure and a subsequent helper insertion error were repaired only in CPU analysis code, with their failure receipts preserved. Native execution/source and raw results were not changed.

ARS interpretation coverage: 11/11 categories checked within this work/provenance scope. No scientific effect or causal conclusion is drawn here.
