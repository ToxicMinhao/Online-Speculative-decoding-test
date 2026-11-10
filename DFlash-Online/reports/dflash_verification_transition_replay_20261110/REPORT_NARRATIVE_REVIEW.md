# Final report narrative and accounting review

No concrete arithmetic or interpretation error was found in the reviewed report. Execution remains `d7b1311e7b2f910e935bf81b4e40282767d7540c`.

Actual first-progress advances remain distinct from verified first-error repair; all 71 observed advances separately pass the latter check. The report retains regressions, isolated positive restart effects and all selected requests. It correctly limits the result to the fixed checkpoint pairs and reused panels, with no serving-speedup, unique percentage attribution or general training-mechanism claim. Its completed-run conclusion is consistent with the prespecified generally-zero-or-beneficial branch.

An additional independent CPU pass through all 520 raw records exactly reconciles the native rounds, original prefills, reconstruction segments and context tokens to the complete ledger. Details are in `RAW_OPERATION_RECONCILIATION.json`. Actual source-cache reconstruction accounts for 1,081 segments and 28,922 processed tokens; the ledger tag `rebuild=True` also includes later canonical suffix appends and must not be labeled pure reconstruction work. The report's current aggregate totals are correct.

This is a same-team evidence review with native-implementation authorship overlap, not an independent GPU reproduction. No model or remote work and no source changes were performed. The JSON receipt binds the reviewed report bytes; later packaging changes require the executor's ordinary final check.

Final additions rechecked: 23,888 target-prefill + 221,306 ordinary/continuation draft + 28,922 constructor token occurrences sum to 274,116. The 1,081 actual constructor segments remain distinct from the broader `rebuild=True` metadata. The three retained CPU-audit errors are accurately described as a trace-schema mistake, a helper lookup error exposed by a fixture, and a helper insertion error. None was a native execution failure. The JSON hash has been refreshed to the final reviewed report bytes.
