## Material Passport

ARS experiment-agent validation; independent local source review. Verification
status: **ANALYZED**. This review did not execute or restart any scientific work,
access the server, or change the sealed campaign code. Continuation outcomes
were unavailable when these checks were performed.

Only `analyze_results.py` and `test_analyze_results.py` changed during the code
review. The fixes bind the original 64-request panel to the continuation's
sealed PLAN bytes, bind evaluated exports to the corresponding training
receipts and event1 parents, require completed evaluator statuses before
labeling the analysis complete, and distinguish reported counters from unknown
interrupted optimizer work even when a CLOSED receipt exists. Completed
training receipts now check event/objective/budget identity. Malformed repeat
values are excluded without crashing; native accepted-token bounds respect the
15 draft positions. Incomplete or inconsistent analysis returns a nonzero exit
code while retaining partial output.

Pooled comparisons are correct: one count trajectory per arm/request after
three consistent repeats; sum of per-request mean timings; historical F counts
use the same valid cohort. O/K timing comparisons remain within each event.
Training exposure sums optimizer-boundary accumulation windows once. Nested
campaign timers remain separate and are not summed into an invented total.

**Validation:** 15 focused CPU tests passed in 0.646 seconds. The captured
command and result are in [ANALYZER_CPU_TESTS_REVIEW.log](ANALYZER_CPU_TESTS_REVIEW.log);
that log was transcribed from the completed tool output, not produced by an
additional test run. A separate read-only check of actual retained event1
records accepted all 64 requests, reconstructed O/K calls as **3075/3027**, and
found no issues in either event1 training receipt. These checks do not certify
the pending continuation results or replace the independent state/runtime
audit.

Analyzer SHA256: `5b50efb532afcc915da596a88eac08bf1f3967d6814bc7dcc3e43c273109aaa3`.
Tests SHA256: `34316a4cbfea923c7d86fbbd3c593459281a15bf1e5d759d4e9b16b5af92507e`.
Retrieval must include `sealed/PLAN.json` and `sealed/SEAL.json` alongside
`campaign_001`; missing provenance prevents a complete analysis verdict.

The initial read-only inspection of post-run `audit_final_states.py` found that
nonempty O/K RNG mismatches were merely recorded under a passed status, and
snapshot/hash exceptions could abort without a durable failure audit. The
parent corrected both; rereading the final source confirms RNG disagreement
now adds a failed-audit error, per-arm exceptions preserve partial inventory,
and `finally` writes the audit. Failed/incomplete audits return nonzero.
Supervisor exit and postflight outcome are also recorded and checked separately
from tensor validation. These corrections received source review only; no
additional test or scientific execution was performed in this follow-up.
`collect_compact.py` covers the expected compact evidence extensions, including
the sealed plan and seal, while excluding tensor payloads. Run it only after the
state audit finishes so its receipt and log cannot race archival; the parent
confirmed this serial ordering. Both scripts require the supervisor EXIT_CODE
marker; collection reserves 8 GiB plus twice the uncompressed compact size and
creates hashes for each archived evidence file. Tensor states and exports stay
on the server and receive the separate state/inventory audit rather than being
silently described as part of the compact archive. No remaining blocking issue
was identified for this post-run sequence. No post-run scripts were executed or
edited by this reviewer.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_ok_continuation_20261027/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.

Historical-source gap: this run archive contains 2 unresolved source-pin occurrences; the missing versions remain explicitly identified.
