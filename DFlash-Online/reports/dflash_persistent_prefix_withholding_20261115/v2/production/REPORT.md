# PR49 production continuation — incomplete run

**The experiment stopped on a packaging bug; there is no final N/E/S result.** N8 and E8 finished, while S8 was interrupted after23 of128 event-eight Adam steps. The final evaluation never started. This is an implementation failure, not a null result or evidence against prefix withholding.

Scope: **DEVELOPMENT_FEASIBILITY_REUSED_DATA**. Execution commit: `db6e1221d8d1b234d9d07e246025d1afd5dc30d5`. Original algorithms, seed, model/runtime, panels, scientific binding and tolerances were unchanged. The execution commit covers the staged DFlash-Online components; the exact separate preparation/staging helpers and sealed payload are retained privately with identities, and were not included in that commit. The historical training/serving qualification was authenticated and reused; none of its8training cases or8serving generations was repeated.

## What failed

The private continuation-package builder changed `plan.references` to the new run directory but did not copy the104authenticated reused canonical records into `references/CANONICALS.json`. The24missing-reference identity join itself passed. The subsequent read of the missing seed file raised `FileNotFoundError` before backend construction or any reference generation. The supervisor then interrupted the concurrently running S8 task, as required after a task failure.

This was our staging omission. The consumer tests created the seed destination explicitly, and the continuation tests mocked the campaign; neither tested this actual assembled package end to end. The CPU preflight checked immutable inputs but missed this writable task prerequisite. The passed187-test suite and32additional focused tests therefore did not establish complete production staging. See [failure analysis](REFERENCE_STAGING_FAILURE.md).

A CPU-only reproduction using the retained package produced the same exception with zero backend constructions or ledger operations. In a disposable directory, copying the exact104-entry seed made the actual loader return24complete requests. That identifies a narrow repair; it is not an applied production fix or authorization to restart.

## Completed and missing work

| Task | Result | New Adam steps | Durable endpoint |
|---|---|---:|---|
| E7 | Complete |128| Export retained; optimizer retired only after validated E8 |
| S7 | Complete |128| Recovery and export retained |
| N/E/S event-eight feedback |384responses and384teacher preparations complete|0| All own-response records/features retained |
| N8 | Complete |128| Recovery and export retained |
| E8 | Complete |128| Recovery and export retained |
| S8 | Interrupted |23| No partial recovery or event-eight export |
| Missing K6 references | Failed before model creation |0|0/24generated |
| Final evaluation | Unattempted |0|0/72warmups;0/1152scored generations |

Production completed535/640Adam steps and1070/1280ordinary training presentations. Five startup initializations added10discarded response forwards/backwards and no Adam steps. Including the preserved prior qualification attempts, cumulative work is:

| Work | Planned | Attempted | Completed | Unknown | Unattempted |
|---|---:|---:|---:|---:|---:|
| Adam applications |650|545|545|0|105|
| Response forwards |1316|1107|1106|1|209|
| Response backwards |1316|1106|1106|0|210|
| Training initializations |8|8|8|0|0|
| Feedback generations |384|384|384|0|0|
| New reference generations |24|0|0|0|24|
| Historical serving qualification generations |8|8|8|0|0|
| Final warmup generations |72|0|0|0|72|
| Final scored generations |1152|0|0|0|1152|

The interrupted S8 forward remains explicitly unknown. Its ledger reservation was opened but not closed; no backward or Adam step was entered for that presentation. A stale copied `unknown=0` field is not used: unknown work is recomputed from attempted/completed/failed records. See [work accounting](WORK_ACCOUNTING.csv).

S8's last durable failure receipt reports globalstep919/presentations1838. These are counters, not a resumable checkpoint. Remote inventory confirms noS8`.pt`/`.safetensors` state; the last complete S state is S7 at896. Retrying S8 from S7 would repeat23already-consumed Adam steps. Even without further failures, total work would become673Adam,1365attempted forwards,1364backwards and9initializations, exceeding this addendum's caps. No retry was launched.

## Duration and resources

The full-run estimate at admission was64.6minutes raw and126.9minutes conservative. The production wrapper stopped after**21minutes12seconds**; this is time to failure, not completion time. CPU preflight took49.56seconds. Production occupied-card leases totaled40.24card-minutes across both A800s; including all earlier qualification/startup failures, cumulative occupied-card time is47.49minutes. These are resource-lease times, not a kernel-time breakdown. Original first-load epoch1794802603.911516 and its four-hour deadline remain unchanged.

| Training task | Task wall(s) | Training component(s) | Startup(s) | Save(s) | Export(s) |
|---|---:|---:|---:|---:|---:|
| E7 |483.47|346.81|25.64|17.99|14.38|
| S7 |437.72|332.11|21.87|17.82|13.85|
| N8 |420.19|319.39|18.38|19.47|15.07|
| E8 |437.89|321.88|27.80|18.78|14.93|
| S8 interrupted |115.96|Incomplete|Recorded separately|Not reached|Not reached|

Task wall times contain their component costs and overlap across cards; do not add components to totals or treat these as matched performance comparisons. [All task costs](TASK_COSTS.csv) retain loading, publication waits, feedback generation, teacher preparation and persistence separately. S8's interrupted components remain incomplete rather than filled with zero.

At postflight both A800s were idle. Diskfree was670.23GB before compact-delivery packaging (670.22GB afterward); budget-charged allocation was38.862GB, including the full4GiB compiler-growth reservation. Actual allocated run/source roots plus observed compiler growth were34.599GB; both are below110GB. No broad cleanup occurred. Original K6/N7 and all surviving completed endpoints passed fresh tensor-file identity checks; all4448staged source entries were unchanged. E7's bulky optimizer state was retired only after validated E8; its serving export remains. The S7 recovery needed for a possible restart is preserved.

## Evidence and next decision

All2867compact delivery files were transferred and verified against their recorded sizes and digests. An independent CPU audit passed4,121,758checks of raw observations, state/accounting receipts and resource records, with zero errors; this verifies the partial evidence, not experiment completion. See [partial audit](PARTIAL_AUDIT.md). Raw feedback tokens, traces, failure records, work journal, interrupted observations, states and provenance remain outside Git in the project evidence directory. [Machine-readable summary](RESULTS.json) and the scalar accounting tables contain no prompt text or token arrays.

The required code repair is to stage the authenticated104-entry seed and invoke the real reference-readiness consumer on the assembled package before starting any worker. Its regression must assemble/unpack the actual package without manually supplying the missing file in a fixture. A retry would additionally require explicit cumulative-budget and original-clock reconciliation because S8 cannot resume at step919. No repair/retry, additional qualification, generation, shortened comparison or algorithm change was silently applied after failure.

No latency, acceptance, verification-call or E-versus-S conclusion is available. Qualification remains passed; production feasibility and E efficacy remain unestablished. This report stops at the failure, as the handoff requires.
