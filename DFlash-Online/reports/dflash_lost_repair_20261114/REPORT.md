# K+ lost-repair gate: recurrence is real, matched weight is below the screen

## Material Passport

Completed November 12, 2026. ARS experiment-agent run/validate. Scientific execution **29b6a15c5f5f16135663ea61e3d31f21b3f1d465**; independent raw-audit/publication helper code **8a5d98e1b50d1a60c9d9bbe5b5b1550659d095af**. Verification status: analyzed native evidence with an independent raw-record arithmetic reduction; this is not independent GPU replication. [PR42](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/42).

**The full probe is complete. The prespecified gate fails its weight-mass threshold.** Lost repairs recur across many requests and are present in the native training schedule. The required matching leaves sufficient anchor/request support, but insufficient projected weight under the fixed screen. No algorithm was trained or changed.

## What was measured

Used the original seed 17 GSM8K event 007 feedback requests, manifest rows [768, 896), with authenticated K+4/5/6 exports at 512/640/768 steps. No compatible exact-K+6 capture existed, so 128 responses were generated once. Censused every natural K+6 start, including opening and terminal rounds. Replayed each older checkpoint at the exact same K+6 parent history, with neutral K+6 parity. Models/runtime, BF16, greedy block 16, concurrency 1, full prompts, 2,048-token-or-EOS cap and native cache construction remain pinned.

A lost repair H means the same genuine slot first fails under K+4 and K+6, while K+5 accepts through it: a4=a6=j−1 and a5≥j. A persistent error M first fails at that slot under all three. Seeds are excluded, and EOS/cap clipping is preserved. This is a census on K+6's histories, not three independent serving trajectories.

Loaded a disposable copy of K+6's exact recovery RNG and reproduce its two-pass native schedule. Ran 256 read-only response forwards at fixed K+6; retained native full-block training conditioning, masks, 512-anchor ceiling, positional decay and response-wide coefficients. Matched H/M only within presentation, response, first-error depth and high/low/ineligible priority, using the fixed seeded order, nearest available wrong-minus-canonical margin and no replacement. The 0.25 factor is descriptive arithmetic only.

## Census

| Measurement | Result |
|---|---:|
| Requests / natural starts |128 /4,877|
| Genuine K+6 rejections |4,252|
| Lost repairs H |252: **5.1671%** of all starts; **5.9266%** of genuine rejections|
| Requests containing H |105/128|
| Largest single request's H contribution |10/252 (3.9683%)|
| Persistent errors M |2,147|
| Other rejection histories |1,853|
| All-correct or terminal starts |625|
| Converse: K+6 passes a blocker shared by both older checkpoints |341|
| Previously attained accepted-prefix occurrences lost across H, Σ(a5−a6) |764|

H is broad across requests rather than concentrated in one example. The 764 occurrences and 341 converse cases are fixed-start observations; they are not output-token loss, recoverable serving savings or a net request-speed estimate. Depth and a5−a6 distributions are in [SUMMARY.json](SUMMARY.json); every history is retained in [ALL_STARTS.csv](ALL_STARTS.csv).

## Training exposure and matching

| Stage | Scheduled occurrences | Distinct H anchors | Requests |
|---|---:|---:|---:|
| Native replay |252|252|105|
| Appears in two-pass schedule |503|252|105|
| Same genuine training first-error j |479|240|104|
| Comparable M available |237|119|69|
| Matched without replacement |229|115|69|

All 252 H anchors appear in the schedule. Twelve fail the same-first-error agreement under training conditioning; this probe does not identify the cause of that difference. Among 479 same-j occurrences, 242 have no comparable M and 8 lose their comparator through matching without replacement. Thus **there is no missing-anchor exposure gap on this batch**; the prescribed matching removes substantial support.

The full schedule contains **64,564 kept anchor occurrences**, representing 32,356 distinct response/start identities. There are 51,846 first-error occurrences, including 1,939 ineligible-successor cases.

Before matching, H receives 315.502452 native first-error weight: **1.141200%** of all 27,646.563011 first-error weight, and **0.109491%** of all 288,152.924683 selected loss weight. Distinct-anchor support is counted once; weight counts both scheduled presentations. Native response-wide FP32 mass is the denominator; serialized per-anchor sums agree within the unchanged native tolerance.

Matching is not exact margin balance: mean H margin 0.883734 versus M 1.696507; mean absolute difference 1.107533, maximum 4.625. High/low/ineligible classes remain separate, and unavailable successors retain the native low coefficient. See [NATIVE_WEIGHTS.csv](NATIVE_WEIGHTS.csv),[PRESENTATION_WEIGHTS.csv](PRESENTATION_WEIGHTS.csv) and [MATCHED_PAIRS.csv](MATCHED_PAIRS.csv).

## Prespecified gate

| Criterion | Observed | Required | Result |
|---|---:|---:|---|
| Distinct matched H anchors |115|≥32|Pass|
| Distinct contributing requests |69|≥16|Pass|
| Projected matched weight / all native first-error weight |**0.146918%**|≥0.25%|**Fail**|

Projected mass is 40.617696 = Σ[0.25×min(wH,wM)]. No weights were changed. The gate decision is **FAIL**, not INCOMPLETE. The result supports recurrence and scheduled exposure on this batch, but does not meet the agreed feasibility screen for the particular matched allocation. It does not prove that another correction would fail. Any mechanism proposal requires a separate discussion; this execution stops here.

## Correctness and independent audit

All 128 requests, 4,877 neutral replays, 9,754 historical replays and 256 read-only presentations completed. **Zero invalid, failed, unknown or unattempted prescribed observations.** Conservative budget ceilings left unused are not missing natural starts.

Neutral K+6 matches all 15 proposals, acceptance, commits, positions, termination and clipping at all 4,877 starts. All 14,631 reconstructions pass target-prefix cache checks; neutral draft caches also agree at every start. Historical checkpoints rebuild their own draft caches. Proposal-equality columns for K+4/K+5 are intentionally not applicable, not missing parity evidence. Source, checkpoints and recovery identities are unchanged after execution. Caller RNG/modes, model parameters and ordinary gradients remain intact. All 71 focused execution CPU tests and 5 publication tests passed. A further independent scalar/table reduction passed 202,150 checks; its script and [review](FINAL_ANALYSIS_REVIEW.md) are included with this publication.

A separately implemented raw-record reduction passes **1,031,607 checks**, with zero errors/missing evidence, and reproduces labels, full denominators, schedule bindings, weights, matching and gate. It imports no production analysis helper and executes no model. Reviewers are on the same team and have some code authorship overlap. Original ingestion ranges are independently derived from prompt/commit records; individual reconstructed ranges are not separately serialized, so exact execution of those ranges additionally relies on constructor assertions and cache parity. Training argmax observations are checked from retained forward receipts rather than independently regenerated logits. See [INDEPENDENT_AUDIT.json](INDEPENDENT_AUDIT.json),[PARITY_SUMMARY.json](PARITY_SUMMARY.json) and [PARITY_RECEIPTS.csv](PARITY_RECEIPTS.csv).

## Work and duration

| Work / phase | Actual |
|---|---:|
| New complete feedback responses |128|
| Natural feedback rounds |4,877|
| Neutral / historical replay rounds |4,877 /9,754|
| Read-only response forwards / vocabulary chunks |256 /8,170|
| Backwards / optimizer steps / applied weight changes / timed endpoints |0 /0 /0 /0|
| Census phase, including initialization/capture/preparation/reconstruction |2,673.613 s (44.56 min)|
| Read-only phase, including process setup |150.098 s (2.50 min)|
| Read-only forward loop within that phase |109.792 s|
| Read-only initialization / schedule construction |33.409 s /1.651 s|
| Aggregate occupied GPU time |**2,853.504 s (47.56 min)** of 14,400 s|
| Supervisor wall time, excluding earlier CPU preflight |2,853.545 s|
| Read-only training peak allocated / reserved memory |5.20 /6.36 GiB|

Executed on physical A800 GPU0 (UUID GPU-9986735a-3e68-d1bd-5998-9634ef24341e); both cards are idle afterward. Initial estimate 1–2 hours was refined to 45–60 minutes from observed throughput; actual GPU time 47.56 minutes. CPU preparation, source staging, evidence transfer and publication are separate.

Native draft operations 19,559 include 19,508 feedback/replay rounds plus 51 startup operations; do not add 51 again. Target calls 19,812 include 19,508 verification calls, 128 prefills, 128 terminal teacher-preparation calls and 48 startup calls. There are 366,527 context operations. Training performs no teacher-head forward, backward or startup optimizer work. Operation ticket timings in [COST_ACCOUNTING.json](COST_ACCOUNTING.json) overlap, may include asynchronous dispatch and host work, and are not a disjoint kernel-time profile. No serving-speed claim is made.

Storage admission: 624.287 GB free against 48 GiB peak-plus-reserve requirement. Closeout: 619.518 GB free. No artifacts deleted. All 50,093 compact evidence files (482,180,433 bytes) were retrieved and hash-verified on Fenrir; the 41,490,154-byte archive has SHA256 `3bd10ea82c848c415034be4abcf401476088c6596bae9bfbd65dc93555144b6f`. Original tokens/features/recovery state remain private on A800; compact raw prompts/tokens remain private in the verified evidence copy. No tensors were transferred or published.

## Reproducibility and limits

[IDENTITIES.json](IDENTITIES.json) records all checkpoint/recovery/config hashes, manifest/batch/source identities and the two-pass schedule hash `6ac18ca2c16ded078f831ab90b72b58333403e5bece9553943de491e5193a1d0`. The schedule contains 256 presentations and independent native draw replay reaches the same final RNG state.

[PER_REQUEST.csv](PER_REQUEST.csv),[ANCHOR_COVERAGE.csv](ANCHOR_COVERAGE.csv),[GATE.json](GATE.json) and the full executor receipt report retain all outcomes. The additional public tables include no prompts, token arrays or tensors. The same-team training audit separately passed 742,225 scalar/schedule checks; its authorship overlap and receipt-level scope are disclosed in [FINAL_TRAINING_REVIEW.md](FINAL_TRAINING_REVIEW.md). Review coverage includes all 11 ARS interpretation checks; this is a descriptive fixed-batch census with prespecified screening thresholds, not a significance test or a causal evaluation of training. Single batch/seed, K+6-conditioned histories, imperfect margin balance, future repair retention and complete-request performance remain limitations. No tuning, new arm or follow-up run was performed.

Public CSV line endings were normalized to LF for Git; cell values and row order are unchanged. The original byte-identical receipts remain in the verified private archive.
