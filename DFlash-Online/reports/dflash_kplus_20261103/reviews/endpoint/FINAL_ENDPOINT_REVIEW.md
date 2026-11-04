# Independent endpoint calculation and interpretation

## Material Passport

- Mode: ARS experiment-agent validate; verification status **ANALYZED**.
- Scientific execution: `e0a9d3c235e1efd15404c549282d21c99243a21c`. CPU reporting repair: `181bf736aee231e36bd82a3bc1f73aa9f1eeb0f2`.
- Separate private auditor SHA256: `a6803396f87ad9fef28960aaa17ca7f8e7c99b7e9937444893854151e0340700`.
- Scope: separate endpoint/statistical implementation, executed CPU-only against the original A800 saved evidence. Its author implemented the K+ objective; this is same-team reconstruction, not independent GPU replication or fresh tensor decoding. The auditor imports only standard Python libraries and NumPy, never the production statistics/report/validation implementation.
- Calculation finished before the regenerated report's endpoint numbers were opened. No new scientific generation, optimization, parameter tuning or output filtering occurred. The full compact local copy is a separate ongoing operation; this review does not depend on that copy having finished.

## Result

The prespecified drafting and complete-request serving criteria both pass on this panel. The primary benefit is small, with a lower interval bound very close to zero.

| Endpoint | Retained K58, measured fresh | K+58 | Reduction, paired 95% interval |
|---|---:|---:|---:|
| Verification calls | 43,387 | 43,267 | **0.276581% [0.004505%, 0.545713%]** |
| Sum of per-request mean complete time | 794.789485 s | 788.247974 s | **0.823049% [0.272580%, 1.379705%]** |
| Output tokens | 314,847 | 314,847 | Exact canonical outputs required |
| Accepted draft tokens / verification | 6.255030 | 6.275499 | Pooled diagnostic ratio |
| Committed output progress / verification | 7.233111 | 7.253172 | Pooled diagnostic ratio |

K+ saves **120 net calls**. Requests improve/regress/tie on **437/360/227 of 1,024**; **792 gross saved calls** are offset by **672 added calls**. A pooled gain does not imply improvement for every request. Total accepted draft tokens are 271,387/271,522; committed progress is 313,823 for both, excluding one initial output token per request.

Counts use one trajectory per request after verifying equality across all three repeats. Times use the per-method, per-request arithmetic mean of three complete timings before pooling and resampling. All intervals were independently recalculated using 10,000 paired 1,024-request draws, PCG64 seed 20261102 and linear 2.5/97.5 percentiles. This quantifies conditional request variation for these fitted checkpoints; it does not quantify training-seed or benchmark-selection uncertainty.

| Repeat | K complete time | K+ complete time | Reduction |
|---|---:|---:|---:|
| 0 | 793.389517 s | 788.941356 s | 0.560653% |
| 1 | 794.996434 s | 787.502072 s | 0.942691% |
| 2 | 795.982504 s | 788.300495 s | 0.965098% |

The fixed 16 blocks are heterogeneous: K+ saves calls in 11 and adds calls in 5; complete time improves in 12 and worsens in 4. These are descriptive summaries of all predeclared blocks, without subgroup hypothesis tests or selection. [Paired comparison and every block](PAIRED_REPORT_COMPARISON.json).

## Evidence and accounting

The audit passed all **6,144 original executions and 259,962 native verification blocks**. It checked sealed request/canonical/count-map identities, every prompt/output token and stop reason, native posterior reconstruction and committed coverage, fixed width, capture-off state, per-request repeated counts and all four historical K58 count fields. It also checked 49 publication transitions, both loaded-weight receipts and all 96 request-batch plus two pre/post serving resource/quiescence receipts. These are retained boundary observations, not continuous surveillance of other processes.

The work ledger independently reconciles **7,473 online generations, 4 correctness generations, 7,424 terminal preparations, 14,848 response forwards, 14,848 backwards and 7,424 optimizer steps**, with zero unknown operations. Adding final evaluation gives exactly **13,621 complete generations**. The training exposure/cost reviewer separately reconstructs event-level receipts; this endpoint audit does not claim to re-decode retired training tensors.

All **12,311 original files consumed** match the earlier 78,924-file remote inventory byte sizes and SHA256 hashes. [Source reconciliation](SOURCE_INVENTORY_RECONCILIATION.json). The small audit outputs were independently hash-verified on transfer to Fenrir. [Transfer verification](RETRIEVAL_VERIFIED.json). The regenerated report agrees on **76 endpoint/repeat checks with zero numerical difference**, and its paired CSV agrees on **1,024 identities and 14,336 per-request scalar checks**. [Endpoint comparison](REPORT_ENDPOINT_COMPARISON.json), [paired comparison](PAIRED_REPORT_COMPARISON.json).

The original wrapper/campaign/report failure remains preserved: exits **1/1/1**, postflight **0**, and the original aggregate audit contains the missing-path error with zero audited rows. An explicit recovery receipt binds six original failure-file hashes; the auditor independently validates those original bytes and all scientific evidence. It never changes a failed operational status to completed. [Original failure](../../EXECUTION_AND_PROVENANCE.md), [strict recovery receipt](../../REPORTING_RECOVERY.json). The first real independent audit passed without an auditor modification or repeat. A later report-schema inspection mistakenly indexed a repeat mapping as a list; that inspection error is retained separately and had no effect on the calculation or data. [Inspection record](REPORT_COMPARISON_ATTEMPT_001.json).

The CPU audit took 12.635 s internally, 12.828 s including interpreter startup/exit. CUDA was hidden, with no GPU processes at admission and 47.919 GB free against an 8 GiB reserve plus 128 MiB analysis allowance. Its output was confined to the authorized analysis sibling; production remained unchanged. [Execution receipt](artifacts/EXECUTION.json), [preflight](PREFLIGHT.json), [calculation](artifacts/result_001/ENDPOINT_RECALCULATION.json).

## Costs and interpretation limits

The separate trainer-author cost reconstruction reports K+ native training **19,272.238 s**, update calls **22,383.303 s**, worker wall **31,887.917 s**, and campaign wall **36,973.308 s**. These scopes nest and must not be added together. The endpoint result measures checkpoint-ready C1 serving; it does not establish an adaptation-inclusive or asynchronous deployment speedup. A historical K training-time comparison is descriptive because training trajectories and operational timing conditions differ. [Cost reconstruction](../../COSTS_EXPOSURE_RETENTION.json).

This compares complete own-feedback online recipes. The original K training features/intermediate states were retired; exact generated tokens or naturally matching anchors cannot establish identical hidden features. The result does not isolate the redistributed-weight mechanism from the subsequent feedback trajectory. The single fitted seed and reused benchmark also limit generalization. The positive primary interval is narrow at its lower edge, so the appropriate claim is a small supported benefit here, not robust superiority across seeds, domains or natural traffic. No additional experiment or adoption decision follows automatically.

## Eleven statistical-fallacy categories assessed

| # | Category | Finding and limit |
|---|---|---|
| 1 | Simpson's paradox | No direction reversal across repeats: calls are repeat-stable and complete-time reductions are positive in all three. Fixed blocks show mixed effects, retained in full; no subgroup-based superiority claim. |
| 2 | Ecological fallacy | Controlled by reporting 437 wins, 360 regressions and 227 ties plus gross gains/harms. The pooled benefit is not attributed to every request. |
| 3 | Berkson's paradox | Selection caution remains: this is the original eligible, train-disjoint but previously used GSM8K panel, not a representative fresh sample of production traffic. No inference outside that scope. |
| 4 | Collider bias | No endpoint conditioning on treatment activity, favorable calls, latency or successful-only records. All sealed requests and repeats enter; incomplete/invalid evidence would block the endpoint. |
| 5 | Base rate neglect | Absolute totals, the 1,024-request denominator, 120 net/792 gross saved/672 added calls and nested adaptation costs accompany percentages. The small magnitude is explicit. |
| 6 | Regression to the mean | Both final checkpoints were measured fresh with balanced 24/24 precedence. Historical counts authenticate K58; old timings are not used. No selection of prior extreme requests. |
| 7 | Survivorship bias | Full work and 6,144 executions reconcile, with zero invalid/unknown/unattempted execution exclusions. The operational failure is preserved and repaired only through complete saved-evidence validation. |
| 8 | Look-elsewhere effect | Primary calls and conditional secondary complete time were fixed in advance; all repeats and blocks are reported. Diagnostic slices are not confirmatory tests. Intervals do not adjust for the wider history of experiments on this reused benchmark. |
| 9 | Garden of forking paths | Seed, horizon, panel, count-once/mean-three rule and bootstrap were fixed before outcomes. CPU path/schema repairs and their prior bytes/failures are retained; no scientific retry, tuning, outlier removal or best-repeat choice. |
| 10 | Correlation versus causation | A complete own-feedback recipe comparison supports this measured endpoint contrast, not isolated weight-mechanism causality. Request resampling omits fitted-seed uncertainty; diagnostic activity cannot prove why the gain arose. |
| 11 | Reverse causality | Training precedes final evaluation. Priority/mixed-response activity also reflects model correctness and feedback; its association with endpoint performance is not used as directional causal evidence. |

Coverage: **11/11 assessed**. The main cautions are the small near-zero-margin primary result, one fitted seed, reused selected panel, heterogeneous request effects, and absence of isolated-mechanism or deployment-net-benefit evidence.
