# Paired C confirmation: completed, incremental benefit unresolved

The complete paired campaign finished. C has small favorable point estimates on GSM8K and MathQA, but **neither dataset demonstrates a C-over-native-K+ improvement** under the prespecified paired intervals. This is not evidence of equivalence or certified preservation.

Production executed commit `0dc9538e8f445faa8bc7471dc9f82d885e98d529`; native qualification executed `31b36fd27976fb46bb6754d8b8b7e769c3d6e5f6`. [PR53](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/53). The original [qualification-only stop report](../REPORT.md) and failed time-admission receipt remain unchanged. The user subsequently authorized completion beyond the elapsed-time limit; the original clock, consumed work and every other scientific/resource limit were retained.

## Results

Each dataset has 256 paired requests, two final checkpoints and three fresh timing repeats. Counts use one validated trajectory per request; times are sums of per-request means over all three repeats. Positive relative values favor C. Intervals are paired 95% request-bootstrap intervals.

| Outcome | GSM8K | MathQA |
|---|---:|---:|
| Native K+ verification calls |11,251|17,952|
| C verification calls |11,231|17,935|
| Calls saved, net |20|17|
| Call reduction |+0.178% [−0.240%, +0.597%]|+0.095% [−0.259%, +0.452%]|
| Native K+ accepted proposals / call |5.772020|6.365586|
| C accepted proposals / call |5.784080|6.372568|
| Acceptance gain |+0.209% [−0.280%, +0.706%]|+0.110% [−0.299%, +0.526%]|
| Native K+ total mean request time |193.098374 s|313.272089 s|
| C total mean request time |192.910034 s|313.005839 s|
| Request time saved |0.188340 s|0.266251 s|
| Time reduction |+0.098% [−0.330%, +0.524%]|+0.085% [−0.263%, +0.437%]|
| Requests improving / regressing / tied in calls |95 / 85 / 76|91 / 90 / 75|
| Gross calls saved / added |153 / 133|179 / 162|

Every prespecified effect is `DIRECTION_UNRESOLVED`. Both arms have identical final output tokens and termination on every evaluated request. All 3,072 scored executions and 96 final warmups are valid; counts are consistent across repeats, with no invalid, missing or outcome-excluded rows. GSM8K has one cap-truncated request and MathQA nine, retained in all totals; neither has a zero-call request.

The earlier GSM8K benefit is **not demonstrated again** in this new paired realization. MathQA remains unresolved. New continuation randomness and new evaluation inputs changed together, so this comparison cannot explain why its effect differs from historical screens. It also does not prove that C cannot help elsewhere.

## What was compared

N is native K+. C is the unchanged bounded prefix-allocation intervention: smoothing 0.5, relative coefficient cap 0.25, detached FP32 allocation and the native BF16 KL graph. Scientific objective/allocation source files retain their qualified identities. C redistributes existing correct-prefix mass without changing direct first-error treatment, support, denominator or response averaging.

Both GSM8K arms restored the complete KPLUS6 origin and ran events 7 and 8. Both MathQA arms restored complete KPLUS4 and ran events 5 and 6. Every first event used the same dataset-specific retained pool; each arm generated and prepared its own 128-response pool for its second event. No packet was exchanged merely because the generated answer matched. Each arm completed 256 new Adam applications over two events; all eight event recovery states are retained.

The one-time continuation fork replaces only the specified Torch CPU and assigned logical-CUDA RNG fields with seed 29 after exact origin restoration. Historical initialization remains seed 17; Python/NumPy state, weights, Adam moments/counters and history are preserved. Second events restore their own first-event state without reseeding. LR 0.000300, horizon 7,424, BF16, sampling, clipping and all other qualified settings remain unchanged.

The intervention was active: C changed all eligible prefix blocks (prefix length at least two): 56,115 / 53,425 in GSM8K E7/E8 and 88,141 / 89,592 in MathQA E5/E6. Moved scalar prefix mass was approximately 3.32% / 3.27% and 3.16% / 3.16%, respectively; this is half the absolute coefficient change divided by retained prefix mass. All recorded mass, coefficient-bound, support and finite-value checks pass. [Event summaries](COEFFICIENT_EVENTS.csv), [prefix-length rows](COEFFICIENT_PREFIX_LENGTH.csv) and [length/slot rows](COEFFICIENT_LENGTH_SLOT.csv) pool original response observations once. Native-arm allocation histograms were disabled, not fabricated. Activation demonstrates that C reached training; it does not establish useful drafting or explain the unresolved endpoint result.

Serving uses the pinned Qwen3-8B/DFlash native runtime, greedy block-16 decoding, 2,048 tokens or EOS, context 4,096, concurrency one, disabled capture and established cache handling. Both datasets' training/preparation finished before final timing. All final measurements used the same physical A800; the other was idle. Arm order was balanced across the four 64-request blocks and three repeats. Loading, activation, preparation and publication are separate from request latency.

## Inputs and interpretation limits

The GSM8K panel is new to the bounded-allocation C family, **not globally unused**. Of all 258 selected scored/warmup inputs, 255 have known older-project exposure and three lack positive exposure evidence in the bounded inventory; those three are not certified globally fresh. MathQA is unexposed within the enumerated project inventories. IDs, rendered inputs, exclusions and order were sealed before generation. Neither answers nor endpoint performance selected requests.

MathQA's current common feedback pool is authenticated and covered by the retained PR52 functional replay. Its `historical_feature_identity_verified=false` limitation remains; fresh hashing does not prove historical byte equality. Canonical outputs come from the frozen common-origin draft plus the qualified native target verifier, not a separate target-only run.

Native acceptance uses committed output progress minus verification calls; the first emitted token is outside that progress. Stop-clipped matching counts remain separate. With identical outputs, acceptance and call improvements are linked accounting views, not independent causal evidence. The three timing repeats do not supply three independent training replications.

Intervals use 10,000 paired whole-request draws, PCG64 seeds 1729/1730 for GSM8K/MathQA, lexicographic source-ID order and linear percentiles. Both arms and all repeats stay together in each draw. No undefined ratios/draws occurred. These intervals condition on these fixed endpoints and inputs, not a population of training seeds or origins. No non-inferiority margin, 0.5% advancement threshold or requirement to beat S was added.

## Storage deviation and preservation

The terminal metadata package contains **1,032,792,335 logical bytes across 15,161 files**, exceeding the handoff's 1 GB compact-evidence sub-budget by **32,792,335 bytes (3.279%)**. Its compressed archive is 94,891,583 bytes; compression does not erase the raw allowance overrun. This is a storage-accounting deviation, not a claim of complete budget compliance. The total 125 GB campaign allocation and 400 GB shared-volume reserve remained satisfied. The evidence is preserved rather than deleted to make the completed run appear compliant.

All eight new complete recovery states, four unique own-feedback prepared banks and four final serving exports remain required through result review. Retired first-event exports are verified derived copies reconstructible from retained recoveries; qualification payloads and exactly duplicate capture files have separate retirement receipts. No historical dependency was deleted. Private prompts, responses, tokens and payload tensors remain outside public Git.

Post-completion CPU hashing verified all 524 payload files against their recorded identities: eight recoveries, four final exports and 512 prepared packets, totaling 71.131002 GB of logical payload. At execution completion, run/source allocation was 72.510460 GB (76.805427 GB charged including the unused compiler-growth reservation); the admitted peak was 124.745481 GB. The shared volume had 456.699458 GB free then and 456.058368 GB after compact packaging/postflight. These are observed endpoints, not a measured whole-run disk high-water mark. Compiler-cache growth was zero. No payloads were copied back to Fenrir; the verified compressed metadata archive and extracted evidence left approximately 5.760 GB free there.

## Adaptation cost

These are fresh measured costs for the two production events per arm, distinct from serving and from the disposable qualification. Training timers include the actual native/C objective and update execution; there is no inferred kernel breakdown. Nested event/load/publication timers must not be summed twice.

| Dataset / arm | Training | Checkpoint persistence | Export | Event wall time |
|---|---:|---:|---:|---:|
| GSM8K N |651.640 s|38.636 s|32.464 s|733.234 s|
| GSM8K C |713.434 s|38.472 s|32.552 s|795.902 s|
| MathQA N |1,034.303 s|39.924 s|32.523 s|1,132.078 s|
| MathQA C |1,140.265 s|43.609 s|33.276 s|1,256.060 s|

C's measured training time was **9.48% higher on GSM8K** and **10.24% higher on MathQA**. This is the cost of these realized trajectories, including their naturally sampled exposure and own feedback; it is not an exactly compute-matched kernel benchmark. The tiny, unresolved serving effects do not establish repayment of this extra adaptation cost. Loading/startup, capture/teacher preparation, final activation, memory and physical forward/head counts are retained separately in the cost audit.

## Completion and decision

The resumed production phase took **152.713 minutes (2 h 32 min 43 s)**, compared with the admission estimate of 161.708 minutes raw / 220.220 minutes with contingency. Aggregate occupied-card time, including the retained qualification, was **200.301 card-minutes**, below the unchanged five-card-hour ceiling. The original elapsed clock includes the administrative wait for the user's elapsed-only authorization; it was never reset.

Completion was November 17 at 23:29:35, America/Cambridge_Bay (November 18 at 06:29:35 UTC). The original first-model clock to completion was **28,314.065 seconds (7 h 51 min 54 s)**, including the long paused authorization/preparation interval. It must not be reported as 7.9 hours of active GPU execution. The later compact-evidence transfer reached its administrative timeout, then resumed from a verified matching byte prefix; no training or generation was repeated.

Actual cumulative work is 1,036 Adam applications (1,024 production + 12 disposable qualification), 2,092 response forwards and 2,092 response backwards, ten trainer initializations, 512 feedback generations/preparations, 516 canonical references, 16 qualification generations, 96 final warmups and 3,072 scored generations. Total serving attempts are 4,212. No qualification was repeated after resumption, and no new experiment followed the final comparison.

Native K+ remains the primary method. C remains a candidate for contribution assessment with performance preservation unresolved under the agreed policy. This run supplies no demonstrated additional serving benefit for C. It does not isolate confidence from position shape, establish seed/origin robustness, or show training-inclusive speedup. No automatic tuning, further events, dataset expansion or algorithm change is authorized by completion.

## Validation status

Material Passport: ARS experiment-agent validation; completed native execution with CPU-only retained-evidence analysis. Verification of scientific effectiveness is `ANALYZED`; record authentication/reduction is audited separately and is not a new model reproduction. Independent readers and publication checks are assistant-team reviews, not external peer review. Detailed audit, cost, storage and per-request/repeat tables accompany the finalized report.

The independent raw reader passes **2,321,962 checks**, the training/resume audit **48,915**, the data/provenance audit **19,611**, and the publication audit **7,070,188**, all with zero remaining errors. The independent bootstrap reproduces the reported effects and intervals. All 11 ARS interpretation categories were reviewed. Original implementation tests comprise 92 CPU checks; the elapsed-only continuation adds 27 focused checks. The training reader's first attempt used a nonexistent histogram field; its preserved failure was corrected in the reader with regression coverage (13 audit fixtures pass), without altering experiment code or evidence. Three coefficient-reduction fixtures pass. Public copies contain token identities/counts only; an initial export check stopped unsanitized arrays before Git, and a separate checker was narrowed to distinguish typed tensor identities from arrays. Original review attempts remain private. The training auditor also helped implement the trainer; that authorship overlap limits independence. The separately implemented raw/statistical reader does not import the production reducer. These checks establish retained-record consistency, not broad training reproducibility.

Numeric evidence: [GSM8K summary](gsm8k/SUMMARY.json), [per-request rows](gsm8k/PER_REQUEST.csv), [all repeats](gsm8k/ALL_REPEATS.csv); [MathQA summary](mathqa/SUMMARY.json), [per-request rows](mathqa/PER_REQUEST.csv), [all repeats](mathqa/ALL_REPEATS.csv). Each dataset directory also contains all warmups, repeat totals, native trace counts, validity rows and evaluation costs. Checkpoint paths in public tables are aliases; source/recovery identities remain bound in the provenance receipts.
