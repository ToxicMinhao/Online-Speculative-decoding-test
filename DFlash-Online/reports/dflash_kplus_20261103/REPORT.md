# K+ versus retained K58: full GSM8K result

## Material Passport

ARS experiment-agent, run → validate. Scientific execution: `e0a9d3c235e1efd15404c549282d21c99243a21c`. CPU reporting repair: `181bf736aee231e36bd82a3bc1f73aa9f1eeb0f2`. Source and results are associated with [PR #25](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/25). Status: **complete and ANALYZED**. Corrected raw-evidence audit and separate endpoint/cost reconstructions passed. This is analysis of retained records, not an independent GPU reproduction.

K+ passes the prespecified endpoint criterion, but its gain is small: **120 fewer calls across 1,024 requests, a 0.2766% reduction**. Its paired request-bootstrap interval is **[0.0045%, 0.5457%]**, only narrowly above zero. Complete-request latency is **0.8230% lower [0.2726%, 1.3797%]** in the contemporaneous comparison. These results support a small endpoint improvement for this fitted seed on this reused mathematics panel; they do not justify a claim of a large or broadly robust advantage.

## What was tested

K+ changes only the distribution of K's first-error KL weight within a response. A block's first error receives priority 2 if its immediate valid successor is currently canonical-correct, otherwise priority 1. The fixed half-uniform, half-priority rule preserves total first-error mass and K's response denominator. Correct-prefix coefficients, full-vocabulary forward KL, full-block attention, sampler and optimizer remain unchanged. The successor is not added to supervision.

One new K+ arm started from released Qwen3-8B-DFlash-b16 weights, fresh native BF16 AdamW and the authenticated original seed-17 RNG. The Qwen3-8B target remained frozen. The 7,473 ordered GSM8K requests supplied 58 windows of 128 responses, followed by 49 requests with no update. Every window used two shuffled passes, response microbatch 1, accumulation 2, up to 512 anchors, constant LR 3e-4, gamma 7, weight decay 0.01, clip 1 and training context 2,272. Training used Flex attention. K58 was reused only as the final serving control. K+ collected its own online feedback; original K intermediate features have been retired. This is an end-to-end recipe comparison, not shared-feature replay.

The final panel is the original 1,024 train-disjoint requests, previously evaluated in this project. Both exports were measured freshly with greedy BF16 block 16, FA3, concurrency 1, context 4,096, 2,048 tokens or EOS and full per-request cache flushing. Capture was disabled. The three repeats followed the sealed 64-request block schedule, with each arm preceding the other 24 times. Training and transfers were absent during timing.

## Final endpoint

| Measure | K58 | K+58 |
|---|---:|---:|
| Verification calls, one trajectory/request | 43,387 | 43,267 |
| Identical output tokens | 314,847 | 314,847 |
| Accepted draft tokens | 271,387 | 271,522 |
| Committed output progress | 313,823 | 313,823 |
| Calls / 1,000 output tokens | 137.8034 | 137.4223 |
| Accepted draft tokens / call | 6.2550 | 6.2755 |
| Committed progress / call | 7.2331 | 7.2532 |
| Complete-request seconds, sum of three-repeat means | 794.7895 | 788.2480 |
| Native-generation seconds, same aggregation | 740.4919 | 737.2847 |
| Cache-maintenance seconds, same aggregation | 40.7966 | 37.6179 |

Accepted draft tokens count proposals accepted by the verifier. Committed progress also includes committed verifier repair/bonus tokens and excludes the first output token produced during prefill; EOS/cap clipping can affect their difference.

K+ improves 437 requests, worsens 360 and ties 227. It saves 792 calls gross and adds 672, leaving 120 net saved. All 1,024 requests remain in the result. Counts repeat exactly across all three runs, and fresh K58 reproduces every historical per-request count. All 6,144 executions match canonical token IDs and termination. The panel contains 1,021 EOS responses and 3 cap hits; output lengths range 94–2,048 (mean 307.47).

| Repeat | K complete-request seconds | K+ complete-request seconds |
|---|---:|---:|
| 1 | 793.3895 | 788.9414 |
| 2 | 794.9964 | 787.5021 |
| 3 | 795.9825 | 788.3005 |

The prespecified estimator divides summed paired measurements. Bootstrap units are 1,024 paired requests carrying both arms; timing repeats contribute means rather than additional samples. There are 10,000 draws from PCG64 seed 20261102, using linear 2.5/97.5 percentiles. The native-generation point reduction is 0.4331%, with interval [−0.0119%,0.8837%]; that component alone does not establish a reduction. Cache-maintenance variation accounts for part of the complete-request difference. Aggregate clocks do not identify kernel-level causes.

## Training activity and cost

All 58 events completed, with exactly 7,424 new AdamW steps, 14,848 response presentations/forwards/backwards and 7,424 terminal teacher preparations. The stream 7,473, four inherited correctness controls and final 6,144 executions total 13,621 complete generations. No additional scientific retry, disposable Adam step, intermediate efficacy panel or tuning occurred.

K+ used 3,728,050 sampled anchor occurrences and 33,411,824 selected supervised-position occurrences out of 54,368,391 available occurrences. Historical K selected 33,320,668 occurrences; matching presentations and natural anchors does not imply equal realized supervision as the model trajectory changes. All 14,848 presentations contained both priority classes. Across response first errors there were 1,874,744 priority-1 and 810,496 priority-2 occurrences; pooled moved first-error mass was 7.9211% (mean response share 7.8831%). This confirms that weighting was active, not why the endpoint changed. The maximum absolute conservation residual was 3.05176e-5 within the declared tolerance. All 14,848 natural presentation identities/order/anchor draws match the available old K records; no anchors were forced, and this does not establish hidden-feature identity.

| Cost scope | K+ | Retained K measurement | Historical difference |
|---|---:|---:|---:|
| Native training | 19,272.24s | 18,707.20s | +565.03s (+3.02%) |
| Blocking update calls | 22,383.30s | 21,735.12s | +648.18s (+2.98%) |
| Checkpoint persistence | 1,114.78s | 1,161.56s | −4.03% |
| Export | 897.14s | 897.49s | −0.04% |
| Teacher preparation | 288.58s | 287.35s | +0.43% |
| Prepared-feature persistence | 325.46s | 322.01s | +1.07% |

The additional K+ instrumented chunk/reduction regions total 235.80 s, already inside native training time. They include synchronization and chunk recomputation; they are not a pure kernel-overhead measurement. These cost scopes overlap and must not be added. Historical K training overlapped archival I/O and included separate intermediate evaluations elsewhere in its worker lifetime; K+ had neither. Consequently, +3.02% is an observed historical training-time difference, not a fresh matched estimate of algorithm overhead. No training-inclusive or asynchronous deployment speedup is established.

K+ online worker wall was 31,887.92 s. Final evaluation took 5,051.53 s; final loading 22.01 s and checkpoint switching 82.57 s are separate from request times. Campaign wall was 36,973.31 s; the wrapper ran 09:08:10–19:26:27 UTC on November 3 (10 h 18 m 17 s). Trainer peaks were 19.2543 GiB allocated and 22.2949 GiB reserved; these are scoped allocator peaks, not total host/device memory.

## Reporting failure and recovery

The scientific run finished, but original wrapper/campaign/report exits were 1/1/1 (postflight 0). Both audit callers resolved native traces relative to the evaluation folder instead of the production root. The original failure records remain unchanged. The two-callsite repair and strict opt-in CPU reporting recovery passed 50 focused CPU tests; independent path/recovery coverage passed 20 tests. Recovery rechecked all saved endpoint rows and native sidecars, canonical outputs, repeated counts and historical K58 counts, along with complete training receipts. It added zero model work and wrote a separate report. The corrected report process passed in 17.30 s. Analysis and execution commits remain distinct.

## Interpretation and retained evidence

The fixed criterion is satisfied, but practical improvement is modest and the primary lower bound is close to zero. Keep K as the incumbent pending any separately agreed adoption decision. No subsequent experiment is launched by this report. One training seed and a reused benchmark do not establish training-seed robustness, broad-domain generalization or the proposed causal explanation. Own-feedback trajectories also prevent an identical-feature ablation claim.

Initial and final complete states and the K+58 serving export remain. Under the admitted rolling policy, obsolete tensors/states were retired after successor validation, not archived; compact raw results, token IDs, native traces, anchors and validity/failure receipts remain. The retention review reconciled 22,386 retired paths: 7,424 native captures, 7,424 terminal tensors, 7,424 prepared packets, 57 intermediate complete states and 57 intermediate exports. Fresh stat verifies retained initial/final states, final export and the latest-state hardlink; tensor contents were not freshly decoded or rehashed by the reviewers.

Companion tables: [paired requests](PAIRED_REQUESTS.csv), [all repeats](ALL_REPEATS.csv), [training exposure](TRAINING_EXPOSURE.csv), [generated summary](SUMMARY.json), [training receipts/costs](TRAINING.json). Full private raw evidence remains intact on A800, outside Git. The separately admitted full compact copy to Fenrir is still in progress at publication; the completed independent audits ran directly against the original A800 records, and their small outputs were hash-verified on transfer. See the [endpoint/statistical review](reviews/endpoint/FINAL_ENDPOINT_REVIEW.md), [cost/retention review](reviews/cost/FINAL_TRAINING_COST_REVIEW.md), [handoff coverage](HANDOFF_COVERAGE.md), [event costs](EVENT_COSTS.csv), [cost/exposure reconstruction](COSTS_EXPOSURE_RETENTION.json), and [execution provenance](EXECUTION_AND_PROVENANCE.md).

Separate same-team scripts checked 12,311 endpoint input files and 30,171 training/cost input files against the original inventory; the two sets may overlap. Endpoint reconstruction checked all 259,962 native verification blocks and reproduced the report exactly. All 11 statistical-fallacy categories were assessed. Both reviewers also contributed to implementation, which limits independence; these are separate record calculations, not blind independent implementations or GPU reproductions.
