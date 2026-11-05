# K with released-prefix protection v1 — completed coding experiment

**KP-v1 did not meet the prespecified useful-improvement criterion. The lower-learning-rate K control was better on this panel and required less training time.**

## Material passport

- Status: completed, supervisor exit 0; all planned work and both included native gates passed.
- Execution code: [`d10521a1aa7d256bb75b37a6aae07586ae7cc8b3`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/d10521a1aa7d256bb75b37a6aae07586ae7cc8b3), [PR #21](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/21).
- Original retained coding execution: `2705d992b133196be162229e1d5a08dd9c4f9cbb`, actual run `production_002`.
- Execution: 2026-10-31 12:34:48–17:49:27 UTC; **5 h 14 m 39 s** including the bounded qualification and automatic final checks. Earlier CPU preparation and subsequent evidence transfer/review are separate.
- Verification status: **ANALYZED**. Separate CPU reconstruction checked the retained raw evidence; it did not repeat the GPU campaign or independently decode the retained model tensors.
- The execution source remained unchanged. This report and its publication commit are subsequent documentation, not a different execution revision.

## What was tested

The hypothesis was that protecting released DFlash's correct prefixes would remove more harmful changes than useful K gains. Partial harm reduction accompanied by lost gains, including a modest or unresolved result, was anticipated.

| Method | Fixed definition |
|---|---|
| F | Released DFlash, unchanged; serving control and KP's frozen protection reference. |
| K | Retained original K4 for each seed: current-prefix-plus-first-error forward KL, LR `3e-4`; no retraining. |
| K_HALF | The same K training, with only constant LR changed to `1.5e-4`. Weight decay stays `0.01`, so this is not an exactly half-sized displacement control. |
| KP | K at LR `3e-4`, plus `0.10 × R_F`. On the fixed released model's correct proposal prefix, `R_F` is the separately normalized weighted hinge `ReLU(0.50 × min(m_F, 1.0) − m_current)`. |

Qwen3-8B and its qualified DFlash-b16 draft were used with the pinned model revisions, BF16 training and native AdamW, Flex training attention, and SGLang/FA3 serving. Four new trajectories—two arms × seeds 17/29—each started from that seed's original released-weight/empty-Adam state and continued four events. Each event reused its original 128-response pool for two passes, 256 presentations and 128 optimizer steps. Natural presentation order and anchors were checked against the original records. Only response positions supplied supervision; source dataset answers were unused.

Serving used the new, presealed 128-request CodeAlpaca panel, three balanced repeats, greedy block-16 decoding, a 2,048-token-or-EOS cap, context 4,096, concurrency 1 and capture disabled. Training and artifact transfers were absent during timing. F and retained K were measured freshly alongside both new methods.

Selection used seed `20261101` over 206 eligible requests in the original project's final-1,024-row reservoir. Exclusions covered retained historical allocations, the completed coding study's 512 training/128 evaluation requests and later recoverable allocations, using source IDs and normalized prompt identities. This is a project partition, not an official CodeAlpaca test split; freshness is relative to those inventories, not proof of absence from pretraining.

## Final drafting and serving results

Positive reductions mean fewer calls or lower complete-request latency than the named comparator. Calls use one validated trajectory per seed/request. Timing uses the sum of per-request three-repeat means.

| Comparison | Call reduction | Paired 95% interval | Request-time reduction |
|---|---:|---:|---:|
| **KP vs K — primary** | **+0.3435%** | **[+0.0182, +0.6919]%** | **+0.1770%** |
| KP vs F | +0.3252% | [−0.6386, +1.2365]% | +0.3620% |
| KP vs K_HALF | −2.4078% | [−3.2861, −1.6281]% | −2.2350% |
| **K_HALF vs K** | **+2.6866%** | **[+1.9233, +3.5338]%** | **+2.3592%** |
| K_HALF vs F | +2.6687% | [+1.8411, +3.5566]% | +2.5402% |
| K vs F | −0.0184% | [−1.0284, +0.8982]% | +0.1853% |

| Seed | F calls | K calls | K_HALF calls | KP calls | KP/K reduction | K_HALF/K reduction |
|---|---:|---:|---:|---:|---:|---:|
| 17 | 8,150 | 8,140 | 7,949 | 8,144 | −0.0491% | +2.3464% |
| 29 | 8,150 | 8,163 | 7,916 | 8,103 | +0.7350% | +3.0258% |
| Combined | 16,300 | 16,303 | 15,865 | 16,247 | +0.3435% | +2.6866% |

KP saved 56 calls versus K: four extra calls for seed 17 and 60 fewer for seed 29. Its pooled interval is above zero, but its point estimate is below the prespecified 0.5% practical target, its seed directions differ, and its advantage over F is unresolved. Its request-time interval versus K is **[−0.3119, +0.6729]%**, also unresolved. Therefore this is a small relative call benefit, **not** the prespecified supported useful improvement.

K_HALF saved 438 calls versus K and 435 versus F. Its call reductions were positive in both seeds. KP used 382 more calls than K_HALF and was slower; the KP/K_HALF interval also lies entirely outside the ±0.5 percentage-point equivalence band. The result does not support superiority or practical equivalence of KP to this control. Secondary intervals are descriptive and unadjusted.

KP/K wins/regressions/ties were **102/80/74**, with 203 gross calls saved and 147 added. K_HALF/K was **175/48/33**, with 570 saved and 132 added. These denominators are 256 seed/request pairs, **not 256 independent sampled requests**. All uncertainty uses the same 10,000 PCG64 draws, seed `20261102`, resampling 128 joint request identities carrying both fixed seeds and every method. Ratios are computed from summed measurements. The intervals do not estimate population-level training-seed robustness.

All methods produced the same 44,133 output tokens per seed: 37–2,042 tokens/request, mean 344.789; all 128 requests terminated at EOS and none hit the cap. Full calls/1,000 tokens, accepted tokens, committed progress, per-seed intervals and repeat totals are in the linked machine-readable records.

## What the preservation accounting shows

The following is computed within each seed before pooling. K had 369 gross call gains and 372 gross harms relative to F.

| Candidate | Same K gain retained | Lost K gain | Added gain | Prevented harm | Added harm | Net calls saved vs K |
|---|---:|---:|---:|---:|---:|---:|
| KP | 301 / 369 (81.57%) | 68 | 91 | 112 | 79 | 56 |
| K_HALF | 261 / 369 (70.73%) | 108 | 299 | 271 | 24 | 438 |

Both exact identities hold: `added gain − lost gain + prevented harm − added harm = net calls saved`. KP retained more of K's original gains, but K_HALF created substantially more new gains and left much less harm. KP's gross gain/harm versus F was 392/339; K_HALF's was 560/125. These are endpoint call-accounting associations, not proof of a particular gradient or preservation mechanism.

The KP penalty was active: 151,802 of 3,470,177 protected training occurrences (4.3745%), across 2,048 response presentations. Current native predictions were incorrect on 112,870 protected occurrences; 6,461 wrong-at-tie occurrences were recorded. A wrong tie with zero reference threshold can remain unpenalized; do not interpret every recorded wrong tie as an unpenalized one. KP selected 4,292,000 K-loss occurrences and K_HALF selected 4,245,261, from 8,117,722 available occurrences per arm. Their anchors were matched; selection depends on each arm's current predictions. These counts include overlapping sampled occurrences, not independent tokens.

Different feedback and anchors across events prevent reading these training rates as a matched before/after preservation test. No additional K or F training replay was run to manufacture such a counterfactual. Loss magnitudes do not identify which gradient dominates.

## Added cost

Totals below cover both seeds and all four events. K_HALF is the fresh contemporaneous updater comparator. Original K training times remain historical context.

| Scope | K_HALF | KP | KP increase |
|---|---:|---:|---:|
| Training, including cold compilation and objective work | 2,983.410 s | 3,695.343 s | **711.933 s; 23.86%** |
| Event wall time, including save/export | 3,294.258 s | 4,004.590 s | 710.331 s; 21.56% |
| Complete fresh subprocesses (`EXEC.wall_s`) | 3,862.644 s | 4,609.036 s | 746.393 s; 19.32% |
| Maximum post-load allocation: training, persistence and validation | 19.286 GiB | 21.239 GiB | 1.953 GiB |

KP's reference draft/head work took 688.867 s inside training; its separately measured reference setup took 88.470 s inside loading. These are nested measurements and must not be added again to their parent totals. Loading/resume validation, checkpoint persistence, exports, state checks, actual recomputation and objective subcosts are retained separately.

The subprocess row includes interpreter/import/exit overhead. The memory peak is read after endpoint validation and is not a training-only peak.

Pooled complete-request times were F 281.665 s, K 281.143 s, K_HALF 274.510 s and KP 280.645 s. These are sums of three-repeat means, not campaign wall time. Evaluation activation/switching took 5,779.404 s across the two cells and is separate from serving. The entire campaign includes benchmarking and cannot be presented as a deployment episode or a training-inclusive speedup.

## Correctness, work and evidence

Fifty-seven new CPU tests passed on both hosts. The sole GPU qualification used exactly four disposable steps, eight current forwards/backwards and six reference forwards, with no generations. The neutral wrapper with reference work reproduced native K's losses, pre-clip gradients, BF16 weights, Adam, RNG, counters and anchors exactly. Production KP was finite and active. The forced-active fixture changed gradients, Adam moments and 35,065,175 BF16 parameter entries; it was never used in production. Qualification maximum allocation was 24.958 GB, a different scope from the production memory table.

Production completed all 16 jobs: **2,048 Adam steps, 4,096 current response forwards/backwards and 2,048 frozen-reference forwards**. Including qualification gives 2,052 Adam applications, 4,104 current forwards/backwards and 2,054 reference forwards. There were no new feedback generations, terminal-teacher preparation forwards, retries, extra warmups or unknown scientific work. Teacher-head projection from saved features was still performed and costed. Actual chunks and checkpoint recomputation are recorded separately from these response equivalents.

All four initial states and all 16 new event states/exports remain retained. All 3,072 serving executions passed exact tokens/termination, repeated counts and global F consistency, including both 96-execution gates. Final source/config/runtime checks passed. A separate reviewer reconstructed 1,973,211 checks over 3,072 raw executions and 194,145 native blocks, plus joint bootstraps and gain/harm identities. Every one of 9,634 retrieved files (747,691,808 bytes) hash-matched; the eight original schedules and all native captures are preserved. No new GPU reproduction or independent local decoding of large recovery tensors was performed.

## Decision and limits

**Stop KP-v1 here; do not advance it as an improvement over the tested lower-LR control.** Retain K_HALF as the stronger comparator for the next research discussion. This is a recommendation from this fixed screen, not authorization to change the project stage, deploy a new default, tune the loss or launch another experiment.

The result is conditional on this short-prompt coding panel, two fitted trajectories and fixed O-generated feedback pools. It does not establish globally optimal learning rates, closed-loop adaptation, multi-turn/domain-switching behavior, mathematical-domain benefit, stochastic verification, convergence or general failure of prefix protection. K_HALF changes native weight decay's effective update as well as gradient step size; it is not a pure gradient-scaling intervention.

## Files

- [Full measured tables](MEASUREMENTS_REPORT.md), [structured summary and intervals](SUMMARY.json), [per-request table](PAIRED_REQUESTS.csv), [every timing repeat](ALL_REPEATS.csv), [repeat totals](REPEAT_TOTALS.csv).
- [Training costs](TRAINING_COSTS.csv), [cost summary](COST_SUMMARY.json), [evaluation costs](EVALUATION_COSTS.json), [response summaries](TRAINING_RESPONSE_SUMMARY.csv), [window records](TRAINING_WINDOWS.csv), [event/pass summaries](TRAINING_EVENT_PASS_SUMMARY.json).
- [Qualification review](QUALIFICATION_REPORT.md), [independent raw-record review](INDEPENDENT_RECORD_REVIEW.md), [audit](INDEPENDENT_RECORD_AUDIT.json), [interpretation and 11-point ARS review](FINAL_INTERPRETATION_REVIEW.md), [training and cost review](FINAL_TRAINING_COST_REVIEW.md), [source postflight](POSTFLIGHT.json), [provenance and full retained locations](PROVENANCE.json).

Exact prompts/tokens, anchor arrays, raw traces, complete training audit and model/optimizer tensors remain at the retained private locations. Compact publication does not retire them. The original automatic report is preserved unchanged as `MEASUREMENTS_REPORT.md`.
