# KP-v1 completed training and cost review

## Material Passport

- Workflow: ARS 3.24.0 experiment-agent, validate; accounting and cost scope only.
- Verification status: **ANALYZED — compact evidence passes; no independent model rerun or tensor reread.**
- Execution: `d10521a1aa7d256bb75b37a6aae07586ae7cc8b3`; remote run `dflash_k_prefix_protection_v1_20261031/production_001`.
- Reviewer provenance: this reviewer contributed the training/qualification adapter. The arithmetic below was independently reconstructed from raw receipts, separately from the serving-statistics reviewer; this is not a blinded external implementation audit.
- Evidence: 272 consumed compact files (163,965,350 bytes) rechecked against the retrieval manifest, with no identity mismatch; 41,809 bounded assertions, no failures. The complete retrieval independently preserved 9,634 files / 747,691,808 bytes with zero mismatch.
- No SSH, artifact transfer, model loading, gradient evaluation, training, generation or executed-source modification was performed for this review. Retrieved bytes were not edited.

## Finding

**Training and qualification accounting are consistent with the handoff.** Four complete production trajectories each reached 512 AdamW steps and 1,024 response presentations. KP adds measurable update cost; the lower-learning-rate control is the appropriate contemporaneous cost comparator. Timing changes do not establish kernel-level causes or efficacy.

| Scope | Current forwards | Backwards | AdamW steps | Frozen-reference forwards | Complete generations |
|---|---:|---:|---:|---:|---:|
| Production training, all 16 events | 4,096 | 4,096 | 2,048 | 2,048 | 0 |
| Bounded qualification, Q1–Q4 | 8 | 8 | 4 disposable | 6 | 0 |
| Fresh final evaluation | — | 0 | 0 | — | 3,072 |
| Training plus qualification | 4,104 | 4,104 | 2,052 | 2,054 | 0 |

The reference forwards are additional draft forwards, not additional response presentations or teacher-model executions. Complete serving generations contain their own variable verification/block forwards and must not be equated with one response-level training forward. All work-ledger starts have matching completions; no unknown, automatic-retry or extra qualification optimizer work is recorded.

## State, schedule and numerical evidence

All four initial migrations record original step-zero weights, RNG, moments/counters and history preserved, with the declared new arm contract and K_HALF learning rate as the only changes. All 12 later resumes point to that arm’s own preceding event. The 16 live and serialized endpoints agree, have empty accumulation, 58 draft tensors and 58 Adam-state entries, and match their saved export. The inherited horizon remains 7,424. Every arm closes at the expected event, step and presentation count; no reset or fake event is present.
All 4,096 original response observations match their independently durable JSONL records. Each request identity, pass, and sampled-anchor record matches the original O schedule, whose eight source identities match the sealed plan. Features are unchanged and event pools remain disjoint within seed. Observations are recorded before two-response accumulation; additive exposure is summed from those original observations, never from both single-response and merged boundary rows. Every one of the 2,048 step boundaries reports two presentations and a finite gradient norm. All stored numeric training records are finite. KP’s 2,048 response records pass the stored loss decomposition, coefficient 0.10, prefix histogram, threshold bounds and finite-margin/threshold checks; production never uses the forced-active qualification override.
Q1–Q4 are four fresh processes on the same physical A800, with six initially empty compiler/cache directories each and no extra warmup model work. They restore K17-event4 at 512 steps / 1,024 presentations, take one disposable step to 513 / 1,026, retain four completed events, and validate serialization/restore. Q2 exactly matches Q1 in losses, normalized preclip gradients, updated weights, Adam, RNG/counters, anchors and response metrics. Q3 activates 106 of 9,546 protected occurrences with finite outputs. Q4 changes gradients and Adam moments, and changes 35,065,175 BF16 parameter coordinates, proving the forced-active path reaches the update. These are correctness/plumbing results, not performance gains. The original qualification states are discarded from production continuity.
All 20 qualification/production worker PIDs and all 120 cache-directory paths are distinct. All workers record the same GPU UUID `c6943f33-bcd0-a724-0b35-f77af6b0a3f1`, CUDA device allocation 1, and initially uninitialized CUDA. The compact records substantiate that the native checks ran; full state tensors remain remote and were not independently reopened in this review.

## Matched cost scopes

All times below are seconds, summed over the four events for each seed. The pooled row is a ratio of summed timers, not the mean of event percentages. Both new arms use the same hardware and cold-process policy; execution order was fixed rather than randomized.

| Scope | Seed | K_HALF | KP | Additional KP seconds | Additional KP % |
|---|---|---:|---:|---:|---:|
| Update loop | 17 | 1490.584 | 1847.024 | +356.440 | +23.913% |
| Update loop | 29 | 1492.825 | 1848.319 | +355.494 | +23.813% |
| Update loop | pooled | 2983.410 | 3695.343 | +711.933 | +23.863% |
| Complete event | 17 | 1642.604 | 2000.769 | +358.166 | +21.805% |
| Complete event | 29 | 1651.655 | 2003.820 | +352.166 | +21.322% |
| Complete event | pooled | 3294.258 | 4004.590 | +710.331 | +21.563% |
| Complete worker | 17 | 1881.155 | 2271.813 | +390.658 | +20.767% |
| Complete worker | 29 | 1904.260 | 2292.546 | +388.286 | +20.390% |
| Complete worker | pooled | 3785.415 | 4564.359 | +778.944 | +20.578% |
| Full subprocess | 17 | 1910.940 | 2293.986 | +383.046 | +20.045% |
| Full subprocess | 29 | 1951.704 | 2315.051 | +363.347 | +18.617% |
| Full subprocess | pooled | 3862.644 | 4609.036 | +746.393 | +19.323% |

The update-loop increase is **711.933 s (+23.863%)** across the two seeds. Complete-worker increase is **778.944 s (+20.578%)**. Full subprocess increase is **746.393 s (+19.323%)**; its smaller difference reflects unequal interpreter/import/exit overhead and should not replace the directly scoped update-loop result.

| Recorded component, both seeds | K_HALF seconds | KP seconds | Scope |
|---|---:|---:|---|
| Load/init/restore validation | 351.356 | 407.932 | Before event; includes frozen reference setup |
| Frozen reference setup | 0.000 | 88.470 | Nested in load/init; do not add again |
| Recovery-state persistence | 160.644 | 159.701 | Nested in complete event |
| Model export | 123.172 | 122.742 | Nested in complete event |
| Endpoint validation and hashing | 134.711 | 146.277 | After event; nested in complete worker |

**Timer overlap:** update-loop time already includes cold compilation, reference inference/head computation, current/teacher projection, loss/backward/recomputation, clipping/Adam and training-record work. Complete event includes the update loop, feature/event preparation, persistence and export. Complete worker includes load/init/restore, the complete event, endpoint validation/hash and cleanup. Full subprocess additionally includes interpreter/import/exit time. Never sum these nested totals. Frozen reference setup is counted inside initialization, not again inside the update-loop difference.
KP records 231.963 s for reference draft forwards, 456.904 s for reference head computation, and 37.300 s for protection arithmetic including actual checkpoint reentry work. These are nested observed subtimers, not an exhaustive kernel profile or an exact causal decomposition of the 711.933-s difference. The two reference subtimers sum to 688.867 s; changes in other work and timing noise mean one cannot equate their sum with total added runtime. K_HALF uses LR 1.5e-4 and KP 3e-4, so even this contemporaneous cost comparison is not an isolated one-component runtime ablation.
Retained ordinary-K update-loop totals are 1,486.487 s (seed 17) and 1,486.279 s (seed 29), with event totals 1,630.890 / 1,637.600 s. These are historical context only, not fresh matched overhead measurements. Historical-versus-fresh serving timings are not used as a learning curve.

## Memory, chunks and exposure


| Recorded maximum across eight events | K_HALF GiB | KP GiB |
|---|---:|---:|
| Post-load allocated | 19.286 | 21.239 |
| Post-load reserved | 25.943 | 27.811 |
| Load/init allocated | 10.133 | 12.086 |
| Load/init reserved | 10.176 | 12.148 |
KP increases the maximum observed post-load allocation by 1.953 GiB. The peak counters reset after loading and are read after training, persistence and boundary validation: these are **not pure training-only peaks**. Reserved memory is allocator reservation, not live parameter size. Q3 has the largest qualification allocation: 23.244 GiB; qualification is not mixed into production peak comparisons.
KP’s actual counters reconcile to 2,048 current response/draft forwards and 2,048 reference response/draft forwards. There are 10,240 current and 10,240 reference layer forward calls (five layers per response), 70,140 reference projection calls, and 140,280 current/teacher projection calls and objective-chunk entries. The latter are exactly twice the 70,140 original sampled-block chunks: projection checkpoint recomputation occurs during backward and is counted as recomputation, not a new response presentation. These instrumented runs show no additional draft-layer replay beyond the five layers per response. The unchanged K_HALF path lacks comparable per-chunk counters; that work is unrecorded at this granularity, not zero.
Seed 17: 1,735,384 protected position occurrences; 77,287 active (4.4536%); 57,776 currently incorrect, including 3,140 wrong ties. Repeated and overlapping sampled anchors make these occurrence counts, not independent tokens or requests. They do not establish that protection prevented native regression.
Seed 29: 1,734,793 protected position occurrences; 74,515 active (4.2953%); 55,094 currently incorrect, including 3,321 wrong ties. Repeated and overlapping sampled anchors make these occurrence counts, not independent tokens or requests. They do not establish that protection prevented native regression.

## CPU checks, wall time and limits

The retained remote CPU log reports 42 experiment tests plus 15 objective tests, all passed. Native preflight and postflight receipts declare zero model work. The published supervisor runs report generation and postflight with CUDA hidden. This review likewise used only standard-library parsing, hashing and arithmetic. CPU fixture operations are not native qualification or campaign work. No separate report-wall timer is retained; do not invent one from the aggregate supervisor residual.
Qualification lasted 546.859 s. Production campaign wall time was 17902.115 s (4.973 h). Supervisor markers span 2026-10-31 12:34:48–17:49:27 UTC, 18879 s (5.244 h), including qualification, CPU checks/admission, campaign, report and postflight. These are elapsed envelopes, not sums to add to serving/training totals.
Scope limits: no independent full-tensor reread, loss/gradient recomputation or GPU rerun; no proof of every parameter’s finiteness beyond the executed finite gates and recorded scalars; no statistical timing confidence interval or kernel attribution; no fresh ordinary-K training cost; no claim of asynchronous or training-inclusive serving speedup. The observed extra cost must be weighed against the separately audited native outcomes, not against lower training loss alone. Strongest available serving result/control and all negative/mixed outcomes remain the responsibility of the companion statistics review.

## Reproducible audit sources

- `FINAL_TRAINING_COST_REVIEW.json`: all per-event costs, checks, actual work, compact-file SHA identities and limits/provenance.
- `final_training_cost_review.py`: CPU-only reconstruction script; no scientific runtime import. `FINAL_TRAINING_COST_REVIEW_EXEC.json` contains its console summary.
- `compact_evidence_001/campaign_001/seed_{17,29}/resume_{001..004}_{K_HALF,KP}/`: WORK, WORKER, COSTS, EVENT_CHECK, LIVE, LIVE_BOUNDARY, CLOSED, MIGRATION/SAVED, SAVED_ENDPOINT and STATE_IDENTITIES receipts.
- `compact_evidence_001/campaign_001/seed_{17,29}/{K_HALF,KP}/events/event_{001..004}/RESULT.json` and original response-observation JSONL; eight `audit_inputs/original_schedules/{seed}_{event}.json` files.
- `compact_evidence_001/qualification_001/Q{1..4}/`: complete compact qualification checks and per-case work ledgers. `final_report/TRAINING_AUDIT.json`, `COST_SUMMARY.json` and `TRAINING_COSTS.csv` agree with the independent reconstruction.
- Executed timer/hook scope: `DFlash-Online/experiments/k_prefix_protection_v1/training_worker.py:96`, `:106`, `:117`, `:129`, `:148`; `online_feedback/prefix_protection.py:125`, `:169`, `:287`; CPU report/postflight isolation in `experiments/k_prefix_protection_v1/run_a800.sh:38`.

## Final-delivery wording check

The assembled `final_delivery_001/REPORT.md` budget, cost numbers and nested-reference explanation agree with this review. The parent was asked to replace “Maximum training allocation” with the actual post-load scope (training, persistence and validation). The 3,862.644 / 4,609.036-s comparison is the full fresh subprocess timer, including interpreter/import/exit, not the narrower `WORKER.worker_wall_s`. No report or retrieved evidence was edited by this reviewer.
