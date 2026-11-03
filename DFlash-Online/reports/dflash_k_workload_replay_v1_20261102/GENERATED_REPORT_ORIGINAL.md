# Workload-conditioned replay with K

## Material Passport

Mode: run→validate. Verification status: raw-evidence-analyzed.

CPU reconstruction of retained raw/native/work records; not an independent GPU reproduction.

Source handoff: TRACK2_K_REPLAY_EXECUTOR_INSTRUCTIONS_20261102.txt; user handoff dated 2026-11-02.

Status: valid_complete

Execution source: `8a4cbcc8b78adfee845a0a1409db4a0d29c54058`.

Source receipt: {"commit": "8a4cbcc8b78adfee845a0a1409db4a0d29c54058", "dirty": false, "repository": "https://github.com/ToxicMinhao/Online-Speculative-decoding-test", "scientific_base": "4d94f30c42701d0f86ac04e2829c834593c16d8e", "source_path": "/data/sources/minhao/DFlash-Online"}

Primary contrast: R/U; required usefulness contrast: R/K. Counts use 256 distinct requests; time uses the per-request mean of three repeats. All 2,304 planned endpoint slots remain in ALL_FINAL_ROWS.csv.

| Panel | Contrast | Call reduction % [95% interval] | Wins / regressions / ties | Gross saved / added | Time reduction % |
|---|---|---:|---:|---:|---:|
| pooled | R/U | 0.0920 [-0.4775, 0.6432] | 101 / 107 / 48 | 241 / 229 | -0.6691 |
| pooled | R/K | -0.1615 [-0.8027, 0.4487] | 101 / 112 / 43 | 241 / 262 | -0.6441 |
| pooled | U/K | -0.2538 [-0.9200, 0.3989] | 110 / 106 / 40 | 250 / 283 | 0.0249 |
| math | R/U | -1.3259 [-2.1867, -0.5011] | 38 / 66 / 24 | 73 / 147 | -2.0587 |
| math | R/K | -0.7842 [-1.7928, 0.1968] | 45 / 63 / 20 | 104 / 148 | -2.2101 |
| math | U/K | 0.5347 [-0.5361, 1.6081] | 62 / 45 / 21 | 146 / 116 | -0.1484 |
| coding | R/U | 1.1534 [0.4136, 1.8845] | 63 / 41 / 24 | 168 / 82 | 0.3937 |
| coding | R/K | 0.3111 [-0.5140, 1.0909] | 56 / 49 / 23 | 137 / 114 | 0.5501 |
| coding | U/K | -0.8522 [-1.6863, -0.0428] | 48 / 61 / 19 | 104 / 167 | 0.1570 |

Conditional request-panel intervals for one trained trajectory/seed on a reused development panel; not seed robustness, deployment-workload uncertainty, or uncertainty over the adaptive trajectory.

Per-domain, U/K and timing intervals are descriptive and unadjusted.

Raw totals, throughput ratios, accepted draft tokens/call, committed progress/call and every repeat are retained in SUMMARY.json; FINAL_PAIRED.csv preserves each request's counts and mean times.

Future-serving events 3–8 are descriptive only. Capture-on K and capture-off U/R times are not speedup comparators.

Observed training exposure: {"active_anchors": 1631357, "optimizer_steps": 3072, "prompt_tokens": 346925, "replay_presentations": 448, "response_presentations": 6144, "response_tokens": 1761381}.

Training forward/backward, token/anchor exposure, per-pass loading/bookkeeping, updater and runtime costs remain in TRAINING_COSTS.json. Campaign and final evaluation statuses preserve recorded preparation, publication, resource and total-time fields separately; missing values are not fabricated. Nested timers are not additive.

[Work accounting](WORK_ACCOUNTING.json) audits the root generation/preparation ledger and all 24 training-worker ledgers, including attempted, completed and unknown partial operations. Required totals are 3,072 pre-update generations, 4 parity generations, 1,024 preparations, 6,144 forwards, 6,144 backwards and 3,072 optimizer steps; the final evaluator separately accounts for its 2,304 generations.

[Evidence index](EVIDENCE_INDEX.json) links every event's selection, before/after memory, capture parity and neutral-state receipts, the full descriptor ledger and exact retirement receipts. [Selection steps](SELECTION_STEPS.csv) preserve marginal gains and reporting-only domain composition; memory mixture, similarity distributions, concentration and ages are in the evidence index.

Recommendation: Do not call this an upgrade to K. Propose 16 old plus 112 fresh only if recurrence gains and lost fresh adaptation are specifically supported; otherwise stop this allocation. Do not execute a changed dose here.

- One seed and intentionally recurrent workload; no natural-traffic claim.
- Previously used development panel; no fresh confirmatory-panel claim.
- Common K-generated supervision; not three independent self-generated streams.
- Cannot add these gains to the separately trained PR23 K58 gain.
- Equal presentations do not establish equal token FLOPs or training-inclusive speedup.
- Coverage is a proxy; its increase alone does not demonstrate serving utility.

Recorded component wall times (seconds; nested scopes may overlap):

| Component | Seconds |
|---|---:|
| initialization_s | 81.28954487107694 |
| backend_loading_s | 20.50684519484639 |
| training_worker_wall_s | 10016.63027044572 |
| activation_s | 91.32158985547721 |
| teacher_preparation_s | 84.0358947198838 |
| feature_persistence_s | 48.736396880820394 |
| descriptor_s | 0.45300007052719593 |
| selection_s | 0.6827728357166052 |
| memory_maintenance_s | 23.95966560393572 |
| export_s | 434.0040175821632 |
| resource_check_s | 1384.5276044029742 |
| adaptation_ready_wall_s | 15327.051032852381 |
| evaluation_wall_s | 2191.9385773949325 |
| campaign_wall_s | 17525.486541213468 |

Resource checks remain at their original granularity in campaign_status.resource_checks or campaign_status.costs.resource_checks in SUMMARY.json; these do not infer unmeasured peaks.
