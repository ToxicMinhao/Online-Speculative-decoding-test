# Persistent prefix-withholding campaign

DEVELOPMENT_FEASIBILITY_REUSED_DATA

Execution commit: `af708012e729437068fbef93cb11c9996a9109a9`. Status: **COMPLETED_VALID_COMPARISON**.

Decision: **NO_PRACTICAL_ADVANCEMENT_THIS_PILOT**.

| Contrast | Request-time reduction | Native acceptance change | Requests with fewer/more/tied calls |
|---|---:|---:|---:|
| E_vs_N | -1.304% (95% CI [-2.317, -0.319]%) | -1.370% (95% CI [-2.550, -0.186]%) | 41/72/15 |
| E_vs_S | -2.537% (95% CI [-3.518, -1.570]%) | -2.714% (95% CI [-3.843, -1.588]%) | 40/77/11 |
| S_vs_N | +1.203% (95% CI [+0.590, +1.822]%) | +1.382% (95% CI [+0.608, +2.164]%) | 62/33/33 |

| Arm | Calls | Native accepted drafts/call | Committed progress/call | Complete request time (s) |
|---|---:|---:|---:|---:|
| N | 5835 | 5.810797 | 6.810797 | 100.762379 |
| E | 5904 | 5.731199 | 6.731199 | 102.075845 |
| S | 5767 | 5.891105 | 6.891105 | 99.550287 |

The fixed decision uses paired whole-request bootstrap intervals and all three repeat directions. Counts use one validated trajectory per request; times use three fresh repeat means.

Full method totals, native/progress histograms, repeat/block totals and all three contrasts are in SUMMARY.json. Audited per-request/repeat tables and warmup records are in tasks/final/evaluation.

| Work | Planned | Completed | Failed | Unknown | Unattempted |
|---|---:|---:|---:|---:|---:|
| adam | 673 | 673 | 0 | 0 | 0 |
| response_forward | 1365 | 1364 | 0 | 1 | 0 |
| response_backward | 1364 | 1364 | 0 | 0 | 0 |
| feedback_generation | 384 | 384 | 0 | 0 | 0 |
| reference_generation | 24 | 24 | 0 | 0 | 0 |
| qualification_generation | 8 | 8 | 0 | 0 | 0 |
| warmup_generation | 72 | 72 | 0 | 0 | 0 |
| scored_generation | 1152 | 1152 | 0 | 0 | 0 |
| teacher_preparation | 384 | 384 | 0 | 0 | 0 |
| training_initialization | 9 | 9 | 0 | 0 | 0 |

Recorded completed occupied-card time: 4400.162635803223 seconds. Campaign elapsed time: 14054.324910640717 seconds. Live leases and each task's wall time remain separate in the machine-readable report.
Storage, source/CPU/native gates, physical cards, actual schedule and idle reasons remain recorded. TASK_COSTS.csv separates training, loading, startup, teacher preparation, persistence and publication; nested timers are not additive.

One training origin and seed on deliberately reused development data.
No independent confirmation, long-horizon claim, or training-inclusive speedup.
N7 was reused; N has one new event while E and S each have two new events.
Native accepted draft tokens and stop-clipped matching counts are distinct.
Partial or invalid execution provides no scientific comparison; missing totals are not zero.

No automatic retry, additional event or mechanism change is authorized.
