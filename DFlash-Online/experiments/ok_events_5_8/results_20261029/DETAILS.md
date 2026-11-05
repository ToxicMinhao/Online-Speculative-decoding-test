# OSD/K continuation through event 8

Preserved scientific source and original execution revision: `04fac901f7af5478b62ff730833c026637ddd2cd`.

Manual operational continuation revision: `1cfb44fa3989e5ed2c5e13805007bb20f22189f8`; recorded scope events 6–8, receipt `recovery_001/RECOVERY.json`. Prior stopped-segment evidence is retained separately.

Development: persists at all three measured endpoints; event-4 to event-8 net call reduction shrinks; no monotonicity claim. Additional: persists at all three measured endpoints; event-4 to event-8 net call reduction shrinks; no monotonicity claim. Fresh: K has fewer calls on the fresh event-8 panel.

| Event | Panel | Valid requests | O calls | K calls | K call reduction | Wins/losses/ties | Request-time reduction | Native-time reduction |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 6 | development | 64/64 | 2997 | 2904 | +3.1031% | 36/18/10 | +3.2116% | +3.0213% |
| 6 | additional | 64/64 | 3086 | 3062 | +0.7777% | 31/21/12 | +0.5287% | +0.7093% |
| 8 | development | 64/64 | 2914 | 2880 | +1.1668% | 31/21/12 | +0.8542% | +1.1284% |
| 8 | additional | 64/64 | 3079 | 3031 | +1.5589% | 33/25/6 | +1.4392% | +1.4972% |
| 8 | fresh | 64/64 | 2997 | 2897 | +3.3367% | 37/13/14 | +3.2227% | +3.2092% |

| Event | Arm | Event wall (s) | Training (s) | State save (s) | Export (s) | Peak allocated (GiB) |
|---:|---|---:|---:|---:|---:|---:|
| 5 | O | 352.352 | 312.613 | 21.401 | 15.278 | 19.147 |
| 5 | K | 356.195 | 317.088 | 20.813 | 15.274 | 25.306 |
| 6 | O | 334.858 | 297.593 | 19.089 | 15.303 | 19.200 |
| 6 | K | 341.024 | 301.413 | 19.987 | 15.396 | 25.318 |
| 7 | O | 342.718 | 307.080 | 17.414 | 15.355 | 25.298 |
| 7 | K | 359.893 | 324.243 | 17.553 | 15.240 | 25.298 |
| 8 | O | 325.493 | 289.956 | 17.384 | 15.121 | 25.298 |
| 8 | K | 342.715 | 306.411 | 17.629 | 15.342 | 25.298 |

Training, state save and export are components of event wall time; they are not added to it. Loading/resume checks and their memory peaks remain separate in COSTS.json. Recorded campaign process wall time: 8995.588 s; backend loading: 47.772 s.

| Execution segment | Revision | Process wall (s) | Backend loading (s) |
|---|---|---:|---:|
| original_through_idle_event5 | `04fac901f7af5478b62ff730833c026637ddd2cd` | 2750.277 | 21.757 |
| manual_event5_boundary_resume6_through8 | `1cfb44fa3989e5ed2c5e13805007bb20f22189f8` | 6245.311 | 26.015 |

Sum of recorded campaign process segments, including their idle waits; gaps between stopped and restarted processes are excluded. Segment timers overlap their contained phase timers and must not be added to them.
Recorded campaign total equals summed segment walls: True.

Recorded work: 512 returned feedback responses, 1024 added optimizer steps, 2048 added presentations, 1920 returned evaluation generations. Required event-6/8 retention receipt: passed; mode `retained_locally`. Local retention refers to the campaign destination recorded in the receipt. These are recorded execution counters; null denotes unavailable evidence.

Primary reduction is 100 × (sum O calls − sum K calls) / sum O calls, using one validated repeat-0 trajectory per request. Three timing repeats are repeated measurements, not independent examples. Panels are never pooled.

CALL_CURVE.csv retains event-4 historical O4/K4 counts beside fresh events 6 and 8. Historical times are excluded. Each timing comparison uses O/K measured together at the same fresh endpoint; native generation and request-cache time are nested in complete request time.

PER_REQUEST.csv retains incomplete and rejected requests; ALL_REPEATS.csv independently rechecks raw tokens, termination, native traces, counters, timings and identities. EVENT_PANEL.csv also reports accepted draft tokens per call and committed progress per call. COSTS.json preserves numeric update/save/export/loading and memory costs separately from serving latency. PROVENANCE.json indexes source/input/launch/state/work receipts by byte identity, including available execution-command evidence.

Coverage: 1920/1920 valid timed executions; 5/5 complete valid panels. The event-6 first-eight-request gate is included in the main budget. Fresh-panel canonical outputs come from the first O8 execution, with no extra F/reference generation.

Validity and interpretation limits: one continued O/K trajectory, no seed replication; all-valid-position O and current-prefix-plus-first-error K retain their qualified objectives. No KRC arm, tuning or asynchronous deployment speedup claim. Update costs do not establish adaptation-inclusive acceleration.

ARS fallacy scan, 11/11 design checks: Simpson’s paradox (panels separate); ecological fallacy (paired requests retained); Berkson’s paradox (fixed selection limits generalization); collider bias (no performance filtering); base-rate neglect (pooled denominators shown); regression to the mean (no extreme-request selection); survivorship bias (missing/invalid cases retained); look-elsewhere effect (all fixed endpoints shown); garden of forking paths (sealed design, no tuning); correlation/causation (no training-age causal claim); reverse causality (no causal claim from the curve). No significance test or seed-population inference is made.

Material Passport: ARS experiment-agent validate; ANALYZED by raw-record reconstruction, not an independent GPU replication. Stop after event 8 and this report; Stage 1 / project None unchanged.
