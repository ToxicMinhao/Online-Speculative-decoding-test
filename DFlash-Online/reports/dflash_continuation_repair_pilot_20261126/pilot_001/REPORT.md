# Continuation-repair pilot: final result

**TAIL did not demonstrate an improvement over native C+K+PAIR.** It saved only two verification calls across 128 requests and was 0.098% slower in complete-request latency. Both 95% intervals cross zero. The prespecified outcome is **NO_DEMONSTRATED_BENEFIT**: retain **BASE (native C+K+PAIR)** and close this fixed refinement branch. This is not an equivalence result or proof that every continuation-based allocation fails.

## Material Passport

Experiment: continuation-repair pilot, 2026-11-26. Scope: one matched warm event plus fixed endpoint evaluation. ARS modes: run and validate. Native execution and independent CPU evidence checks completed; no independent GPU replication was added. Reused OpenCode development panel; one training trajectory. Private source IDs, prompts, token arrays, anchors, features, recoveries and raw validity evidence remain outside Git.

## Serving result

| Metric | BASE | TAIL | TAIL reduction | Paired 95% interval |
|---|---:|---:|---:|---:|
| Summed per-request mean complete latency (s) | 135.777717 | 135.910396 | -0.09772% | [-0.49671%, +0.27973%] |
| Verification calls | 7,639 | 7,637 | +0.02618% | [-0.36009%, +0.39771%] |
| Native accepted draft tokens |42,459|42,461|—|—|
| Native accepted tokens / verification |5.55819|5.55991|—|—|
| Committed progress / verification |6.55819|6.55991|—|—|

Calls: 40 request improvements, 40 regressions and 48 ties; 60 gross calls saved and 58 added. Complete latency improves on 57 requests and worsens on 71. All 128 requests produce identical canonical token IDs and termination across both arms and all repeats. Every request ends in EOS. Both arms return 50,226 output tokens (mean 392.39; range 144–1,314); committed progress is 50,098 because the first prefill token is excluded from verification progress. Native acceptance and the separately retained stop-clipped acceptance counter are not interchangeable; both appear in the tables.

| Complete latency total (s) | Repeat 1 | Repeat 2 | Repeat 3 |
|---|---:|---:|---:|
| BASE | 135.796026 | 135.818138 | 135.718986 |
| TAIL | 136.042372 | 135.862024 | 135.826791 |

All 768 scored generations and 24 warmups completed and passed; invalid, unknown and unattempted counts are zero. Each arm’s call counts agree across its three repeats. Both endpoints were freshly evaluated on the same physical A800 with the other card idle and no concurrent training, preparation or artifact transfer. Greedy BF16 SGLang/FA3, block 16, 2,048 tokens/EOS, logical 4,096/physical 4,112 context, concurrency 1, disabled capture and qualified cache handling remain unchanged.

The primary estimate averages repeats within request, then pools by summing request means. Counts use one validated trajectory per request. The 95% intervals use 10,000 paired whole-request draws, PCG64 seed 20261126, linear percentiles and common draw indices. Timing repeats are not independent training replicates. The latency point estimate is negative, so the tentative rule is not met even though its lower bound lies just above −0.5%.

## What changed locally

TAIL reallocates only existing HIGH first-error repair weight within each response using the detached running-minimum continuation score. It preserves HIGH total mass, LOW/preservation coefficients and the native FKL+PAIR objective. It adds no scoring forward or direct suffix loss. Exact native/disabled qualification, matched retained BASE reuse and independent FP64 coefficient checks passed. All 256 training presentations were active; 22,781 HIGH coefficient occurrences changed, and final serving parameter tensors differ from BASE.

| Fixed-parent outcome | TRAIN: BASE → TAIL | Retention: BASE → TAIL |
|---|---:|---:|
| Raw old-first-error repairs | 5,065 → 5,065 | 10 → 10 |
| Repairs with preceding matches preserved | 4,861 → 4,860 | 9 → 9 |
| Reachable original followers | 5,301 → 5,327 | 12 → 12 |
| Lost consecutive-prefix occurrences | 1,464 → 1,539 | 32 → 18 |
| Damaged individual prefix positions | 564 → 589 | 14 → 13 |
| Blocks with prefix damage | 401 → 409 | 10 → 8 |
| Capped signed prefix progress | 8,698 → 8,648 | -11 → 3 |

TRAIN covers 128 responses/8,192 matched blocks, including 6,329 parent first errors. Retention covers 8 earlier parent-training responses/64 blocks, including 31 parent first errors. In TRAIN, 26 additional reachable follower occurrences are outweighed by 75 additional lost-prefix occurrences and one fewer useful repair: signed progress falls by 50. Retention improves by 14 through fewer lost-prefix occurrences. These mixed training-conditioned descriptions do not explain a percentage of serving time or establish generalization. “Raw repair” does not imply preserved preceding matches. The historical new-buffer witness overlaps TRAIN and is not independent evidence. PR64’s observational conclusion remains INCONCLUSIVE; no regression or predictor fitting was repeated.

## Work, costs and preserved failures

| Work | Completed |
|---|---:|
| Disposable qualification Adam steps |3|
| TAIL event presentations / Adam steps |256 /128|
| Startup response forwards/backwards / Adam steps |8 /8 /0|
| Total qualification+training+startup response forwards/backwards |270 /270|
| Readout forwards |137 (74 +0 +63 across three processes)|
| Scored generations / warmups |768 /24|
| New feedback generations / teacher-feature preparations |0 /0|

TAIL starts from complete PAIR1 step 128/presentations 256/event 1 and ends at 256/512/event 2. Natural presentation/anchor sampling matches retained CONTROL; optimizers and RNG were not reset. Retained BASE was not retrained. The 270/270 figure counts response-level forwards/backwards; physical projection and checkpoint-recomputation work is recorded separately in COST_AUDIT.json (main event: 23,580 objective chunks and 47,160 vocabulary-head calls).

Main training took 607.428 s; loading 48.219 s, disposable startup 39.865 s, checkpoint persistence 19.076 s and export 15.791 s are separately retained nested timers. Historical BASE training was 564.545 s, making TAIL 7.60% longer in those historical measurements; this is not a fresh matched overhead control. No training-inclusive speedup is established. Recorded training peak allocated/reserved memory was 20.555/27.781 GB. Qualification and serving CUDA peaks were not recorded, so these are not campaign-wide peak claims.

Serving scored requests totaled 815.064 s across all repeats, with 32.998 s warmups, 21.761 s loading and 63.844 s activation recorded separately. Summed per-request mean native-generation times were 131.754180 s BASE and 131.800432 s TAIL; cache-maintenance means summed to 0.054260 s and 0.054744 s. Detailed repeats and timers are retained; aggregate request latency is not a kernel-time breakdown. Nested timers must not be added again to process charges.

Total charged process lifetime was 2,401.552 s (40.026 minutes). Original-clock elapsed time was 31,880.663 s (8 h 51 m 21 s), including approval/repair idle time. The user explicitly extended only elapsed allowance from 4 to 12 hours; the original clock, four GPU-card-hour cap and settings remained unchanged.

The original readout process hit a concurrent storage-receipt temporary-file race after 74 completed forwards. Its first repair then rejected the error receipt before model loading because compact FileNotFoundError text omitted a filename; this added 8.846716 s and zero model work. Both failures remain preserved. The approved continuation ran only 63 unexecuted readouts and the original serving schedule. No training, qualification or successful readout was repeated. A CPU-only reporting correction counts all three processes separately from two forward-producing attempts. A later audit found six extensionless process receipts omitted from compact retrieval; their 30 bytes were retrieved and authenticated against the original preserved hashes, with no native rerun.

Before the pilot, 19.243552768 GB was freed by retiring exactly three obsolete CODE event 57 rollback states after verifying retained final event 58 states and compact evidence. Direct event 57 optimizer/RNG rollback is lost, not archived. All pilot inputs, final states and scientific evidence remain. Postflight verified 492 original/input/endpoint files and all four source snapshots unchanged, with 414.287 GB free on A800, preserving the 400 GB reserve. Remote charged storage is 10.447 GB; observed all-location transport overlap was 11.728 GB, below 24 GB. The conservative observed peak plus the entire shared Git history and publication allowance is 11.798 GB in [RESOURCE_STORAGE.json](RESOURCE_STORAGE.json); storage figures describe observed boundaries plus prospective peak admission, not continuous filesystem sampling.

## Audit, provenance and decision

Independent CPU audits pass: 6,957 work/state checks; 7,787,452 serving/validity/statistical checks; 1,856,640 mechanism checks; 337 cost checks. All 54 final focused CPU tests pass. Separate agents independently reduced receipts, tokens and metrics; they share task context, source and some implementation authorship. This is not a blinded independent GPU reproduction. Compact data cannot independently recreate the full model backward or prove full-vocabulary rival optimality; those scopes rely on preserved qualification and source evidence. All 11 ARS statistical fallacy categories were considered in AUDIT_SERVING.json.

Training source `b3637cfe4ac3fb4096368cc0c93896b420bdb9a8`; remaining readout/serving source `b059d0ea7595f24a9db114b46a98cae60f0602b0`; CPU analysis source `00ac1944e8d8d535c24c690917e2a2890af64659`. Publication commits only add audited reporting/audit code after execution. [Commands](COMMANDS.md), [identities](PROVENANCE.json), [failures](FAILURES.json), [audit](AUDIT.md), [costs](COST_AUDIT.json), [request table](PER_REQUEST.csv), [all repeats](ALL_REPEATS.csv), [bootstrap](BOOTSTRAP.json), [pool activity](POOL_ACTIVITY.csv) and [mechanism totals](MECHANISM_SUMMARY.json) accompany this report.

**Decision: retain BASE (native C+K+PAIR); this fixed TAIL refinement has no demonstrated endpoint benefit.** The small count change, uncertain timing and mixed training readouts do not justify expansion or adoption. This fixed event does not establish equivalence, training-seed robustness, fresh-panel generalization, calibrated repair utility or failure of all continuation-based methods. No tuning, further event or new experiment was run or is authorized by this report.
