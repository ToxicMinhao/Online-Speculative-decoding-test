# Second-follower training pilot

Decision: **No demonstrated improvement over native K+ on this fixed panel.** The candidate used one more verification call overall (11,523 versus 11,522), with a paired 95% interval spanning benefit and harm. Its complete-request time was 0.579% longer at the point estimate, also unresolved. The experiment completed correctly; this is a completed development test without a demonstrated benefit, not an inactive intervention or evidence of equivalence.

All 128 main optimizer steps, 1,024 timed executions and 128 matched starts completed. Every timed output matched the saved canonical tokens and termination, and counts were consistent across repeats. Invalid, unknown and unattempted timed executions are zero. Both GPUs were idle at completion.

Analysis source: `279333255b32c907b1ba05b4a399d994a4cacabb`; scientific execution source is recorded below. Independent arithmetic and resource checks are described in `RESULT_REVIEW.md`. Private token records, tensor-bearing failure details and the full source/input manifests remain outside Git.

Execution commit: `95fdbe1eb599b2cbc26f904bdfca46a3b7b501d2`. One GSM8K continuation from step 512 to 640; native KPLUS5 weights are reused as NATIVE. Both serving arms use the same native algorithm. The exact PR39 panel has known prior project evaluation; this single inherited seed does not establish generalization.

| Sealed input / checkpoint | SHA256 |
|---|---|
| panel_manifest | `9d58a5ed0eff2720a4df8964770ba55e50b6025ace22d4b6a348b29a34777531` |
| references | `750636e5feb9941e1073dc7e8c2a57205304469e9faeb89ec90e8e3724eb2393` |
| matched_start | `c6d29914fd132b4b3cf576380c0f63ea5c3d8f264633abbfcb0a7e087c709476` |
| origin_state_identity | `5e113f62a6c3d519f7f0f9224a924a7fec3ce06f7a30e49eb682e9363b8a2d74` |
| origin_weights_identity | `4f3010e961bd7dcf35fa100486c355ffbc7629d816dda6e2b8fa001878255701` |
| original_schedule_identity | `31c36c7f7bb3cbc5795f85de720e39ba0c1739488eaff003444f38aed4c9b196` |
| sealed_plan | `48601f4a5fc0c30355f463296cdc6d8bfa68099fb1791188dfeb6caa97e0a777` |
| NATIVE serving weights | `9dcb0d637eabef6bd4e34c552da0448faa9510e6bcaf5c753a6bd1b43674b795` |
| SECOND serving weights | `3c3579b66820e5554c1fed2b2a40c8febfdaa3311e4296ad5d14b03e3adf416a` |

| Endpoint, SECOND versus NATIVE | Reduction and paired 95% interval |
|---|---:|
| Verification calls | -0.0087% [-0.3738%, +0.3548%] |
| Complete request time | -0.5789% [-1.4864%, +0.3505%] |
| Native generation time | -0.6054% [-1.2496%, +0.0521%] |
| Cache maintenance time | +1.4639% [-18.0746%, +17.9341%] |

Gross calls saved/added: 125/126; net saved -1. Improved/worse/tied requests: 78/91/87. Counts use one validated trajectory; times average the two repeats within each request. Intervals use 10,000 paired whole-request PCG64 draws, seed 20261113, linear percentiles.

Complete-time sweep reductions: sweep 1: -0.4324%, sweep 2: -0.7251%.
A latency benefit requires a positive lower confidence bound and both favorable sweeps. A call interval spanning zero is unresolved; it does not establish equivalence.

| Arm | Calls, one trajectory per request | Mean-repeat request-time sum (s) | Capped requests |
|---|---:|---:|---:|
| NATIVE | 11522 | 207.676335 | 0 |
| SECOND | 11523 | 208.878591 | 0 |

The reallocation was active in 255/256 response presentations and all 128 steps. It moved **0.11419% of total selected loss weight**, or **3.53253% of native high-priority weight**. Qualification demonstrated a change in BF16 weights. These facts establish that the treatment was applied; they do not explain why the endpoint comparison remained unresolved.

Exposure audit: **passed**. Observed presentations 256/256; optimizer steps 128/128. Active responses/steps: 255/128. SHORT/LONG/censored occurrences: 5215.0/8783.0/799.0.

Transferred mass: 339.6559110060334; sum transfer / sum selected weight: 0.0011418980473852998; sum transfer / sum native high weight: 0.035325262921654624; mean response transfer / selected denominator: 0.001104168076411344.
Largest response/depth shares of total transfer: 0.014825654443386968/0.16984021591409706; mixed/active depth strata: 2151.0/2151.0.
The scalar audit independently recomputes the capped depth transfer, donor floor, mass, native coefficients, numerator correction and unchanged denominator. It cannot reconstruct unsaved per-anchor proposals or class membership. Per-response and depth metrics are in EXPOSURE.csv and DEPTH_EXPOSURE.csv.

Qualification: passed; actual positive-transfer window checked: True. Exact state/gradient/RNG comparisons are retained execution receipts, not a second numerical replay by this report.

- NATIVE: transferred mass 0.0; changed BF16 gradient/weight coordinates None/None; exact comparison None; next anchor draws 1.
- ZERO: transferred mass 0.0; changed BF16 gradient/weight coordinates 0/0; exact comparison True; next anchor draws 1.
- ACTIVE: transferred mass 2.142814040184021; changed BF16 gradient/weight coordinates 975920969/5971778; exact RNG and next-sampling identity passed. Its weights and gradients intentionally differ. ZERO separately passed the full loss/gradient/weight/Adam/RNG identity comparison; one next-anchor draw was checked in each case.

Matched later-parent-start diagnostic: **complete**. These local accepted-prefix changes are separate from full-request calls and latency; other parent starts remain unmeasured.

| Class | NATIVE repairs / earlier breaks | SECOND repairs / earlier breaks | Paired net prefix tokens |
|---|---:|---:|---:|
| SHORT | 18.0/10.0 | 18.0/9.0 | +0 |
| LONG | 15.0/9.0 | 17.0/9.0 | +5 |

Census-weighted LONG+SHORT candidate-minus-control change per later parent start: +0.011674; request-cluster 95% interval [-0.019431665929602233, 0.05218658545072964]. LONG weight 403/2697; SHORT 263/2697. The positive local direction remains uncertain: its interval includes zero. Full follower acceptance, donor losses, paired changes and contributing request counts are available in `MATCHED_STARTS.csv` and `MATCHED_SUMMARY.csv`; the complete summary is retained privately.

SHORT donor starts, candidate versus control: gross prefix tokens gained/lost 1/1; improved/worse/tied 1/1/62. These losses remain reported regardless of endpoint direction.

Campaign status: completed. Conservative occupied-GPU clock: 2598.9550755023956 s of 4,500; unused ceiling 1901.0449244976044 s. Unattempted main steps/timed completions: 0/0.

Training, qualification/startup, publication/loading, endpoint requests and matched replays have separate recorded clocks. Nested timers are not added. Native control training is historical reuse, so this run provides no contemporaneous NATIVE training-cost comparison. Unknown or missing work is not imputed as zero. The private summary retains phase receipts and planned versus observed work. Candidate training took **342.53 s (5.71 min)**; the conservative total GPU clock, including the failed attempt, was **2,598.96 s (43.32 min)**, below the 75-minute limit. Historical native-control training time is not a matched overhead estimate.

Candidate event clocks (seconds): training 342.53250353597105; checkpoint save 19.513621646910906; export 15.17760973982513; enclosing event wall 379.76119854301214.

| Recorded phase | Status | Wrapper wall seconds |
|---|---|---:|
| EVALUATE_EXEC.json | completed | 1046.8306716829538 |
| MATCHED_EXEC.json | completed | 521.3674044422805 |
| qualification/ACTIVE_EXEC.json | completed | 124.81031202338636 |
| qualification/EXPOSURE_EXEC.json | completed | 77.80627684667706 |
| qualification/NATIVE_EXEC.json | completed | 116.75221784599125 |
| qualification/ZERO_EXEC.json | completed | 135.9770633932203 |
| training/TRAIN_EXEC.json | completed | 462.24650564976037 |

| Qualification phase (startup excluded) | Returned forwards | Returned backwards | Disposable updates |
|---|---:|---:|---:|
| exposure | 16 | 0 | 0 |
| NATIVE | 2 | 2 | 1 |
| ZERO | 2 | 2 | 1 |
| ACTIVE | 2 | 2 | 1 |

Discarded-startup receipts retained: 5. Their work ledgers are separate from the table. Next-anchor checks are sampler operations, not model forwards. Failed qualification returned counters do not bound unknown in-flight work. Deferred native-twin extra work: None; its nested diagnostic wall is None s.

Source/input/checkpoint identities, raw-audit validity failures and evidence stability are retained in private `SUMMARY.json` and `INPUT_MANIFEST.json`. All 4,841 compact evidence files (255,012,475 uncompressed bytes) were streamed and hash-verified after retrieval; the complete compressed archive and selected readable records are retained on Fenrir, with original raw artifacts and model/recovery tensors on A800. Raw token records and tensors remain in the private run directory; exported tables contain request identifiers and metrics only. No new seed, tuning, configuration, training event or follow-up is authorized by a favorable result.

Preserved prior attempt `c1362db680494d77042a61986392cb394df3a615`: OrderedDict-versus-dict type check stopped before tensor comparisons. Charged 18 forwards, 2 backwards, 0 Adam steps and 74.05223560333252 occupied-GPU seconds. This cost is included in the same 4,500-second ceiling, not added to its remaining allowance. Retrieved identity check: passed.

This handoff is complete and stopped. No additional event, tuning or panel selection follows from this result. A new experiment would need separate authorization and a concrete reason to revisit this fixed second-follower rule.
