## Material Passport

ARS academic-research-suite, experiment-agent run/validate. The declared run completed; numerical analysis and independent compact-evidence audit passed. No independent scientific rerun is claimed. Stage 1/project None is unchanged.

**K’s advantage persists through four events, but it does not grow steadily.** Its verification-call advantage over ordinary OSD is 1.56% after event 1, 2.81% after event 2, and 1.74% after event 4. Both arms improve relative to retained frozen DFlash. Between events 2 and 4, O saves another 65 calls and K another 31 calls on the same evaluation inputs, narrowing the gap from 86 to 52 calls. This is evidence for a modest sustained benefit on this trajectory, not a growing advantage or a general robustness result.

**What was run.** Continue each arm’s own complete event-1 weights, BF16 Adam moments/counters and Python/NumPy/Torch/CUDA RNG. Events 2–4 each collect 128 responses with O’s preceding checkpoint, prepare one immutable shared pool, and train O and K separately for two passes, 256 presentations and 128 steps. K recomputes first-error selection from its own current logits. Both updates finish before the next collection. This controls feedback while comparing objectives; it is not two independently measured online deployments.

Qualified Qwen3-8B target / DFlash-b16 pair; constant LR 3e-4, BF16 AdamW and KL arithmetic, weight decay 0.01, clipping 1.0, shuffled responses each pass, up to 512 anchors per response, two-response accumulation, Flex training attention, response averaging and full-block attention are unchanged. O supervises every valid proposal; K includes the current correct prefix and first error, with selected-weight normalization. Training length is 2,272. The inherited 7,424-step horizon is preserved while orchestration stops each arm at 512 cumulative steps. No seed reset or event-1 repeat occurred.

Feedback inputs are the next sealed GSM8K train indices 128–511, excluding the original 128 feedback inputs and both current evaluation panels, with exact pinned preprocessing and prompt deduplication. All 384 have disclosed older-project exposure, which the user permitted; fresh responses were generated for this continuation. Dataset answers are unused. Evaluation reuses the original 64-request development panel; the later confirmation panel is not pooled here.

Native evaluation retains greedy decoding, block size 16, 2,048 tokens or EOS, context 4,096, concurrency 1, capture disabled, and the pinned SGLang/FA3 runtime with cache invalidation. Three fresh alternating-order repeats were measured after events 2 and 4, with training closed. Event-1 measurements are retained.

**Learning curve.** Positive percentages mean reductions. Frozen uses 3,140 retained calls on the identical 64 inputs. Calls use one validated trajectory per request, not all three repeats as independent samples.

| Event | Cumulative steps/arm | O calls | K calls | K/O call reduction | O/F reduction | K/F reduction | K/O request-time reduction |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 3,075 | 3,027 | 1.56% | 2.07% | 3.60% | 1.43% |
| 2 | 256 | 3,059 | 2,973 | 2.81% | 2.58% | 5.32% | 3.02% |
| 4 | 512 | 2,994 | 2,942 | 1.74% | 4.65% | 6.31% | 1.88% |

Each time ratio compares O and K measured together at that event. Historical-versus-fresh raw latency is not interpreted as learning progress, and no fresh F latency claim is made. These are available-checkpoint serving measurements; training, loading, export and activation are excluded. All three timing repeats favor K at each event.

| Event | Requests with fewer K calls | More K calls | Ties | Net calls saved |
|---|---:|---:|---:|---:|
| 1 | 33 | 21 | 10 | 48 |
| 2 | 37 | 12 | 15 | 86 |
| 4 | 34 | 20 | 10 | 52 |

At event 4, K improves over retained F on 50 requests, regresses on 8 and ties on 6. All evaluated checkpoints produce exactly the same 19,547 output tokens across the panel, with EOS termination on every request and no cap hits. O/K event-4 calls per 1,000 output tokens are 153.169/150.509; accepted draft tokens per verification are 5.526/5.643 and committed progress per verification is 6.507/6.622. Request-level regressions remain in the tables.

Two feedback responses reached the 2,048-token cap: train_00187 in event 2 and train_00399 in event 4. Their total training lengths were 2,133 and 2,231, within the unchanged 2,272 limit. They were retained without truncation or substitution.

**Exposure.** The arms use identical feedback, normal shuffled presentation order and anchor draws within every event. K’s selection changes which occurrences and weights receive loss; its mask and selected-weight renormalization are jointly tested. These are repeated position occurrences across blocks and passes, not unique tokens.

| Event | Sampled anchors per arm, both passes | Available occurrences | K selected occurrences | Selected fraction | K selected weight fraction |
|---|---:|---:|---:|---:|---:|
| 1 | 65,096 | 949,560 | 534,063 | 56.24% | 70.52% |
| 2 | 66,938 | 977,434 | 556,911 | 56.98% | 71.12% |
| 3 | 66,818 | 975,476 | 564,886 | 57.91% | 71.74% |
| 4 | 65,996 | 963,417 | 554,913 | 57.60% | 71.68% |

O uses all available occurrences. Per-event weighted numerators/denominators, no-error blocks and all 15 first-error-position counts are retained in TRAINING.csv and the raw event receipts. Boundary accumulation rows are counted once. Raw O/K losses supervise different selected distributions and are not direct efficacy comparisons.

**New continuation costs, events 2–4 only (seconds).**

| Arm | Training | Checkpoint persistence | Export | Complete update events | Loading + resume validation |
|---|---:|---:|---:|---:|---:|
| O | 962.039 | 53.050 | 41.555 | 1065.730 | 125.576 |
| K | 999.706 | 50.625 | 42.355 | 1101.760 | 120.500 |

K’s measured training total is 3.92% higher than O’s. Its recorded selection computation is 40.639 s, already included in training. Teacher-head work is also nested in training, not added again. Complete update-event time includes training, persistence, export and event bookkeeping; loading/resume validation is separate.

| Shared feedback or benchmark component | Seconds | Accounting |
|---|---:|---|
| Shared O feedback collection wall | 323.441 | Collected once across the system |
| Feedback request latency | 319.928 | Included in collection wall |
| Capture host work | 235.811 | Nested in captured requests; not separately added |
| Capture persistence | 19.415 | Nested in captured requests |
| Feedback preparation wall | 46.277 | After collection |
| Teacher preparation | 19.552 | Nested in preparation; 384 terminal feature forwards |
| Prepared-feature persistence | 17.864 | Nested in preparation |
| Activation for O feedback collection | 5.050 | Outside captured request latency |
| Paired benchmark activation/switching | 554.842 | Benchmark overhead; outside serving latency |
| Paired benchmark wall | 1193.749 | Includes repeats, activation and validation |
| Initial native model loading | 20.258 | One-time campaign setup |

Campaign driver wall time was 4050.667 s. The supervisor, including preflight, CPU checks and postflight, ran from 02:05:43 to 03:15:03 UTC on October 27 (69 min 20 s). Timers above overlap as labeled and must not be summed indiscriminately. Benchmark repeats are measurement cost, not a deployed serving episode. No independent adaptation-inclusive O/K deployment speedup is established by this shared-feedback design. Peak training-process allocation/reservation was 25.48/29.80 GiB; these are process peaks, not an isolated algorithm-memory attribution.

**Validity and accounting.** Exactly 384 feedback plus 768 evaluation generations = 1,152 new complete-response generations; 1,536 presentations and 768 optimizer steps. No extra optimizer qualification steps, generation warmups, retries, invalid comparisons, unknown or unattempted work. The initial 48 event-2 evaluations formed the native correctness gate within the budget. Both arms finish at 512 steps / 1,024 cumulative presentations.

All six saved/live restore checks passed for own-arm weights, Adam, counters, config, training mode and complete RNG, with zero mismatch. Final CPU reload independently checks both completed states against event-4 exports and 512-step Adam counters; final O/K RNG payloads match. All 185 sealed-artifact checks and the seven external runtime source checks passed before and after execution. Independent compact auditing reconstructs 52,270 native blocks and verifies all 1,152 response records, 384 terminal feature forwards, shared feedback and matched exposure. All 4,001 retrieved compact files match their manifest. Original event-1 artifacts remain unchanged; final full states, all six new exports and feature pools remain on A800.

Only the experiment orchestration, state validation, two-arm evaluator and local analysis were added. The 117 inherited scientific source files remained unchanged. The 36 focused execution CPU tests, five selection tests and separate reviewed analyzer/auditor tests passed. Pre-launch test-fixture and dictionary-representation corrections are preserved; they did not change learning settings or introduce GPU work. Exact state restoration does not prove an uninterrupted GPU execution would be bitwise identical; no extra warmup or tolerance change was introduced.

**What this establishes.** With this seed, shared O-generated feedback sequence and fixed development panel, K remains modestly better than ordinary OSD after more feedback. Both continue improving versus frozen; O catches up partly between events 2 and 4. It does not establish monotonic growth, independent training-seed/feedback-window robustness, that masking alone caused the benefit separately from renormalization, or net serving-plus-adaptation acceleration. The original development panel is reused. The retained O/K checkpoints remain the agreed starting points; no full-trajectory bitwise-equivalence claim is made. No algorithm change or additional run was made.

**Evidence files.**

ALL_REPEATS.csv contains 384 retained event-1 O/K measurements plus the 768 new event-2/4 measurements. PER_REQUEST.csv contains 64 reused requests at each of three checkpoints, not 192 independent questions. The 384 new feedback generations are retained separately in the raw window records.

- [Event comparison](/home/minhao2/EAGLE_SPEC/research/dflash_ok_continuation_20261027/analysis/EVENTS.csv)
- [Per-request results, including regressions](/home/minhao2/EAGLE_SPEC/research/dflash_ok_continuation_20261027/analysis/PER_REQUEST.csv)
- [Every O/K timing measurement, events 1/2/4](/home/minhao2/EAGLE_SPEC/research/dflash_ok_continuation_20261027/analysis/ALL_REPEATS.csv) and [repeat totals](/home/minhao2/EAGLE_SPEC/research/dflash_ok_continuation_20261027/analysis/REPEATS.csv)
- [Training exposure and source costs](/home/minhao2/EAGLE_SPEC/research/dflash_ok_continuation_20261027/analysis/TRAINING.csv) and [separate cost accounting](/home/minhao2/EAGLE_SPEC/research/dflash_ok_continuation_20261027/analysis/COSTS.csv)
- [Independent execution audit](/home/minhao2/EAGLE_SPEC/research/dflash_ok_continuation_20261027/FINAL_AUDIT.md) and [final full-state CPU audit](/home/minhao2/EAGLE_SPEC/research/dflash_ok_continuation_20261027/raw/FINAL_STATE_AUDIT.json)
- [Pinned execution plan](/home/minhao2/EAGLE_SPEC/research/dflash_ok_continuation_20261027/raw/sealed/PLAN.json) and [configuration](/home/minhao2/EAGLE_SPEC/research/dflash_ok_continuation_20261027/raw/sealed/config.yaml)
- [Selection and older-exposure provenance](/home/minhao2/EAGLE_SPEC/research/dflash_ok_continuation_20261027/data_preparation/DATA_PREPARATION_REPORT.md)
- [Raw compact artifacts](/home/minhao2/EAGLE_SPEC/research/dflash_ok_continuation_20261027/raw) and [retrieval verification](/home/minhao2/EAGLE_SPEC/research/dflash_ok_continuation_20261027/RETRIEVAL_VERIFIED.json)
- [Implementation](/home/minhao2/EAGLE_SPEC/research/dflash_ok_continuation_20261027/IMPLEMENTATION.md). Authoritative A800 module: /data/sources/minhao/DFlash-Online/experiments/ok_continuation; artifacts: /data/sources/minhao/runs/dflash_ok_continuation_20261027.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_ok_continuation_20261027/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.

Historical-source gap: this run archive contains 2 unresolved source-pin occurrences; the missing versions remain explicitly identified.
