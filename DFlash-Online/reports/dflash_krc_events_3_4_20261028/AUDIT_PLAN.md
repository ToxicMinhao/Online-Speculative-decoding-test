# KRC events 3–4 independent audit plan

## Material Passport

ARS academic-research-suite 3.24.0, experiment-agent validation. Scope: same-team independent protocol, integration and retained-data audit of the authorized KRC2→KRC3→KRC4 continuation and fresh event-4 comparison. This is not blinded external review or an independent training rerun. The auditor performs no SSH, GPU work or scientific optimizer steps. Stage 1 / project None is unchanged. The user handoff is the authorization; this document supplies verification procedures, not additional scope.

## Required evidence and identities

Local receipts below live under `/home/minhao2/EAGLE_SPEC/research/`; their exact bytes are bound in `AUDIT_SOURCE_RECEIPTS.json`. Historical remote paths are read from those receipts. Remote presence, state identity and available capacity still require launch-time checks by the executor.

| Role | Source receipt | Receipt-resolved remote artifact |
|---|---|---|
| Qualified KRC2 recovery and export | `dflash_krc_continuation_20261028/raw/campaign_001/KRC2/events/event_002/RESULT.json` | `/data/sources/minhao/runs/dflash_krc_continuation_20261028/campaign_001/KRC2/training_state.pt` and `KRC2/events/event_002/export` |
| KRC2 stopping boundary | `dflash_krc_continuation_20261028/raw/campaign_001/TRAINER_CLOSED.json` | Global step 256, 512 ordinary presentations, event history 001/002, no accumulated gradients, horizon 7424 |
| Independently sealed KRC2 export | `dflash_krc_off_20261028/raw/sealed/SEAL.json` | Model SHA-256 `89e12d038799a4b8c2de24fc2e783fc9a075f4d8574528d4e512b1d96004efa9`, 2097259104 bytes |
| O4/K4 exports | `dflash_ok_continuation_20261027/raw/FINAL_STATE_AUDIT.json` | `/data/sources/minhao/runs/dflash_ok_continuation_20261027/campaign_001/{O,K}/events/event_004/export` |
| Event 3/4 feedback and prepared features | `dflash_ok_continuation_20261027/raw/campaign_001/event_00{3,4}/POOL.json` | Retained O2/O3-generated responses and 128 prepared feature identities per event |
| Ordinary presentation/anchor bindings | `dflash_ok_continuation_20261027/raw/campaign_001/{O,K}/events/event_00{3,4}/RESULT.json` | O/K features and normal sampled schedules match in local retained evidence |
| Panel inputs and canonical outputs | `dflash_krc_continuation_20261028/raw/sealed/PLAN.json` and its two `evaluation/{development,additional}/CANONICAL.json` files | Original development and seed-20261028 additional panels, 64 requests each |
| Development O4/K4 count controls | `dflash_ok_continuation_20261027/raw/campaign_001/evaluation_004/MEASUREMENTS.jsonl` | Exact repeat-0 per-request maps, pooled counts 2994/2942 |
| Event-2 history | KRC2 `evaluation/{development,additional}/MEASUREMENTS.jsonl` | K2/KRC2 count maps; development 2973/2967, additional 3136/3116 |

O4 model SHA-256 is `17d63cfaa19686d61ee5a70952202578887bf5b6edf941c6843255f96f47d105`; K4 is `0ed5d8e29e3bfd21eed975d1a7927cfa8a24c38613f99fb347968ada62da7788`. Both are 2097259104 bytes. KRC2's complete state is not in the compact local mirror; do not imply a local full-payload hash check. The executor must seal the actual retained payload and establish snapshot/export and live restoration equality before training.

## Prelaunch implementation review

1. Compare inherited scientific files to their qualified source pins; only additive continuation/evaluation orchestration may change. Preserve historical O/K `objective.py` and `training.py` mapping to archived bytes, rather than falsely treating the publication revision as their execution revision. Record actual execution commit, dirty-tree patch identities, configuration and command.
2. Restore KRC2 corrected BF16 draft, BF16 Adam moments and counters, all Python/NumPy/Torch/CUDA RNG, complete configuration/contract, event history and empty accumulation. Reject KRC2_OFF, K2 and serving-only exports. Restore the already-disabled probe contract exactly; event-1 probe migration cannot be replayed blindly.
3. Advance event-local windows 1–128 at global steps 257–384 and 385–512. Build each cyclic reference schedule from its original O receipt, using a validated local counter view only where required by the inherited builder. Normal ordinary shuffle and anchor draws must match all 256 retained bindings per event; no seed reset, forced anchors or foreign-arm RNG.
4. Save independent complete KRC3 and KRC4 states and exports. The inherited `training_state.pt` path is mutable; a pointer to that slot does not preserve KRC3. Verify saving/exporting KRC3 preserves the live state and RNG used for event 4. Do not add event-3 serving generation.
5. Check runtime storage admission against two retained full states, two exports, atomic replacement and export temporary overlap, serialization slack, compact retrieval staging and required reserve. The last known 17.8 GB free is not sufficient evidence of admission: two approximately 6.3 GB states plus two approximately 2.1 GB exports already consume about 16.8 GB before temporary overlap/reserve. Only exact newly measured capacity and authorized cleanup receipts can resolve this.
6. Check the three-arm order against all 1152 slots. Rotations across three repeats can give each request each order position once. With 64 requests, a single repeat's first-position counts are necessarily 22/21/21; require the declared rotation and exact aggregate balance rather than impossible equality within that repeat. The first native gate is part of the declared generation budget.

## Postrun training audit

Independently rebuild both reference schedules using the previously qualified retained-record parser in `dflash_krc_event_20261027/audit.py`. Reuse its `BASE.score_window` and `TRACE.training_exposure` helpers to reconstruct reference prefix statistics, margins, pooled triangular denominator, correction branches and supervised exposure from stored labels, validity, selected positions and native decisions. Never infer post-correction prefix benefit when optional replay probes are disabled.

For each event require 128 distinct responses, two passes, 256 ordinary presentations, 128 Adam updates, 256 original forwards, 256 pre-reference forwards and 256 post-K-reference forwards; optional diagnostic forwards are zero. Each event has 256 K backwards and at most 256 violation backwards. Across events: 512 ordinary presentations, 256 Adam steps, 1536 core forwards and at most 1024 backwards. Startup is separately reported (qualified two disposable forward/backwards, zero Adam). Reference exposure is 256 presentation occurrences per event, with two reference phases; do not report the phases as 512 independent examples.

Check every window's disjoint A/R pairs, same-pass cyclic wrap, R's own anchors, native prefix decisions and FP32 correction accounting. Confirm all reference operations preserve ordinary RNG, mode and cache, and direct correction preserves Adam counters/moments. Sum window work and nested timers to event evidence. Compare naturally drawn ordinary records against both retained O and K schedules. Preserve event-4 capped `train_00399` at 2048 response tokens / 2231 total tokens; event 3 has no capped response. Verify feature and prompt exclusion of both evaluation panels.

Require full snapshot/export identity and live resume receipts at KRC2, complete saved KRC3/KRC4 checks, exact global boundaries 256→384→512, presentations 512→768→1024, horizon 7424, completed event history and no new feedback or qualification optimizer steps. Training must close before fresh native evaluation starts.

## Postrun serving and analysis audit

Reconstruct each of the 1152 native block trajectories with the qualified `trace_record` parser from `dflash_prefix_ranking_one_event_20261025/audit_results.py` and `validate_generation` from the pinned `online_feedback/native.py`. Check exact prompt/response tokens, termination, accepted draft tokens, committed progress, verification calls, clipping and final block; raw worker capture must equal the campaign record and be unique. Validate publication order, loaded weight identities, cache maintenance and zero feedback capture. Require all three repeats to agree on deterministic counts and canonical outputs.

Each panel must contain exactly 576 chronological measurement slots: O4/K4/KRC4 × 64 requests × 3 repeats, with no extra or unaccounted complete generation. Development enforces retained per-request O4 and K4 counts. Additional-panel O4/K4 counts are new and have no historical gate. No count from the earlier seed-20261027 O/K panel is admissible here.

Pool counts once per request using a validated trajectory, and use the three-repeat mean for request-time totals. Compute all three comparisons, with O4 primary: KRC4/O4, K4/O4, KRC4/K4. Independently reconstruct pooled reduction, per-request wins/losses/ties and gross calls saved/added; compare these to report CSV/JSON only afterward. Keep panels separate. Retain each repeat's request, native-generation and cache times, noting that native/cache timers are nested in request time. Loading, publication, training and persistence stay outside latency.

Reproduce event-1/2/4 count curves from source maps and fresh event-4 counts. Historical absent O values on additional remain absent. Distinguish within-arm change from changing relative gaps, historical from fresh measurements, and correction-history attribution from the narrower unmeasured effect of event-3/4 corrections. One training trajectory and three timing repeats do not establish seed robustness or independent deployments.

## Integrity and meaningful CPU fixtures

Verify local retained receipt hashes, all retrieved files, remote source/artifact pre/postflight seals, execution commit/patch bindings, final counters and zero failure/unknown/unattempted work. Source/runtime tensors that remain remote must be explicitly labeled as supported by remote checks, not locally rehashed. A failed or missing gate is an audit issue, never silently converted to passed.

New CPU fixtures will exercise three-arm rotation/budget, one-trajectory pooled arithmetic (including valid regressions), missing/duplicate/repeat-inconsistent measurements, development-only O4/K4 gates, and event-local/global boundaries. Existing qualified native and correction parsers remain reused. The auditor will inspect final integration after trainer/evaluator APIs settle and save a separate prelaunch review before execution.

Current status: protocol and retained compact artifacts inspected; no scientific run or remote operation performed by this reviewer. No budget contradiction found. Full-state availability, live restoration and current capacity remain execution-time gates.

Execution source commit: `11b7718a04416a89025a6df69c0c33e558e5fe1b`. No scientific source changes after launch.
