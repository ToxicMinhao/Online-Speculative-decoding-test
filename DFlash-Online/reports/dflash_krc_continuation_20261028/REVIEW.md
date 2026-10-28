# Independent KRC2 continuation implementation review

**Verdict: no remaining implementation blocker for the authorized bounded run.** Native resume equality, storage admission and the first actual window remain required execution gates. CPU tests do not replace those checks or establish performance.

Material Passport: ARS academic-research-suite / experiment-agent validation; independent source and CPU review by a separate agent within the same project team. No SSH, GPU, model-scale scientific run, training campaign or serving generation was performed by the reviewer.

## Verified behavior

The additive trainer first reconstructs the exact saved KRC1 contract, then delegates restoration to the unchanged strict parent loader. The campaign binds the complete recovery payload to the corrected KRC1 export and checks live weights, native Adam moments/counters, full RNG, configuration, completed-event history and empty accumulation state. KRC1 remains read-only; new outputs go to a separate directory. The old 7,424-step horizon and ordinary K optimizer/objective remain intact.

Only after exact restoration does the continuation manager change `probe_steps` from `[1,64,128]` to `[]`. The before/after contract is recorded and every other field must match. The manager starts at global128, uses local windows1–128 and advances native Adam to256. It retains the tested reference correction implementation and state-integrity checks. No A-prefix recording or post-correction probe is created. The first actual window checks completed work, true native Adam counters and integrity; it does not require event-2 loss or weight equality with K2.

The original O2 receipt remains unchanged. A validated event-local view translates its global optimizer counters solely for the inherited schedule builder. Actual training uses the original `PresentationSchedule`: shuffle and anchor draws occur normally and must match the retained records. A mismatch stops; no anchors are forced and no RNG reset is introduced. R uses its own same-pass next-pair anchors, including wraparound. The parent preparation preserves all128 responses, the immutable teacher-feature identities, and capped `train_00187` (2048 response tokens,2133 total tokens); evaluation prompts and IDs are excluded.

The complete main budget is256 K presentations,256 pre-R forwards and256 post-R forwards,128 new Adam steps and at most256 violation backwards/128 corrections. Optional diagnostic forwards are zero. Disposable startup remains separate at two forwards/backwards and zero Adam steps. Training closes before native evaluation.

The two-arm evaluator measures retained K2 and final KRC2 on both unchanged64-request panels, three balanced repeats each. Only the first8 development requests form the included48-generation gate. Both panels enforce exact canonical tokens/termination and repeat-consistent counts. Only development enforces retained K2 counts; the additional panel has no historical K2 gate. K1/KRC1 historical counts support count-only learning curves and cannot change validity or fresh timing totals. Raw, invalid, unknown and unattempted work remains visible on failure; harmful valid results continue.

Parent preflight preserves the8GiB reserve and budgets a fresh output's3W state plusW export plusW possible export temporary overlap,10% serialization slack and4GiB auxiliary allowance. Existing KRC1 occupies already-accounted disk; no extra copy is created in the new output. Exact source/artifact checks precede launch and run again afterward. The launcher has no automatic retry or additional qualification optimizer steps.

## Independent tests and source check

**22 focused CPU tests passed in9.554s**, with CUDA uninitialized. The suite includes an actual tiny SpecForge KRC1 serialization followed by exact resumed-versus-uninterrupted comparison of draft weights, Adam, RNG, sampled anchors and ordinary training metrics. It also covers active/empty reference windows, local/global counters, diagnostic-only contract migration, non-forced anchor failure, first-window failure accounting, development-only K2 count gates, both panel budgets and report revalidation.

The reviewer rechecked **181 inherited `.py`, `.yaml` and `.md` files: no changes or missing files**. Scientific changes are confined to the additive continuation integration and its evaluator namespace. Evidence: `CPU_FINAL_INDEPENDENT_REVIEW.log`, the additive source diff and `REVIEW_CHECKLIST.md`.

## Findings resolved before execution

1. The parent preparer imported `validate_plan` from the wrong module. It now imports the actual `evaluation_protocol` implementation.
2. The campaign passed an unsupported `ranking_lambda` argument to the additive trainer. The call now matches its interface.
3. The initial trainer assumed embedded checkpoint events contained a completed `status` field. The inherited serializer stores that event before the external receipt adds status. The trainer now checks the real saved boundary schema; the campaign separately validates the completed receipt/export. The CPU fixture explicitly exercises an actual checkpoint event without `status`.
4. The launcher's test-module name and a stale storage-budget label were corrected.

The reviewer's initial command additionally named a nonexistent `test_protocol` module after the16 real campaign/evaluator tests. That invocation-only error is preserved in `INITIAL_CPU_REVIEW_INVOCATION.log`; the corrected combined22-test run passes. No scientific failure is inferred from that collection error.

Preparation additionally found two historical O/K source paths whose current bytes reflect the already qualified optional guard hooks/reporting. The preparer now validates exact archived O/K copies against the old seal and validates the live execution copies against the later KRC1 seal, recording both identities. Only `online_feedback/objective.py` and `online_feedback/training.py` are eligible for this explicit mapping; every other mismatch still stops. Current scientific source remains unchanged. This preserves historical provenance rather than claiming those old source bytes are still the live files.

The independent retained-data auditor is prepared outside the executed source. Its four CPU fixtures pass, covering one-trajectory count pooling, three-repeat timing means, balanced order, signed regressions, additional-panel gate exclusion, and the distinction between a larger relative advantage and actual within-arm improvement. It will run only after complete raw evidence arrives; no result is inferred before that audit.

## Interpretation boundary

This run tests whether KRC's relative advantage survives a second shared-feedback event and whether KRC itself improves. Those are separate questions. Shared O-generated feedback is not two independent online deployments. Historical and fresh timing differences are not evidence of learning progress. No result authorizes tuning, event3/4 or a mechanism change.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_krc_continuation_20261028/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
