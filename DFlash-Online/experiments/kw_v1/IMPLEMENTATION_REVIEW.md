# KW-v1 separate numerical-core and trainer review

## Material Passport

- Date: 2026-10-29.
- Workflow: ARS-Codex 3.24.0 experiment-agent, implementation validation.
- Reviewer role: campaign/selection/resume plumbing author; separate from the numerical-core and KW-trainer authors. This is a read-only review of their code, not an independent review of my own campaign implementation.
- Verification status: ANALYZED source; independently rerun CPU fixtures passed. Full-model A800 numerical/resource qualification and production outcomes remain pending.
- Scope: the fixed HANDOFF.txt contract, numerical core, ordinary TrainerCore integration, checkpoint/controller restoration and bounded qualification flow. No change to Stage1/projectNone or experiment scope.

## Finding

No unresolved blocking numerical or trainer-flow discrepancy was found in the reviewed source. The implementation remains conditional on the mandatory bounded full-model qualification; CPU fixtures do not establish A800 memory fit or full-model equality.

The five feature definitions use detached per-response StepOutput values before TrainerCore merges metrics. FP32 features, CPU FP64 controller/scalars, the no-error location16, excluded empty blocks, loss-only feature clamp, response-swap sign and finite checks agree with the handoff. Native hook captures return None and preserve BF16 accumulation. At zero allocation the native gradient object is retained; nonzero weighting uses separate eager FP32 subtract/multiply/add and one BF16 cast. Native None-gradient membership is preserved.

The preview executes torch.optim.AdamW on separate BF16 parameters and cloned BF16 moments/counters with the native groups/options and clipping. Both proposals reset to the same current pre-step state. The secant converts each proposal to FP32 before subtraction and uses the clipped actual allocation difference. Query dot products use bounded FP64 chunks and CPU FP64 accumulation in named-parameter order. Preview isolation and actual optimizer equality remain part of the full-model gate.

Window allocation freezes from entry phi before credit updates it. Current unweighted/unclipped native G supplies preceding-window credit. Producing derivatives are cached, controller updates use descent and prescribed RMS normalization, and zero increments leave phi bitwise unchanged. The first two windows remain neutral. Eligibility clears at windows64/128, while controller/RMS state persists. Complete-event snapshots require no pending eligibility, exact KW contract and event/controller counts; the sole K8 migration verifies native scientific equality before adding metadata.

## Issues resolved during this review

1. Production preview/credit work originally appeared only in completed KW_WINDOWS rows. A failure after one preview could therefore omit consumed scratch work. The trainer now writes KW_WORK start/completion records before dispatch; campaign reconciliation charges attempted previews/credit, reports completed lower bounds and unknown work, and accounts for a logged completed live call even if the higher-level counter had not yet advanced. A dedicated failed-preview fixture passes.
2. A complete KW trainer checkpoint roundtrip was initially unverified in the files first inspected. The latest CPU integration test now runs a complete128-window/two-pass event, checks126 credits/256 previews and both boundary clears, restores native weights/Adam/RNG/counters plus exact controller/migration state, and checks that the resumed first window has no stale credit. I inspected and independently reran this test successfully.
3. Qualification admission now binds its nested work counters and top-level validity flags to original K8 SHA256, the canonical sorted complete QUALIFICATION_INPUTS object hash, and the sealed execution commit. A stale passed receipt cannot authorize changed inputs or execution pin.

## Independently executed checks

- `test_response_weighting`:16 CPU tests passed, including features/EOS/empty blocks, raw hook capture, zero neutral storage, response swap, delayed timing, zero credit, controller checkpoint/projection, native preview parameters/moments/counters and RNG isolation, clipped secants, None-gradient semantics, FP64 reduction and nonfinite rejection.
- `test_kw_training`:10 CPU tests passed. These include two real tiny DFlash/TrainerCore windows exactly matching ordinary K, complete-event controller/native state roundtrip, the actual qualification harness consuming8 response forwards/backwards and14 disposable optimizer applications, durable budget admission, and preserved original errors across cleanup/receipt failures.
- Campaign author's checks:8 campaign/selection/budget CPU tests plus2 tiny ordinary O/K continuation tests passed. The latter compare uninterrupted execution with own complete event8-to12 restores. They do not constitute a separate review of this reviewer's own plumbing.
- Actual local question-only prepared plan and4032-slot evaluation schedule validate. The computed full-capacity threshold rejects one byte below215,574,560,848 and admits the exact threshold when host memory is sufficient. This is a budgeting check, not remote capacity evidence.

All reported numerical reruns used `CUDA_VISIBLE_DEVICES=''`. An earlier legacy continuation-fixture invocation omitted that setting and its teardown detected local CUDA initialization through inherited RNG capture; it was preserved in the task transcript, then corrected and rerun with CUDA hidden. This review performed no A800 operation and makes no full-model GPU qualification claim. CPU tiny-model work is separate from the handoff's bounded full-model budget.

## Remaining required evidence

The original remote O8/K8 states/features and pinned dependencies still need the remote read-only preflight; full gradient/eligibility/scratch memory and exact two-window full-model neutral agreement need the bounded qualification. No efficacy, native serving result, outcome label or final scientific conclusion follows from this source/CPU review. Production must retain fixed budgets and report negative/mixed outcomes without tuning or retry.

## Source identities at review completion

The source hashes below bind this review to the inspected versions. A later source change requires a scoped follow-up review; final execution commit is pinned separately before qualification.

- `online_feedback/response_weighting.py`: `863db4dbee0848107fefcf8b0057dabb75d39387b9a9a1768d13e642b2f679b8`
- `online_feedback/kw_training.py`: `a96f2e2979eb6b7d4844dfc5fadf5c4cb812425e53528b281b871997f389019d`
- `online_feedback/optimizer.py`: `fd4032d438f72214fd124a34e678aeef45c83457bafb0453fec364d4b211c2d5`
- `online_feedback/training.py`: `315db6b9a494c3cebf1b8e3f23ff9193ffac15ad546399ced7644ffc20097c2b`
- `experiments/kw_v1/qualify.py`: `4e99c24db88a901b4edda14575db2690a19023ff95918a6bb89fdc2782730b91`
- `experiments/kw_v1/campaign.py`: `296eba0ab2650a759aae3a6d1b58ae051d25139e759e7cafef3f3180e0df3793`
- `tests/test_response_weighting.py`: `1893db3718c63104337519b4f54d09d7196aea599742f8dfe9ae5df0409a7e41`
- `tests/test_kw_training.py`: `2434bd314fb932a0d2c9fc18e20fbf4228c943ca1b4699e28959b735dd311d1c`
