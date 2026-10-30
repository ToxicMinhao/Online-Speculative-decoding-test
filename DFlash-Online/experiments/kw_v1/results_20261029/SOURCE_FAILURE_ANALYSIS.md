# KW-v1 neutral-window failure: bounded source and evidence review

## Material Passport

ARS experiment-agent, read-only failure interpretation, October 29, 2026. Reviewed by the numerical-core author within the same team; this is not an independent implementation review or a GPU replication. No model forward/backward, optimizer application, GPU test, remote operation, protocol change or implementation repair was performed for this review. Stage 1 / project None is unchanged.

Executed source: `9fd9fe6f7db9796a49e2a6879b5a79e7f01fbd22`. The local files inspected below match their SHA-256 entries in the executed seal (`private: raw/sealed/SEAL.json`): `online_feedback/response_weighting.py`, `online_feedback/kw_training.py`, `online_feedback/optimizer.py`, `online_feedback/training.py`, `experiments/kw_v1/qualify.py`, `experiments/kw_v1/resume.py`, and the pinned SpecForge `training/backend.py` and `training/controller.py`.

## Confirmed outcome

The first full-model KW optimizer window failed the exact native-K state comparison. The sole recorded check is `neutral_window_1`, with differing draft-weight and Adam-moment paths. This is an implementation-qualification failure, not evidence for or against learned response weighting. Production remained unstarted.

The saved KW response records (`private: raw/implementation_qualification/KW_RESPONSES.json`) equal the first two K response records (`private: raw/implementation_qualification/K_RESPONSES.json`) as complete decoded JSON objects. That equality includes response IDs `train_00947` and `train_00916`, all reported metrics, all sampled anchors, stepped flags and reported steps. Both paths report boundary gradient norm `0.2099609375`.

The KW window record (`private: raw/implementation_qualification/KW_WINDOWS.jsonl`) establishes:

- Allocation `a=0`, response weights `(0.5, 0.5)`, entry and final `phi` both zero, no pending credit consumed, and RMS/count `v=0`, `n=0`.
- Ordinary and weighted FP64 gradient norms both `0.2095905596013031`; the native pre-clip norm is below the clipping threshold of one.
- Two previews completed at `+0.0625` and `-0.0625`, followed by one real neutral update. These previews produced a nonzero secant, but the controller had not used it.

The work ledger (`private: raw/implementation_qualification/WORK.json`) records six completed response forwards and six backwards: four for K and two for KW. Five disposable optimizer applications completed: two ordinary K updates, two KW previews, and one KW live update. Serving generations and persistent optimizer updates are zero. The failure receipt (`private: raw/IMPLEMENTATION_QUALIFICATION.json`) reports 106.5904 seconds and the process log (`private: raw/QUALIFICATION.log`) ends at the exact-state assertion, not an out-of-memory error.

## Source write-path audit

No direct live-state mutation or storage alias was identified in the inspected neutral path:

1. `HalfGradientCapture` clones the incoming hook contribution. Any repeated contribution is added only into that clone. The hook returns `None`; it does not assign or modify the incoming gradient.
2. `weighted_gradients(..., a=0)` returns the existing native gradient references. `apply_weighted_gradients_` performs its `parameter.grad.copy_` only when `a != 0`. Thus the inspected neutral allocation branch contains no gradient rewrite.
3. `NativeAdamWPreview` constructs new parameters from `live.detach().clone()`, replaces every optimizer-group parameter with its scratch counterpart, recursively clones optimizer options/state tensors, and clones scratch gradients before clipping. Its optimizer step targets this distinct scratch optimizer. Clearing its state dictionary does not clear the live optimizer's dictionary.
4. Preview differences are calculated in FP32 arrays obtained from BF16 scratch parameters; subtraction and division modify those FP32 arrays. They do not target live parameters or gradients.
5. The trainer invokes the original direct-BF16 adapter once for the actual neutral update. The adapter still runs its original global clipping, AdamW step and gradient clearing. The observation wrapper returns the original response `StepOutput` unchanged.

This is a source-level finding, not an observed full-model proof of preview isolation. The separate full-model nonzero/isolation comparisons are later in `qualify.py`; execution stopped before reaching them.

## What the retained evidence cannot establish

The failure receipt retains only the first 100 mismatch paths (`differences[:100]`). Its stored list contains exactly 100 entries: 31 weight paths and 69 Adam-moment paths. The source visits the complete weight mapping before the optimizer mapping, so the 31 differing weight tensors are identified; the 69 moment paths are a lower bound and the full mismatch count is unavailable. Absence of RNG, mode or counter paths from this truncated list does not prove those fields matched at the failed boundary. The per-response records separately establish equality of the reported step counters.

The ordinary pre-step gradient vector was not retained. The KW first-window pre-step state and captured gradients were held in process memory for later optimizer-only checks, without persistence. The ordinary endpoint states also lived only in memory. The retrieved artifacts contain no tensor pair from which to reconstruct the first differing gradient coordinate, its magnitude, or live-state equality immediately before and after previews. Equal losses, statistics and scalar gradient norms do not imply coordinatewise equal gradients.

Consequently, the records do not isolate whether the discrepancy arose during backward/accumulation, through an unobserved runtime interaction, or during the optimizer path. GPU numerical variation is a possible explanation, but no native repeatability comparison or saved before/after gradient evidence establishes it. A particular kernel, compiler, DDP hook ordering or BF16 nondeterminism must not be named as the cause from these records alone. No causative source defect has been confirmed by this bounded review either.

## Disposition

The exact neutral-state requirement remains unmet. Preserve this failure, its work counts and the original source/input bindings; do not reinterpret the CPU fixtures or equal forward metrics as passing GPU qualification. The fixed production comparison has no measured outcome. This review proposes no precision change, gate relaxation, retry or additional probe, and makes no claim about KW efficacy.
