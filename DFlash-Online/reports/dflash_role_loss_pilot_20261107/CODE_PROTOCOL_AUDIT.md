# Role-loss pilot: code/protocol audit

## Material Passport

- Workflow: ARS-Codex 3.24.2 experiment-agent, validate.
- Status: ANALYZED; same-team source review and CPU fixtures, not independent native reproduction.
- Scope: local read-only audit of the submitted handoff and published scientific implementation. No algorithm edits, model generation, training, remote operations or new experimental arms were performed.
- Execution source: `cdb640543d8fe3a43c8e63af4ccaaa002e09a599`.
- Reviewed local HEAD: `f653b762c9865f570486796324d718bb29dbcdbf`; `online_feedback/`, `experiments/` and `tests/` have no diff from the execution commit.

## Finding

**No new scientific implementation blocker was found in the inspected paths.** The source implements the three prescribed objectives and the fixed six-trajectory, four-event protocol. This conclusion concerns code and CPU fixtures. Completion, native correctness, resource usage and efficacy must be established separately from the actual execution records; they are not inferred here.

## Checked contracts

| Requirement | Source evidence and assessment |
|---|---|
| Preserve ordinary K+ | `experiments/role_loss_pilot/training.py:113` installs the existing `install_kplus` for KPLUS. The new extension is only installed for MIX/ROLE. `online_feedback/objective.py:202` dispatches the new reducer before the unchanged K+ branch; without the new attribute, ordinary execution is unchanged. |
| Distinct roles | `online_feedback/role_loss.py:79` reuses current detached proposals, the native selection and `first_error_groups`; P is selected minus first errors, E is the two error-priority groups. The successor affects priority only. Native validity, EOS and seed handling are inherited rather than independently redefined. |
| Full-response D versus selected Z | `role_loss.py:71` captures full original FP32 weight mass plus one epsilon before projection chunking. `objective.py:443` sums chunk terms; `objective.py:454` adds the selected epsilon once. `role_loss.py:150` uses that supplied Z rather than substituting Z into the cap. |
| Released repository reverse-KL arithmetic | `online_feedback/adaflash_repo.py:69` uses BF16 student softmax, FP32 teacher log-softmax cast to student dtype, vocabulary-summed F.kl_div, in-place division by D and the 0.01 position cap. The new reducer calls these existing row operations only on the nonempty mixture group. It does not claim full A_repo equivalence. |
| Exact MIX/ROLE formulas | `role_loss.py:139` redistributes CE and clipped RKL separately using the native two-priority correction across response-wide sums. `role_loss.py:158` evaluates the prescribed component expressions. ROLE's prefix term needs no error redistribution because the changed coefficients apply only to E. No separate P/E normalizers are introduced. |
| Neutral branches | `role_loss.py:153` returns the native numerator graph for a ROLE response with no errors. Empty mixture groups skip CE/RKL evaluation (`role_loss.py:100`), preventing a nominal zero-weight branch from creating underflow-gradient failures. |
| Own-feedback trajectories and state continuity | `campaign.py:110` builds a new backend/trainer per dataset/arm; `training.py:180` checks genuinely new released weights, empty Adam and original RNG without loading another trained arm. `campaign.py:139` collects each next 128 responses from the current arm, updates the persistent trainer, publishes, then advances. It does not reset between the four events. |
| No more than four events | `training.py:224` rejects a fifth event; `contract.py:14` fixes four events and all work budgets. Event receipt/boundary validation checks counters, actual response observations and immutable selected exports. |
| Nonfinite policy | `training.py:125` rejects a nonfinite loss before backward; `training.py:155` checks gradients before the native optimizer step. Nonfinite observations are not silently stabilized or excluded. |
| Evaluation follows closed training | `campaign.py:262` waits for all six training subprocesses to finish before creating the evaluation backend. `campaign.py:267` and each 32-request boundary invoke the inherited quiescence check for training, transfers and foreign GPU work. Event-2 saved weights are evaluated after training has closed, not mixed with event-4 state. |
| Timing and checkpoint activation | `evaluation.py:196` applies each sealed block schedule, runs boundary checks and activates/checks exports before request generation. `campaign.py:231` verifies event-2/4 export identities before and after evaluation. Native serving remains the shared backend, with captured training data disabled for evaluation. |
| Raw validity before conclusions | `evaluation.py:210` saves each returned raw record before validation. It checks exact canonical tokens/termination and repeat-consistent native counts, preserves failures, and stops without automatic retry. `evaluation.py:275` reconstructs and rechecks saved values and native sidecars rather than trusting a valid flag alone. |
| Same-context diagnostics do not train | `diagnostics.py:136` uses fixed canonical features/anchors; the inherited `online_feedback/vfd_scoring.py:72` context manager applies eval/no_grad and restores RNG, mode, sampler, buffers and caches. `diagnostics.py:194` checks restored state; campaign boundaries verify the full trainer state afterward. |
| Statistical unit and aggregation | `statistics.py:15` validates full per-request repeats and uses one count trajectory plus within-request mean time. `statistics.py:88` uses the prescribed PCG64 seed, 10,000 paired-request resamples and ratios of sums. The two datasets remain separate. |

## Focused CPU recheck

Executed locally with CUDA disabled, from DFlash-Online:

```text
python -m unittest test_role_loss_objective test_role_loss_evaluation test_role_loss_diagnostics -q
Ran 36 tests in 1.318s — OK
```

These cover a separately written blockwise expression oracle, exact BF16 logit gradients, cap decisions, response-wide priority conservation, no-error/empty neutral branches, EOS/gaps, teacher-argmax targets, checkpoint denominator lifetime, excluded-position gradients, fixed-context state restoration, validity failures and the paired bootstrap. The local environment warns that optional FlashAttention is absent and uses FlexAttention for its fixtures; this is not a test of A800's native FA3 serving kernel. A test-only scalar conversion warning was also emitted; no assertion failed.

## Limits for the result interpretation

1. This audit does not independently rerun GPU qualification or the experiment. Actual work counts, native outputs, final state/export identity and source pins require the run records.
2. Each arm generates its own feedback. Differences are comparisons of complete online recipes; shared prompts and seed do not isolate a loss intervention on identical post-divergence teacher features.
3. One training seed, four events and reused development panels do not establish a 58-event ranking or broader robustness.
4. Fixed-context repairs explain local prediction changes, not a percentage of serving-time savings. Statistical intervals are unadjusted descriptive intervals for the prespecified contrasts.
5. KPLUS does not compute CE/RKL merely for logging; its clipping counters and teacher-CE discrepancy counters are therefore unavailable, as disclosed by the report code. This avoids adding unrequested objective work.
6. The static quiescence checks and output-validity code must still be corroborated by execution receipts. Source inspection is not evidence that every gate actually ran or passed.

No implementation modification or additional experiment is recommended by this source audit alone.
