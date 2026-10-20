# Implementation validation — 2026-10-20

**Status: code submitted for review; no GPU qualification or model experiment.**
The 48 real requests have not yet been selected. No historical timing or
scientific result is reused as evidence for this screen.

## Local checks

Environment: Python 3.10.18, PyTorch 2.11.0+cu130 (CPU tensors), pytest 8.3.5.

| Check | Outcome |
|---|---:|
| New learning-curve package | 65 passed |
| Existing two-update native wrapper | 11 passed, 4 skipped |
| Existing author loss and follow-up native wrapper | 22 passed |
| Existing per-turn training and framing/dataset contracts | 18 passed, 2 skipped |
| Python compilation and whitespace checks | Passed |

The six skipped inherited checks require unavailable private installed artifacts.
Torch emitted a local CUDA compatibility warning while probing availability;
all numerical fixtures used CPU tensors. These checks do not establish native
A800 integration, memory fit, output preservation or actual model performance.

Commands (run historical native suites separately because they import modules
by the same top-level names):

```bash
python -m pytest -q experiments/dflash_baseline_learning_curve
(cd experiments/dflash_two_update && python -m pytest -q test_native.py)
(cd experiments/dflash_two_update_followup && python -m pytest -q test_author_training.py test_native.py)
python -m pytest -q experiments/dflash_per_turn/test_training.py experiments/dflash_per_turn/test_framing_dataset.py
python -m compileall -q experiments/dflash_baseline_learning_curve
git diff --check
```

## What was checked

- Distinct source/role partitions, fixed selection seed, global prompt and
  conversation deduplication, source hashes, exact wording, full prompt budgets,
  shortfall reporting, and exclusion of dataset answers.
- The mean of two independently normalized response losses, one clipping
  operation and one AdamW step per pair. FP32 and BF16 toy differential checks.
- A continuous 0→2→8 trajectory, persistent Adam/sampler state, four immutable
  frozen-generated feedback responses, fixed diagnostic anchors, and restoration
  of RNG/modes after source-loss measurement.
- A real CPU `FixedFeedbackTrainer` exercised through the production native
  service's scheduling/export-receipt path and controller accounting validator;
  only model loading/export storage and CUDA counters were replaced by fixtures.
- Training process-group shutdown before timing; rejection of surviving trainer
  children; capture-disabled identical-input comparisons with frozen evaluated
  first, middle or last; separate publication/cache/request costs.
- Native trace offset/count reconciliation and the distinction between accepted
  draft tokens, verification-committed tokens and the first prefill token.
- A harmful but correct first group continues through all six groups. Output
  mismatches and invalid measurements remain durable and stop execution.
- Reporting rechecks measurements and training accounting rather than trusting
  success flags. Aggregate ratios use summed paired values; source fitting,
  held-out transfer, extra training and costs remain distinct.
- The qualified sampler's legitimate zero-supervision short-answer samples are
  retained with explicit zero exposure. They neither trigger resampling nor
  remove scheduled optimizer steps.
- Storage admission includes both exports and both optimizer states for all six
  groups, staging and the reserve. No automatic checkpoint deletion is added.

## Remaining execution checks

After review, prepare and seal the actual sample manifest with the pinned local
corpora and tokenizer. CPU preparation additionally requires
`langdetect==1.0.9`; missing packages are not silently replaced. The exact native
model/runtime bindings and source hashes must be available on A800.

The first complete group is the first of six, not an additional pilot. It must
pass actual output equality, masks/exposure, continuous training accounting and
measurement checks. A null or harmful adaptation result is valid and does not
block the remaining five. No GPU execution starts automatically from this PR.
