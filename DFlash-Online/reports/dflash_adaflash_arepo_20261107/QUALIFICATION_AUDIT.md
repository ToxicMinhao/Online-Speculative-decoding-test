# A_repo native qualification evidence audit

## Material Passport

- Mode: ARS experiment-agent validate.
- Verification status: **ANALYZED**. This is a read-only retained-record audit, not an independent GPU reproduction.
- Scientific execution: `7dac02b0fc6686d7129c5d29699ee9bfe5be2e7a`.
- Evidence: A800 `/data/sources/minhao/runs/dflash_adaflash_arepo_20261107/production_001/qualification/`, specifically `QUALIFICATION.json`, all three `BATCH_*_PARITY.json`, six `batch_*/RESULT.json`, and `FIXED_LOSS_CHECKS.json`.
- Authorship overlap: this auditor implemented the loss and part of the CPU coverage. Numerical aggregation was independently recomputed from retained per-parameter records; this is same-team verification.

The bounded native qualification passed. In all three fixed pairs, literal-source and ported losses produced exactly matching recorded gradients and actual BF16 Adam displacements. Adam state, final RNG and response metrics also matched. Six exact initial-state restorations and six one-step branches completed, with no uncompleted branches.

I independently recomputed 54 checks: for each pair and each of gradients/displacements, the eight global fields (`coordinates`, `different`, `left_nonzero`, `right_nonzero`, `left_squared`, `right_squared`, `difference_squared`, `dot`) were checked against sums of the per-parameter records, and zero discrepancy was checked separately. All pass. Floating aggregate summation checks use relative/absolute tolerance 2e-12 solely for JSON re-summing; scientific gradient/displacement equality remains exact, with zero differing coordinates and zero discrepancy norm.

| Fixed pair | Eligible position occurrences | Capped occurrences | Smallest positive native draft probability | Zero probabilities |
|---|---:|---:|---:|---:|
| 1 | 11,400 | 1,478 | 1.1258623e-26 | 0 |
| 2 | 6,630 | 2,588 | 5.4283728e-25 | 0 |
| 3 | 7,125 | 2,226 | 4.4428646e-26 | 0 |
| Total | 25,155 | 6,292 | 1.1258623e-26 | 0 |

These are sampled training-position occurrences, counted once per pair rather than doubled across the two identical implementations. The native cap count differs from the earlier common-arithmetic clipping proxy and must not be substituted with that proxy's count. No zero probabilities were observed in the six native qualification branches; CPU tests still establish the literal `F.kl_div` path can produce nonfinite backward values when native softmax underflows to zero. This qualification does not rule out that boundary later in training.

Work accounting reconciles: six disposable Adam steps; twelve core response forwards/backwards; two startup response forwards/backwards and zero startup Adam steps; six small fixed loss-fixture forwards/backwards. Total recorded model forwards are fourteen, layer forwards including recomputation seventy, and head forwards including recomputation 2,136. Wall time was 242.276 seconds; maximum allocated GPU memory was 23,290,610,176 bytes. No complete-response generation or permanent serving checkpoint belongs to this check.

**Metadata correction for interpretation:** shared helper fields still say `left="paper_entry"` and `right="repo_position"`. In these files, left actually means **literal_source** and right means **ported_loss**, as established by the runner's branch order and branch directory identities. This is an obsolete label, not a comparison with the earlier paper-clipping implementation. No raw file was changed during this audit.

The supported conclusion is implementation equivalence on the prescribed fixed native cases, sufficient for the numerical qualification gate. There is no serving-effectiveness conclusion. The `main_training_authorized` receipt expresses the numerical gate only; full storage admission remains separate. At audit time the GPU was idle and 128,572,325,888 bytes were free, below the separately planned full-campaign requirement. The auditor did not start or authorize main training.

## Interpretation checks

All eleven ARS checks were considered: Simpson's paradox (per-pair equality accompanies aggregate equality); ecological fallacy (no future-request inference from training occurrences); Berkson selection (fixed initial pairs, not outcome-selected); collider bias (no conditioned performance model); base-rate neglect (cap denominators shown); regression to the mean (no improvement estimate); survivorship (all six planned branches present); look-elsewhere (all three pairs reported); forking paths (source commit/tolerances frozen); correlation versus causation (no efficacy claim); reverse causality (paired implementation verification only). No statistical inference or p-value claim is made.
