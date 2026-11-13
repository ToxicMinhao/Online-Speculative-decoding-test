## Material Passport

- Type: bounded experimental result and evidence audit.
- Verification: ANALYZED; raw native serving records rechecked when available.
- Execution commit: `562f63fdff6955d5ffcf6086bf9b2f8f948dba8b`.
- Run status: `failed`; decision: `incomplete_or_invalid_no_gain_claim`.
- Private prompt text, canonical/rival token IDs and tensors are excluded from this package.

## Question and fixed design

Does a fixed 0.10 parent-margin floor on recent repairs improve one K+6→K+7 continuation? NATIVE continues ordinary K+; RECENT protects prospective K5→K6 repair slots; CONTROL moves matched assignments to established slots and retains unmatched assignments. All start from the same complete K+6 state and use the same 128 feedback responses, two-pass order and anchors. No hyperparameter was tuned.
Sealed panel: 256 requests; shortfall from 256: 0. Recorded prior-exposure tiers: {'1': 256}. This is one seed and a shared-feedback fork, not independent online deployments.

## Completion and validity

Raw serving audit: **unavailable**; audit errors: 0; report reduction errors: 0.
Budgets: at most 384 scientific Adam steps / 768 presentations, three disposable qualification Adam steps / six presentations, and at most four first-firing diagnostic Adam steps / eight presentations. Timed serving budget: 1536 completions; fixed-start parent walks: 64. Startup and canonical preparation are separate.

**No endpoint gain claim is made.** The run is incomplete, invalid, or stopped at the pretraining feasibility gate. Completed measurements and failure accounting remain in the tables; unattempted work is not imputed. Correctness/resource failure requires resolution before interpretation.

## Eligibility, placement and actual treatment

Eligible distinct recent slots: 815; occurrences: 1629; matched slots: 114; matched requests: 70. Switched weight fraction: 0.1646; absolute margin SMD: 0.0125; placement screen: True.
Frozen assigned/native selected weight: 0.00335772. Raw weight is exposure, not gradient influence. Final parameter differences and first-natural-firing ACTIVE/OFF gradient, moment and BF16 displacement receipts are separate evidence.

| Arm | Presentations | Adam steps | Firing presentations / windows | Final coordinates differing from NATIVE |
|---|---:|---:|---:|---:|
| NATIVE | 256 | 128 | 0 / 0 | not measured |
| RECENT | 256 | 128 | 160 / 107 | 296587104 |
| CONTROL | 256 | 128 | 164 / 107 | 297981102 |

| Arm / pass | Assigned / reachable / firing occurrences | Firing wrong / still correct | Assigned / firing raw weight | Auxiliary loss sum |
|---|---:|---:|---:|---:|
| CONTROL / 0 | 814 / 762 / 215 | 163 / 52 | 483.6779 / 135.8533 | 0.009637 |
| CONTROL / 1 | 815 / 781 / 107 | 80 / 27 | 483.8580 / 63.1364 | 0.003081 |
| NATIVE / 0 | 0 / 0 / 0 | 0 / 0 | 0.0000 / 0.0000 | 0.000000 |
| NATIVE / 1 | 0 / 0 / 0 | 0 / 0 | 0.0000 / 0.0000 | 0.000000 |
| RECENT / 0 | 814 / 758 / 233 | 171 / 62 | 483.6779 / 150.4025 | 0.010286 |
| RECENT / 1 | 815 / 780 / 108 | 78 / 30 | 483.8580 / 65.9861 | 0.002932 |

Activation/exposure sums use each presentation's original response observation exactly once; accumulation-boundary metrics are not counted a second time. `ordinary_selected_kl` is kplus_N_K/(kplus_S+1e-6); `retention_native_kl_loss` is the redistributed K+ objective, not that ordinary selected KL. `retention_auxiliary_loss` remains separately reported.

Native numerical qualification: NATIVE=passed; DISABLED=passed; ZERO=passed; ACTIVE=passed.

## Costs and interpretation

| Arm | Main training s | Checkpoint persistence s | Export s | Diagnostic state capture s | First-firing replay s |
|---|---:|---:|---:|---:|---:|
| NATIVE | 333.39 | 18.60 | 14.50 | 0.00 | unavailable |
| RECENT | 339.37 | 19.30 | 14.51 | 55.72 | 107.53 |
| CONTROL | 338.98 | 20.82 | 15.71 | 54.99 | 104.77 |

costs.csv separates frozen preparation, disposable qualification, startup, main training, first-firing replays, diagnostic state capture, persistence/export, fixed-start work, canonical preparation, loading, activation/cache maintenance and timed serving. Parent/child timers overlap and must not be added as independent costs; compiler timers may also nest. Training includes the realized objective workload, while diagnostic replays and snapshot capture are recorded separately. The bounded first-firing preclip-gradient CPU copy remains included in training_s; other reversible state captures are timed separately. These are measured process costs, not exact intrinsic-kernel costs. No kernel-time breakdown is inferred.
Recorded campaign wall time: 2325.63 s. This does not establish training-inclusive speedup.

Recorded allocated phase windows: 0.6243 card-hours on GPU-9986735a-3e68-d1bd-5998-9634ef24341e; peak allocated / reserved bytes: 20514508800 / 25811746816. Serialized child-process windows plus canonical/serving driver windows; includes CPU loading/cleanup, excludes gaps; not kernel-busy time.

**Decision: incomplete_or_invalid_no_gain_claim.** Advance requires the prespecified native accepted-length gain and interval, an identifiable placement advantage, fewer exact held-out repair relapses and more accept-through decisions versus both controls, and no demonstrated slowdown. Lower protected loss alone is insufficient. An unresolved interval is not equivalence. No tuning or additional experiment follows automatically; reopening requires a separately authorized study addressing the failed or unresolved link.

Files: SUMMARY.json, per_request.csv, all_repeats.csv, fixed_start_origins.csv, fixed_start_outcomes.csv, eligibility.csv, matching.csv, assigned_slots.csv, activation.csv, gradient_receipts.csv, endpoint_differences.csv, costs.csv and REPORT_SELF_CHECKS.json.
