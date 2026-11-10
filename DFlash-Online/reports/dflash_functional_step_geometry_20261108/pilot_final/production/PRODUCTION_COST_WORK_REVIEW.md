# Final production, numerical and cost audit

## Material Passport

- Workflow: ARS experiment-agent, validate mode; status **ANALYZED**. CPU reproduction of compact-record arithmetic passed; no GPU or model rerun.
- Execution: `6440e8ff5af1a845f830d65c884f6711d7e4a687`; scientific binding `8ec5fed0d4ed9b40dab7c26d5089cd42b01f357ab0379193da42baa93f0b3dc5`.
- Sources: authenticated final MathQA/global compact snapshot and the unchanged authenticated GSM8K interim snapshot. The two small evaluation worker envelopes are separately authenticated in the final snapshot.
- Coverage: all 12 arm-events, 1,024 FSG/SCALAR production windows, 3,072 ordinary presentations, 16 diagnostic work ledgers, four shared feedback collections, and both evaluation envelopes.
- Reviewer overlap: this same-team reviewer implemented parts of the runtime and analysis. This is a source/receipt audit, not an independent native reproduction.

## Completion and state

All 12 arm-events completed their 128 native AdamW steps and 256 ordinary response presentations. All six trajectories start this pilot at cumulative step 512, resume their own event-5 boundary at 640, and end at 768. The recorded live-state, startup restoration, historical-history preservation and final export checks pass with no reseeding. Source `native_pilot.py` passes each arm's own preceding `state_path` into its event-6 constructor. The final controller lists all 18 tasks complete, no incomplete/unknown/unattempted task, exit 0 and a passed postflight.

These checks review retained execution certificates and source, without loading large recovery tensors again. Event-5 bulk states/exports were subsequently retired according to retained retirement receipts; the final event-6 recovery/exports and selected diagnostic states remain. Retired event-5 tensors are not described as archived or presently replayable.

## Production behavior

| Dataset | Arm | Genuine writes, event 5 / event 6 | Combined writes | Native fallback reasons | Feasibility searches |
|---|---|---:|---:|---|---:|
| GSM8K | FSG | 128 / 128 | 256/256 | None | 254/256 |
| GSM8K | SCALAR | 23 / 11 | 34/256 | 188 numerical ambiguity; 34 increased protected KL | 252/256 |
| MathQA | FSG | 128 / 128 | 256/256 | None | 255/256 |
| MathQA | SCALAR | 23 / 17 | 40/256 | 182 numerical ambiguity; 34 increased protected KL | 254/256 |

A genuine write is a final represented endpoint distinct from both its own native AdamW endpoint and its pre-step parameters. Every filtered window has a nonzero native displacement, protected positions, nonempty sketch and actual final-candidate replay. All 1,024 independently recalculated norm/KL/deadband decisions match the recorded decision. Every search uses exactly 24 probes and the retained lower bracket; the realized lower-bracket norm matches the selected endpoint and does not exceed its native norm. No empty or lambda-zero bypass occurs in this campaign.

FSG is active throughout both datasets, so an ineffective/no-op implementation does not explain its endpoint outcome. This does not identify a unique cause of the observed regression.

| FSG quantity | GSM8K | MathQA |
|---|---:|---:|
| Search-only median scale | 0.9936348200 | 0.9935396910 |
| Search-only scale range | 0.9855061770–0.9989002347 | 0.9824869633–0.9986535311 |
| Median continuous raw norm / native norm | 0.9991733475 | 0.9992461156 |
| Median raw BF16 norm / native norm | 1.0000604616 | 1.0000598655 |
| Median committed BF16 norm / native norm | 0.999999999578 | 0.999999999570 |
| Median committed direction cosine to native | 0.9998812442 | 0.9998867610 |
| Median candidate/native protected KL ratio | 0.7116840654 | 0.7113720452 |
| Median changed coordinates where native displacement was zero | 11,934 | 9,959 |
| Median coordinates differing from native endpoint | 1,363,434 | 1,281,790.5 |

The realized FSG update norms remain essentially native in scale. The lower protected KL is partly enforced by acceptance and cannot establish useful learning or held-out benefit. The SCALAR control matches its **own** hypothetical FSG feasible norm; its maximum relative norm mismatch is `3.7048238686e-7` on GSM8K and `1.7731567123e-8` on MathQA. Its frequent native fallbacks do not establish equivalence to KPLUS or to FSG. Cross-arm protection/risk comparisons involve diverged states and protected sets, not paired identical fitting objectives.

All 1,024 windows have recorded native/candidate KL; missing measurements are never replaced by zero. Full per-response `c`, auxiliary-gradient norms, `g^T u / c`, Gram matrices/eigenvalues, sigma, solve residuals, projected energy, cross-response cosine, all endpoint identities/changed-coordinate counts, realization scales, and clipping/exposure observations are retained in `scientific_trace/`. These are auxiliary-gradient products; ordinary training-gradient directional products were not recorded and are not inferred. Recorded preclip norms exceed the fixed 1.0 threshold in 14/256 GSM8K FSG windows and 7/256 MathQA FSG windows; neither SCALAR trajectory has such a window. This association does not establish clipping as a cause.

## Work reconciliation

| Scope | Completed work |
|---|---:|
| Main AdamW steps | 1,536 |
| Main ordinary response forwards / backwards | 3,072 / 3,072 |
| Main auxiliary response forwards / backwards | 6,144 / 2,048 |
| Fixed diagnostics, response forwards / backwards | 640 / 64 |
| Disposable startup forwards / backwards / Adam steps | 24 / 24 / 0 |
| New shared feedback generations / teacher preparations | 508 / 508 |
| Retained qualified feedback packets reused | 4 |
| Final evaluation generations | 3,072 |
| Failed or unknown work in these main-pilot scopes | 0 |

Thus main/diagnostic/startup draft work totals 9,880 response forwards and 5,208 backwards, separate from feedback/teacher/native generation. Every diagnostic used its full 40-forward/4-backward allowance. Recorded projection/head/checkpoint-recomputation/layer and parameter-scan counters appear separately in `work/PHYSICAL_COUNTERS.json`; they overlap the response-level work and are not added as extra responses. Resource profiler `extra_forwards=0` describes profiler overhead only, not total auxiliary work. Earlier qualification and failed debug work are separate historical scopes.

## Training cost and memory

| Dataset | Arm | Intrinsic training, s | Ratio to KPLUS | Diagnostic observer, s | Peak allocated / reserved, GiB |
|---|---|---:|---:|---:|---:|
| GSM8K | KPLUS | 635.517 | 1.000 | 0 | 19.133 / 27.027 |
| GSM8K | FSG | 9,933.515 | 15.631 | 1,182.472 | 32.756 / 37.359 |
| GSM8K | SCALAR | 13,661.100 | 21.496 | 0 | 25.784 / 33.871 |
| MathQA | KPLUS | 1,027.006 | 1.000 | 0 | 19.382 / 23.023 |
| MathQA | FSG | 11,216.977 | 10.922 | 1,353.065 | 33.155 / 36.176 |
| MathQA | SCALAR | 14,851.921 | 14.461 | 0 | 26.069 / 28.785 |

Intrinsic training is the recorded event training envelope minus diagnostic observer wall time; it includes ordinary updates and production geometry. The two-dataset intrinsic totals are KPLUS **1,662.523 s**, FSG **21,150.492 s**, SCALAR **28,513.020 s**. FSG's **2,535.537 s** diagnostic observers remain separate. Peak memory is the measured event peak, which can include diagnostics and publication; no isolated intrinsic-memory peak is available.

FSG's feasibility-realization envelope is 6,699.570 s on GSM8K and 6,827.958 s on MathQA; its probe subcomponent is 5,429.358 s and 5,535.193 s. SCALAR also incurs its own realization work and another 3,805.338 s / 3,801.475 s scalar bisection envelope. These timers are nested in intrinsic training and must not be added again. They establish measured implementation costs, not a detailed GPU kernel-time attribution.

Loading, startup, persistence, export, worker envelopes, compiler counters, feedback preparation and endpoint activation/serving are separately preserved in `work/COST_COMPONENTS.csv` and event cost tables. Shared new-feedback envelopes are GSM8K **291.635 s** and MathQA **569.321 s**, paid once physically per dataset across arms. New teacher preparation is 9.519 s / 18.888 s and feature persistence 11.017 s / 23.275 s, nested in those envelopes. Four reused packets' historical preparation costs are labeled separately, not charged as new work.

Endpoint evaluation envelopes are **3,828.938 s** GSM8K and **4,163.371 s** MathQA, including checkpoint activation totals **2,295.516 s** / **1,992.124 s**. Actual serving comparison uses summed per-request mean latency, not these activation envelopes. Final controller wall time is **64,567.090 s (17.935 h)**; this excludes earlier qualification/debugging and is not a deployment speedup. No adaptation-inclusive benefit can be inferred from protected-KL reduction; endpoint analysis is reported separately.

## CPU audit provenance and limitations

The dataset-aware analyzer reproduces every prior GSM8K scientific output exactly. One JSON object's key ordering changed bytes only. Retained CPU attempts `WORK_AUDIT_001.log`, `TRACE_001.log`, and `TRACE_003.log` contain postprocessing failures: a worker envelope omitted from the original snapshot; an overly narrow expectation that every metrics row was a presentation row; and an inappropriate comparison between separately summed response metrics and FP32 accumulation-window metrics. None was a failed scientific run. The corrected exporter uses each `one_original_response_forward` observation, reproduces TrainerCore's FP32 two-response boundary sum exactly, and reconciles every recorded event aggregate without changing tolerances or data. Both reduction totals and their small rounding differences are preserved in `EXPOSURE_RECONCILIATION.csv`.

ARS fallacy checks: retained negative results; missing is not zero; protected KL is not an efficacy endpoint; no causality from correlations; no repeated-update/repeat pseudoreplication; no unsupported equivalence; no averaging ratios instead of pooling; no nested cost addition; no post-selection of favorable windows; no cross-dataset/seed generalization; and no conflation of analysis correctness with independent native replication. No new GPU/model work, algorithm change, tuning, checkpoint or feedback was produced by this review.

Reusable CPU scripts: `analyze_production.py --dataset {gsm8k,mathqa} --evidence ROOT --output OUT`; `audit_work_costs.py --gsm8k GSM_ROOT --mathqa FINAL_ROOT --output OUT`; `export_scientific_trace.py` takes the same latter arguments. All inputs are command-line paths; no private prompt/response text or model tensor is embedded in code.
