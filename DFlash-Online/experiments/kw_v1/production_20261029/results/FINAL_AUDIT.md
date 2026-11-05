## Material Passport

- Origin Skill: ARS experiment-agent; Mode: validate/statistical interpretation.
- Origin Date: 2026-10-30; Verification Status: **ANALYZED**.
- Version Label: kw_production_final_audit_v1.
- Source: production_001 retained raw records, sealed plan and automatic report; scientific execution `708e974e06f5d9f4b186c32607c16b37cbd4b14f`.
- Review provenance: separate same-team `kw_eval` agent, local read-only analysis. This reviewer previously authored the evaluator/report; this is independent numerical reconstruction, not a blind external review. Native protocol helpers were reused for structural validation. No remote operation, model rerun, scientific-code edit or bulk checkpoint reload occurred.

## Finding

**No blocking numerical, output-validity or completed-work discrepancy found. The prespecified result is Mixed/inconclusive.** KW12 uses 2,780 versus K12's 2,786 calls on new final64: six fewer calls, **0.21536%**, with paired bootstrap percentile 95% interval **[−0.68967%, +1.05266%]**. Request time is 0.23019% lower. The final four-panel pool instead uses eight more calls for KW, **−0.06997%** reduction, and 0.07661% more request time. These observations do not establish an incremental serving improvement over K.

The four-event measured training total is **1,802.709259 seconds for KW versus 1,277.774415 seconds for K**: **524.934844 seconds (8.749 minutes), or 41.08197%, extra**. This cost is substantial relative to the small, inconsistent serving differences. It is a measured implementation cost on one trajectory, not a universal complexity estimate or deployed asynchronous-speedup result.

## Raw and work checks

The local audit rehashed **14,495 retained compact files, 698,386,662 bytes**, against the retrieved identity manifest. All campaign/report exits are zero. Recorded read-only postflight passed all 2,670 sealed files and eight original artifacts, with zero model passes/optimizer steps/generations. The retrieved plan and successful qualification are unchanged from their reviewed local copies; qualification source/input/K8 bindings pass. Original failed qualification remains separate.

All **4,032 evaluations and 183,864 native verification blocks** pass structural, exact canonical output/termination, count/trajectory, checkpoint, publication, ordering and repeated-count checks. I independently reconstructed emitted tokens and count totals from every trace, and recomputed panel reductions and timings from raw measurement records. All seven planned panels contain 64 complete valid common requests. The included event-10 eight-request gate uses its planned 72 executions. Final canonicals are first O12 outputs; retained F/K1/O8 outputs remain fixed.

All 12 native training receipts show 128 steps/256 presentations each; all 3,072 presentation identities and sampled anchor draws match across O/K/KW within their shared event pools. The four immutable pools contain 512 chronological feedback requests from the preceding O checkpoint. Their existing shared preparation includes **512 terminal teacher forwards**, separately recorded; these are neither full-response generations nor extra controller passes.

Completed production work is exactly **512 feedback generations, 1,536 live Adam steps, 3,072 presentations, 4,032 evaluations, 1,024 previews and 504 credits**. All 512 KW window rows match training identities. Raw work logs independently reconcile 1,024 previews, 512 KW live steps and 504 credit operations, with every attempt completed and zero unknown work. There is no extra generation, production optimizer step, controller model pass or automatic retry. Every arm ends at cumulative step 1,536/presentation 3,072. These production counts do not erase earlier separately retained qualification work.

## Endpoint results

Positive reductions favor KW. Each row has 64 requests; timing uses the three-repeat mean per request.

| Event | Panel | O calls | K calls | KW calls | KW/K call reduction | KW/K request-time reduction |
|---:|---|---:|---:|---:|---:|---:|
| 10 | Development | 2,884 | 2,871 | 2,856 | +0.52247% | +0.94352% |
| 10 | Additional | 3,080 | 3,019 | 3,030 | −0.36436% | −0.36066% |
| 10 | PR18-fresh | 2,980 | 2,921 | 2,927 | −0.20541% | −0.18774% |
| 12 | Development | 2,925 | 2,831 | 2,822 | +0.31791% | +0.38258% |
| 12 | Additional | 3,091 | 2,958 | 2,971 | −0.43949% | −0.56071% |
| 12 | PR18-fresh | 2,979 | 2,859 | 2,869 | −0.34977% | −0.33364% |
| 12 | New final64, primary | 2,849 | 2,786 | 2,780 | +0.21536% | +0.23019% |
| 12 | Four-panel pool, 256 requests | 11,844 | 11,434 | 11,442 | −0.06997% | −0.07661% |

On primary final64, KW improves/regresses/ties **25/20/19** requests: gross 40 calls saved and 34 added. Pooled final counts are **93/98/65**, gross 160 saved and 168 added. K and KW both beat O in final pooled calls, by **3.46167% and 3.39412%**, respectively; that common gain does not identify an added KW benefit.

Independent NumPy PCG64 seed-20261030 reconstruction exactly reproduces the 10,000 paired-request ratio-of-sums resamples and linear percentile endpoints. Calls use one repeat-0 trajectory after equality checks; the 192 timed executions per arm/panel are **64 request observations**, not 192 independent samples. The interval is conditional on this trained trajectory and selected panel. No training-seed replication, multiplicity-adjusted secondary claim or general-population guarantee follows. “New final” means presealed for this study; the disclosed 61/64 older unrelated-project exposures remain a limit.

## Training cost scope

| Event | K training seconds | KW training seconds | Extra |
|---:|---:|---:|---:|
| 9 | 326.158238 | 457.609627 | +40.30295% |
| 10 | 327.501686 | 458.418439 | +39.97438% |
| 11 | 313.070903 | 443.956815 | +41.80712% |
| 12 | 311.043588 | 442.724378 | +42.33516% |

All training totals reproduce the sum of raw presentation timers and the native/campaign receipts. KW adds approximately **1.0253 seconds per live update** in this run. Recorded KW components sum to 295.506516 seconds for previews, 103.797570 for credit/controller work, 96.713562 for weighting, and 34.046991 for the live update and its checks. They describe the implementation's timed regions; they do not constitute an isolated causal cost experiment.

The ordinary-window observation (1,265.873100 seconds) includes intervening loading, metrics and capture work and is not directly interchangeable with K's summed presentation timer. Capture host enqueue/clone time (1.019443 seconds) is nested; total window-observed time (1,804.372538 seconds) overlaps all components. **Do not add these rows together or add components to training_s.** Loading/resume validation is separately 147.037764/176.908459 seconds for K/KW; event wall time, which already contains training/save/export, is 1,410.207196/1,937.878644 seconds. The post-load peak allocation across training, persistence and boundary validation is 27.2419/35.9273 GiB. The raw receipt’s “training-only” label is imprecise: campaign.py reads the peak after the saved-state load and boundary verification. Timing repeats assess serving measurement variation, not repeated training-cost experiments.

## ARS fallacy scan: 11/11 checked

| Pattern | Assessment |
|---|---|
| Simpson's paradox | Panel directions differ; the pooled loss is shown beside every panel, not used to hide them. |
| Ecological fallacy | Inference remains at paired requests within one trained trajectory; no per-user or seed claim. |
| Berkson/selection bias | Fixed selected requests and prior exposure limit population generalization. |
| Collider bias | No observed-performance filtering or covariate adjustment; all planned requests retained. |
| Base-rate neglect | Exact call/token denominators and gross saved/added totals retained; diagnostic prevalence inference is inapplicable. |
| Regression to the mean | No outcome-extreme panel selection; checkpoint trends alone are not controller evidence. |
| Survivorship | All planned work completes; zero invalid/missing evaluations; earlier qualification failure remains visible. |
| Look-elsewhere effect | One fixed primary; all seven endpoint panels and O comparisons visible; secondary differences descriptive. |
| Forking paths | Endpoints, panel selection, order, statistic, seed and CI fixed before outcomes; no tuning or reruns. |
| Correlation/causation | Arm comparison supports a scoped observed difference, not attribution to any one controller feature or timing component. |
| Reverse causality | Training-window credit precedes later windows, but its association with results is not proof of the serving mechanism. |

Overall statistical confidence: **CAUTION**, owing to one trajectory, panel-selection scope, a primary interval crossing zero and discordant final panels. Reproducibility method: local arithmetic/receipt reconstruction only; independent model replication **not run**. `FINAL_AUDIT.json` and `independent_audit.py` preserve the check outputs and method. Stage 1 / project None remains unchanged; this audit authorizes no further experiment.
