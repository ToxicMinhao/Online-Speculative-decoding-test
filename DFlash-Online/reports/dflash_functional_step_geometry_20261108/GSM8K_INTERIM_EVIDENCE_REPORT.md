# GSM8K interim evidence: FSG R1 events 5–6

**FSG6 regressed on the completed GSM8K panel.** Against KPLUS6 it used **6.11% more verification calls** (paired 95% interval **4.38–8.53%**) and **6.26% more complete-request time** (**4.44–8.84%**). Against SCALAR6, the increases were **6.49% calls** and **6.48% time**, also with intervals above zero. The small SCALAR/KPLUS difference remains unresolved; this does not establish equivalence.

**The mechanism was active:** FSG committed a nontrivial correction in **256/256** production windows. In the eight fixed same-state diagnostics, FSG usually sacrificed useful repair/progress while reducing old-draft KL. Its realized update norms were almost unchanged from native. These observations support harmful local redirection and suppressed useful learning, rather than an explanation based on inactive corrections or broad step-size shrinkage. They do **not** identify a unique cause or quantify how much endpoint regression each mechanism explains.

This is an additive **GSM8K-only interim analysis**. MathQA continues unchanged; its evidence and the overall two-dataset conclusion remain pending. No additional model work was performed.

## Scope, provenance and validity

Execution commit: [`6440e8ff5af1a845f830d65c884f6711d7e4a687`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/6440e8ff5af1a845f830d65c884f6711d7e4a687). Run: A800 `dflash_functional_step_geometry_20261108/pilot_r1_001`; binding `8ec5fed0d4ed9b40dab7c26d5089cd42b01f357ab0379193da42baa93f0b3dc5`. The [input/source manifest](gsm8k_interim/INPUT_SOURCE_MANIFEST.json) records exact checkpoint, source, model/runtime, panel and input identities. Publication is additive in a separate analysis checkout; the live execution source and raw evidence are unchanged.

All **256 requests × three methods × two sweeps = 1,536 executions** passed the retained native validator. Request identities, canonical token IDs and termination, checkpoint identities, serving settings, raw trace sums, cap handling and repeat counts agree. There are **zero invalid or missing GSM8K executions**. Qualification/startup records are not endpoint samples. The stable snapshot contains 6,341 files / 314,544,756 bytes, authenticated before and after retrieval; no model/state tensors were loaded for this review. [Validity](gsm8k_interim/endpoint/VALIDITY.json).

## Endpoint result

Times below sum the two-repeat mean **within each request**. Counts use one validated deterministic trajectory per request, not both repeats.

| Method | Total request time, s | Mean/request, s | Verification calls | Matched accepted/call | Committed progress/call | Output tokens | Capped requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| KPLUS6 | 205.469 | 0.802612 | 12,018 | 5.6962 | 6.6764 | 80,493 | 1 |
| FSG6 | 218.321 | 0.852817 | 12,752 | 5.3107 | 6.2921 | 80,493 | 1 |
| SCALAR6 | 205.037 | 0.800928 | 11,975 | 5.7201 | 6.7004 | 80,493 | 1 |

Committed progress totals **80,237** per method, excluding each request's initial seed token. Matched accepted proposals and the runtime's `progress−1` accepted counter differ at stopping boundaries; both remain separately recorded. Matched totals are 68,457 / 67,722 / 68,498; native accepted totals are 68,219 / 67,485 / 68,262. Identical output tokens mean this is a drafting-efficiency regression, not degraded target answer quality.

| Contrast | Metric | Reduction %, 95% interval | Increase %, 95% interval | Request wins / regressions / ties |
|---|---|---|---|---|
| FSG / KPLUS | Calls | −6.11 [−8.53, −4.38] | +6.11 [4.38, 8.53] | 51 / 175 / 30 |
| FSG / KPLUS | Time | −6.26 [−8.84, −4.44] | +6.26 [4.44, 8.84] | 65 / 191 / 0 |
| FSG / SCALAR | Calls | −6.49 [−8.92, −4.78] | +6.49 [4.78, 8.92] | 39 / 187 / 30 |
| FSG / SCALAR | Time | −6.48 [−9.09, −4.68] | +6.48 [4.68, 9.09] | 53 / 203 / 0 |
| SCALAR / KPLUS | Calls | +0.36 [−0.08, 0.81] | −0.36 [−0.81, 0.08] | 105 / 88 / 63 |
| SCALAR / KPLUS | Time | +0.21 [−0.28, 0.69] | −0.21 [−0.69, 0.28] | 136 / 120 / 0 |

Reduction is `100*(1−candidate/reference)`; increase has the opposite sign and reversed interval endpoints. Unchanged `experiments/functional_step_geometry/statistics.py` and a separately coded CPU bootstrap reproduce the retained summary exactly: 10,000 paired-request resamples, PCG64 seed 20261108, linear percentiles, identical sampled indices across contrasts. The preliminary +6.26% time / +6.11% calls were correctly rounded. These unadjusted intervals concern this reused panel and continuing training seed, not seed robustness or population-wide generalization.

Complete-request time includes native generation, recorded cache maintenance and the existing request wrapper; checkpoint loading/activation is outside it. Native-generation sums are **201.635 / 213.919 / 201.007 s** and cache-maintenance sums **0.1214 / 0.1202 / 0.1200 s**. Those components are already included and do not exhaustively explain the wrapper's latency. No kernel-time attribution is inferred. Sweep totals, KPLUS/FSG/SCALAR: **205.606 / 218.184 / 205.078 s** and **205.331 / 218.458 / 204.997 s**. The FSG regression occurs in both. [All requests, repeats, components and comparisons](gsm8k_interim/README.md#endpoint).

## What the updater did

All windows had nonzero native displacement, protected support, a nonempty sketch and an actual candidate replay. No candidate KL was missing. Every filtered trace matched its event receipt and an independent recalculation of its norm/KL gate.

| Arm/event | Windows | Genuine corrections | Native fallbacks | Numerical ambiguity | KL-increase rejection | Feasibility searches |
|---|---:|---:|---:|---:|---:|---:|
| FSG5 | 128 | 128 | 0 | 0 | 0 | 126 |
| FSG6 | 128 | 128 | 0 | 0 | 0 | 128 |
| FSG combined | 256 | 256 | 0 | 0 | 0 | 254 |
| SCALAR5 | 128 | 23 | 105 | 90 | 15 | 125 |
| SCALAR6 | 128 | 11 | 117 | 98 | 19 | 127 |
| SCALAR combined | 256 | 34 | 222 | 188 | 34 | 252 |

“Genuine” requires a committed endpoint distinct from **both** pre-step and native endpoints. FSG's median norm/native ratios were **0.99917335 continuous**, **1.00006046 raw BF16**, and **0.99999999958 feasible/committed**. Median committed direction cosine was **0.99988124**; median candidate/native protected-KL ratio **0.711684**. Search-only median scale was **0.993635**. A smaller scale is not an equally large reduction of the rounded update norm.

SCALAR exercised its own norm matching in all 256 windows (median relative mismatch **5.19×10⁻¹³**, maximum **3.70×10⁻⁷**) but usually restored native. It neither borrows FSG's evolving state nor establishes the behavior of a substantially smaller-step training recipe. Lower protected KL is partly enforced by acceptance and is not evidence of successful preservation. [Per-event/combined distributions, all 512 windows and eligibility denominators](gsm8k_interim/production/PRODUCTION_REVIEW.md).

## Eight fixed same-state diagnostics

All events 5/6 × windows 1/32/65/128 are present and valid. **A** is the fitting pair; **D** is other buffered training data, not held-out evaluation. The complete [160-row table](gsm8k_interim/diagnostics/window_pair_candidates.csv) retains pre-state, native, raw FSG, feasible FSG, committed FSG, scalar, raw/feasible uniform and raw/feasible expanded candidates. It includes repair identities, KL numerators/denominators, norms and guard outcomes; [320 response rows](gsm8k_interim/diagnostics/response_candidates.csv) preserve further detail.

The table below sums four sampled windows per event/role. Entries are overlapping block **occurrences**, not independent tokens or test requests. Each cell shows native → committed FSG; all eight committed endpoints equal their feasible candidates.

| Event/role | Blocks / original first errors / old-prefix occurrences | Useful repairs | Old-prefix occurrences lost | New earlier errors | Net prefix gain |
|---|---|---|---|---|---|
| 5 / A | 2,329 / 2,013 / 15,496 | 415 → 381 | 64 → 58 | 16 → 16 | +1,154 → +1,016 |
| 5 / D | 1,854 / 1,359 / 15,158 | 73 → 39 | 147 → 166 | 45 → 50 | +44 → −59 |
| 6 / A | 1,945 / 1,475 / 15,661 | 357 → 355 | 22 → 24 | 9 → 11 | +1,194 → +1,174 |
| 6 / D | 1,873 / 1,406 / 15,004 | 44 → 40 | 164 → 162 | 54 → 56 | −10 → −33 |

The [full diagnostic review](gsm8k_interim/diagnostics/FINDINGS.md) addresses all six hypotheses:

- **A — Over-preservation:** lower A drift in 8/8 windows accompanies worse A net progress in 7/8. Useful-prefix preservation is inconsistent; fewer repairs is the clearer pattern. Lower KL alone does not establish useful preservation.
- **B — Reference fitting:** A improves over its pre-state, but FSG underperforms the same-state native update on A in 7/8 and D in 6/8. This is not a clean “A helped, D harmed” explanation for the added mechanism.
- **C — Directional harm:** the same-state scalar matches FSG's norm within at most 2.43×10⁻¹² relative error and retains all native repair identities. FSG has worse net progress than that scalar in 7/8 A and 6/8 D pairs. This supports harmful redirection locally. These are **pre-guard scalar counterfactual scores**, not the trained SCALAR trajectory; only two local scalar candidates pass their guard, six reject.
- **D — BF16 realization:** feasible-minus-raw net changes are +26/+21 on A but −18/−21 on D for events 5/6. Realization matters, with mixed effects. Raw FSG already trails native in aggregate A and D, so feasibility scaling alone does not explain the local deficit.
- **E — Protection weighting:** uniform weights improve D net progress in four windows and worsen it in four. Aggregate D becomes −42/−25, better than FSG's −59/−33 but still worse than native's +44/−10; A also worsens. This is mixed evidence, not a selected alternative.
- **F — Sketch expressiveness:** expanded-sketch D progress improves in five windows and worsens in three; aggregate −36/−25 still trails native, and useful repairs do not consistently increase. Changed norms/directions prevent attributing improvement specifically to sketch rank.

No significance is manufactured from eight windows. Their repair frequencies cannot be converted into a percentage of total time saved or lost. A retained free-text description mislabeled the `raw_fsg` candidate as an expanded direction; source construction, candidate key and measurements identify it as raw FSG. The original text is preserved and this annotation does not alter any measurement. [Cross-review](gsm8k_interim/reviews/CROSS_REVIEW.md).

## Origin counts and costs

The retained KPLUS4 records pass exact panel, tokens/termination, runtime and checkpoint-origin checks. Its **12,116 calls** compare with **12,018 KPLUS6**, **12,752 FSG6**, and **11,975 SCALAR6**. Thus FSG uses **5.25% more calls than its common origin**: actual deterioration on this panel, not merely smaller improvement. No historical latency is used. [Origin checks and per-request counts](gsm8k_interim/origin/ORIGIN_VALIDATION.json).

| Arm | Events 5–6 training excluding diagnostics, s | Relative to KPLUS | Separate diagnostic observer, s |
|---|---:|---:|---:|
| KPLUS | 635.52 | 1.00× | 0 |
| FSG | 9,933.52 | 15.63× | 1,182.47 |
| SCALAR | 13,661.10 | 21.50× | 0 |

FSG's 5,429.36 s parameter probes are **inside** its 6,699.57 s feasibility envelope, which is inside training—not additive costs. Loading, startup, persistence/export, worker envelopes and observed memory peaks remain separate in the [cost tables](gsm8k_interim/production/TRAINING_COST_SUMMARY.json). These are implementation costs, not fundamental lower bounds. Endpoint evaluation separately recorded 2,295.52 s activation and 8.85 s weight validation within a 3,828.94 s evaluation envelope. None is added to request latency or presented as a kernel decomposition.

## Analysis work and disposition

ARS experiment-agent status: **ANALYZED**. Separate CPU arithmetic and same-team source/numerical reviews reproduced the endpoint, all 512 production decisions and 4,320 diagnostic count/KL cells. Implementation-author overlap is disclosed; this is not an external GPU replication. All 11 statistical-fallacy categories were checked. [Review](gsm8k_interim/reviews/NUMERICAL_REVIEW.md), [analysis work and reproduction](gsm8k_interim/README.md).

Snapshot packaging took 3.96 s, local identity verification 2.21 s, primary endpoint arithmetic 2.93 s and independent endpoint/origin arithmetic 2.42 s. Production and diagnostic CPU passes, portability checks and test costs are retained separately. Peak primary endpoint host RSS was about 788 MiB; reading, review and writing time is not represented as instrumented computation. **Additional GPU/model work: zero**—no loads, feedback, teacher preparation, optimizer steps, forwards/backwards, generations or new diagnostic windows.

The supported interim conclusion is a **GSM8K endpoint regression despite active, almost norm-preserving corrections**, with local evidence of suppressed repair and harmful redirection. Weighting, sketch capacity, rounding, trajectory accumulation and their interactions are not causally resolved by these records. No alternative, tuning or further experiment is selected. MathQA and the complete two-dataset review remain pending.
