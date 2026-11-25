# Point1: does predicted allocation survive the actual update?

**Completed, November 25, 2026.** [PR61](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/61). Actual execution: `269f05644eeed2f0396602c47beb2a71166f8dd0`. Four separately restored native updates; no retry, new feedback, serving, checkpoint export or full event.

**Answer.** The allocation effect survives on this update window, but its predicted magnitude is poorly calibrated. FRONTIER_P minus native C has score benefit **+0.249655** and actual native-gradient projection **+0.249596**. The finite FP32 shadow remains close. The first large change occurs when its endpoints are cast to BF16: the relevant projection falls to **+0.085117**, while the differential vector grows and changes direction. Actual native assessment benefit remains positive, **+0.080694**, with **21 additional consecutive-prefix occurrences and six additional useful repairs** versus C. However, the eight disjoint TRAIN witnesses have **no discrete advantage**, and their assessment is slightly worse (**−0.002119** benefit). Moving only the assessment cotangent to native beta barely improves calibration. These results identify a local numerical mismatch and limited transfer; they do not identify a unique cause of PR60's unresolved serving result or demonstrate a successful new method.

## Material Passport

- Mode: ARS experiment-agent `run` + `validate`; bounded update-fidelity diagnosis.
- Scope: exact retained PAIR1 parent and PR60 qualification window; primary F−C, secondary F−BETA.
- Verification: native integrity checks passed; independent offline scalar/receipt audit passed. The audit cannot reconstruct unsaved full-vector contractions.
- Evidence: [contract](CONTRACT.json), [identities](IDENTITIES.json), [contrasts](CONTRASTS.csv), [state](STATE_AUDIT.json), [steps](STEP_LEDGER.json), [independent audit](INDEPENDENT_OFFLINE_AUDIT.md).
- Project stage unchanged. Points 2/3 remain provisional; no further experiment is launched.

## Fixed comparison and integrity

All branches restore the same corrected PAIR1 weights, warm Adam moments/options/counters, RNG, histories and empty accumulation at step 128/presentations 256/event 1. Each disposable endpoint reaches step 129/presentations 258/event 1, then the parent is restored. Cases are BETA, native C, FRONTIER_P (F), and exact native C_REPEAT. C is captured from its actual training path. BETA's observed preclip gradient equals scoring g0 exactly; C_REPEAT's gradients, endpoints, Adam state, counters, RNG and scientific rows match C exactly. All initial/final scientific-state checks pass; ancillary runtime restoration is implemented and source-reviewed. No historical failed gate is reclassified.

The five-layer drafter, BF16 model/head/Adam path, attention, forward KL, PAIR, positional gamma 7, K+ eta 0.5, C alpha 0.5/cap 0.25, FRONTIER_P cap 0.25, LR 3e-4, decay 0.01, clipping 1, accumulation 2 and eight-block head chunks stay fixed. The retained two-response window has **512+269=781 sampled block occurrences**. All **762 nonempty prefix pools / 8,979 prefix occurrences** are used. Coefficients, labels, role boundaries, denominators and assessment origins are sealed before native updates. Clipping is inactive at the reference gradient norm **0.152821422**. No serial token oracle is rerun.

The independent CPU helpers test warm clipped/unclipped Adam JVPs against autodiff and adjoint identities, readout boundaries and reductions, export/privacy, restoration and resource/step admission. **41 new focused CPU tests pass on A800**, with **31 inherited mathematical/scoring/integration tests passing on Fenrir**. The native code is unchanged after the execution commit.

## Differential chain

Positive values below predict or measure lower assessment for F. All diagnostic vector products/reductions use FP64; represented operands are promoted to FP32 before subtraction. The linearization for **both contrasts is at BETA**, not at C.

| Quantity | F−C, primary | F−BETA, secondary |
|---|---:|---:|
| Score sum | +0.249654631 | +0.167018852 |
| Actual native preclip-gradient contraction | +0.249596208 | +0.167047970 |
| Linear shadow projection | +0.249596208 | +0.167047970 |
| Finite FP32 shadow projection | +0.249428149 | +0.166664538 |
| BF16 cast-shadow projection | +0.085117243 | +0.055776542 |
| Actual native differential projection | +0.085168166 | +0.055858743 |
| Actual cast-shadow assessment benefit | +0.092064426 | +0.055300389 |
| Actual native assessment benefit | **+0.080693995** | **+0.043241774** |
| Native consecutive-prefix change | **+21** | **+8** |
| Native useful-repair change | **+6** | **−2** |

The JVP/VJP scalar differences are 1.50e−12 and 1.13e−12. Score/native-gradient discrepancies are approximately 0.0234% and 0.0174% of the score magnitudes; these are descriptive, not newly invented gates. Raw versus beta-centered score totals differ by only −2.48e−10 / −1.04e−9. Raw positive/negative score contributions are +0.254831/−0.005176 for F−C and +0.168625/−0.001606 for F−BETA. Detailed cancellation and mass residuals are retained in [CONTRASTS.csv](CONTRASTS.csv).

| Adjacent vector comparison | Reference norm | Candidate norm | Difference norm | Relative difference / reference norm | Cosine |
|---|---:|---:|---:|---:|---:|
| F−C linear → finite shadow | 0.088516 | 0.088455 | 0.000381 | 0.004302 | 0.999991 |
| F−C finite shadow → cast | 0.088455 | 0.275185 | 0.280190 | 3.167620 | 0.103615 |
| F−C cast → native | 0.275185 | 0.275183 | 0.137286 | 0.498887 | 0.875555 |
| F−BETA linear → finite shadow | 0.054991 | 0.054854 | 0.000584 | 0.010612 | 0.999947 |
| F−BETA finite shadow → cast | 0.054854 | 0.216070 | 0.218601 | 3.985115 | 0.080528 |
| F−BETA cast → native | 0.216070 | 0.216213 | 0.133069 | 0.615859 | 0.810484 |

Casting attenuates the **relevant projection** by about 66%; it does not simply shrink the update. Native optimizer arithmetic then changes the differential vector further while barely changing this fixed-cotangent projection. Large vector differences alone therefore do not establish large task-relevant harm. Per-layer contractions, zero cases and coordinate counts are in [VECTOR_LAYERS.csv](VECTOR_LAYERS.csv) and [VECTOR_SUMMARY.json](VECTOR_SUMMARY.json).

Absolute BETA shadow/native displacement norms reproduce the historical pattern: **1.968026737 / 1.230760086**, difference **1.759311677** (0.893947 relative to shadow). Q_BETA versus N_BETA differs in **1,420,195** of 1,048,626,432 coordinates; C and F differ in 1,414,482 and 1,427,696. This is a consistency diagnostic, not a bit-exact gate: the new products/reductions are FP64, whereas the old norms squared in FP32 before FP64 summation. Differential comparisons remain primary.

## Fixed outer-evaluation-point probe

The probe uses the same J_beta, gradients, coefficients and assessment origins, changing only the assessment cotangent from represented Q_BETA to exact N_BETA. No new Adam step or priority is created.

| Contrast | Original prediction | Native-point prediction | Actual native benefit | Absolute error, original → native point |
|---|---:|---:|---:|---:|
| F−C | 0.249596208 | 0.248989596 | 0.080693995 | 0.168902213 → 0.168295601 |
| F−BETA | 0.167047970 | 0.166788807 | 0.043241774 | 0.123806195 → 0.123547032 |

Both predictions already have the correct sign. Error improves by only **0.36% / 0.21%** and predictions still overstate the realized benefit by roughly 3.09× /3.86×. The evaluation point matters slightly here but does not resolve the scale mismatch or validate the continuous optimizer Jacobian. [Full probe comparison](PREDICTION_ERRORS.json).

## Native agreement and disjoint TRAIN readouts

Assessment R uses fixed original K and triangular weight K−i+1 through K: native FP32 softplus margin contributions divided by log 2, summed over each response and divided by its count of nonempty blocks, then averaged across responses. This is the existing assessment, not a new normalized protected-position loss. Strongest wrong rivals are recomputed at each endpoint. We retain the scorer-compatible FP32-reduced values and matched per-position FP64 contrasts; their small reduction differences remain visible.

| Native endpoint | Window R | Window matching prefix | Useful repairs /371 original errors | Damaged prefix positions /8,979 | Reachable followers /295 | Disjoint TRAIN R | Disjoint matching prefix |
|---|---:|---:|---:|---:|---:|---:|---:|
| Parent | 5.937279 | 8,979 | 0 | 0 | 0 | 6.844945 | 488 |
| BETA | 5.330328 | 9,084 | 65 | 1 | 41 | 6.901863 | 483 |
| C | 5.367780 | 9,071 | 57 | 2 | 38 | 6.919527 | 483 |
| F | 5.287086 | 9,092 | 63 | 0 | 40 | 6.921646 | 483 |

The window has 9,645 assessed occurrences. F versus C yields 12 block prefix gains, 3 losses and 766 ties; one response gains and one loses. Both response assessments improve. F−C matched assessment benefit is +0.131846 positive contributions minus 0.051152 negative contributions. Raw and useful repairs happen to coincide at these endpoints, but they are separately computed: a useful repair requires all preceding original matches to survive. Fixed-K shortfalls for parent/BETA/C/F are 666/571/583/563; surviving usable followers are 295/292/293/292. These counters are overlapping occurrences, not independent token samples or verification calls.

The authenticated disjoint set is the original **eight new-buffer TRAIN responses ×eight retained anchors =64 blocks**, with 44 original first errors, 488 prefix occurrences, 60 usable followers and 592 assessed occurrences. It excludes the two update responses by request identity and resolved feature path; the overlapping historical parent-training witness set is not substituted. All 64 discrete F−C and F−BETA block comparisons tie. Each native endpoint loses five prefix occurrences relative to the parent, has one damaged original prefix position/block, and repairs no original first error. F−C assessment benefit is **−0.002119269**, from +0.078539774−0.080659044; three response assessments improve and five worsen. This is no demonstrated transfer benefit on this witness set, not a population claim.

[Per-response endpoint table](ENDPOINT_READOUTS.csv), [all endpoint blocks](ENDPOINT_BLOCKS.csv), [paired readout comparisons](READOUT_COMPARISONS.json) include Q endpoints, repair/follower definitions, gains/losses/ties and denominators. No confidence intervals or significance claims are made over overlapping anchors.

## Execution, costs and audit

| Item | Actual |
|---|---:|
| Native Adam calls | 4 completed /4 planned /6 permitted |
| Attempts / retries | 1 /0 |
| Native update presentations | 8 |
| Endpoint readout forwards | 46:14 window +32 disjoint TRAIN |
| Disposable startup | 2 forwards/backwards;0 Adam;19.488 s |
| Instrumented draft forwards including startup | 68 |
| Instrumented vocabulary-head calls including startup/recomputation | 3,952 |
| Native diagnostic wall time | 328.238s |
| Supervised new phase / occupied card time | 336.257s /335.811s (0.09328 card-hours) |
| Prior +new occupied card time | 5,175.047+335.811 =5,510.858s |
| Post-startup Torch peak allocated /reserved | 37.658 /48.566GB |
| Sampled device usage /child peak RSS | 49.599 /68.467GB |
| Minimum sampled available host RAM | 41.937GB |
| Charged new storage, including 500 MB local envelope | 0.895GB, below 2 GB |
| Postflight A800 free space | 407.576GB, above 400 GB |

[WORK.json](WORK.json) preserves attempted/completed operation categories, head chunks and startup; no pending work remains. Module counters include instrumented recomputation and are not extra response presentations. [RESOURCES.json](RESOURCES.json) and [POSTFLIGHT_AUDIT.json](POSTFLIGHT_AUDIT.json) retain resource bindings and limits. Exact launch-to-exit GPU/host peaks are unavailable: Torch peaks reset after startup, and device/RSS samples are 30 s apart. CPU implementation/publication was not continuously timed; postflight CPU authentication took 10.412 s and is separate from GPU occupancy.

All 5,256 staged source entries and 15 protected input files—including parent recovery/export and ten feature packets—remain unchanged. All 57 raw files (63,030,015 bytes) were copied and rehashed. Both A800s are idle; only GPU B was used. No inherited artifact deletion, full-vector file or checkpoint export occurred. Operational issues resolved before/after GPU execution are recorded in [OPERATIONS.json](OPERATIONS.json); none caused a scientific retry.

The independent offline audit passed **798,571 assertions**, with all 11 interpretation-fallacy checks covered. It verifies token-derived metrics, FP64 matched reductions, score cancellation, per-layer/vector scalar identities, source/input bindings, repeatability/restoration receipts and budgets. Full gradients and endpoints deliberately existed only in RAM, so the audit cannot independently reconstruct every contraction or endpoint difference. The historical cross-method numerical failures remain disclosed rather than relabeled passes. Prior PR60's unresolved serving comparison remains unchanged.

The published [stdlib audit script](audit_scalars.py) operates on the private retained run layout and an untouched execution checkout at `269f056`; report-only metadata changes in later commits are not the source input to that audit. No private inputs are published.

## Next specific question; stop

**Remain at Point1 and ask whether a bounded preview of the represented/native update can predict F−C assessment contrasts more accurately than the continuous shadow.** Moving only the outer cotangent was insufficient; simply raising precision or removing a guard is not authorized by these data. The current evidence also leaves assessment transfer to other responses unresolved. A subsequent proposal should distinguish that transfer limitation from numerical calibration, instead of assuming either uniquely caused the earlier 128-update outcome.

This one-window result neither proves a code defect nor establishes production-wide optimizer fidelity or serving efficacy. No additional window, optimizer step, loss, allocation arm or event follows this report automatically.
