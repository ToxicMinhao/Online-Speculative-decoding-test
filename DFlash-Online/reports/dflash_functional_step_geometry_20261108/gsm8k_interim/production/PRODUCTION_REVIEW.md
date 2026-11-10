# GSM8K production-update evidence review

## Material Passport

- Status: ANALYZED; completed GSM8K production records only.
- Execution: `6440e8ff5af1a845f830d65c884f6711d7e4a687`; realization `norm_feasible_bf16_r1`.
- Scope: events 5 and 6, 128 windows per event per arm; no MathQA or full-pilot classification.
- Inputs: authenticated compact snapshot; every consumed file is hashed in INPUTS.json.
- Zero model loading, GPU work, optimizer steps, forwards, backwards, feedback or serving generations.
- Review: same-team source/numerical review with implementation-author overlap; not independent GPU replication.

## Main finding

FSG committed a distinct correction in all 256 windows. Each committed endpoint differs from both the pre-step and ordinary native optimizer endpoint. All 256 satisfy the recorded strict realized-norm/actual-KL rule. Thus, the reported endpoint regression is not explained by predominantly inactive FSG or failed guard acceptance. Production traces alone do not identify why the changed updates harmed final drafting.

SCALAR committed 34/256 windows (13.28125%) and restored native weights in 222/256 (86.71875%). Its own target norms were generally almost the native norm. This is the specified own-state scalar control, but its predominantly native trajectory cannot establish equivalence or the behavior of a substantially smaller-step training recipe.

## Activity and eligibility

Eligibility is shown as three distinct conditions: nonzero native displacement, nonempty protected support, and nonempty sketch. All were satisfied in every recorded filtered window. Every candidate received an actual BF16 replay; no candidate KL was missing in these 512 windows. No zero-native, empty-sketch, lambda-zero, exact-native-candidate, or rounded-to-prestate branch occurred.

| Arm | Event | Windows | Eligible: nonzero / protected / sketch | Genuine accepted writes | Native fallbacks | Numerical ambiguity | Measured KL increase | Feasibility search |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FSG | 5 | 128 | 128 / 128 / 128 | 128 | 0 | 0 | 0 | 126 |
| FSG | 6 | 128 | 128 / 128 / 128 | 128 | 0 | 0 | 0 | 128 |
| FSG | combined | 256 | 256 / 256 / 256 | 256 | 0 | 0 | 0 | 254 |
| SCALAR | 5 | 128 | 128 / 128 / 128 | 23 | 105 | 90 | 15 | 125 |
| SCALAR | 6 | 128 | 128 / 128 / 128 | 11 | 117 | 98 | 19 | 127 |
| SCALAR | combined | 256 | 256 / 256 / 256 | 34 | 222 | 188 | 34 | 252 |

Every genuine write has reason `actual_risk_decreased`. Ambiguous intervals and a measured KL increase remain separate rejection reasons; neither is silently counted as a successful correction. Geometry metrics are inapplicable to KPLUS.

## Step size, direction and protected KL

| Arm | Quantity, combined 256 windows | Minimum | Median | Maximum |
|---|---|---:|---:|---:|
| FSG | continuous_raw_norm_ratio | 0.993755386096 | 0.999173347527 | 0.999931952624 |
| FSG | raw_bf16_norm_ratio | 0.999893208515 | 1.00006046158 | 1.000363929 |
| FSG | feasible_norm_ratio | 0.999893208515 | 0.999999999578 | 0.999999999998 |
| FSG | actual_candidate_norm_ratio | 0.999893208515 | 0.999999999578 | 0.999999999998 |
| FSG | committed_norm_ratio | 0.999893208515 | 0.999999999578 | 0.999999999998 |
| FSG | committed_direction_cosine | 0.998734856007 | 0.999881244204 | 0.999993735616 |
| FSG | candidate_native_kl_ratio | 0.368731167433 | 0.711684065398 | 0.912172500073 |
| SCALAR | continuous_raw_norm_ratio | 0.995865182125 | 0.999244776214 | 0.999918675721 |
| SCALAR | raw_bf16_norm_ratio | 0.999851257265 | 1.00003546695 | 1.00015819481 |
| SCALAR | feasible_norm_ratio | 0.999851257265 | 0.999999999653 | 1 |
| SCALAR | actual_candidate_norm_ratio | 0.999851256847 | 0.999999999652 | 1 |
| SCALAR | committed_norm_ratio | 0.999851256847 | 1 | 1 |
| SCALAR | committed_direction_cosine | 0.999998681857 | 1 | 1 |
| SCALAR | candidate_native_kl_ratio | 0.990394159755 | 1 | 1.01570519369 |

Norm ratios use that same window’s own native optimizer displacement. The raw continuous direction, its unadjusted BF16 endpoint, norm-feasible endpoint, actually scored candidate and committed endpoint are distinct measurements. The final FSG norm is nearly unchanged from native; a feasibility scale below one must not be read as an equally large reduction in the rounded displacement norm.

| Arm | Search-only denominator | Alpha minimum | Alpha median | Alpha maximum |
|---|---:|---:|---:|---:|
| FSG | 254 | 0.985506177 | 0.993634820 | 0.998900235 |
| SCALAR | 252 | 0.989433646 | 0.995215029 | 0.998896301 |

SCALAR performed its own matching operation in all 256 windows. Relative target/realized norm mismatch: median 5.18549703e-13, maximum 3.70482387e-07; the maximum is a fraction, not a percentage. This close norm match does not make its direction or accepted endpoint identical to FSG. SCALAR-specific columns are inapplicable to FSG.

FSG’s median candidate/native protected-KL ratio is 0.711684065. Lower measured protected KL is partly imposed by acceptance and is not independent evidence of preserved useful learning. Values across independent evolving FSG/SCALAR trajectories are descriptive, not same-prestate causal comparisons. The recorded direction cosines can exceed 1 by floating-point roundoff; raw values are retained rather than clipped.

Protection coverage counts overlapping sampled-block occurrences, not unique response tokens. The margin floor uses the specified probability margin, and protection weight is separately normalized per response.

| Arm | Protected occurrences | Margin-floor occurrences | Margin-floor weight fraction | Correct-tie occurrences | Tie weight fraction |
|---|---:|---:|---:|---:|---:|
| FSG | 911,213 | 5,887 | 0.273225 | 3,025 | 0.156151 |
| SCALAR | 971,969 | 5,905 | 0.267424 | 3,076 | 0.152380 |

## Recorded costs

Intrinsic training here means the recorded training envelope minus the separately measured diagnostic observer. It still includes ordinary training, the production geometry/correction process and their bookkeeping. All other timers below have explicit scopes; they are not additive estimates of kernel time.

| Arm | Event | Intrinsic training s | Extra diagnostics s | Intrinsic/KPLUS ratio |
|---|---|---:|---:|---:|
| KPLUS | 5 | 325.919 | 0.000 | 1.0000 |
| FSG | 5 | 4926.178 | 595.636 | 15.1147 |
| SCALAR | 5 | 6777.056 | 0.000 | 20.7937 |
| KPLUS | 6 | 309.598 | 0.000 | 1.0000 |
| FSG | 6 | 5007.337 | 586.836 | 16.1737 |
| SCALAR | 6 | 6884.044 | 0.000 | 22.2354 |
| KPLUS | Combined | 635.517 | 0.000 | 1.0000 |
| FSG | Combined | 9933.515 | 1182.472 | 15.6306 |
| SCALAR | Combined | 13661.100 | 0.000 | 21.4960 |

| Arm | Feasible-realization envelope s | Its parameter-only probes s | Additional SCALAR matcher s |
|---|---:|---:|---:|
| FSG | 6699.570 | 5429.358 | n/a |
| SCALAR | 6622.020 | 5359.787 | 3805.338 |

Probe time is inside the feasible-realization envelope, and both are inside intrinsic training. Hook totals include the native optimizer and, for FSG, the diagnostic observer; do not add hook totals to training. Per-category norm/construction/identity scans overlap and are not counts of unique passes. Loading, startup, persistence/export and worker envelopes are retained separately in EVENT_COSTS.csv and TRAINING_COST_SUMMARY.json. Memory peaks are measured event peaks, including diagnostic/export activity where present, not isolated production-algorithm memory. This is not an equal-compute comparison or training-inclusive serving-speedup claim.

## Integrity, analysis work and limits

- Six complete event receipts, 256 presentations/128 Adam steps each, verified against the declared counters.
- Every arm used the same recorded presentation order/anchors as KPLUS for its event.
- All 512 filtered step files matched their corresponding event-embedded records exactly.
- All 512 recorded gate decisions were independently recalculated from saved norms, identities, risk values and deadbands. Per-response risk pooling and 24-iteration search receipts were checked.
- Both ANALYSIS_001.log and ANALYSIS_002.log are successful CPU passes. The second added phase-timing/flattened tables, with no changed scientific inputs or findings. No failed analysis or scientific inconsistency was encountered.
- The final CPU assembly wall time was 0.317005 s. Source review and human/agent reasoning are outside that timer. No model work was performed.
- Optional ordinary-gradient directional products were not recorded and are not inferred from auxiliary-gradient products.
- The production analysis does not attribute the endpoint regression to a particular mechanism. The separately analyzed eight same-prestate windows are needed for bounded mechanism evidence, and cannot establish training-seed robustness.

ARS safeguards checked (11/11): dataset/ event strata retained; update/occurrence/request units distinguished; reused-panel selection disclosed; no accepted-window-only efficacy claim; all scheduled windows in denominators; no selected best step; all fallbacks retained; all requested measures shown; no settings changes; divergent trajectories are not same-state causal controls; protected-KL and endpoint associations do not prove the underlying cause.

Files: PRODUCTION_WINDOWS.csv; ACTIVITY_SUMMARY.csv; SCALAR_DISTRIBUTIONS.csv; PRODUCTION_SUMMARY.json; EVENT_COSTS.csv; TRAINING_COST_SUMMARY.json; CHECKPOINT_IDENTITIES.json; INPUTS.json; ANALYSIS_RECEIPT.json.
