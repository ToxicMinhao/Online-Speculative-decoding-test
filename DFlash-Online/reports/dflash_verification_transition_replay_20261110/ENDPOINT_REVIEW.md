# Independent arithmetic and result review

## Material Passport

ARS experiment-agent / validate. Scientific execution: `d7b1311e7b2f910e935bf81b4e40282767d7540c`. Verification status: **ANALYZED**, with independently recomputed CPU arithmetic and retained native parity checks. No new model, GPU, training or remote work. This is a same-team review: the reviewer authored the production analysis, while the separate `audit_endpoint.py` reconstructs the arithmetic directly without importing it. It is not an independent-team replication.

## Verdict

**The proposed expensive-farther-restart bottleneck is not supported on either dataset.** Genuine advances generally give zero or fewer remaining verification calls. All four conditional mean intervals are below zero, and every leave-one-request-out mean stays negative. Under the prespecified handoff, these results do not justify the newly reached-start allocation training pilot.

One MathQA request does demonstrate that the phenomenon can occur: `mathqa-transfer-0583` has C44/C46/C64/C66 = 10/11/11/12, hence R4 = R6 = +1. Both successors are interior (69 and68 canonical tokens remain). This isolated counterexample is preserved; it does not establish a recurrent bottleneck or outweigh the negative average effects.

## Audited results

| Dataset | Selected | First advances / regressions / no divergence | Verified repairs among advances | Both restart effects positive | Conditional R4 mean [95% CI] | Conditional R6 mean [95% CI] |
|---|---:|---:|---:|---:|---|---|
| GSM8K |64|31 /33 /0|31 /31|0 /31|−0.387097 [−0.700000, −0.071429]|−0.612903 [−0.857143, −0.390210]|
| MathQA |64|40 /24 /0|40 /40|1 /40|−0.525000 [−0.727273, −0.303013]|−0.700000 [−1.025641, −0.410256]|

All selected transitions are interior, so the all-advance and interior-advance analyses coincide. No boundary exclusions were made. All actual advances remain eligible regardless of the repair flag; in the observed data all71 advances separately pass the first-rejected-position repair check. Among regressions, GSM8K has33 checked nonrepairs; MathQA has19 checked nonrepairs and five cases with no valid KPLUS4 first rejection. The latter are explicitly not applicable, not successful repairs.

GSM8K R4 effects are12 negative/17zero/2positive; R6 effects17negative/14zero/0positive. Both GSM8K positive-R4 cases have nonpositive R6, matching the handoff's distinction between a weakness under KPLUS4 and one still present under KPLUS6. MathQA R4 effects are23negative/14zero/3positive; R6 effects25negative/12zero/3positive, with only the single shared-positive case above.

Unconditional contributions per selected request are GSM8K R4=−0.187500 [−0.343750, −0.031250], R6=−0.296875 [−0.437500, −0.171875]; MathQA R4=−0.328125 [−0.468750, −0.187500], R6=−0.437500 [−0.656250, −0.250000]. These are signed call effects, not percentages of endpoint improvement.

GSM8K conditional medians are0 and−1 calls (R4/R6), ranges [−3,+2] and[−3,0]. MathQA medians are−1 and−1, ranges [−2,+1] and[−5,+1]. Largest absolute contributor shares are16.67%/15.79% for GSM8K and7.41%/14.71% for MathQA. All64 leave-one-out means are negative for each effect/dataset; there are no empty-stratum bootstrap draws. No result is presented as equivalence or a universal monotonicity theorem.

## Endpoint context and distinctions

Endpoint calls KPLUS4→KPLUS6 are2761→2748 on GSM8K and4330→4252 on MathQA. Request improvements/regressions/ties are29/29/6 and39/19/6. Of the29 GSM8K endpoint regressions,19 begin with a first progress regression; of19 MathQA endpoint regressions,11 do so. This is a descriptive timing-of-divergence classification, not an attribution of final harm to that one round.

Earlier proposals already differed before the first actual-progress divergence on54/64 GSM8K and30/64 MathQA requests. Those differences remain disclosed; equal earlier committed progress is not claimed to mean identical proposal history or identical BF16 target state. The crossed measurements use each actual source successor, not a token-offset equivalence assumption.

## Correctness and reproducibility

All128 rows,256 fresh native walks and256 crossed suffixes were checked. Fresh native records exactly match their authenticated retained original proposals, targets, actual commits, positions, tokens and termination. All eight neutral suffixes match their original suffix records and report exact target/draft valid-prefix cache reconstruction. All five effects and both decomposition identities reproduce for every row; no missing value is replaced by zero. Completion is128/128 with no recorded failures or unknown work; all1,421 postflight identities pass, and all four checkpoints remain unchanged.

The independently generated10,000 PCG64 request-bootstrap draws use seed20261110 separately for each dataset, resample all64 complete request rows, and recompute the eligible stratum per draw. Shared index matrix SHA256: `e2ff03adccd7eb5350a4f846d5138879723b9e1826ef93862a75a50c1c3be82b`. All intervals, eligible counts, concentration shares and64 leave-one-out means match production. Intervals are prespecified pointwise percentile intervals; no familywise multiplicity-adjustment claim is made.

No endpoint/statistical discrepancy was found. Detailed computed values are in `RESULT.json`; independently reconstructed public-safe rows are in `DERIVED_ROWS.json`; native parity/postflight checks are in `PARITY_REVIEW.json`. These are retained-evidence checks, not new native executions.

## Interpretation boundaries and fallacy coverage

All11 ARS categories were checked: Simpson's paradox, ecological inference, outcome-conditioned selection/Berkson, collider bias, base-rate neglect, regression to the mean, survivorship, look-elsewhere effects, forking paths, causal overstatement and reverse causality. The per-category audit is recorded in `RESULT_REVIEW.json`.

The fixed-panel conditional decomposition does not identify a unique percentage contribution of learning, prove that allocating training weight is learnable or effective, or establish serving speedup from untimed replay. Reused development panels and one fixed checkpoint pair do not establish broader generality. No new training or automatic follow-up is recommended under this handoff.
