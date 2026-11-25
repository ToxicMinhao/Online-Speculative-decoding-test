## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-11-25
- Verification Status: ANALYZED
- Version Label: point1_independent_offline_v1
- Native execution commit: `269f05644eeed2f0396602c47beb2a71166f8dd0`

## Validation Report

The independent standard-library audit passed 798571 assertions across 64 retained files. It imported no production reducer and ran no model. This verifies retained scalar arithmetic, token-based readout reconstruction and receipt consistency; it does not independently reproduce unsaved tensor operations.

All four separately restored one-step cases, full endpoint coverage, raw/centered score cancellation, per-layer vector aggregate identities, paired response/block summaries, source/export bindings and recorded time/storage budgets were checked. No confidence intervals, p-values or independent-anchor inference were computed. Overall interpretation confidence: CAUTION, because this is a fixed TRAIN mechanism diagnostic.

## Fallacy Scan

**Coverage: 11/11 checked.**

| Check | Status | Evidence and limit |
|---|---|---|
| Simpson's paradox | CHECKED_DESCRIPTIVE | Matched response means, every response benefit, and block gains/losses are retained. Mixed subgroup signs are reported; no aggregate-to-subgroup inference. |
| Ecological fallacy | CHECKED_NO_INFERENCE | Fixed response/block results do not imply individual requests, populations, or serving gains. Anchors overlap and are not independent observations. |
| Berkson's paradox | NOT_APPLICABLE | No estimated association or selected-sample correlation. Fixed TRAIN selection limits external validity and is disclosed. |
| Collider bias | NOT_APPLICABLE | No regression adjustment or conditioning on post-treatment covariates; roles are frozen at the common parent. |
| Base rate neglect | CHECKED_DESCRIPTIVE | Raw repair/follower counts accompany original opportunity totals; no diagnostic accuracy or population predictive probability is claimed. |
| Regression to the mean | CHECKED_CONTROLLED_LOCAL | F and native C are matched one-step interventions from the same fixed warm parent. Selection is fixed; no selected-extreme cohort efficacy inference. |
| Survivorship bias | CHECKED_COMPLETE_RECEIPTS | All four planned cases, all selected endpoint readouts, every reserved call and attempt are audited. C_REPEAT is an integrity check, not an independent replicate. |
| Look-elsewhere effect | CHECKED_NO_INFERENCE | F-C is primary, F-BETA secondary, C-BETA derived context. All fixed contrasts and native-point probes are retained; no p-values or selective significance claims. |
| Garden of forking paths | CHECKED_BOUNDED_DIAGNOSTIC | The fixed handoff, contract, source commit and ledger constrain this local diagnostic. Historical development selection prevents a confirmatory population interpretation. |
| Correlation versus causation | CHECKED_SCOPE_LIMITED | Controlled local endpoint differences support only this exact intervention/window. A discrepancy does not uniquely explain the preceding 128-update serving result. |
| Reverse causality | NOT_APPLICABLE | Frozen coefficients precede independently restored one-step endpoints; no cross-sectional directional association is estimated. |

## Reproducibility and limits

Model reproducibility verdict: CANNOT_VERIFY. Offline scalar/token reconstruction: passed. C_REPEAT is an exact execution repeat receipt, not a statistical replicate.

- Full model vectors were not saved, so tensor contractions cannot be independently reconstructed.
- FP32 softplus/head outputs are retained native measurements, not independently recomputed logits.
- Source-input and parent model identities are compared to pinned receipts; this offline audit does not rehash remote model/feature files.
- Score per-pool sums and cancellation are checked; unsaved per-token coefficient/score products cannot be independently reconstructed.
- Resource receipts are sampled and do not prove the absence of a transient peak between samples.
- Few fixed overlapping TRAIN blocks support no population or serving claims.
