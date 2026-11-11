# ARS interpretation checks

Verification status: ANALYZED (CPU arithmetic reproducibility is checked separately; no new independent GPU replication). This is an inline review by the implementation author because native agent dispatch hit the platform thread limit. It is not an external review.

11/11 statistical fallacy categories covered:

1. Simpson reversal: report GSM8K and MathQA separately, plus common-(j,v)-support estimates; do not pool conflicting workloads.
2. Ecological inference: sampled native starts describe local checkpoint transitions; they do not measure training-anchor weight allocation or request latency.
3. Selection: sample from parent-defined SHORT/LONG strata, sealed before child reads; inference is conditional on that definition and reused development requests.
4. Collider conditioning: repair-conditional B is reported with unconditional realized counts and its denominator; it is not a causal selection effect.
5. Base rates: A includes exact parent LONG prevalence among all later starts; separate conditional and unconditional rates.
6. Regression to mean: parent errors define the sample, but no intervention efficacy or future repair probability is established.
7. Survivorship: preserve all selected starts, invalid/pending status, no replacements; incomplete coverage cannot pass the joint screen.
8. Look-elsewhere: fixed two thresholds/datasets and reported comparisons; nominal intervals are not simultaneous family-wise guarantees.
9. Forking paths: fixed hash order, fixed checkpoints, fixed native layout/cap/budgets; no retuning, resampling or added model work.
10. Causality: actual child repair and acceptance describe frozen checkpoints; do not claim that moving training weight would reproduce them.
11. Reverse causality: source weights precede child weights, but the observational SHORT/LONG association does not identify why training changed a proposal.

Sparse repair denominators, undefined bootstrap draws and degenerate zero-event intervals remain explicit. A/B thresholds are engineering screens, not predicted speedups. Follower agreement is computed from canonical IDs after rejection, not wrongly conditioned verifier logits. All-correlated blocks are resampled by request cluster, not independently.
