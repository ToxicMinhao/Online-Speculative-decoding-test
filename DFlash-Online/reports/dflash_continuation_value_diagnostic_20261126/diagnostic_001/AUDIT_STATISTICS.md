# Independent statistical audit

Execution commit: `b6f2a2c8d487bcc2d32a5e716b870c10eb5067b3`.

**Status: ANALYZED — all 25,372 checks passed.** This is an independently written reconstruction of retained results, not a new GPU experiment or an external blinded review. The audit used no experimental analysis helpers. Its linear algebra uses the same NumPy FP64 SVD family, which is a disclosed shared implementation dependency.

The analyzed transition contains **8,192 matched blocks**, of which **2,114 parent-HIGH blocks from 128 responses** enter the prespecified model. The auditor independently rebuilt response-equal beta weights, parent-only standardization, exact joint `(e,r,n)` categories, all difficulty/fragility/tie controls, and the four immediate-confidence terms. Both weighted within-response centering and an explicit response-fixed-effect design recover the main coefficient:

| Quantity | Independently checked result |
|---|---:|
| Full-model theta, original u units | 3.0627201412 |
| Explicit response-FE WLS theta | 3.0627201412 |
| Primary response-bootstrap 95% interval | [−2.5389875433, 8.2484679515] |
| Effect per 0.1 increase in u | 0.3062720141 [−0.2538987543, 0.8248467952] |
| Effect per frozen residual-u SD | 0.1636404690 [−0.1356575505, 0.4407138431] |
| Supported response clusters | 127 |
| Residual degrees of freedom | 1,824 |
| Identifiable bootstrap draws | 10,000 / 10,000 |
| Positive identifiable leave-one-response-out estimates | 128 / 128 |

All 10,000 PCG64 draw identities, status counts and saved-coefficient percentile intervals reproduce. The first **20 actual bootstrap draws** and **all 128 leave-one-response-out fits** were independently recomputed by direct weighted-design SVD. A second full 10,000-fit bootstrap was not run. The full-baseline residual-u energy, each response's D range/support status, effect-size scale and degrees of freedom also reproduce. The reduced immediate-confidence-control fit and descriptive useful-repair/prefix-damage coefficients agree with their retained results.

The independently applied decision rule yields **INCONCLUSIVE**, because the primary confidence interval crosses zero. Adequate support and positive leave-one-response-out point estimates do not override that interval. This is neither demonstrated absence of information nor equivalence. The diagnostic does not show that reweighting these repairs would help training or serving; it describes one retained training-buffer transition.

The first audit passed 25,364 checks. A second CPU-only pass added eight explicit budget, effect-size, influence-count and decision-rule checks; it also passed. These passes used **11.618 seconds total CPU wall time**, with zero GPU work, model forwards, optimizer steps, new bootstrap draws or scientific retries. Four synthetic tests cover known coefficients, explicit/centered fixed effects, whole-response multiplicities and exact redundancy.

[AUDIT_STATISTICS.json](AUDIT_STATISTICS.json) retains input-file hashes and compact numerical results without response identifiers, prompts, tokens or private paths. Native-role binding, model immutability, resource accounting and retrieval authentication belong to the companion native/provenance audit. The shared assistant-team authorship and shared solver family limit independence; this audit is not an independent training replication.
