# Completed experiment audits

These are sanitized copies of CPU audits of the verified execution002 delivery, bound to scientific commit `d51718a7fa993a1494e593d15a8539090227d8b8`. They ran no new model operations and did not alter retained results. The delivered inventory SHA-256 was `7bba748079052cee3dc9125861ee27dd8381ecb04b3fd2bdc31fdc71aaac310a`.

| Audit | What it checks |
|---|---|
| [Raw probe scoring](RAW_PROBE_AUDIT_001.json) | Independently rescores canonical tokens, seed positions, EOS/cap treatment, all fifteen neutral proposals, 16,128 physical probes, 256 exact aliases and every one of 16,384 public outcome rows. |
| [Execution evidence](EXECUTION_EVIDENCE_AUDIT_001.json) and [review](EXECUTION_REVIEW.md) | Recomputes memory eligibility, threats and selections; checks branch restoration, shared layouts and normalization, update counters, reconstruction receipts and final native parity. |
| [Work accounting, current](WORK_AUDIT_002.json) | Cross-checks both attempts' operation ledgers, occupied-card time and the cumulative 170/171 Adam budget. |
| [Work accounting, original](WORK_AUDIT_001.json) | Preserved prior audit version. Its field labelled “scientific Adam” included two duplicate checks; version002 explicitly separates 159 native/disposable steps and two duplicate checks. Raw values and experiment evidence were not corrected or rerun. |
| [Independent scalar reduction](PUBLIC_SCALAR_AUDIT_001.json) | Recomputes public scalar estimands and checks internal consistency. |
| [Interpretation review](INTERPRETATION_REVIEW.md) and [structured assessment](INTERPRETATION_REVIEW.json) | Checks 2,251 scalar values, both primary contrasts, the fixed advancement decision and all eleven ARS statistical interpretation categories; notes conditional scope and authorship overlap. |
| [Interpretation calculations](INTERPRETATION_REDUCTION.json) | Separate reduction of both primary contrasts and paired intervals, acquisition/loss decomposition, denominators, support and intervention effects. |
| [Publication privacy](PUBLICATION_PRIVACY_REVIEW.md) and [receipt](PUBLICATION_PRIVACY_REVIEW.json) | Reviews the original sixteen scalar/identity staging files. Numerical arrays retained there are allowed anchor offsets/window indices; private token content and absolute paths are excluded. |

[Audit provenance](PUBLIC_AUDIT_PROVENANCE.json) records identities of the original private audits and their sanitized copies. [Original audit identities](AUDIT_IDENTITIES.json) retain source-script and original receipt fingerprints; they describe the unsanitized source artifacts, not replacement hashes for the public copies. Private path maps, prompt/answer text, tokens, model tensors and recovery state are not published here.

Reviewers were separate agents of the same model family, with authorship overlap. They used separately written scalar checks; they did not perform a blinded external review or repeat the GPU experiment. Native KV/state equality remains supported by retained in-process receipts, not by reloading billion-parameter states or replaying the model in these CPU audits. Passing these checks establishes internal consistency within the stated evidence scope, not efficacy or a causal explanation.
