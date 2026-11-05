# Separate final record audit: KP v1

**Verdict: the completed fixed campaign and published numerical results pass independent compact-record reconstruction. KP shows a small, mixed-seed improvement over K; the simpler lower-learning-rate control performs substantially better.**

Execution: `d10521a1aa7d256bb75b37a6aae07586ae7cc8b3`. Audit completed 2026-10-31T17:54:03.266835+00:00.
Analysis script was frozen before final retrieval: `final_record_audit.py`, SHA-256 `c70d4b0d25403f4336246dce93d9944e0481f026eedddbf1fe75542e68b65d19`. The script imports no authored experiment statistics or report helper. It used CPU/NumPy with CUDA hidden; there was no model execution, remote command, retraining, request replacement or execution-code change.

## Validity and evidence scope

- 1,973,211 independent checks passed with no discrepancies. All 3,072 raw executions, all 194,145 full proposal/target/commit blocks, balanced ordering, canonical tokens/termination, repeat-consistent counts and both included 96-execution gates were checked.
- Reconstructed 256 seed/request pairs from 128 shared request identities. Counts use one validated trajectory per method/request. Timing uses each request’s three-repeat arithmetic mean.
- Recomputed all six prespecified comparisons and within-seed gain/harm components using one joint 10,000-draw PCG64 index matrix, seed 20261102, with linear 2.5/97.5 percentiles. Both seeds and all methods travel together under each request resample.
- Independently SHA-256 checked all 9,634 retrieved files (747,691,808 bytes), with zero mismatches. All 3,072 separately retained native captures exactly equal their wrapper captures, tokens and traces. All 3,242 primary audit inputs match the retrieval inventory.
- All eight original O schedule receipts match their sealed source hashes. All 16 new events retain the original response order, anchors and immutable features. Complete boundaries record 512 steps/1,024 presentations per trajectory; production totals are 2,048 Adam steps, 4,096 ordinary response forward/backwards and 2,048 reference forwards. Separate qualification totals are four Adam steps, eight current forward/backwards and six reference forwards.
- Original remote model/Adam tensors were not independently decoded locally. Exact restore, unchanged/exported weights, cold-process setup and endpoint equality were reviewed through retained receipts; the earlier native qualification remains its own evidence. This audit is separate record analysis, not independent GPU reproduction.

## Independent endpoint reconstruction

| Scope | F calls | K calls | K_HALF calls | KP calls | KP/K reduction and 95% interval |
|---|---:|---:|---:|---:|---:|
| Seed 17 | 8,150 | 8,140 | 7,949 | 8,144 | -0.049140% [-0.483754, +0.403670]% |
| Seed 29 | 8,150 | 8,163 | 7,916 | 8,103 | +0.735024% [+0.127636, +1.344161]% |
| Combined | 16,300 | 16,303 | 15,865 | 16,247 | +0.343495% [+0.018152, +0.691892]% |

| Combined comparison | Call reduction | 95% interval | Request-time reduction | 95% interval |
|---|---:|---:|---:|---:|
| KP_vs_K | +0.343495% | [+0.018152, +0.691892]% | +0.176987% | [-0.311890, +0.672927]% |
| KP_vs_F | +0.325153% | [-0.638583, +1.236479]% | +0.361958% | [-0.626710, +1.293488]% |
| KP_vs_K_HALF | -2.407816% | [-3.286106, -1.628051]% | -2.234990% | [-3.222188, -1.371493]% |
| K_HALF_vs_K | +2.686622% | [+1.923292, +3.533751]% | +2.359249% | [+1.566849, +3.252813]% |
| K_HALF_vs_F | +2.668712% | [+1.841082, +3.556608]% | +2.540176% | [+1.710953, +3.429243]% |
| K_vs_F | -0.018405% | [-1.028394, +0.898187]% | +0.185299% | [-0.822291, +1.097031]% |

KP versus K: 102 improved, 80 regressed and 74 tied seed/request outcomes; 203 calls saved and 147 added, for 56 net saved. Seed 17 adds four calls and seed 29 saves 60. Those 256 outcomes are descriptive counts, not 256 independent bootstrap units.
KP’s combined +0.343495% is below the prespecified 0.5% practical target, one seed is negative, and the KP/F interval crosses zero. Therefore the handoff’s “supported useful improvement at this budget” criterion is **not met**, despite the primary interval’s positive lower endpoint. Complete-request timing improvement is also unresolved. Secondary intervals are descriptive/unadjusted and conditional on these two fitted trajectories.

## Same-request gain and harm

Positive parts were formed within each seed before summing. K has 369 gross gain calls and 372 gross harm calls versus F.

| Candidate | Retained K gain | Lost gain | Added gain | Prevented harm | Added harm | Retained fraction |
|---|---:|---:|---:|---:|---:|---:|
| KP | 301 | 68 | 91 | 112 | 79 | 81.5718% |
| K_HALF | 261 | 108 | 299 | 271 | 24 | 70.7317% |

KP has net gain 91−68+112−79=56 calls versus K. K_HALF has 299−108+271−24=438. KP retains more of K’s existing gains, while K_HALF generates more different gains and prevents more harm. These are endpoint decompositions; they do not identify a causal gradient mechanism.

## Cost reconstruction

| Four-event trajectory | Training seconds | Event wall seconds | Initialization/resume seconds | Peak allocated GiB |
|---|---:|---:|---:|---:|
| 17/K_HALF | 1490.584 | 1642.604 | 169.329 | 19.286 |
| 17/KP | 1847.024 | 2000.769 | 202.442 | 21.239 |
| 29/K_HALF | 1492.825 | 1651.655 | 182.026 | 19.259 |
| 29/KP | 1848.319 | 2003.820 | 205.490 | 21.212 |

Across both seeds, KP training takes 3695.343 s versus 2983.410 s for fresh K_HALF: +711.933 s (+23.8631%). This is the observed cold-training comparison to K_HALF, not a freshly matched ordinary-K overhead measurement. Historical K costs remain historical. Nested reference/setup/head timers must not be added to their parent training/loading timers.

Evaluation incurred 2,818 checkpoint activations, 5779.404 s switching, 74.086 s loading, and 23.965 s weight validation, all separate from reported request latency. The exact activation schedule and cache/publication receipts reconcile. No training-inclusive speedup is established.

## Supported decision

The fixed KP candidate does not justify replacing K or the tested K_HALF control. Provisionally prefer the simpler lower-update-strength control for any separately agreed next design discussion; this completed screen does not authorize a new experiment or establish an optimal learning rate. Stop at the agreed campaign. The evidence supports a small, mixed-seed relative effect for KP and a stronger endpoint benefit for K_HALF on this panel.

## Artifacts

- `compact_record_audit_001/AUDIT.json`: complete reconstructed statistics and work/cost checks.
- `compact_record_audit_001/INDEPENDENT_PAIRED_REQUESTS.csv`: independently rebuilt paired table.
- `compact_record_audit_001/INPUT_IDENTITIES.json`: all primary audit input hashes.
- `compact_record_audit_001/RETRIEVAL_NATIVE_SOURCE_AUDIT.json`: complete retrieved-byte and separate native-capture verification.
- `compact_evidence_001/RETRIEVAL_MANIFEST.json`: remote-to-local file identities.
