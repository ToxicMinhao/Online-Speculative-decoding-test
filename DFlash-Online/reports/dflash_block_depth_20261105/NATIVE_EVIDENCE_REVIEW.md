# K+ depth first-milestone evidence review

Execution: `9150cc3b20714153b8e4fb3d70e8a32861c5921f`. Source records are under `../retrieved_002/qualification_002`; the complete audit is [native_audit_002_final/AUDIT.json](native_audit_002_final/AUDIT.json), with per-start, per-prompt and context-bin CSVs alongside it. The standalone calculation imports no experiment or model code. Seven synthetic CPU tests passed, including partial execution, false parity receipts, layer execution, head counts and mixed-depth schedule failures.

All **11,955 audit checks pass**. The evidence supports completed native qualification: 32 planned and returned complete generations, all valid; zero invalid, unknown or unattempted generations. This comprises 16 original FIVE requests, four forced-FIVE requests, four static-THREE requests and eight alternating-depth requests. No optimizer steps, readout/gate fitting or FINAL executions occurred.

Original FIVE supplied 530 verification rounds. The additional qualification supplied 1,007 rounds: 173 forced-FIVE, 501 static-THREE and 333 alternating. Every qualification round has an original/forced-five shadow pair with all 15 proposals equal at the exact same canonical start. Those checks used 2,014 shadow proposal forwards and vocabulary projections. The qualification receipts report exactly 3,021 vocabulary projections: two shadow heads and one selected live head per round. All eight mixed requests follow the full 3,5,3,5,… depth sequence. Their outputs and termination match original FIVE; matching shallow proposal IDs or verification counts is not required.

The four new captured graph segments have the expected layer-call vectors: static-three and prefix `[1,1,1,0,0]`, shallow `[0,0,0,0,0]`, deep `[0,0,0,1,1]`. Each records two setup warmups and one capture, totaling eight segment warmups and four captures. These are additional depth-graph setup operations, not complete generations or a claim to enumerate all internal normal-runtime initialization kernels. All 16 pricing receipts pass the six state/cache restoration checks. Before/after weight receipts pass unchanged-weight checks.

Pricing contains **128 sampled starts × four routes × four repeats = 2,048 timed measurements**, in the declared balanced order. There are separately 512 route warmups, 128 per route. Sampling uses eight evenly spaced starts per original request; all exact native FIVE and adaptive forced-FIVE proposal reconstruction checks pass. Timed proposal readiness includes the route dispatch/head work and excludes the shared committed-KV append/allocation. It is not a final serving comparison.

The independent `N/n` expansion and 10,000 paired-prompt bootstrap with seed `20261105` reproduce the production calculation:

| Quantity | Reconstructed value |
|---|---:|
| Same-progress optimistic saved time | 245.296644687 ms |
| Original complete-request denominator | 9,295.291725546 ms |
| Optimistic request-time reduction | 2.638934333% |
| Paired 95% interval | [2.544976237%, 2.703472869%] |
| Adaptive forced-five minus adaptive forced-three | 0.517600885 ms/start |

The declared early stop is **not triggered**: its upper bound exceeds 2%. This only says the measured skip-cost headroom clears that budget screen. It does not establish an effective readout, an accurate depth selector, or a serving-time improvement. The run correctly stopped before fitting, as authorized.

Coverage is narrow: sampled committed prefixes range from 47 to 502 tokens. There are 96 starts below 256 tokens across all 16 requests and 32 starts at 256–511 tokens across 11 requests. Longer-context behavior is unmeasured here. The per-context and per-prompt tables retain this limitation rather than extrapolating it away.

Attempt 001 remains preserved separately. [Its audit](native_audit_001/AUDIT.json) reconciles zero returned/unknown generations, 32 unattempted, and no pricing or training. Its startup failed before generation. Attempt 002 changes the launcher environment to supply the missing compiler setting; neither the failed record nor the first planned request was replaced. The successful source identities are checked against the new published commit.

This is a separate CPU evidence reconstruction by a same-team author who also implemented data, pricing and parts of qualification. It is neither a blinded review nor an independent GPU replication. No new generation, network operation or experiment-source change was made during this audit.
