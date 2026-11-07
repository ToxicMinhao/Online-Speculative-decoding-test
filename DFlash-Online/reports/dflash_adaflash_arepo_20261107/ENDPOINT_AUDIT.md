# A_repo final endpoint evidence audit

## Material Passport

**Verification status: ANALYZED; endpoint audit passed on its first invocation.** Scientific execution: `7dac02b0fc6686d7129c5d29699ee9bfe5be2e7a`. The campaign executor ran the pre-result, frozen CPU audit on the original A800 records; this reviewer inspected the returned receipt locally. The script uses only the Python standard library and NumPy and imports no production evaluation/statistics/report functions. No model generation, optimizer update or scientific rerun occurred.

Authorship overlap is explicit: this reviewer authored earlier VFD endpoint/report modules and native audit logic. A separate child reviewer checked the script and final interpretation. This is a separate same-team recomputation, not blind independence or an independent GPU replication.

All **12,288 executions** (1,024 requests × four methods × three repeats) passed. The audit completed **4,310,189 checks**, recounted **530,403 native verification rounds**, and recorded **24,597 source-file identities**: 24,596 original production files plus the frozen-inventory carrier. These include all 12,288 raw request records and their 12,288 original native sidecars. All **186 checkpoint publications** and **192 timing boundaries** passed; the latter confirm no concurrent training, transfer or foreign GPU work at the recorded serving boundaries. Exact canonical tokens/termination and repeated counts passed; invalid, unknown and unattempted executions are zero. Each method retains all three capped requests and 1,021 EOS responses.

Input identities were pinned before results. The audited original files match the frozen inventory; the separately recorded full original-server rehash matched **97,485 compact files / 5,764,699,468 bytes** with zero changes. Whole-mirror transfer completion is tracked separately by its retrieval receipt. This endpoint review does not claim a second tensor-level state/weight reproduction or repeat the separate training audit.

The reconstruction matches the original SUMMARY.json within the declared numerical comparison tolerance (relative and absolute tolerance 1e-10); native count and token comparisons are exact.

| Candidate / reference | Complete-request reduction | Paired 95% interval | Call reduction | Call wins / regressions / ties |
|---|---:|---:|---:|---:|
| KPLUS / AREPO | 1.214010% | [0.572209%, 1.861134%] | 1.038403% | 472 / 370 / 182 |
| KPLUS / O | 2.863861% | [2.175156%, 3.563900%] | 3.270736% | 633 / 250 / 141 |
| AREPO / O | 1.670126% | [0.971617%, 2.354444%] | 2.255757% | 593 / 258 / 173 |
| AREPO / A | 2.538986% | [1.837100%, 3.241924%] | 3.021094% | 663 / 202 / 159 |

The sealed primary **K+/A_repo complete-request-time** interval is strictly positive, so **favors_KPLUS** follows the declared rule. K+ saves 454 net verification calls (1,256 saved, 802 added) and 9.655657 seconds of summed mean request time. The call reduction is 1.038403% [0.653797%, 1.409650%]. Wins/regressions/ties above concern verification calls, not latency. The result does not establish a true latency gain above 1%; the primary lower bound is 0.5722%.

Three timings are averaged within each request/method before 10,000 shared paired-request bootstrap draws: PCG64 seed 20261102, linear 2.5/97.5 percentiles. Ratios use summed counts/times. Repeats are not additional independent requests. The three secondary comparisons are descriptive and unadjusted. All three repeat-specific complete-request comparisons favor K+ (1.1880%, 1.2539%, 1.2001%).

K+/A_repo native-generation time reduction is 0.863624% [0.349739%, 1.363109%]. Its cache-maintenance reduction is 7.448536% [−2.378517%, 16.594512%], which crosses zero. These component timers neither explain the primary gap causally nor provide a detailed kernel-time decomposition.

| Method | Verification calls | Matched accepted tokens | Native accepted counter | Committed progress | Output tokens |
|---|---:|---:|---:|---:|---:|
| O | 44,730 | 270,037 | 269,093 | 313,823 | 314,847 |
| KPLUS | 43,267 | 271,522 | 270,556 | 313,823 | 314,847 |
| A | 45,083 | 269,678 | 268,740 | 313,823 | 314,847 |
| AREPO | 43,721 | 271,025 | 270,102 | 313,823 | 314,847 |

Both acceptance counters were recomputed from original per-round draft proposals, posterior tokens and committed output, then checked against saved measurements. Native acceptance is committed progress minus one per round. Matched acceptance is the canonical-matching draft prefix clipped at the actual stopping boundary. EOS/cap clipping can remove the ordinary bonus-token slot, so the counters can differ. Committed progress excludes the initial target-prefill token for each request; every ratio uses summed counts. The draft report states these definitions correctly.

All **11/11 ARS statistical-interpretation risks** were checked:

| Risk | Finding / limit |
|---|---|
| Simpson / aggregation | All three repeats favor K+ on complete-request time (1.1880%, 1.2539%, 1.2001%). No subgroup or temporal-causal explanation is inferred. |
| Ecological inference | The estimate is a ratio of pooled times, not the average percentage speedup of individual requests. |
| Berkson / selection | The fixed GSM8K panel is train-disjoint but previously used; inference remains conditional on this selected panel. |
| Collider adjustment | No outcome-conditioned filtering or covariate adjustment enters the sealed comparison. |
| Base-rate neglect | All 1,024 paired requests remain in each denominator; call wins, regressions and ties are explicitly distinguished from latency outcomes. |
| Regression to the mean | Controls were retimed contemporaneously. Historical K+/paper-A changes are not an isolated causal before/after estimate. |
| Survivorship | All 12,288 planned executions are required and validated; all three capped requests per method remain included. |
| Look-elsewhere | One latency primary and all three prescribed descriptive controls are reported. Secondary intervals are unadjusted. |
| Forking paths | The input hashes, order, primary, bootstrap generator/seed/draw count and decision rule were fixed before endpoint results; no outcome-based tuning occurred. |
| Correlation and causal mechanism | This is an own-feedback recipe comparison. It does not isolate clipping, probability arithmetic, particular training examples, or cache behavior as the cause. |
| Reverse causality | Training precedes the frozen checkpoint endpoint. Endpoint outputs were not fed back into training. |

The supported conclusion concerns the fixed checkpoints in this one-seed, reused mathematics panel and timing session. The A_repo online recipe has lower measured calls and latency than OSD and paper-A in the contemporaneous comparisons, while K+ retains the primary advantage. Own-feedback trajectories remain part of each recipe. This does not isolate clipping or other components, establish the full AdaFlash system, show other-seed/domain robustness, or establish training-inclusive/asynchronous speedup. No tuning, extension or research-stage advancement is authorized by this result.

Two wording refinements are recommended for the final prose: name the **A_repo online recipe** as the measured treatment rather than “the loss itself”; use its contemporaneous paper-A comparison instead of implying an isolated causal shrinkage from historical K+/paper-A timings. The current draft correctly labels call wins and discloses the primary/secondary CI scope. No numerical discrepancy or remaining endpoint-validity blocker was found.

Evidence: [compact audit summary](ENDPOINT_AUDIT.json), original returned receipt `REMOTE_AUDIT_001.json` (SHA-256 `1452d1bbe3890794b1b348fd6810fe6b419ea282e7b5799d586f26824e89e79e`), frozen script `audit_endpoint.py` (SHA-256 `f71051e6a3476e189fb0246d292f55661cf0bc977091c782af3b0dad4e4abda6`), and full-inventory rehash receipt `REMOTE_EVIDENCE_REHASH.json`. The original scripts, input pins, readiness record and returned receipt remain unchanged.
