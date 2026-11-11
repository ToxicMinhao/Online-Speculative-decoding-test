# K+ longer-run headroom: completed frozen-checkpoint diagnostic

**Decision: both datasets pass the prespecified opportunity/payoff screen. This supports proposing one capped reallocation test within K+’s existing high-priority weight budget. It does not show that reweighting improves learning or serving.** SHORT also retains useful gains, and training-weight headroom remains unknown.

Native execution: **`c504d48b05f79368e18bd9ec9c76e0e59c658205`**. CPU analysis/audit code: **`c6c787497e7e16c2616745f009cc1217ad13faf4`**. [PR #40](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/40). The diagnostic completed on November 11 UTC; `20261112` is the run label/bootstrap seed. No training, new checkpoint, latency benchmark or candidate fitting occurred.

## Main result

A estimates the share of **all later parent starts** that remain blocked while at least the first two originally correct followers survive. It is exact LONG census share multiplied by the sampled LONG blocked-with-two fraction. B measures the fraction of **actually repaired LONG blockers** for which both followers survive and are natively accepted.

| Dataset | LONG share of later starts | Still blocked with both followers | A [95% interval] | Both followers accepted / LONG repairs | B [95% interval] |
|---|---:|---:|---:|---:|---:|
| GSM8K | 403/2,697 = 14.94% | 21/64, across 15 requests | **4.90% [2.92%, 6.96%]** | 18/23; repairs on 18 requests, payoff on 15 | **78.26% [60.71%, 93.75%]** |
| MathQA | 651/4,266 = 15.26% | 30/64, across 22 requests | **7.15% [5.39%, 9.19%]** | 15/17; repairs on 15 requests, payoff on 13 | **88.24% [69.23%, 100%]** |

Both lower bounds exceed A’s **1%** and B’s **50%** engineering thresholds, with events recurring across requests. All 10,000 draws are defined for each primary quantity; none of these intervals is degenerate. Repair samples are still modest (23 and 17), so these are bounded diagnostic findings, not broad generalization guarantees.

A missed repair contributes zero to unconditional realized payoff: **18/64 = 28.13% [16.67%, 40.00%]** for GSM8K and **15/64 = 23.44% [13.24%, 35.71%]** for MathQA. These denominators are sampled LONG starts. Neither A nor B is a predicted speedup, and local token gains are not converted into saved verification calls.

## What was measured

The probe uses PR34’s frozen KPLUS4 and KPLUS6 exports at Adam steps **512 and 768**, on its original 64 GSM8K and 64 MathQA development requests. The target is pinned Qwen3-8B with native five-layer DFlash, BF16/FA3, greedy block 16, context 4,096, concurrency 1 and each original 2,048-token/EOS budget. These are reused development walks, not fresh held-out requests or a new learning trajectory.

At each parent start, the 15 proposals exclude the known seed. Canonical response IDs define correctness after the first error; verifier logits after a wrong proposal are not used as correct follower labels. Valid EOS is included; the saved cap censors unavailable continuation. With first error j and v available follower slots, SHORT requires exactly one consecutive correct follower followed by an observed error; LONG requires at least two. Both require v≥2.

The parent-only census and hash-selected sample were sealed **before KPLUS6 outcome files were read**. Each dataset/group uses the lowest 64 SHA256 keys of the prescribed compact JSON tuple. No selected state was replaced because reconstruction was difficult.

| Parent census | GSM8K | MathQA |
|---|---:|---:|
| All rounds | 2,761 | 4,330 |
| Opening rounds, descriptive only | 64 | 64 |
| Primary later rounds | 2,697 | 4,266 |
| LONG | 403 | 651 |
| SHORT | 263 | 381 |
| R=0, with adequate lookahead | 1,552 | 2,283 |
| All correct | 359 | 746 |
| Insufficient lookahead | 120 | 205 |
| Requests containing both eligible groups | 61/64 | 62/64 |

Opening counts are separate: GSM8K LONG 5 / SHORT 4 / R0 55; MathQA LONG 13 / SHORT 8 / R0 38 / all-correct 5. Selected SHORT/LONG inclusion fractions are 64/263 and64/403 for GSM8K,64/381 and64/651 for MathQA. Samples cover 41/39 and 37/35 distinct requests respectively. Full request and (j,v) counts are in [CENSUS.json](CENSUS.json); the exact parent-only sample is [SELECTED_STARTS.json](SELECTED_STARTS.json).

## Repairs, remaining blockers and donor cost

A repair requires an intact preceding prefix, a canonical match at the old error, and native acceptance beyond the parent’s accepted prefix. An advance alone is not substituted for this check. Counts below retain every sampled start, including regressions. Accepted-prefix counts include an accepted EOS; unavailable suffix positions are not scored.

| Dataset / group | Starts | Still blocked | Verified repairs | Earlier prefix breaks | Gain / loss / tie starts | Gross tokens gained / lost | Net accepted-prefix tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| GSM8K SHORT | 64 | 38 | 12 | 14 | 12 / 14 / 38 | 49 / 44 | **+5** |
| GSM8K LONG | 64 | 29 | 23 | 12 | 23 / 12 / 29 | 111 / 32 | **+79** |
| MathQA SHORT | 64 | 41 | 18 | 5 | 18 / 5 / 41 | 68 / 11 | **+57** |
| MathQA LONG | 64 | 43 | 17 | 4 | 17 / 4 / 43 | 109 / 14 | **+95** |

SHORT is a potentially costly donor: it has useful repairs and positive net prefix gain, especially on MathQA. The evidence does not justify discarding its priority or assuming that moving its weight is free. [All per-start records](PER_START.csv), [per-request reductions](PER_REQUEST.csv).

## Common-position comparison and uncertainty

Primary intervals use 10,000 request-cluster bootstrap draws, PCG64 seed 20261112 independently reset for each dataset. Every draw resamples all 64 request clusters and recomputes parent census shares and sampled ratios together. Blocks are not treated as independent observations. Intervals are nominal percentile intervals, not simultaneous family-wise guarantees.

For the additional SHORT/LONG comparison, common (j,v) cells receive the smaller parent census count as weight:

| Dataset | Observed common cells | SHORT / LONG sampled starts covered | SHORT / LONG parent census covered | Standardized repair rate, SHORT / LONG | Standardized net tokens per start, SHORT / LONG |
|---|---:|---:|---:|---:|---:|
| GSM8K | 13 | 62/64 / 62/64 | 257/263 / 389/403 | 17.83% / 35.97% | +0.094 / +1.382 |
| MathQA | 12 | 61/64 / 62/64 | 360/381 / 617/651 | 27.34% / 27.53% | +1.272 / +1.426 |

The full census has 13 common cells for GSM8K and 14 for MathQA. Some observed cells contain only one sampled start. Keeping the observed support fixed makes **7,965/10,000 GSM8K** and **8,726/10,000 MathQA** bootstrap LONG-minus-SHORT draws undefined. Conditional intervals are retained in the machine-readable summary, but **must not establish a standardized difference**. These descriptive comparisons do not establish that LONG is uniformly more repairable or a better causal recipient of training weight. [Cell table](STANDARDIZED_CELLS.csv), [all rates/intervals and undefined draws](SUMMARY.json).

## Training-anchor coverage

The retained ordinary KPLUS event 5/6 training logs on both datasets contain 1,024 response presentations, sampled anchor positions and aggregate K+ weights/counts. They do not contain current-model per-anchor proposal correctness or SHORT/LONG weight breakdowns. Aggregate immediate-successor priority does not recover the number of consecutive followers or mixed response/depth strata.

Therefore SHORT/LONG shares of native high-priority weight and mixed-stratum share of all selected weight are **unknown**. No new model forward was used to backfill them. Serving starts use a different anchor distribution. A later allocation candidate would first need to show that it actually moves a nontrivial amount of training weight. [Coverage and limits](TRAINING_COVERAGE.json), [inspected schemas](TRAINING_SCHEMA_INVENTORY.json).

## Native validity, work and costs

All **256/256 pairs are valid**, with zero invalid, failed, unknown or unattempted selected starts. One child block qualified for retained reuse because its complete prior native walk matched; 255 starts required reconstruction across 106 parent requests. A shared output offset was insufficient.

Every reconstructed state retains target-prefix KV and original feature-ingestion boundaries, rebuilds the destination draft cache through its native path, and preserves the original cap. All 255 neutral one-block replays reproduce every proposal, accepted prefix and committed output. All 510 reconstruction receipts report exact target-prefix KV; all 255 neutral draft-prefix KV checks pass. Every child emits the corresponding canonical prefix. No full child continuation is generated.

All 106 fresh parent walks match the archived full traces across 6,385 rounds, including final canonical tokens and termination. The postflight preserves the execution source, retained walks and all four draft exports. Admission checked 1,253 source Python files and 238 artifact identities under Python 3.12.14/Torch 2.9.1+cu130/Transformers 4.57.1.

| Work / elapsed time | Actual | Limit |
|---|---:|---:|
| Fresh parent-walk rounds | 6,385 | Included below |
| Neutral / child blocks | 255 / 255 | Included below |
| New native rounds | 6,895 | Included below |
| Startup draft / standalone verifier blocks | 3 / 48 | Included below |
| Total counted blocks | **6,946** | **8,000** |
| Occupied GPU time, including loading/publication/rebuild | **625.188s (10m25s)** | **2,700s (45m)** |
| Wrapper wall, including CPU admission/postflight | **668.386s (11m08s)** | — |
| Optimizer steps / backwards / candidate training | **0 / 0 / 0** | **0** |

The ledger also records 213 initialization/publication operations, 7,049 target calls and 29,781 context-forward operations. These are overlapping categories. Their host-operation clocks are not GPU kernel timings, and this diagnostic provides no serving-latency comparison. [Costs and work accounting](REPLAY_COSTS.json), [complete work ledger](WORK.jsonl.gz), [KV receipts](KV_PARITY_RECEIPTS.json).

CPU census took 0.504 s, child binding 0.239 s, final statistics 0.330 s and separate arithmetic audit 0.292 s. Preparation, tests, transfers and report writing are outside occupied GPU time. The initial end-to-end estimate was 2–3 hours with a 45-minute native ceiling; actual native work finished in about 10.4 minutes.

A800 admission had 665.201 GB free against a 2 GiB output allowance plus 8 GiB reserve. Both GPUs are idle afterward. All 985 selected evidence files / 41,673,058 bytes were retrieved and hash-verified; original raw outputs and runtime logs remain on A800. Only a verified redundant Fenrir PR39 transfer archive was removed during preparation; its 5,014 members matched the retained unpacked evidence, and no replay capability was lost. [Retrieval](RETRIEVAL.json), [cleanup receipt](STORAGE_CLEANUP_FENRIR.json).

## Review and decision boundary

23 focused CPU tests pass locally, 18 native-adapter tests pass on A800, and 57 inherited constructor/budget tests pass locally. A separately coded raw-block/census/sample/denominator audit passes 4,667 assertions and reproduces the primary intervals; an additional full-parent audit passes 6,915 checks. Review is same-author: delegated agent execution was unavailable because of the platform’s thread limit. Separate arithmetic code is not independent human review or another native replication. The 11-category ARS interpretation check is retained. [Arithmetic audit](ARITHMETIC_AUDIT.json), [parent audit](FRESH_PARENT_AUDIT.json), [implementation review](IMPLEMENTATION_REVIEW.md), [interpretation checks](INTERPRETATION_CHECKLIST.md).

**What this establishes:** on these two fixed checkpoint transitions, longer correct follower runs leave repeated, still-blocked opportunity, and actual repairs often realize their first two followers. The predeclared screen supports proposing one capped test within K+’s existing high-priority budget.

**What remains unestablished:** whether moving weight causes more useful repairs, how much training weight is movable, whether donor damage offsets the gains, and whether complete-request speed improves. Keep native K+ as the control in any separately authorized candidate test. No new mechanism or training run is implemented here. This diagnostic is complete and stopped; research stage remains unchanged.

## Material Passport

Origin skill: academic-research-suite / experiment-agent. Modes: run and validate. Origin date: 2026-11-11 UTC. Verification status: ANALYZED, with exact native parity and separate CPU arithmetic checks; no independent GPU replication. Version: longer_run_headroom_v1. Scientific execution and analysis commits are recorded above. Private prompts, canonical token arrays and runtime authentication data remain outside Git.
