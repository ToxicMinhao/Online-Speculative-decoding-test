# PR8 numerical repair: equivalence passes; N still blocks qualification

The direct-full-gradient repair passes its native comparison against unchanged H. The complete first-stream qualification **stops at N's unchanged 1% norm gate**. No campaign was started and no tolerance, precision, learning rate or optimizer behavior was changed to force continuation.

Executed code: `a1428029ba54a182fbf2d8f97d4ce17abf55bc0c`, [PR8](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/8). See the [revised numerical protocol](NUMERICAL_PROTOCOL_V2.md). The run used the pinned Llama-3.1-8B-Instruct / released DFlash pair, native BF16 and existing runtime. Only the planned first-three-request WildChat00 qualification was authorized. Controller wall time: **921.62s (15.36 minutes)**, including initialization, verification, temporary trials, training, persistence and cleanup; preflight is separate.

## Qualification results

| Requirement | Native result |
|---|---|
| Direct full-loss proposal, filtering disabled, versus unchanged H | **Pass: zero differences** in final weights, Adam first/second moments and actual cumulative displacement across 1,048,626,432 coordinates after two steps |
| Adam counters, response/position/block counters, sampled anchors and exposure | Exact match; each trial uses 7,848 supervised-position presentations and 880 eligible-block presentations across two passes |
| Sampler/global RNG and restoration | Exact match; initial model, optimizer, counters and RNG restored after four temporary steps |
| Historical gradient-sum comparison | **Still fails in all 8 recorded G/N presentations**, including two temporary disabled-filter presentations; zero nonfinite inputs. It is explicitly diagnostic under v2, not a passed old gate |
| Native output preservation | All 8 executed paired checkpoint scores have matching canonical input/output token IDs; H/L/G finish their planned sequences |
| N's actual/intended norm match | **Fails both steps of its first event**; N stops before request 2 and its paired scores |
| Recipe metadata | All 4 runtime receipts say 1 response, 2 passes, 2 optimizer steps; G/N record 3 backwards/step and 6/event; H/L 1/step and 2/event |
| Campaign | Absent; `campaign_started=false`, `campaign=null`; both GPUs idle after clean worker shutdown |

The first disabled-filter sum diagnostic reproduces 120,111 violations (0.0114541%) and maximum error/tolerance 15.06498. Its second pass has 83,274 violations. These observations coexist with **exact ordinary-update equivalence** because AdamW now receives the direct native full-loss gradient, while raw A/B gradients only determine the subsequent filter or N's own scale. Existing equivalence tolerances were retained, though the observed errors are zero. This is an explicit numerical-protocol change.

### N's unchanged BF16 control

| Step | Ordinary proposal norm | Intended filtered norm | Stored N displacement norm | Relative error | Limit |
|---|---:|---:|---:|---:|---:|
| 1 | 10.800965 | 7.737605 | 10.150817 | **31.1881%** |1%|
| 2 | 6.799957 | 5.077536 | 6.281747 | **23.7164%** |1%|

N used its own gradients/state, with scales 0.716381 and 0.746701. BF16 storage did not retain the intended norm match. The measurements and failed checkpoint/restart state were preserved. No N request 2/3 evaluation was performed. Beating this unqualified control would not establish an advantage beyond reducing update magnitude; no selective-filter efficacy claim is made.

## Backward cost and memory

G/N now compute A, B and full backwards from each shared forward: **three per step, six per response event**. H/L retain one per step. The additional direct full backward cost **0.282–0.336s per normal G/N pass**, or **0.564–0.672s per two-pass event**, in this run. These are synchronized phase measurements, not timing repetitions or deployment-speedup estimates.

| Arm/event | All backward phases (s) | Added full backward (s) | Complete instrumented training (s) | Peak trainer Torch allocation (GiB) |
|---|---:|---:|---:|---:|
| H/1 |0.753|—|3.110|31.692|
| H/2 |0.671|—|2.341|32.519|
| L/1 |0.827|—|2.951|31.692|
| L/2 |0.669|—|2.337|32.519|
| G/1 |1.697|0.564|20.186|42.481|
| G/2 |2.010|0.672|37.558|42.479|
| N/1 (control failed) |1.929|0.565|22.266|42.479|

Training time includes teacher preparation, diagnostics, state identity copying/hashing, gradient handling and optimizer work. Gradient-sum summaries alone took about 1.515–1.529s/pass. The remaining elapsed time is not assigned a detailed kernel/copy breakdown; it must not be attributed entirely to the extra backward. Export took 12.46–12.70s/event and optimizer-state persistence 21.21–23.93s/event, outside the training column. Complete execution wall time includes those costs once.

The native full-gradient buffer holds **2,097,252,864bytes (1.953GiB)** and is cloned before clipping. Overall G/N trainer peak allocation is 42.48 GiB versus 31.69–32.52 GiB for H/L; this is the complete pipeline difference, not the memory attributable only to the third backward. The running event peak did not rise during the measured full-backward phase, which releases saved graph state; that does not mean the added gradient costs zero memory. Reserved-memory peaks and live phase snapshots are retained separately.

The isolated matched reference trial used 2.935 s training and 31.692 GiB peak allocation; the disabled-filter grouped trial used 29.311 s and 42.480 GiB. Its complete qualification including snapshots/restoration/comparisons took 56.395 s. Ordinary-reference warm-up and grouped diagnostics differ, so these are observed implementation costs, not a controlled training-throughput benchmark.

## Scope, validity and evidence

Completed: 10 canonical generations, 8 valid scored replays, 7 qualification update events/14 steps, plus 4 temporary restored steps. H/L/G complete two events each; N completes one event and fails its control. The planned remaining N event and two scores were not run. No main-campaign update or generation occurred.

Validation: 164 local CPU tests and 47 focused tests on installed A800 Torch 2.9.1 pass. Independent source review confirms historical dependencies, H/L, loss/anchors/weights/conditioning, BF16/LRs, equivalence tolerances and `controls.py` are unchanged. 59 dependency files, 10 author files and 8 runtime files passed preflight checks. A separate evidence audit passed 300 assertions against the saved receipts and raw records. That audit validates recorded comparator results; the compact archive does not contain the full tensors needed to independently recompute every weight comparison. Numerical and output validity remain distinct from efficacy.

Compact evidence: 221 files verified by size and SHA-256 on Fenrir after retrieval. Archive SHA-256: `3f922f6992260272bbf94a3b5bb4d361b3996f330ad702f333e9611ee85f6538`. Private responses, token arrays, logs and restart states remain outside Git. The archive contains compact evidence, not all model/optimizer tensors. Failed N recovery stays on A800; completed-arm bulk retirement is recorded in each `FINAL_RETENTION.json`, so those retired states are not described as archived. No manual storage deletion was required. A800 had 35,232,116,736 bytes free after evidence collection; the 8 GiB reserve remains intact.

Private machine-readable derived tables retained with the execution include: `QUALIFICATION_UPDATES.csv`, `QUALIFICATION_PASSES.csv`, `DISABLED_FILTER_PARAMETERS.csv`, `GRADIENT_SUM_DIAGNOSTICS.csv`, `N_NORM_CONTROLS.csv` and `QUALIFICATION_STREAMS.csv`, with `QUALIFICATION_SUMMARY.json`. Raw evidence and those private tables are retained outside Git with the execution archive. All nested timings retain their original boundaries and must not be added twice.

**Decision:** the ordinary-proposal numerical repair is supported in the checked native case. Full qualification remains blocked by N's BF16 norm mismatch. This does not establish that agreement filtering helps or fails. Return these results for review; no further repair, retuning, qualification restart or campaign is authorized by this result.
