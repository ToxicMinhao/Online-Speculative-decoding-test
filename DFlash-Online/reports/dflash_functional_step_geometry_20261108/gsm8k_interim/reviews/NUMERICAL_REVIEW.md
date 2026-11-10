# Independent GSM8K endpoint arithmetic review

Verification Status: **ANALYZED — retained numerical results reproduced on CPU.** Same-team review with authorship overlap: the reviewer previously contributed numerical helpers, CPU tests and pilot sealing. Raw structural validation reuses the qualified native-record validator; pairing, aggregation, bootstrap and origin comparison are independently coded in `reproduce.py`. No model/GPU rerun was performed.

Execution: `6440e8ff5af1a845f830d65c884f6711d7e4a687`. Binding: `8ec5fed0d4ed9b40dab7c26d5089cd42b01f357ab0379193da42baa93f0b3dc5`. Authenticated snapshot: 6,341 files / 314,544,756 bytes. This review independently rechecked 1550 consumed current-snapshot inputs against its identities and verified they remained unchanged. Historical origin inputs are separately checked against the original remote inventory.

## Confirmed endpoint

All **256 requests × 3 checkpoints × 2 repeats = 1,536 executions** validate. Exact canonical prompts, output token IDs, termination, checkpoint labels and serving settings match. Counts agree across repeats. Raw trace sums reconstruct the recorded calls, matched accepted tokens, native accepted tokens and committed progress. Each method produces 80,493 output tokens, with one capped request. Committed progress is 80,237 because the initial token of each request is excluded from verification progress.

FSG6 uses **6.107505% more calls** and **6.255246% more complete-request time** than KPLUS6; both paired intervals exclude zero. It also underperforms SCALAR6. SCALAR6's small apparent improvement over KPLUS6 remains unresolved by the paired intervals; this is not evidence of equivalence.

| Method | Sum of within-request mean time (s) | Mean time/request (s) | Calls | Matched accepted/call | Progress/call | Output tokens | Capped requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| KPLUS6 | 205.468576 | 0.802612 | 12,018 | 5.696206 | 6.676402 | 80,493 | 1 |
| FSG6 | 218.321140 | 0.852817 | 12,752 | 5.310696 | 6.292111 | 80,493 | 1 |
| SCALAR6 | 205.037496 | 0.800928 | 11,975 | 5.720084 | 6.700376 | 80,493 | 1 |

| Contrast | Metric | Reduction and 95% interval | Increase and 95% interval | Request wins/regressions/ties |
|---|---|---|---|---|
| FSG_vs_KPLUS | verification_calls | -6.107505% [-8.532407, -4.375513] | +6.107505% [+4.375513, +8.532407] | 51/175/30 |
| FSG_vs_KPLUS | request_latency_s | -6.255246% [-8.843421, -4.435203] | +6.255246% [+4.435203, +8.843421] | 65/191/0 |
| FSG_vs_SCALAR | verification_calls | -6.488518% [-8.918225, -4.777069] | +6.488518% [+4.777069, +8.918225] | 39/187/30 |
| FSG_vs_SCALAR | request_latency_s | -6.478641% [-9.089769, -4.675150] | +6.478641% [+4.675150, +9.089769] | 53/203/0 |
| SCALAR_vs_KPLUS | verification_calls | +0.357797% [-0.084081, +0.809171] | -0.357797% [-0.809171, +0.084081] | 105/88/63 |
| SCALAR_vs_KPLUS | request_latency_s | +0.209803% [-0.279643, +0.691298] | -0.209803% [-0.691298, +0.279643] | 136/120/0 |

Reduction is `100*(1−candidate/reference)`; increase is its negative with interval endpoints reversed. The original production summary's bootstrap points and intervals reproduce **exactly**. The preliminary +6.26% time / +6.11% calls are correctly rounded versions of the reproduced results, with no substantive discrepancy.

Bootstrap: 10,000 paired-request resamples, PCG64 seed 20261108, linear 2.5/97.5 percentiles, identical sampled request indices for every contrast and metric. Two repeats are averaged within request; they are not treated as independent samples. Intervals are unadjusted, conditional on this continuing seed and reused panel.

Sweep complete-request totals (KPLUS/FSG/SCALAR): sweep 0 = **205.605848 / 218.184243 / 205.078327 s**; sweep 1 = **205.331303 / 218.458037 / 204.996666 s**. FSG's regression occurs in both sweeps. Full native-generation and cache-maintenance totals are retained separately in `ENDPOINT_INDEPENDENT.json`; those components must not be summed again into complete-request time.

## Authenticated origin count comparison

KPLUS4 is a valid count-only reference: all 512 retained KPLUS4 raw records match the exact current panel and canonical outputs/termination, both repeats agree, model/runtime pins and serving settings match, and the historical export identity equals the pilot's original KPLUS4 initialization. Historical source commit: `cdb640543d8fe3a43c8e63af4ccaaa002e09a599`.

KPLUS4 used **12,116 calls**, with 68,361 matched accepted tokens, 68,121 native accepted tokens and 80,237 committed progress. No historical timing is used.

| Endpoint | Calls | Call reduction versus KPLUS4 | Wins/regressions/ties | Gross calls saved/added |
|---|---:|---:|---:|---:|
| KPLUS6 | 12,018 | +0.808848% | 120/107/29 | 353/255 |
| FSG6 | 12,752 | -5.249257% | 69/166/21 | 159/795 |
| SCALAR6 | 11,975 | +1.163750% | 119/93/44 | 376/235 |

FSG therefore shows **actual deterioration in drafting efficiency relative to its common origin on this panel**, not merely less improvement than ordinary KPLUS. This does not mean target output quality degraded: output tokens and termination remain identical. Origin count differences are descriptive; no additional origin hypothesis test or historical timing comparison was introduced.

## Interpretation limits and accounting

The endpoint establishes a GSM8K regression for this trained FSG6 checkpoint. It does not establish whether weighting, off-fit transfer, geometry, precision or preservation caused it; those hypotheses require the separate retained-window analysis. Lower protected KL is not an endpoint success criterion. MathQA and the complete two-dataset conclusion remain pending.

No feedback, teacher preparation, optimizer steps, forwards/backwards, serving generations or diagnostic windows were added. Main independent endpoint/origin arithmetic took **2.415 s** on CPU, plus file-identity verification and report preparation. Input hashes, complete per-request/repeat tables and exact origin counts remain in this private review directory.

ARS statistical fallacies: **11/11 checked**.

- **Simpson:** GSM8K only; no pooling with pending MathQA or historical timing.
- **Ecological:** Pooled effects accompanied by request-level wins/regressions/ties; not all requests benefit or regress.
- **Berkson:** Fixed reused development panel; no inference of population-wide or fresh-data generalization.
- **Collider:** No conditioning on speed improvement or chosen favorable subset; all 256 requests retained.
- **Base_rate:** Denominators: 256 paired requests, 1536 executions, one deterministic trajectory per method/request.
- **Regression_to_mean:** No selection or checkpoint replacement based on this endpoint; unchanged original origin retained.
- **Survivorship:** All completed GSM8K records included; MathQA explicitly pending, not silently omitted from full-study conclusion.
- **Look_elsewhere:** All three declared contrasts and all bounds reported; intervals unadjusted, no familywise claim.
- **Forking_paths:** Original 10000 resamples/PCG64 seed20261108/linear percentiles/shared indices preserved; no new thresholds or tuning.
- **Correlation_causation:** Endpoint contrasts establish observed drafting effects for fixed trajectories; do not isolate a mechanism or explain why.
- **Reverse_causality:** Continued updates precede endpoint serving; sparse training diagnostics cannot be promoted to established causal explanation.
