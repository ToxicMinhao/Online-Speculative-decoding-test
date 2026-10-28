# Token-level projected-target walkthrough

This deterministic CPU diagnostic uses the deployed target constructor and actual `TargetObjectiveModel._chunk_loss`. An identity head exposes four freely adjustable FP32 logits. It is not DFlash model training; no checkpoint, backbone, production optimizer or source was modified. No A800 operation occurred.

ARS experiment supervision and an independent mathematical review were used. Constructor/loss source was byte-compared with the exact bundle previously installed on A800.

## Fixed targets and changing predictions

For the same sampled position and context, the pre-event reference q0 determines r once. Learning changes the current logits and probabilities q, not r. Different sampled positions or contexts may have different targets; there is no global r attached to a token ID. Production recomputes targets from the same reference to avoid retaining vocabulary tensors.

## Projection examples

All vectors use token order (A, B, C, D), with c=1.1.

| Case | Correct token | Reference q0 | Fixed r |
| --- | --- | --- | --- |
| clear_error | C | (0.400000, 0.350000, 0.050000, 0.200000) | (0.258065, 0.258065, 0.283871, 0.200000) |
| near_miss | B | (0.360000, 0.340000, 0.200000, 0.100000) | (0.333333, 0.366667, 0.200000, 0.100000) |
| already_correct | B | (0.150000, 0.600000, 0.150000, 0.100000) | (0.150000, 0.600000, 0.150000, 0.100000) |
| tie_selected | A | (0.400000, 0.400000, 0.100000, 0.100000) | (0.400000, 0.400000, 0.100000, 0.100000) |
| tie_not_selected | B | (0.400000, 0.400000, 0.100000, 0.100000) | (0.380952, 0.419048, 0.100000, 0.100000) |

Ties use the lowest token ID on original logits. A tie already selecting the correct token is preserved without imposing a positive margin; another tied target is repaired.

## Main learning trajectory

The illustrative update is `z <- z - 1.0 * gradient_z softCE(r, z)`, repeated 100 times on free logits. It does not use the real BF16-backbone/FP32-master AdamW training recipe; its step size and convergence speed must not be transferred to DFlash. The production loss is used with a single position of weight one.

| Step | q(A) | q(B) | q(C), correct | q(D) | Selected token | Soft CE |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.400000 | 0.350000 | 0.050000 | 0.200000 | A | 1.679673 |
| 1 | 0.373394 | 0.343471 | 0.067965 | 0.215169 | A | 1.600532 |
| 2 | 0.352329 | 0.333938 | 0.089315 | 0.224419 | A | 1.536820 |
| 5 | 0.306997 | 0.302075 | 0.163633 | 0.227294 | A | 1.423821 |
| 10 | 0.270052 | 0.269253 | 0.249981 | 0.210714 | A | 1.381451 |
| 12 | 0.264368 | 0.263949 | 0.265080 | 0.206603 | C | 1.379368 |
| 20 | 0.258437 | 0.258401 | 0.282276 | 0.200886 | C | 1.378475 |
| 50 | 0.258064 | 0.258064 | 0.283871 | 0.200001 | C | 1.378469 |
| 100 | 0.258065 | 0.258065 | 0.283871 | 0.200000 | C | 1.378468 |

Target r stays (0.258065, 0.258065, 0.283871, 0.200000) at every step. C first becomes the highest-probability token at step 12 in this synthetic trace. Soft CE approaches the nonzero entropy of r, not zero. Raw hard-CE and soft-CE losses are not comparable learning-quality scores.

![Token learning](token_learning.png)

## Already-correct predictions and restoring gradients

An already-correct reference `(0.15, 0.60, 0.15, 0.10)` is preserved. Starting there gives zero logit gradient to numerical precision. A separate synthetic probe manually disturbs current probabilities to `(0.35, 0.35, 0.20, 0.10)` while keeping the original reference and r. The initial gradient is approximately `(0.20, -0.25, 0.05, 0.00)`, and descent returns toward r. This checks the restoring loss term; it does not demonstrate protection under shared-backbone training.

In the main case D has r(D)=q0(D)=0.20 and initially zero logit gradient, but q(D) temporarily rises because other logits change the softmax denominator. An unchanged target probability does not guarantee an unchanged current probability at every step.

## Validation

All six cases passed. Maximum difference between production autograd and q-r: **7.98e-08**. Targets remain detached and exactly unchanged across all 101 observations per case. Probability/margin atol: 2e-6. Gradient atol: 2e-7; rtol: 2e-5. The handoff worked example is checked explicitly. Final maximum distance to r is below 2e-6 in every case.

| Case | First correctly selected step | Final maximum probability error |
| --- | --- | --- |
| clear_error | 12 | 2.98e-08 |
| near_miss | 2 | 1.27e-07 |
| already_correct | 0 | 0 |
| tie_selected | 0 | 1.49e-08 |
| tie_not_selected | 1 | 2.98e-08 |
| correct_but_disturbed | 1 | 2.53e-07 |

Files: [script](trace.py), [all 606 observations](trajectory.csv), [structured results](results.json), [log](run.log), [plot source](plot.py), [SVG](token_learning.svg).

Reproduce with:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=/home/minhao2/DFlash-Research/training /home/minhao2/.venvs/dflash-specforge/bin/python /home/minhao2/EAGLE_SPEC/research/projected_target_token_trace_20261022/trace.py
```

The model environment lacks matplotlib; existing system matplotlib rendered the saved JSON separately, without installation. One report-generation command had a quoting syntax error and was corrected; the numerical trace passed on its first execution.

These examples verify local construction and loss behavior. They do not establish source fitting by full DFlash, future-request transfer, verification-call reduction or serving speedup. No scientific configuration changed; native qualification and the campaign remain pending review.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/projected_target_token_trace_20261022/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
