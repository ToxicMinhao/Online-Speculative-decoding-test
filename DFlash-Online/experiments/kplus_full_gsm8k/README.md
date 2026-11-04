# K+ on the full GSM8K online stream

This is the fixed November 3, 2026 Track 1 experiment. It trains one fresh
seed-17 K+ arm through 58 events and compares its final checkpoint with the
retained K58 checkpoint. It does not retrain K, resume Track 2, or reuse retired
K feedback features. Each recipe learns from its own online serving trajectory.

The scientific reference execution is `0698eeb2ddd97b9e2ac6953989dc82acbc0a1e34`;
the source/report parent is `4d94f30c42701d0f86ac04e2829c834593c16d8e`.
The sealed plan records this implementation's exact execution commit. Reports
must distinguish that commit from later documentation-only publication commits.

## What changes

Native K selects each block's correct prefix and its first error for forward KL.
K+ keeps that selection, the correct-prefix weights, and the response denominator.
It only redistributes first-error weight within the current response. A first
error has priority 2 when its immediate valid successor is already correct;
otherwise it has priority 1. Successor agreement does not select that successor
for training. EOS, gaps and the end of a block close the lookahead.

For original first-error weights `w` and priorities `v`, K+ uses
`w' = w * (0.5 + 0.5*v/weighted_mean(v))`. Across the response, first-error mass
is conserved. Equal-priority and error-free responses take the original native
K numerator directly. For mixed responses, the additive chunk statistics give

```
N_new = N_K + 0.5*(W1*N2 - W2*N1)/(W1 + 2*W2)
loss  = N_new/(S + 1e-6)
```

`online_feedback/kplus.py` implements this small modifier. The objective reuses
its existing BF16 KL rows and current argmax predictions. The two `N` statistics
remain differentiable; memberships and weights are detached. Reduction is once
per response after all eight-block projection chunks, with no second model pass
or additional full-vocabulary projection. The two separately normalized
response losses retain the native averaging/accumulation rule.

## Fixed execution

The pinned Qwen3-8B target and released Qwen3-8B-DFlash-b16, seed 17, native BF16
AdamW, constant LR `3e-4`, two shuffled passes, microbatch 1, accumulation 2,
512-anchor ceiling, gamma 7 and Flex training attention remain unchanged.
Training context is 2,272. GPU 0 serves; GPU 1 updates. Every update is committed
and published before collecting the next window. No asynchronous queue is used.

| New work | Fixed budget |
|---|---:|
| Ordered online requests | 7,473 |
| Feedback responses / terminal preparations | 7,424 / 7,424 |
| Events / AdamW steps | 58 / 7,424 |
| Response presentations / forwards / backwards | 14,848 each |
| Included native correctness generations | 4 |
| Final K58/K+58 generations | 6,144 |
| Total complete-response generations | 13,621 |

The remaining 49 online requests are served without an update. The original
1,024-request evaluation panel is reused unchanged, with three fresh repeats.
Within each of 16 blocks of 64 requests, K58 runs first when `(block+repeat)%2`
is zero. Precedence is balanced 24/24. Timed serving has no training or transfers;
loading and publication are separate. Greedy BF16 block-16 decoding, concurrency
1, FA3, full cache flush, context 4,096 and 2,048 tokens or EOS remain pinned.

Every endpoint output and termination must match its canonical record. Counts
must repeat exactly, and fresh K58 must reproduce every retained request's
counts (43,387 total calls; 314,847 output tokens). Invalid records are preserved
and stop execution. Valid unfavorable results do not stop or change the run.

## Qualification, storage and evidence

CPU fixtures compare the explicit weighted formula with chunked loss and
gradients, including mixed classes across chunks, boundaries, conserved mass,
and exact native neutral branches. Tolerances are fixed in the implementation
and qualification receipt before production. There are no disposable model
training steps or efficacy warmups. The four inherited native checks are part
of the fixed production budget.

Preflight authenticates released initialization, the empty original Adam/RNG
state, K58, manifests/canonical tokens, models and runtimes. Capacity includes
initial state, current/pending model-and-Adam states, old/new exports, the full
128-response buffer, one packet's preparation overlap, compiler growth,
compact evidence, workspace, and an 8 GiB reserve. No planned offload is treated
as free space. During this new run, durable prepared packets replace redundant
raw tensors; after validated publication, obsolete event tensors/state/export
copies are retired with exact deletion receipts. Initial/final states, final
export and all compact provenance/validity/observation records remain.

`seal.py` creates the immutable source/config/input plan. `preflight.py` checks
it before models are constructed. `campaign.py` uses the existing native
controller and trainer. `evaluation.py` records fresh paired executions;
`statistics.py` applies the prespecified ratio-of-sums analysis. `report.py`
returns raw repeat rows, paired request rows, event exposure/activity/costs and
explicit incomplete-work accounting. Historical two-arm guards are unchanged.

The primary comparison is `100*(1-sum(KPLUS_calls)/sum(K_calls))`, counting one
validated trajectory per request. Serving-time reduction uses summed request
means across three repeats. Both use 10,000 paired request bootstrap draws,
PCG64 seed 20261102 and linear 95% percentiles. A positive lower call bound
supports drafting improvement here; a serving-time claim additionally requires
a positive lower time bound. Otherwise K remains the incumbent. One fitted seed
and a reused mathematics panel do not establish broad robustness or an
asynchronous/training-inclusive speedup. No automatic tuning or rescue run follows.

## Saved-evidence reporting recovery

The original production run completed its model work but its aggregate audit
used the evaluation directory as the root for native sidecars. Both callers now
pass the production root explicitly. This changes file resolution only.

For that preserved missing-path failure, `report.py --recover-completed-audit
--output <new-directory>` permits CPU-only report recovery. It requires complete
training, all 6,144 saved executions to pass the ordinary audit, the original
failed exits and audit, successful resource/timing gates and postflight. It
retains the original campaign status and failure hashes in the new report.
An incomplete run or a different failure does not qualify. Recovery creates no
model, optimizer, generation, or replacement scientific record; execution and
reporting commits must be reported separately.
