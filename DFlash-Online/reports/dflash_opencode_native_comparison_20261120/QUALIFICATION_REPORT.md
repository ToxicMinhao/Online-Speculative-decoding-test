# CODE qualification retry: exact-output gate failed

## Material Passport

ARS experiment-agent execution supervision and retained-evidence review. Actual execution: `93a53f11abbfbf00e36c1bcc309a33943f972099`. Report revision: its containing Git commit. **Qualification is incomplete; O/AREPO/C production training has not started.**

## Outcome

The authorized AR startup repair passed on both A800s. Actual engine receipts confirm ordinary CUDA graphs enabled, experimental piecewise graphs disabled, and the pinned BF16/FA3 target-only configuration. The original failed execution `9a7eabe7e3539410ce995a17d7407ea4f8fd495c` and its costs remain preserved.

The retry then stopped at the required exact AR/F output check on reserved request index0, on physical cardB:

| Measurement | Target-only AR | Frozen DFlash |
|---|---:|---:|
| Input tokens | 65 | 65 |
| Output tokens | 290 | 292 |
| Termination | EOS | EOS |
| First differing token ID, at zero-based output index279 | 11 | 323 |
| Verification calls | N/A | 30 |

The prompt token IDs, output cap2048 and stop IDs are identical. Outputs agree for279 tokens before diverging. AR also produced identical tokens and termination for this request on physical cardA. Thus this observed mismatch is between same-card AR and DFlash; cross-card AR variation does not explain it.

The saved DFlash trace contains the differing token as its native target prediction at the first divergence. The adapter's input/termination/trace checks pass; the additional canonical-output check rejects it. This localizes the discrepancy but does not establish its underlying cause. Numerical differences between autoregressive and block-verification execution are a possible explanation, not a demonstrated diagnosis. No new logits, model forwards or serving requests were run to investigate it. Neither recipe performance nor correctness can be inferred from this stopped attempt.

## Work and cost accounting

Of64 planned qualification completions:12 were attempted,11 returned, one remains unknown after peer termination, and52 were unattempted. Ten returned executions passed the checks reached; one returned DFlash execution failed canonical equality. This is **not** ten fully qualified requests or a passed qualification gate. All training initializations, teacher preparations, response-training forwards/backwards and Adam steps remain zero.

The launcher stopped the other card after the mismatch. That worker was forcibly terminated while its ninth AR request was in flight; no completed record survives for that request. The global pending entry is retained as unknown, without replacing it or treating it as completed. Its stale card lease was closed using the saved owned-process cleanup timestamp after confirming the process was absent and both GPUs were idle. The pre-reconciliation clock and the correction receipt remain available; the elapsed occupancy is a conservative upper bound through cleanup completion.

Retry launcher wall time was108.879 seconds. Including the original failed startup, cumulative occupied-card time is193.239 seconds. The original first-model-load timestamp is unchanged. Original piecewise initialization still has an unknown additional forward count; its six recorded ordinary initialization forwards are not claimed as the complete startup total. No budget clock or work counter was reset.

## Storage and retained evidence

A800 has493.410GB free after the stop; both GPUs are idle. The authorized90GB additional-storage ceiling and400GB reserve remain unchanged, with the86.531GB forecast preserved. No model or checkpoint was deleted. Earlier cleanup removed194.314MB of verified redundant historical transport archives and the verified new source-transfer archive.

The private retry-failure bundle is95,437bytes, with87 files and11 completed generation records. Its transferred SHA256 and member inventory were verified. Raw prompts, responses and traces stay outside Git. Public summary: [QUALIFICATION_RETRY.json](QUALIFICATION_RETRY.json). Private source files are retained under the A800 study root, its `attempts/failed_startup_001` archive, and Fenrir's `failure_002` directory. The latter also includes `CLOCK_BEFORE_RECONCILIATION.json`, `CLOCK_RECONCILIATION.json` and delivery verification.

## Remaining blocker

The handoff explicitly makes AR/F disagreement a blocker for this combined study. The run therefore remains stopped: no request replacement, trimming, changed precision, relaxed equality, training or automatic retry. A bounded investigation at the retained common prefix would be needed to distinguish native numerical divergence from an implementation discrepancy before choosing a repair or requesting a protocol revision.

There is no completion-time estimate while this correctness blocker is unresolved. The previous35–45hour estimate remains provisional for the full campaign and is not a promise that it fits the48-hour cap; successful qualification, the full AR/F precheck and measured admission are still required.
