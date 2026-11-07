# A_repo58 training, exposure and cost audit

## Material Passport

ARS experiment-agent validate. **Verification status: ANALYZED.** The prescribed training and accounting checks passed. This is a read-only audit of completed native evidence, not an independent GPU replication. The auditor implemented the released-loss module and part of qualification; authorship overlap is disclosed.

Scientific execution: `7dac02b0fc6686d7129c5d29699ee9bfe5be2e7a`. The independent standard-library audit ran against the completed original A800 evidence at `/data/sources/minhao/runs/dflash_adaflash_arepo_20261107/production_001`, with output in the new `analysis_audits_001/training` directory. It loaded no model or tensor file, ran no forward/backward/optimizer, and changed no scientific artifact. Its compact result was retrieved as `REMOTE_AUDIT_001.json`; the first attempt passed, so no parser repair or scientific retry was needed.

**402,173 / 402,173 checks passed**, with zero failed checks and zero missing required files. A further 29 local independent arithmetic/inventory checks passed (`LOCAL_SUM_CHECKS.json`). After both saved-evidence audits, all **97,485 original compact files / 5,764,699,468 bytes** were rehashed against the sealed completed inventory: **zero changed files** (8.015 seconds; `../REMOTE_REHASH.stdout`, inventory SHA-256 `babe901503bdabdb8157a5038ba90b0bca398e95326b986eee4e8f6150b1f2ac`). Original remote byte identities are therefore validated. The complete Fenrir copy was still transferring at audit time; its whole-copy local identity verification remains pending. These are distinct states, and the pending transfer does not negate the completed original-evidence audit.

## Main trajectory accounting

| Quantity | Attempted | Completed | Unknown |
|---|---:|---:|---:|
| Own online responses | 7,473 | 7,473 | 0 |
| Terminal feedback-preparation forwards | 7,424 | 7,424 | 0 |
| Core response forwards | 14,848 | 14,848 | 0 |
| Core response backwards | 14,848 | 14,848 | 0 |
| Native Adam steps | 7,424 | 7,424 | 0 |
| Inherited first-event correctness generations | 4 | 4 | 0 |

All 58 events have 128 distinct feedback responses, two complete passes, 256 original response observations and 128 optimizer windows. Source identities do not repeat across events. Actual native serving records bind the sealed prompts to the appropriate pre-event checkpoint and the qualified settings. The final 49 requests use A_repo58 without feedback capture or another update. Each final state receipt reports 7,424 cumulative Adam steps and 14,848 presentations, with exact state/export consistency and empty accumulation. Initial receipts attest released weights, empty fresh Adam and the qualified seed-17 RNG. This analysis relies on those saved tensor-validation receipts; it does not repeat tensor-level restoration checks.

Six disposable qualification Adam steps remain separate. The earlier qualification's twelve core response forwards/backwards, two startup forwards/backwards with zero Adam steps, and six small loss-fixture forwards/backwards are not counted as main training. No baseline was retrained. The four native controls are outside the 12,288 endpoint timed completions. The full generation total is 7,473 + 4 + 12,288 = 19,765; endpoint validity is reviewed separately.

## Exposure and released-loss consistency

| Recorded quantity | Total |
|---|---:|
| Original response presentations | 14,848 |
| Active sampled anchors | 3,728,050 |
| Eligible supervised position occurrences | 54,368,391 |
| Capped normalized-position occurrences | 11,559,628 |
| Capped / eligible | 21.2616702% |
| Zero draft probabilities | 0 |
| Projected probability entries | 9,062,800,076,800 |
| Nonfinite reverse-KL entries | 0 |
| Teacher argmax / saved canonical disagreements | 616 |

Exposure sums use only each original response observation. The raw observation files match the observations embedded in event results; the optimizer-window records pair those observations correctly. Neither activation checkpoint recomputation nor the second appearance of metrics in accumulated optimizer rows is treated as extra supervised exposure.

For every presentation, the audit checked the full-response denominator against the CE ratio record; the separate CE numerator and already-normalized clipped reverse-KL component reconstruct the recorded late mixture within the predeclared 2e-6 FP32 scalar audit tolerance. The capped-position numerator and eligible-position denominator match the ratio metrics. This scalar reconstruction does not relax scientific qualification tolerances, nor does it independently reproduce missing activations. Source review and native qualification establish the actual BF16 operations and in-place normalization order; the final audit establishes retained scalar and accounting consistency.

Probability-zero counts cover **all projected vocabulary entries, including masked positions**. Cap counts cover **eligible supervised position occurrences**. They are different denominators. All 54.37 million supervised occurrences are overlapping sampled training positions, not independent served tokens. The 616 teacher-argmax/canonical differences are 0.001133% of supervised occurrences. The literal released CE uses teacher argmax as specified; these differences must not be relabeled as serving output mismatches or assigned an unmeasured cause. Scalar loss diagnostics and optimizer-window gradient norms are finite in the retained records.

No probability zeros were recorded in this run. That observation does not prove the literal BF16 `softmax`/`F.kl_div` path can never underflow or have nonfinite gradients; qualification's CPU boundary fixture remains relevant. No stabilization was introduced.

Recorded model work: 14,848 draft forwards, 74,240 layer forwards, 944,660 objective chunks and 1,889,320 vocabulary-head calls. These instrumentation counters have different scopes and are not interchangeable with response presentations. All events report zero reference operations; A_repo introduces no reference model or K+ selection mask.

## Fresh adaptation costs

| Timer | Total seconds | Interpretation |
|---|---:|---|
| Response training | 19,164.221 | Native training and included compilation |
| Whole update calls | 22,170.554 | Encloses preparation, training, persistence, export, activation and commit |
| Terminal teacher preparation | 288.257 | Inside update calls |
| Feature persistence | 338.837 | Inside update calls |
| Checkpoint persistence | 1,094.266 | Inside update calls |
| Serving export | 808.707 | Inside update calls |
| Host checkpoint-activation wrapper | 89.722 | Inside update calls |
| Native activation subcomponent | 19.286 | Inside host activation; not additional |
| Publication validation | 1,401.258 | Diagnostic correctness work |
| Event retirement | 536.867 | Validated bulk retirement |
| Complete trajectory worker | 31,668.327 | Feedback serving, adaptation and overhead |

Training seconds independently sum the per-response training timers; cost records agree with event results and published training summaries. A further `math.fsum` check of ten cost fields passed. **These timer rows overlap and must not be added together.** Teacher-head work totals 5,935.429 seconds and is already inside response training. Compiler timers also nest within training. Aggregate timing does not supply a detailed kernel-time explanation.

Peak allocated/reserved training memory was 20,674,401,280 / 23,976,738,816 bytes (20.674 / 23.977 decimal GB). Recorded resource observations introduce zero extra forwards, backwards or optimizer steps. The qualification maximum and runtime remain separate from the main trajectory's measurements.

Retained O58, K+58 and paper-A58 training costs are historical context, not fresh matched overhead measurements. Nothing in this audit establishes training-inclusive deployment speedup or equal-compute adaptation.

## Retention and remaining capability

All 58 event-retirement receipts completed with exact plan/removal agreement and consistent byte sums. Across events they retired:

- 7,424 prepared feature packets;
- 57 superseded recovery-state files;
- 57 superseded serving weight files.

Those event-retirement entries total **599,331,995,491 bytes** written and retired across the trajectory. This is cumulative file volume across many events, **not disk space freed at one time** and not the peak storage footprint. Separately, the completed remote inventory contains 7,424 preparation-retirement receipts and 7,424 exact-equivalence receipts for the native/terminal-to-prepared feature conversion. Their tensor contents were not reloaded by this audit, and their deletion bytes are not silently added to the event-retirement total.

The completed remote inventory retains the released initial state, the final `trainer/training_state.pt`, event-58 recovery state and A_repo58 serving export. The final two recovery paths may refer to the same retained hardlinked state; this audit does not add them as independent capacity requirements. Retired intermediate states and feature tensors are **deleted, not archived**: direct intermediate-event optimizer continuation and feature-tensor replay are unavailable. Compact source/configuration, exact output/trace evidence, metrics, validation and retirement receipts remain.

The report must distinguish the earlier capacity-admission failure from the successful authorized operational continuation. The former does not describe the completed campaign outcome.

## Interpretation review

All eleven ARS checks were considered: Simpson's paradox (per-event records accompany aggregate sums), ecological inference (occurrences are not independent requests), filtered-sample/Berkson bias (all prescribed events), collider bias (no conditioned efficacy model), base-rate neglect (separate cap/probability denominators), regression to the mean (no efficacy estimate from losses), survivorship (58/58 events and zero unknown work), look-elsewhere (fixed full accounting), forking paths (unchanged source and protocol), unsupported causality (no isolated-clipping or training-inclusive benefit claim), and reverse causality (causal serving-before-training order checked). No p-value or generalization claim follows from this audit.

The supported conclusion is that the completed A_repo trajectory reconciles with the fixed scientific budget, own-feedback chronology, loss observations and cost records. Endpoint benefit and its uncertainty must be judged from the separate paired serving audit.
