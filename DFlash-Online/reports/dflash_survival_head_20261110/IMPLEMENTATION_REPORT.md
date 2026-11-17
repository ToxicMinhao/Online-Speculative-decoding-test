# Survival-head pilot implementation and scheduling

Execution code: `7f89719956f0256c0916a583d7ff615954cc66f9` ([PR37](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/37)); base PR36 `c64dfe3bf50d8bce84f2cca3576826c8d12a0087`.

The user approved CPU preparation now and a separately admitted GPU window after priority PR36. The user also explicitly permitted a scoped VAT reproduction exception for this pilot. The algorithm is a component adaptation, not a VAT reproduction. Original experiment settings and PR36 remain unchanged.

## Implemented comparison

Restore original complete GSM8K PR32 KPLUS4 at step512 independently for KPLUS, LOCAL and SURVIVAL. LOCAL predicts current per-slot correctness; SURVIVAL predicts consecutive correctness through each proposal. Both use the same scalar head, isolated seed17 initialization, coefficient1, FP32 response-wide BCE and separate BF16 AdamW/clipping. Native K+ and its response averaging remain unchanged. The head is absent from serving exports.

Main work is384backbone Adam steps,256head steps,768response forward/backward pairs, then768native completions across128prespecified requests and two counterbalanced sweeps. Qualification/startup and reconstructed teacher work are separate.

## Checks completed

- 83campaign/data/feedback/qualification/scheduling CPU tests passed.
- 21head/native-trainer CPU tests passed, including a complete synthetic tiny-model event-five trajectory.
- 47existing K+/objective regression tests passed.
- All128retained private canonical JSON identities and trace contracts validate; original256presentation and anchor records are unchanged.
- The independent input-only panel selection yields128requests, disjoint from feedback, with binding `5668f61d01166ffd310ce98d5eb449d13b46258f28ba2ad7aefa5f5661f2cc73`.
- Separate same-team reviews covered root orchestration and feedback reconstruction. Reviewers shared task context; this is not a blind external or cross-model review.

CPU checks do not establish native BF16 feature parity, GPU memory fit or performance. No new pilot GPU execution is claimed in this report.

## Reuse and missing artifacts

Original KPLUS4 recovery/export and all128feedback answers/native traces remain. Only two original teacher packets survive;126were retired. Ordinary PR36 KPLUS5 recovery/export was also retired after its successor, so allthreepilot arms will train. These artifacts are unavailable, not archived.

The new teacher reconstruction walks saved SGLang prefill/verification inputs without drafting or answer regeneration. It first compares both surviving packets bitwise, including terminal rows. Only after both pass may it reconstruct126missing packets. Planned work:128prefills +4,776verification forwards +128terminal-row forwards =5,032target forwards. First-two qualification costs58of these; startup is recorded separately. Old KV tensors and126old full packets are unavailable; no historical KV-byte or missing-packet-byte equality is claimed.

New redundant capture/terminal tensors may be removed only after exact retained-packet row checks and a durable retirement receipt. Original records remain intact. The pool and original schedule are sealed; only owned paths are rebound, never response order or sampled anchors.

## Scheduling, capacity and duration

PR36 has priority. A CPU queue waits for its successful final exit, then repeats exact-source, original-state, runtime, physical-GPU, storage and quiet-host checks. Training binds physicalGPU1; teacher reconstruction and evaluation bind physicalGPU0. Only one physical card is used at a time. Every timing block checks both cards and transfers.

The pilot peak budget is **59,916,015,137bytes (55.801GiB)**, including8GiBreserve, temporary state/export overlap, measured fixed-pool storage, serialization/5%tensor margins, compiler/workspace and raw records. This is not an admission based on current free space; actual capacity is checked after PR36.

Planning estimate after that window opens: **1.5–2.5hours** for reconstruction, bounded qualification, three training arms and evaluation. Reserve roughly45–60minutes of host-quiet time for endpoint serving. These estimates use retained historical costs and must be revised from native measurements; they exclude remaining PR36 work and network staging.

Qualification:4disposable backbone steps +2head steps;8case forwards/backwards,4extra auxiliary backwards,8discarded startup forwards/backwards and6native export checks. Main startup adds6forwards/backwards and0steps. No disposable endpoint initializes main training. Null/harmful valid results continue unchanged; any concrete correctness/resource failure stops with evidence. No automatic retry, tuning or further event.

## Interpretation

The primary contrasts are SURVIVAL versus KPLUS and LOCAL in complete-request latency, supported by paired calls and both sweeps. Report LOCAL versus KPLUS, clipping, BCE, memory/cost, and first-round prefix changes at identical starts. Later rounds contribute only to whole-request endpoints. Lower BCE alone is insufficient. A positive development screen only motivates a separately approved fresh confirmation; it does not establish broader robustness or training-inclusive acceleration.

Source reviewed: [VAT core](https://github.com/naver-ai/VAT/blob/124ad4d05dcf8336d91594122ed3f428bb931223/specforge/core/vat.py), commit `124ad4d05dcf8336d91594122ed3f428bb931223`. The pilot does not adopt VAT's CE/KD mixture, temperature, adaptive weighting, self-coverage or joint optimizer ownership.
