# Frozen event-seven training-repair replay: execution pending

## Material Passport

- Status: implementation and CPU checks complete; A800 staging is blocked by SSH connection failures.
- Reviewed implementation: `10d89bf8ba657cd905d7ad4fb31bf65a0a851656`.
- Execution commit: **none**. No model forward, backward, optimizer step or generation has run for this diagnostic.
- Base: PR44 publication `ae4d7f1ea2eca46a967b4484931c69fbb17c35a0`. Its completed scientific results remain unchanged.

## Authorized comparison

Freeze NATIVE7 and RECENT7 at step 896 and replay the same original 256 training presentations per endpoint. Preserve native BF16/Flex Attention, block 16 including seed, eight-block vocabulary projections, full geometry, original teacher features and the K6 sampler RNG. Verify installed endpoint weights before and after replay.

The predeclared population is 128 requests, 815 slots and1629 paired occurrences: 814 in pass 0 and 815 in pass 1. The singleton `train_00786`, start 91, slot 13 remains eligible only in pass 1. The specified `train_00768`, start 3 sanity case has slot 7, parent margin 0.75 and floor 0.375.

Primary: RECENT7 minus NATIVE7 accepts-through percentage points after averaging eligible presentations within each slot and weighting 815 slots equally. Report native headroom, rescues/losses, 3×3 outcomes, direct correctness, margins and pass checks. The interval resamples 128 whole requests 10,000 times using PCG64 seed 1707 and each draw's slot denominator. This is fitted retention, not free-running acceptance or serving speed.

## Implementation evidence

- 28 focused CPU tests pass: authenticated historical joins, prefix outcomes, EOS/ties, singleton weighting, pairing/normalization, partial final chunks, no-tag forwards, endpoint tensors and shared-component integrity, natural sampling/RNG restoration, failure accounting and statistical reduction.
- Separate source review found no remaining protocol mismatch. An independent exact-rational/bootstrap scalar auditor is prepared; three synthetic auditor tests pass. The final scalar audit cannot run until endpoint observations exist.
- Source and deployment archives prepared locally; no GPU launch attempted.

## Resource check and blocker

The initial successful A800 check found both physical A800s idle (0 MiB each) and 504,121,249,792bytes free. Fenrir had 55,843,102,720bytes free. No cleanup was needed or performed. Recheck before launch; require 2 GiB peak additional writes plus 8 GiB disk reserve.

Two later research-record transfers disconnected; two fresh SSH connections timed out before receiving the server greeting, including a connection without shared-session reuse. The user was asked whether a fresh login currently works. No scientific attempt crashed or needs retry authorization: execution has not started.

Expected scientific work: 512 forwards, 16,340 head chunks, zero backward/optimizer/generation. Based on PR44's 103.909 s one-endpoint frozen preparation, budget **5–10 minutes of A800 execution** once staging succeeds, with a 30-minute safety timeout. Implementation/staging/report time is separate and network delay remains unknown.

## Reproduction

From the pinned qualified training environment and source checkout:

```bash
python -m experiments.frozen_training_repair_replay.campaign --spec /path/to/private/RUN_SPEC.json
```

The private spec names the authenticated original execution, identity tables, source manifest, original receipt hashes and a fresh output directory. Preparation seals the occurrence join, input identities, endpoint identities and reduction contract before any new scores. Each arm runs in a fresh process; failures preserve work and do not retry automatically.

No fitted retention conclusion is available. The pending report will state native headroom, rescues versus losses and the supported explanation, while preserving PR44's serving result. No further training or tuning is authorized.
