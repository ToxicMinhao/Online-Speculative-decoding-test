# PR49 event-seven collection: retrospective CPU analysis

**E7 was already worse than N7 on these 128 requests before E's own event-eight learning.** E needed 66 more verification calls and its pooled native acceptance was 1.726% lower. S7 saved only nine calls; its small favorable estimate remains unresolved. Native K+ remains primary, persistent E stays parked, and S remains a fixed control.

## Material passport and scope

- Status: completed CPU-only evidence reduction; statistical interpretation ANALYZED. Independent raw-reader agreement passed; no new model experiment or scientific replication.
- Collection execution: `db6e1221d8d1b234d9d07e246025d1afd5dc30d5`. Restart `af708012e729437068fbef93cb11c9996a9109a9` imported those collection tasks unchanged. Previous final-panel report: `27e307cbd29efb08106cacb16e7cfe2bbdd54738`.
- Analysis source: `97a3f90b1b80f0d8bdcfc1f1910aa7cd4dd378c9`; source and tests are in [experiments/pr49_collection_cpu](../../experiments/pr49_collection_cpu/README.md).
- Same origin and seed; retrospective reuse of the existing GSM8K training-stream collection. Stage 1 / project None unchanged.

The duplication check found existing collection validity/work audits, but no complete published paired collection reduction. The prior serving result compares final N8/E8/S8 on a different development panel. This analysis adds no responses or updates.

## What was compared and authenticated

N7 is native K+ at LR 3e-4; E7 withholds direct correct-prefix token-loss gradients while retaining the qualified remaining K+ logic at LR 3e-4; S7 is ordinary K+ with constant LR 0.96×3e-4. S is not a trajectory-wise norm match. Subject to the retained state/ancestry receipts, all derive from the same K6 recovery, event-seven prepared feedback and schedule; N7 is the parity-qualified reused native endpoint. The updates differ, so this does not uniquely diagnose optimizer history or gradient direction.

The fixed population is `train_00896`–`train_01023`, source IDs `gsm8k:main:train:896`–`1023`; native dataset IDs remain separately retained. Request indices are 897–1024, not evaluation-panel ordinals. All three arms have exactly 128 ordered requests, equal prompt token arrays and one fixed event-seven checkpoint throughout collection. The source creates one backend and contains no update/publication operation. Each collector finished before **its own** event-eight training; E8 training overlapped S collection. No global all-collection-before-all-training claim is made.

Each of the 384 outer records and independently persisted native JSON captures was authenticated through retained delivery/import manifests. Capture mode, target argmax labels, exact covered rows, proposals, committed outputs, stop flags and separate trace equality all pass. All 128 paired response token arrays and terminations match. The target/tokenizer/runtime remain qualified Qwen3-8B/DFlash, BF16/FA3, greedy, block 16, 2,048 output tokens or EOS, application context 4,096 and concurrency one. The engine reserves 4,112 positions for speculative slack; this does not enlarge the request context budget. Model files and native `.pt` tensors were not read; their existing receipts supply checkpoint identities.

## Counts and paired differences

Each arm produced **30,648 output tokens; 128 EOS responses, zero cap hits and zero zero-call outputs**. These outcomes were checked, not assumed. Fixtures also cover valid cap/zero-call outputs. V is verification calls, P committed progress, A native accepted proposals, and M the separate audit-only matching count.

| Arm | V | A | P | M | A / V | P / V |
|---|---:|---:|---:|---:|---:|---:|
| N7 | 4,402 | 26,118 | 30,520 | 26,234 | 5.933212 | 6.933212 |
| E7 | 4,468 | 26,052 | 30,520 | 26,164 | 5.830797 | 6.830797 |
| S7 | 4,393 | 26,127 | 30,520 | 26,245 | 5.947416 | 6.947416 |

The known prefill token is excluded: P = output tokens − 128. Each block contributes A = committed progress − 1; terminal clipping can make M differ. Hence A = P − V, and acceptance and call changes are algebraically linked—not two independent proofs.

Positive values favor the first named arm. Brackets are the specified paired-request 95% intervals.

| Contrast | Acceptance change, tokens/call | Relative acceptance change, % | Call reduction, % |
|---|---:|---:|---:|
| E / N | -0.102 [-0.182, -0.029] | -1.726 [-3.011, -0.485] | -1.499 [-2.650, -0.417] |
| S / N | +0.014 [-0.028, +0.058] | +0.239 [-0.473, +0.977] | +0.204 [-0.408, +0.829] |
| E / S | -0.117 [-0.192, -0.044] | -1.961 [-3.180, -0.750] | -1.707 [-2.805, -0.645] |

| Contrast | Fewer / more / tied calls | Gross saved | Gross added | Net saved |
|---|---:|---:|---:|---:|
| E / N | 38 / 61 / 29 | 73 | 139 | -66 |
| S / N | 44 / 42 / 42 | 61 | 52 | +9 |
| E / S | 35 / 67 / 26 | 67 | 142 | -75 |

All 128 rows, including regressions and ties, remain in [PAIRED_REQUESTS.csv](PAIRED_REQUESTS.csv); [PER_REQUEST.csv](PER_REQUEST.csv) contains all 384 arm/request rows. Pooled ratios use summed counts, not an average of request ratios. No favorable subgroup was selected and no differing verification-block boundaries were aligned across arms.

## Uncertainty and validity

Sort by sealed source ID, then request ID. Draw 10,000 common 128-request samples using NumPy Generator(PCG64(1708)); each request's three-arm record travels together. Recompute pooled quantities, then use linear 2.5/97.5 percentiles. NumPy version is 2.3.5; source-list and draw-matrix encodings/hashes are in [SUMMARY.json](SUMMARY.json). There were no undefined draws. The implementation retains undefined ratios explicitly and never drops or redraws them.

These descriptive intervals are conditional on this one origin/seed and reused collection population. They are not training-seed replication, multiplicity-adjusted confirmation, or a correction for prior project method selection. No new efficacy threshold is introduced.

Provenance checks: **5,177 passed**. Primary validation: **177,277 passed**, zero discrepancies, 384 rows / 128 complete pairs. A separately implemented raw reader saved its results before seeing the primary reducer; **112,536 raw checks** and **7,523 cross-implementation checks** pass. All integer identities/counts agree exactly; scalar/interval agreement uses declared 1e-12 absolute rounding tolerance. Both source-order and draw hashes agree. Same-family reviewers had source access: this is code/evidence audit, not independent scientific replication.

All 12 primary fixtures pass, plus five independent-reader fixtures. A test-only nested-list assertion typo and a missing-metadata failure-saving gap were corrected before the primary reduction; the initial logs remain private. A preliminary provenance check had confused the prospective 1,024-row ancestry manifest with the 896 already-learned rows; the final audit verifies its first 768/896 learned prefixes and exact final 128 collection entries. This was a checker schema issue, not observed training leakage. No real-data record failed or was replaced.

## Meaning and limits

E's collection deficit predates its own event-eight learning, so that later update is not necessary to produce **this deficit on these requests**. This does not explain the deficit on the different final PR49 panel, identify a harmful gradient component, or prove that all prefix withholding fails. Do not subtract the collection −1.726% and final-panel −1.370% acceptance effects to infer temporal recovery or degradation.

S has a +0.239% acceptance point estimate here, but its interval spans −0.473% to +0.977%; 61 calls saved are mostly offset by 52 added. This analysis does not demonstrate a uniform benefit, establish equivalence, identify an optimal LR, or justify an early/late schedule. Its favorable final-panel result remains a separate observation.

**No collection latency ranking is reported.** Capturing features and overlapping GPU tasks contaminate such a comparison. Call reduction is a decoding-work proxy, not a measured speedup. There is no new target-only AR reference, repair-retention diagnosis, causal optimizer explanation or proposed mechanism. The remaining unresolved links require a separate agreed investigation; this report authorizes none.

## Cost, reproducibility and stop

The primary reduction took **0.707 seconds**, peak RSS **93.54 MB**, one observed thread. Provenance took 1.010 seconds; independent raw reduction took 0.562 seconds and reported peak RSS 810.39 MB. Their combined listed process wall time is 2.279 seconds, excluding development, fixtures and publication. CPU processes used one thread each, at most two concurrently, below the four-thread limit. Initial preparation/review estimate was 30–45 minutes; reduction estimate was seconds to a few minutes. [RESOURCE_USAGE.json](RESOURCE_USAGE.json) records measured process costs and final artifact bytes.

**Zero new GPU work, model loads, tensor deserializations, teacher preparations, optimizer steps, generations, remote server operations or deletions.** All original records, traces, checkpoints/recovery receipts, inputs and failure evidence are preserved in place. Preparation, review and report assembly took 16.5 minutes through publication preparation. New source/report/table files total approximately 3.40 MB, below the 1 GB limit; no raw run tree was copied. Private prompts and token arrays remain outside Git.

The [reproduction instructions](../../experiments/pr49_collection_cpu/README.md) give the CPU-only provenance, reduction and independent-reader commands. [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) records consumed small-file identities and mirror-relative paths; the complete private mapping remains available for audit. The report ends here: native K+ primary, E parked, S control; no automatic GPU run, tuning, replay, new panel or algorithm change.

ARS interpretation coverage: **11/11 categories checked** in [INTERPRETATION_REVIEW.md](INTERPRETATION_REVIEW.md). Population selection, pooled-versus-individual effects, no outcome filtering, all denominators/failures, prior-selection/multiplicity limits, fixed analysis choices and the limited causal chronology are explicit. This is not a claim that every possible subgroup has been checked or that selection limitations disappear.
