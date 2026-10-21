# Related-instance transfer implementation validation — 2026-10-21

Status: **implemented, requests sealed, CPU checks passed; awaiting code review**.
No A800 operation, GPU qualification, generation, training, KL evaluation,
checkpoint export or earlier-campaign restart occurred during this implementation.

## CPU validation

**957 tests passed in 53.52 seconds**, including 221 new checks and 736 inherited
regression checks. GPU visibility was disabled:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  experiments/dflash_baseline_learning_curve \
  experiments/dflash_kl_native_diagnostic \
  experiments/dflash_lower_lr_control \
  experiments/dflash_matched_prefix_diagnostic \
  experiments/dflash_feedback_allocation_control \
  experiments/dflash_related_instance_transfer
```

The existing CPU test environment uses Torch 2.11.0+cu130, transformers 4.57.6
and pytest 8.3.5. No dependency, author implementation or historical scientific
file was changed. Native tests use runtime stubs and synthetic measurements;
they do not establish GPU output equality or memory fit.

Coverage includes:

- Retained-length rank selection, tie breaking, frozen parent order, prompt-only
  exports, twelve-child completeness, new role/ID isolation, answer-relevant
  semantic attestations, revision chains, no original48/child duplicates, exact
  original reencoding, and complete sealed checkpoint/token assignments.
- Separate child-only F/own-S8 admission, standalone input rendering, fixed
  generation order and four-call limit, read-only API restrictions, unchanged
  checkpoint checks, capture disabled, cache cleanup, and no explicit warmups.
- Fresh F canonical references, exact output/termination validation, preserved
  invalid/nonfinite raw measurements, rejected state mutation, persistence errors,
  no retries, and returned/invalid/unknown/unattempted accounting. Synthetic
  harmful results still finish all six groups when correctness passes.
- Report revalidation, ratios from summed valid paired counts, all twelve
  selected-case denominators, exact-repeat/non-exact-repeat subsets, historical
  context separation, real CLI provenance fields, and incomplete-evidence limits.
- Shared/separate filesystem admission, actual nested payload sizing, retrieval
  staging and 8 GiB reserve, unchanged-artifact isolation, and six-group capacity
  accounting at the exact declared threshold.

Integration review found and repaired three operational issues before upload:

1. Non-ASCII metadata used inconsistent JSON digest conventions across native
   admission and request sealing. Group/runtime payloads now use their canonical
   seal digest; historical checkpoint/token hash algorithms remain unchanged.
2. A nonfinite worker measurement could fail JSON persistence before being
   retained. The real compiled wrapper now writes a lossless nonfinite marker
   before rejection; a CPU wrapper regression verifies returned-invalid recovery.
3. Requiring the full storage allowance again between groups double-counted
   artifacts already written. Rechecks now credit verified current-attempt files
   only, preserving all retrieval staging and reserve. Foreign or incomplete
   artifacts receive no credit.

## Actual retained-evidence and request admission

The unchanged PR9/PR10 readers verified the real compact evidence. All six
original groups, twelve expected parents, own-group F/S8 bindings, retained
parent-source and held-out context, sealed children and native child admission
passed CPU integration. The pinned tokenizer reproduced all 48 original inputs
before sealing the twelve child inputs, whose lengths range from 45 to 1,457
tokens. No answers were generated or regenerated.

The private request-manifest file SHA is
`71a6388c44b79f19571542d2fee42dcdbe0c2cbe4d654c2666dde5e49b0e53b6`;
its payload seal is
`3ae6c9adf2248be9158de410c961fcb28a02e1ce6d72affce6c1026c8179621c`.
[REQUEST_SET.md](REQUEST_SET.md) records the complete preparation summary and
other bindings. The full preview, original prompts and runtime paths stay private.

## Artifact-derived storage budget

No new checkpoint storage is included. Conservative sizing covers actual
group/runtime configurations, controller copies, bounded raw/normalized traces,
atomic overlap, logs and an uncompressed retrieval-staging copy.

| Destination arrangement | Required free capacity, including reserve |
|---|---:|
| Shared filesystem | 23,303,129,722 bytes (23.30 GB) |
| Native filesystem, if separate | 11,970,387,557 bytes (11.97 GB) |
| Controller filesystem, if separate | 19,922,676,757 bytes (19.92 GB) |

These are calculated requirements, **not current A800 capacity readings**.
Current capacity and the actual destination paths must be verified before the
authorized future run. Each distinct filesystem retains at least 8 GiB; existing
verified current-attempt artifacts are credited only against primary writes.

## Remaining native checks and measurement limit

The first complete group must still qualify native integration, exact fresh F/S8
outputs and termination, checkpoint integrity, cache handling and measurements.
Its four generations are part of the fixed 24 total. If correct, the remaining
five proceed unchanged even for null or harmful results. No extra qualification
generations or retuning are permitted.

Normal initialization, activation and cache timers remain separately recorded.
The qualified runtime includes prefill in request latency and records its token
work, but does not expose a separate prefill duration. The new result explicitly
marks this unavailable; it does not estimate a kernel-time breakdown. One fresh
fixed-order measurement per arm/request cannot establish serving speedup.

No transfer finding, output-preservation result or performance benefit is claimed
by these CPU tests or the request preparation.
