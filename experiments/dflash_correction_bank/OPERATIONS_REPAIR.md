# Storage accounting and partial-start cleanup — October 19, 2026

The user authorized these repairs before any future campaign. The existing
five-conversation partial result remains available for the immediate research
decision; completing the sixth conversation is not required. This revision
does not restart execution, change learning settings, or retire evidence.
Executed-source provenance remains commit `6dd7536`; this fix is subsequent code.

## Storage mismatch

The inherited rolling-retirement policy preserves both the pre-update and
post-update restart bundles of the latest OSD event. Each bundle contains one
checkpoint and two Adam moments, approximately three model-sized units. Episode
closure does not remove the predecessor. The old whole-campaign estimate charged
only one bundle per completed episode.

The corrected `bank_campaign_storage_v2` receipt charges two bundles per episode
for the qualification plus six main streams. Let `M` be the released draft's
safetensor byte total:

- Retained OSD payload and metadata: `7 × 2 × (3M + 64 MiB)`.
- Active rolling overlap: `original_storage_bound(M, 0) − 6M`, currently
  `5M + 512 MiB`. The six retained units are already included above; the complete
  512 MiB rolling allowance remains, in addition to per-bundle metadata.
- Bank artifacts: six main envelopes of at least 8 GiB and a qualification
  envelope of at least 12 GiB.
- Additional free-space reserve: at least 8 GiB.

With pinned `M = 2,097,259,104` bytes, required free space is
**173,062,016,928 bytes (161.18 GiB)**, versus the former 125.58 GiB.
The recorded initial free space of 136,406,708,224 bytes would now be rejected
before workers start. This is a budget calculation, not a measurement of actual
retained disk occupancy. The original source, admission receipt and partial
measurements are not rewritten.

No retention policy was weakened. The per-episode admission check, artifact
envelopes and reserve remain intact; an explicitly declared campaign budget
below the corrected requirement is also rejected. Future transfers/offloading
still require their own capacity and staging checks.

## Cleanup after failed initialization

The native admission check can reject an episode before assigning `self.out`.
The bank close method previously accessed that missing attribute and generated
a secondary cleanup error. It now always invokes worker shutdown, but accesses
and writes the episode directory only when initialization reached that point.
Its receipt explicitly records whether the episode was initialized.

The original startup failure remains the experiment error. Cleanup does not
create a directory, overwrite a prior episode's close receipt, launch workers,
or delete retained artifacts. Actual shutdown failures remain visible after
both worker handles have been given a shutdown attempt.

## Verification

The local CPU suite reports **193 passed, 2 skipped**: 70 correction-bank tests
and 123 inherited per-turn/follow-up tests passed. The skips are optional
historical artifact/tokenizer checks. The existing JUnit `record_property`
format warning is unrelated to these fixes. Test runtime was 23.00 seconds,
using Python 3.10.18, NumPy 2.2.6, SciPy 1.15.3, pytest 8.3.5 and PyTorch
2.11.0+cu130 with CUDA devices hidden.

Seven new regression tests cover:

- Actual inherited retirement and bank closure preserving both restart bundles.
- All seven episode peaks, including larger configured bank envelopes.
- Rejection of the historical free space and underestimated declared budget.
- The actual native low-space failure through stream orchestration, without
  worker launch or loss of the primary error.
- An earlier campaign-admission failure before episode initialization.
- Worker cleanup, retained evidence and protection against stale output paths.
- Shutdown failure reporting while still attempting the other worker's cleanup.

The existing exact storage-boundary tests still check acceptance at the required
byte count and rejection one byte below it. No GPU qualification or campaign was
performed for this operational revision. A separate read-only code review
checked the retention formula and shutdown changes.
