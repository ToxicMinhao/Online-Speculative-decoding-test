# CODE qualification policy amendment and recovery

## Material Passport

ARS experiment-run supervision; implementation and CPU validation. This is a user-authorized recovery of PR56, not a new scientific experiment. GPU qualification is pending under the revised output-validity contract.

## What changes

The user permits differences between target-only AR and speculative decoding, including tokens and termination, after the retained diagnostic reproduced an exact AR tie versus a unique speculative winner. Both native paths retain their own canonical records. AR must reproduce AR; frozen F and the O/AREPO/C speculative arms must reproduce F exactly. All native trace, stop, repeat-count, capture parity and cross-card checks remain. No algorithm, training setting, model/runtime, input panel or decoding setting changes.

The full-panel AR/F output differences are preserved per request. All-request AR timing ratios are descriptive when outputs differ. Identical-output AR speedup ratios use only the explicitly counted matching subset. Full-panel O/AREPO/C/F comparisons remain unchanged. This protocol amendment does not retrospectively make the original strict AR/F gate pass.

## Recovery and work accounting

Eleven returned ordinary qualification records are copied byte-for-byte and revalidated under the amended policy. They do not invoke engines or count as new generations. The one original interrupted operation remains unknown. The two instrumented diagnostic replays are extra work, not ordinary qualification slots.

| Qualification accounting | Before resume | At successful completion |
|---|---:|---:|
| Original prescribed successful slots | 11 returned | 64 |
| Diagnostic replay returns | 2 | 2 |
| Interrupted unknown attempts | 1 | 1 |
| Total attempts | 14 | 67 |
| Total returned generations | 13 | 66 |
| Remaining new ordinary generations | 53 | 0 |

The user authorizes the 67-attempt ceiling and equivalent operational retries only when scientific scope remains fixed. The implementation admits this specific recovery once, with an authenticated authorization and original ledger. New unknown work or failures are not silently waived. Prior raw records, failed qualification, diagnostic records, initialization accounting and source identities remain intact. Original elapsed and occupied-card clocks continue; time spent debugging is not removed.

The remaining training qualification is unchanged: 17 disposable Adam steps, 40 response forwards/backwards each including startup, 16 teacher preparations and three trainer initializations. Those temporary states never initialize main training.

## Validation and resources

The pure-CPU suite passes 192 tests, including independent-reference routing, differing AR outputs, strict same-path failure, reused-record byte preservation without model/ledger calls, recovery authorization and clock/work preservation. The separate Torch suite passes 18 training and qualification tests (247.359 seconds), for 210 distinct passing CPU tests. An independent integration review passes 42 overlapping checks. The validation receipt records their scopes.

Both A800 GPUs were idle at the preparation check. Free storage was 491.908 GB; the existing peak forecast remains approximately 86.53 GB within the authorized 90 GB study cap, with 400 GB retained free. No new cleanup is required for this recovery. The 48-hour original elapsed and 90 occupied-card-hour caps remain unchanged. A measured full-campaign forecast follows qualification and the fixed 2,048-generation AR/F precheck; no training begins without that admission. Existing historical estimates are not a guarantee that the remaining time fits.

Historical diagnosis: [MISMATCH_DIAGNOSTIC.md](MISMATCH_DIAGNOSTIC.md). Original failed strict qualification: [QUALIFICATION_REPORT.md](QUALIFICATION_REPORT.md). Both remain unchanged.
