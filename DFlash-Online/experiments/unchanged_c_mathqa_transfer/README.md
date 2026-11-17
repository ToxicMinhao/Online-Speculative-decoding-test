# Unchanged C MathQA transfer

This adapter preserves PR50's allocation and objective bytes, and uses PR51's
qualified training/serving infrastructure. C continued through events 5 and 6;
N6/S6 are authenticated retained controls. The fresh panel and balanced N/C/S
evaluation have their own identities and never reuse historical timings.

**Current status: completed; practical timing gain not supported.** Execution
`3b25b068f0caa0035aeb4c8d58ba498f02a814cc` passed both exact N5/S5 control
replays, native qualification and all 2,304 scored executions plus 72 warmups.
C/N request-time reduction was +0.0255%, with a paired 95% interval of
[-0.3028%, +0.3485%], below the prescribed +0.5% practical target. The fixed
candidate does not advance. See the
[completed report](../../reports/dflash_unchanged_c_mathqa_transfer_20261117/replay_transfer/REPORT.md).

The original readiness stop remains preserved in its
[historical report](../../reports/dflash_unchanged_c_mathqa_transfer_20261117/REPORT.md).
The later user-approved amendment replaced the missing historical feature-hash
prerequisite with exact functional replay. Current hashes and matching replays
do not prove historical feature bytes: `historical_feature_identity_verified`
remains false.

The implementation separates input preparation (`data.py`, `prepare.py`),
complete C continuation (`state.py`, `training.py`), disposable qualification,
resource accounting, serial task execution, and three-arm evaluation/analysis.
The existing allocation module, eleven-field reducer, models, optimizer and
historical N/S guards are unchanged. The 100 CPU tests and subsequent native
qualification are reported separately from endpoint performance.

Completed work was 256 replay, 12 qualification and 256 C-production Adam
steps; 1,060 response forwards/backwards each including startup; six trainer
initializations; and 2,770 serving attempts. The amendment's ceilings were 528
Adam applications, 1,088 forwards/backwards each, eight initializations, four
elapsed/card-hours, 90 GB new allocation and 400 GB reserve. All limits passed.
The campaign has stopped after its report; this documentation authorizes no
restart, tuning or further event.

## Functional replay prerequisite (November 17 amendment)

The current route is `prepare.build_prerequisite`, followed by `launch` once.
The typed current-file manifest permits only original N5/S5 replays; it does not
claim historical feature equality. `replay_comparison.py` checks exact retained
scientific receipts and live endpoints. `replay_gate.py` permits main C work only
after both comparisons pass. The original clock/work ledger continues through
main qualification and execution. No replay checkpoint/export is written.
See the report's `REPLAY_ADDENDUM.md` for scope and revised cumulative caps.
