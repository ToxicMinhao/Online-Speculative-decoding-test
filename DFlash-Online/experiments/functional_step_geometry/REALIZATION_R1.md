# Norm-feasible BF16 realization, R1

This explicit numerical realization revision preserves the original continuous
FSG direction, BF16 training and the strict native-update norm and KL guards.
The earlier raw-candidate norm rejections and failed cross-layout tests remain
historical evidence; R1 does not relabel them as passes.

If the rounded unit-scale candidate exceeds the native norm, run exactly 24
parameter-only bisections along the original continuous direction and choose
the feasible lower bracket. Probes and the final write share the same streamed
arithmetic. Score the one resulting nondegenerate candidate using the existing
KL rule. Native/prestate coincidences have explicit bypass/fallback outcomes.
SCALAR derives its own feasible hypothetical target, then performs its original
closest-bracket matching and checks its own final endpoint.

`qualify_realization` runs four live cases on the original two retained windows.
Each case restores the complete dataset-specific KPLUS4 state. Warmed ordinary
reference gradients are captured without an Adam update, and native weights are
checked against authenticated ordinary exports. The new budget is four Adam
applications, eight ordinary forwards/backwards, up to 32/16 additional
forwards/backwards, and four serving generations. Startup and reference capture
are included in that additional allowance. Passed unchanged oracles and maximum
resource evidence are reused without new model work. Failed work consumes the
same budget; no automatic retry is provided.

The original two-event pilot is conditional on all checks, at least one natural
nontrivial accepted FSG correction, native SCALAR matching/final checking, and
separate native integration and storage admission. Its scientific settings,
panels and limits remain unchanged. Qualification states never initialize it.
All diagnostic endpoint pairs and final complete states are retained; rolling
retirement is limited to validated superseded pilot boundaries and shared pools.
CPU tests qualify the implementation logic, not GPU execution or learning benefit.

## Recorded qualification

Execution `95aeb869189da2d7d4f2109b93c05fd3bd9b7944` passed all four
cases and four serving checks, with the exact authorized model-work counts.
Both natural FSG cases accepted nontrivial feasible candidates. Both SCALAR
cases exercised matching and actual final scoring, then correctly restored
native under the existing KL deadband. This passes the activity gate without
showing a held-out drafting benefit. See the
[qualification report](../../reports/dflash_functional_step_geometry_20261108/BF16_REALIZATION_R1_REPORT.md).

The conditional pilot retains the original budget: 512 shared feedback
responses (four already collected), 1,536 Adam steps, 3,072 timed endpoint
completions and 16 separately bounded diagnostic windows. Its current
rolling-retention admission requires 188,444,277,887 bytes before launch;
qualification admission does not grant this storage. Estimated duration after
launch is 12–18 hours, with 15–18 hours the safer planning range if most windows
perform the measured parameter scans. No pilot efficacy result exists yet.
