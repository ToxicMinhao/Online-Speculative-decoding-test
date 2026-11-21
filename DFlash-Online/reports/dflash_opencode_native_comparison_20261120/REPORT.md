# CODE comparison — authorized qualification recovery

The user has authorized AR/speculative output differences while retaining exact F/O/AREPO/C output agreement, repeat checks and all other correctness gates. The retained mismatch remains a documented numerical-path difference, not a retrospectively passed strict equality check.

The recovery reuses eleven completed ordinary records and preserves two diagnostic replays plus one interrupted unknown. It requires 53 further ordinary qualification generations: 67 physical attempts in total, with 66 expected returns and one historical unknown. Training, inputs, models, runtime and settings remain unchanged.

Implementation, independent integration review and 210 CPU tests pass; native qualification resumed on both A800s after CPU preflight passed in 24.677 seconds. No O/AREPO/C performance result exists yet. Execution source is `658d734c422a1a200fe400dceb256492f529204e` (launcher PID3993464). This report update is published separately from that execution revision. The campaign will proceed only after remaining correctness, storage and measured duration checks pass.

See [the amendment and complete accounting](OUTPUT_POLICY_RECOVERY.md), [the retained diagnostic](MISMATCH_DIAGNOSTIC.md), and [the original failed qualification](QUALIFICATION_REPORT.md). Execution commit and validation receipt are recorded alongside the launch report.
