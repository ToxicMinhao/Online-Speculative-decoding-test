# PR52 evidence index

The campaign is complete. Read the [final report](replay_transfer/REPORT.md)
and its [per-request](replay_transfer/PER_REQUEST.csv) and
[all-repeat](replay_transfer/ALL_REPEATS.csv) tables.

Execution commit: `3b25b068f0caa0035aeb4c8d58ba498f02a814cc`.
Both exact E5 control replays passed. The subsequent unchanged-C MathQA test
finished with `PRACTICAL_TIME_GAIN_NOT_SUPPORTED`; it does not advance the fixed
candidate under the agreed rule.

The [original readiness report](REPORT.md) remains unchanged as a historical
record. The later [replay addendum](REPLAY_ADDENDUM.md) replaced only its missing
historical-feature-hash prerequisite. Functional replay now passes, while
historical feature byte identity remains unverified. No further experiment is
automatically authorized.
