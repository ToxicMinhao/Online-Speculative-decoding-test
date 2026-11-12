# Frozen K+ lost-repair gate

This probe measures whether a repair made at K+5 is absent again at K+6,
starting from the exact same K+6 serving prefix. It then measures how often
those natural starts appear in the next native training schedule. It does not
train or change any checkpoint.

The sealed pool is original seed-17 GSM8K training rows 768:896 (event 007).
Authenticated K+4/5/6 exports and K+6 sampler state are mandatory. No compatible
retained K+6 capture was found, so each of the 128 responses is captured once.
Every real verification start, including opening and terminal rounds, receives
neutral K+6 and historical K+4/K+5 reconstruction. Neutral proposals, acceptance,
commits and termination must agree. PR40's constructor is reused with explicit
opening-state capture after native prompt ingestion.

The read-only training worker loads the exact K+6 weights and a disposable RNG
copy, seals the ordinary two-pass shuffle/anchor schedule, then independently
repeats those draws. Native BF16/Flex full-block conditioning and eight-block
head projections remain unchanged. There is no optimizer, backward, candidate,
weight modification, endpoint benchmark or feedback-dependent selection.

`analysis.py` joins by request and canonical seed position, requires the same
first-error depth, and matches within presentation/request/depth/priority.
Unavailable immediate followers remain `ineligible`, with their actual native
low-priority coefficient retained in the denominator. The inherited 0.25 factor
is a descriptive calculation only. Complete census and training coverage are
required before any gate decision. Missing work remains INCOMPLETE.

`prepare.py` seals inputs/source identities, `run.py` owns the four aggregate
occupied A800-hour ceiling and process cleanup, `native_worker.py` performs the
same-start reconstruction, `training_probe.py` records exposure, and `report.py`
produces public-safe tables. Raw prompts, tokens, tensors and connection tokens
remain private. Storage admission reserves 40 GiB of artifacts plus 8 GiB free;
within-request CPU snapshot capture also stops before exhausting host memory.

All three thresholds are reported independently: at least 32 distinct matched
lost-repair anchors, at least 16 of the 128 requests, and projected matched
weight at least 0.25% of native first-error weight. Publication ends this probe,
including when thresholds pass. Passing supports a separate design discussion;
it is not evidence of recoverability, transfer, acceptance or serving speedup.

Post-execution audit: `audit_raw.py` independently reduces retained token records,
schedules and weights without importing production analysis or loading models.
`publish_supplement.py` writes sanitized all-start/parity tables, artifact identities
and cost receipts to a separate directory. These publication helpers were added
after scientific execution29b6a15; they do not change or repeat model execution.
