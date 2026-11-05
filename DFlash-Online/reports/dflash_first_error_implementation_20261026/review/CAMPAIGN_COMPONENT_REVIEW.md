# Prepared campaign component review

ARS experiment-agent, CPU implementation stage. This delegated review examined
the parent agent's `experiments/first_error/campaign.py` and its two shared
`experiments/prefix_ranking/{evaluate,protocol}.py` changes. The reviewer authored
the objective/training integration, but did not author or edit these campaign
components. No GPU work was launched by this review.

No confirmed scientific arithmetic or evaluation defect was found in the
prepared components. The new objective mapping trains only C/K/A and retains O.
The reused startup validator explicitly checks seed 17, pinned released weights
and full RNG, zero counters, no gradients and empty Adam state. It restores RNG
after fresh construction; it never transfers the discarded warmup optimizer or
learned weights. The original presentation schedule is passed to every new arm.
The post-event check requires 128 optimizer steps and 256 presentations per arm.

Evaluation supplies exactly O/C/K/A to the existing three-repeat rotating
evaluator. Native validation compares exact prompt/output tokens, finish reason,
stop IDs and token cap to the retained canonical request. Retained O call counts
are additionally required to reproduce. The four specified comparison pairs are
present: K/O and A/C for selection, C/O and A/K for loss choice. New results use a
common complete valid cohort. Historical F counts are written separately, on
that same cohort; F has no freshly timed arm or latency comparison. Existing
callers retain the previous required F/O defaults.

The peak-storage estimate explicitly includes 16 model-sized copies for three
BF16 parameter/Adam states, three exports, three fresh-state snapshots and one
discarded warmup snapshot, plus failure overlap, staging and reserve. Actual
qualification storage/work remains separate from this proposed main-campaign
estimate. Loss counts come from the trainer's once-per-window totals; nested
objective timers must not be added again to event wall time.

The delivered module is intentionally a set of callable components without a
launch command or GPU qualification driver. Before a future launch, the driver
must pin and verify model/runtime/source/config/feature/panel/schedule identities,
require the real native qualification receipt with separate work/cost accounting,
and enforce the exact 128-response/64-request bounds before starting work. The
current `require_qualification` checks named boolean gates; it does not itself
verify provenance hashes or execute those gates. The parent was informed of
these preparation-stage limits. They are not evidence that GPU qualification
has occurred or permission to launch one.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_first_error_implementation_20261026/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
