# O/C/K/A experiment completed

The supervised A800 run completed on 2026-10-26 at 23:47:41 UTC, exit 0, after 54m52s. All three C/K/A events completed 128 optimizer steps and 256 response presentations each. All 768 O/C/K/A evaluations passed exact canonical-token/termination and native checks on all 64 requests.

Independent audit: passed, zero issues; 36,786 native blocks reconstructed. Postflight confirmed 117 source files, 10 execution files, 32 dependencies and 144 retained artifacts unchanged. All 2,395 compact result files are verified on Fenrir. Both A800 GPUs were idle when checked.

K (KL through the current correct prefix plus first error) used 3,027 calls versus O's 3,075: 1.561% fewer calls and 1.433% less request time. A improved over C by 1.418% calls and 1.505% time. CE performed worse than KL under both position rules in this screen. This is a modest one-seed, reused-development-panel result, not an adaptation-inclusive acceleration claim or an established general mechanism.

No further experiment, tuning or update was launched. [Full report](REPORT.md), [independent audit](FINAL_AUDIT.md), [all per-request results](raw/campaign_001/evaluation/report/PER_REQUEST.csv).


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_first_error_execution_20261026/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
