# Exact native target replay repair

## Material Passport

ARS bounded correctness repair following first-group qualification failure. No precision, objective, anchor, optimizer or dataset setting changed. Original failed attempt and source remain intact.

The feedback recorder now repeats the saved native target prefill and exact verification blocks with fresh caches. It checks every target posterior, including uncommitted suffix predictions, retains only canonical input-prefix feature rows, and makes one final token-only forward to fill the last feature row. That final forward supplies no training label. Response masks and payload fields remain unchanged; a new binding label identifies this path. The trainer accepts the new binding only when strict replay checks pass.

Local coherent regression suite:116tests pass (108unchanged tests, eight exact-native replay/integration tests replacing five full-sequence-specific tests). The five superseded tests and replacements are recorded in SUPERSEDED_TESTS.json/test_native_replay.py; original tests remain archived. Eight campaign-runner checks pass including the restricted recovery guard. Independent review found no blocking logic issue.

Recovery may import only the complete16-record frozen first group from a failed attempt with no prior training and no C/P artifacts, matching the entire sealed manifest and released checkpoint. Original token histories and traces are validated, not regenerated. Totalbudget remains144generations;128remain. Recovery runs in a fresh directory, with original attempt time retained separately.

A800 installation replaces only online_feedback/native.py, adds online_feedback/native_replay.py, extends the receipt validator in online_feedback/training.py, and updates the corresponding tests. Upstream inference/training mathematics are unchanged. Source archives preserve both versions. Native GPU replay and full-group qualification remain pending at report creation.

GPU evidence: all eight GSM8K source responses now reproduce the complete saved native posterior vectors (372 target forwards, including prefill and final-feature completion). Both learned checkpoints passed exact publication and all 16 output comparisons apiece. See [first-group evidence](../retrieved/first_group_recovery/campaign_002/gsm8k/QUALIFICATION.json) via its gsm8k subdirectory. The scientific results may be unfavorable despite correctness passing.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/projected_target_feasibility_20261022/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
