# Final runner review

Date: 2026-10-22.

## Material Passport

- Mode: ARS experiment workflow, bounded implementation review and CPU verification.
- Scope: `runner.py`, `test_runner.py`, `report.py`, and the existing objective/native/trainer contracts.
- Provenance: separate native reviewer execution with shared task context; no external model or remote operation.
- Stage: Stage 1 unchanged. This receipt does not authorize an exception, seal data, or certify native efficacy.

**Verdict: no remaining code launch blocker found in the reviewed version.** Launch remains blocked on the pending user decision about historical GSM8K A reuse and a finalized eligible manifest. The first complete dataset group must still pass the runner's native qualification gates before later groups proceed.

The final changes address the identified integration issues: the manifest gate uses the producer's `selection_seed`; provisional status is rejected; fixed configuration is checked before native loading; completed generations and training events are counted before post-execution validation; training evidence is written before learning checks; trainer closures and frozen-table references are released before evaluation. C/P initialize independently from released weights, retain exact anchor/exposure/RNG checks, and compare both tokens and termination against F. Full optimizer states are retired only after verified publication, complete paired replay, and verified compact metadata preservation.

Independently ran all **13 focused CPU tests**, including the real upstream eight-step BF16Optimizer schedule, output/termination failures, exposure accounting, protocol gates, invalid-completed-generation accounting, and report denominators. All passed after the final fixes. The existing Torch 2.13 nested scheduler uses `6e-4 * (1 + cos(pi*k/8)) / (1 + cos(pi/8))` for updates `k=0..7`, beginning at `0.0006237396779379481`; post-step-eight LR is zero. The runner preserves this actual upstream behavior.

Also checked the actual 48-request conditional manifest: its unapproved status is rejected. A separate in-memory copy with only the status changed passes structural/config validation; no data file was modified or sealed. Initial peak-storage admission is **39,099,160,064 bytes**, including six retained exports, one full state, three 1 GiB feature/staging allowances, and the 8 GiB reserve. Runtime free-space admission remains required.

Limits remain explicit: target mutation checks use parameter identity/version and gradient/optimizer exclusion, not whole-target bitwise comparison; verification traces remain identically enabled, while training-feature capture is outside generation. CPU checks do not establish projected-target GPU correctness, transfer, or speedup. No GPU work or remote operation was performed by this reviewer; this receipt is the only file written.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/projected_target_feasibility_20261022/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
