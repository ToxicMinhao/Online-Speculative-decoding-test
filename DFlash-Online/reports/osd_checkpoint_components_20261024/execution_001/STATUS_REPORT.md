# Native qualification passed; campaign running

## Material Passport

ARS experiment-agent execution, October 24, 2026 (Edmonton). This is a running
status report, not the completed scientific result. User authorized final review
and the fixed 768-generation experiment. No design/settings change was needed.

Final implementation review found no additional launch blocker. The previously
passed 322 CPU tests remain applicable. At launch the runner revalidated the input
seal, actual model/target/hybrid/source artifacts, original successful-run code and
native runtime before initializing the GPU worker.

The detached supervisor started at 2026-10-25T02:18:56Z on A800 GPU 0. The first
eight requests completed all 96 generations and passed native qualification at
02:23:50Z: 294 seconds / 4.9 minutes including preflight and loading.

An independent local audit of the retained 96 raw records confirms:

- Exact saved input and canonical output IDs for all four arms and all repeats.
- F/O verification counts reproduce their retained historical counts.
- Verification counts repeat exactly within each method/request.
- No feedback-feature capture; no optimizer steps or learning.
- All four arms pass 58-source / 43-native tensor equality and publication checks.
- Positive finite request measurements and consistent trace/progress accounting.

See [qualification receipt](qualification/QUALIFICATION.json),
[raw-record audit](QUALIFICATION_AUDIT.json), and [pre-launch review](PRELAUNCH_REVIEW.md).
The remaining requests continued automatically under unchanged settings. At the
02:24:46Z check, 120/768 generations had returned and the next was in progress;
qualification was passed and no failure was recorded. This does not certify future
requests, which retain their own gates and raw mismatch records.

A800 storage remained 48.55 GB free; GPU 1 was not used by this experiment.
No files were deleted. Fenrir's F5 training/debugging environment remains untouched.
The original estimate remains 35–60 minutes total, approximately 30–55 minutes
remaining at this check. The two-hour bound plus 120-second cleanup grace remains.

Authoritative run directory:
`/data/sources/minhao/runs/osd_checkpoint_components_20261024/campaign_001`.
Supervisor PID: 238144. Completion/exit status lives under sibling `execution_001`.
No automatic retry, retuning, training or additional campaign is configured.
The runner will produce its complete/partial report when it exits. No conclusion
about component benefit is drawn from qualification alone. Stage 1 / project None
and other paused research remain unchanged.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/osd_checkpoint_components_20261024/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
