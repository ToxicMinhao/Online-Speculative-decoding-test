# A800 feedback-loop implementation — stopped at user request

Implementation is complete in `/data/sources/minhao/DFlash-Online`. The user
requested stopping at the reusable loop, so development and qualification are
paused. Both A800 GPUs were confirmed idle after the process exited. No scientific
campaign or projected-target mechanism was launched. Fenrir
`/home/minhao2/DFlash-Research` remains unchanged (673 recorded files rechecked).

## What exists

Normal DFlash generation → immutable exact-token feedback buffer → canonical
frozen-target feature replay → persistent SpecForge training → checkpoint export
and publication → read-only evaluation. Only actual serving records enter training;
prompt/history tokens supply context and only the newest response receives loss.
The learning copy starts from released DFlash; the frozen comparison stays fixed.

The normal orchestration is small. The complete new package is 1,553 lines,
including 204 qualification-runner lines; the 158-line controller also contains
strict publication recovery and evaluation checks. Buffer durability, validation,
training-state persistence and compatibility checks account for most of the rest.
These are infrastructure contracts, not new algorithmic components or research
hyperparameters. Future objective/update experiments can reuse them unchanged.

Current learning uses SpecForge's weighted hard-token CE and author sampler,
with explicit response-mean accumulation. This is not the historical full-teacher
forward-KL OSD recipe. Canonical teacher replay has a separately measured cost;
features are not yet reused directly from in-verifier hidden states.

## Verified and unfinished

- All 97 CPU tests pass in the isolated A800 environment.
- Real Qwen3-8B / released DFlash-b16 served four 128-token-or-EOS fixtures.
- The first two responses produced exact-token feedback; replay target argmax
  matched all labels. Event 1 completed two steps/four presentations, exported,
  published, and matched all 58 trained draft tensors exactly.
- Trainer restart restored draft, optimizer/master/scheduler, counters and RNG
  fingerprints exactly. The next two responses used the published update.
- Event 2 reached global step 4/presentation 8, but the user's stop interrupted
  pending-state fsync before export/publication/commit. Its pending state is
  uncommitted evidence, not a qualified serving checkpoint. The last committed
  full state remains event 1. No automatic retry or continuation is enabled.
- The three final identical-input checkpoint replay evaluations were not run.
  Therefore complete replay/output-preservation qualification is unfinished.

The final raw result says `failed` with `KeyboardInterrupt`; the separate
`USER_STOP.json` records that this was user cancellation, not a new correctness
failure. Preserve the earlier actual failures: cache API compatibility, absent
compiler binding, and an implicit-versus-explicit vocabulary config default.
Those were repaired narrowly; none changed the objective or optimizer recipe.
Publication-only recovery reused event 1 without additional learning.

## Source and compatibility

DFlash pin `07ebd93db9f472af339b644bb70221ad8428328a`; SpecForge pin
`1839aae30003e3573d00979eb1edd57347b3578c`. Official inference adds a read-only
trace callback and a narrowly qualified Transformers 5.12.1 full-attention Qwen3
cache path. The latter rejects sliding/hybrid configurations. Publication treats
missing `draft_vocab_size` as `vocab_size`, matching SpecForge's exact default;
explicit differences remain errors. The training source retains only the inherited
feature-cache path fix. Core trainer/model/optimizer code is reused.

Environment: `/data/conda_envs/minhao/dflash_specforge_20261022`; Python 3.12.15,
Torch 2.13.0+cu130, Transformers 5.12.1. Launcher binds the existing GCC 13.4
compiler explicitly. Models match Fenrir's transferred SHA256 manifests.

## Evidence and retention

- [Code snapshot](source_snapshot/README.md), [file hashes](SOURCE_SNAPSHOT_SHA256.json)
- [97 CPU tests](CPU_TESTS_FINAL.log)
- [Native result at stop](remote_evidence/qualification_004_recovery/RESULT.json)
- [User stop receipt](remote_evidence/qualification_004_recovery/USER_STOP.json)
- [Restart before](remote_evidence/qualification_004_recovery/RESUME_BEFORE.json)
  and [after](remote_evidence/qualification_004_recovery/RESUME_AFTER.json)
- [Original first update](remote_evidence/qualification_003/trainer/events/event_1/RESULT.json)
- [Storage retirement record](STORAGE_SUMMARY.json), [operations](OPERATIONS.md)

Full tensors remain on A800. Compact records and source snapshots are on Fenrir
outside its protected clean project. At stop A800 had 14,031,015,936 free bytes;
Fenrir had 85,372,010,496. Both the 14.68 GB committed state and the interrupted
14.68 GB pending state remain preserved. Twelve completed historical optimizer-
only artifacts were retired earlier under standing authorization (50.335 GB),
with compact metadata and serving checkpoints retained; exact historical Adam
continuation was lost, not archived. No further cleanup or launch is implied.

This is engineering evidence for a reusable loop, not a speedup, generalization,
or projected-target efficacy result. Further work waits for the user's direction.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_online_foundation_20261022/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.

Historical-source gap: this run archive contains 7 unresolved source-pin occurrences; the missing versions remain explicitly identified.
