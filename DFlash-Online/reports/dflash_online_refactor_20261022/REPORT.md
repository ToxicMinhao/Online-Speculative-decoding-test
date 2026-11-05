# Online feedback readability refactor

Implemented in `/data/sources/minhao/DFlash-Online/online_feedback/` on A800.
Fenrir's `/home/minhao2/DFlash-Research` is unchanged. This was code refactoring
and CPU verification, not a scientific experiment or restart of qualification.

## What changed

- The controller exposes readable `serve`, `update` and `evaluate` procedures.
- The feedback buffer uses one readable JSON state rather than hashed records,
  lock management, duplicate-key parsing and filesystem-policing machinery.
- Training is divided into setup, event validation, batch construction, training
  passes, checkpoint save and export. It still calls upstream `TrainerCore`.
- Removed file/token/model SHA-256 calculations, repeated frozen-model scans,
  fingerprints and the one-off recovery procedure for the old publication bug.
- Shared atomic-write and RNG helpers replace duplicate implementations.
- Expanded dense one-line statements; added a code-reading guide and local
  coding instructions. Meaningful checks live at input/scientific boundaries.

No new mechanism, loss, sampling rule, optimizer or learning settings were added.
The original five qualification prompts and configuration are byte/content equal.
Upstream inference and training trees were not part of the installed change.

## What remains protected

Exact saved tokens, response-only masks, full context without silent truncation,
correctly conditioned target replay, persistent optimizer/master/scheduler state,
complete accumulation windows, RNG isolation, publication before buffer commit,
and evaluation that cannot feed training. Failed or interrupted events remain
visible and cannot automatically repeat updates.

Expensive direct model/state comparisons remain in explicit qualification, not
normal serving/training. Qualification compares live state with the saved state
before closing and then checks restored state; reproducing a bad save is not a pass.

## Verification

- **86 CPU tests pass locally and on A800**; A800 test-body time 1.701 seconds.
- Small CPU fixtures compare two update events against direct upstream training
  for both existing reductions: exact BF16 draft weights, FP32 masters, Adam and
  scheduler state, RNG state and counters match.
- Native adapter CPU fixtures check teacher conditioning, masks, exact outputs,
  traces, publication and cost boundaries. These do not replace full-model GPU
  output-preservation qualification.
- Cross-file reviews identified and resolved the need to compare live state
  before restart, preserve original errors when diagnostic writing fails, and
  retain post-run free-space reporting.
- Previous GPU qualification remains unfinished and paused. No model generation,
  GPU training, new performance measurement or scientific campaign occurred.

See [A800 tests](CPU_TESTS_A800.log), [local tests](CPU_TESTS_LOCAL.log),
[installation receipt](INSTALL.json), and [code-reading guide](worktree/README.md).

## State and storage

The buffer and trainer have new state formats. Start a new output directory;
old state is explicitly rejected rather than silently migrated. The previous
source is preserved in `before.tar.gz` both here and at
`/data/sources/minhao/runs/dflash_online_refactor_20261022/before.tar.gz`.
All existing run artifacts and checkpoints remain intact. Historical recovery
requires that archived implementation, not the refactored runner.

The runner remains single-process and synchronous. Atomic replacement protects
against partial writes; concurrent-writer support and the earlier ancestor-directory
fsync/recovery machinery are intentionally absent. A complete power-loss recovery
protocol is not claimed.

Installation matched source bytes and checked for intervening edits before
replacement. Only the obsolete recovery test and disposable Python caches were
removed from the active code tree; old source is in the archive. Free A800 space
was 14,032,830,464 bytes before installation and 14,032,998,400 after, above the
8 GiB reserve for this bounded code/CPU-test operation. Full native qualification
requires its separate, larger checkpoint-overlap admission.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_online_refactor_20261022/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
