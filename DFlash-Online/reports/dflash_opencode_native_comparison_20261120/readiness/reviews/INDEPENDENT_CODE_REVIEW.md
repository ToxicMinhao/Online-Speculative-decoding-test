# CODE serving and accounting review

Scope: same-team independent review of `evaluation.py`, `analysis.py`,
`runtime.py`, and `storage.py`, during implementation on November 20, 2026.
The reviewer authored the separate AR adapter and native serving gate; that
authorship overlap is disclosed. No GPU engine, model, training or generation
was executed for this review. CPU fixtures are not native qualification.

## Findings already repaired and checked

1. A returned generation could remain marked as unknown if measurement parsing
   raised after its raw record was saved. The evaluator now clears its execution
   pending marker after the return and records malformed returned evidence as
   invalid. An independent fixture verifies one returned/invalid generation,
   zero unknown work, and 15,599 unattempted slots.
2. The reduction initially accepted a method using a different checkpoint on
   one request, provided each request's three repeats agreed internally. It also
   accepted a different source identity for one arm of a pair. Both behaviors
   were reproduced with the existing CPU fixture. The revised reduction rejects
   both; focused regression tests pass.
3. A genuine frozen one-token EOS output with zero verification rounds passes
   the final measurement path. It remains in output/latency totals, while AR's
   speculative metrics are N/A. This was checked independently of the AR adapter.

## Further fixes reviewed

- The data constructor now supplies the immutable `source_id` and
  `original_index` consumed by the existing measurement interface.
- The earlier `runtime.storage` subtracted all allocated bytes from one fixed
  total peak. This can credit excess compact/compiler growth against future
  tensor writes. With a 70 GB forecast, 50 GB current allocation and
  425 GB free, that formula allowed only 20 GB remaining. If those 50 GB contain 30 GB
  compact evidence and 20 GB tensors, but the forthcoming tensor boundary needs
  60 GB, actual remaining writes are 40 GB, crossing both the 80 GB study cap
  and the 400 GB reserve. The revised implementation uses separate remaining
  tensor, compact and compiler forecasts. A CPU fixture now rejects precisely
  this condition. A measured total free-space check is not by itself a
  future-overlap check.
- The inherited retirement helper records each file's logical size and the
  observed shared-volume free-space change, but not its allocated blocks.
  new study wrapper supplements it with an immutable pre-deletion
  `st_blocks * 512` receipt and separate completion record. A one-byte CPU
  fixture verifies that logical size, allocated size and observed shared-volume
  free-space changes remain distinct.

## Remaining integration and resource checks

- The final runner must supply a real two-card quiet check to the evaluator's
  `boundary` callback, close AR before reloading DFlash, and count all graph
  initialization and teardown work. The evaluator's 120 logical activations
  and 240 mandatory warmups are correct. The lifecycle/quiet callback was not
  yet available for end-to-end review when this note was written.

## Supported implementation observations

- All three repeats are averaged within each request. The bootstrap resamples
  whole requests, carries all arms together, uses shared PCG64 seed 20261120
  draws, and records the lexicographic ID ordering and draw digest.
- Complete worker-request latency and native `Engine.generate` latency remain
  separate; no estimated instrumentation overhead is subtracted.
- The seven-state/four-export rolling overlap accounts for two completed arms
  retaining their event-57/58 states and the active arm's current/predecessor/
  atomic successor. Actual component growth still needs the admission check
  described above.
- AR is a separate speculation-disabled Engine. The adapter does not install
  DFlash observer hooks, uses the supported native cache-flush API, and records
  the runtime's actual overlap mode and physical context capacity. Its only
  scheduler instrumentation counts existing graph initialization and inspects
  idle/cache state without adding model calls.
- In this pinned runtime, the scheduler checks output cap before EOS. The AR
  validator preserves raw native termination while using the existing semantic
  EOS convention for an EOS token emitted exactly at the cap. This exception is
  source-supported, not a waiver for unequal generated tokens.

## CPU verification

Command:

```text
/home/minhao2/miniconda3/envs/moziflux/bin/python -m pytest -q \
  experiments/opencode_native_comparison/test_ar.py \
  experiments/opencode_native_comparison/test_evaluation.py \
  experiments/opencode_native_comparison/test_analysis.py \
  experiments/opencode_native_comparison/test_contract.py \
  experiments/opencode_native_comparison/test_runtime.py \
  experiments/opencode_native_comparison/test_storage.py
```

Result: **51 passed**. No dependency installation or native model execution.

## Pinned native API inspection

Read-only inspection used the installed runtime at
`/data/conda_envs/minhao/adaflash_author/lib/python3.12/site-packages/sglang/`.
Its distribution metadata identifies the original source as
`/data/sources/minhao/EAGLE_SPEC/research/adaflash_reproduction_20260930/vendor/sglang-b8f7b319c634aac88910aec8ae46e6a1f194ac66/python`.

| File | SHA-256 | Relevant inspected API |
|---|---|---|
| `srt/entrypoints/engine.py` | `2bcd4b04c325d4ed65f867490c5bac9e915e02f05471c83dacb5a4a097cf1f2a` | `flush_cache` 708, `get_server_info` 771, `collective_rpc` 981, scheduler factory 157 |
| `srt/server_args.py` | `bd60199803698e4ffe6dc44499a8d42be73f42db28056cb6750411523cf56c04` | speculation defaults to `None` at 480; graph default enabled at 607; native AR overlap default at 621 |
| `srt/managers/scheduler.py` | `72859b070ae0de055917607797fc6562f39511b562b041acb27cf9ead61457c7` | no-request inspection 2715; cache reset 2735; absent AR draft worker 550 |

`TokenizerManager.flush_cache` returns `FlushCacheReqOutput` with a `success`
field, rather than a bare boolean. `Req.check_finished` checks the cap before
token-based stopping at `schedule_batch.py:1065–1095`. These are API/source
checks only; AR/F parity, native memory fit, startup accounting and production
serving remain subject to the bounded GPU gate.

## Controller and packaging follow-up

The final controller checks both cards and transfer processes outside request timing before each of120activations. AR uses its own engine; DFlash publication retains the qualified endpoint reload. Final raw responses precede independent structural/token/termination/count validation. The first-event training guard accounts for its four actual additional generations; no hidden optimizer qualification is inserted.

Two lifecycle defects found during this review are fixed: cleanup errors can no longer leave a phase marked completed, and cleanup attempts all owned engines even when one fails. The CPU preflight now verifies the exact configured zstd executable by an exact binary round trip before model work. A mocked preparation fixture validates all serving/publication plan keys,64qualification generations,17disposable Adam steps,40response forwards/backwards,16teacher preparations,120activations,15,360scores and240warmups. These are implementation contracts, not native qualification.

The lossless event archive is fully scanned against per-member bytes and hashes before loose copies are removed. CPU failures cover absent/duplicate/tampered members, changed archive/index, outsiders/symlinks, active boundaries, changed loose members, and verification failures. No tensors, export metadata, source, or historical artifacts are candidates. The new event package and member reader preserve exact source bytes; old loose-path audit consumers must use that reader.

Combined bounded CPU suite after these fixes: **70passed**. Native AR equality, memory fit, graph behavior and actual execution remain unqualified. Storage admission remains blocked at the unchanged80GBcap; see the separate storage review.

## Protected dependencies during retirement

The CPU preflight now seals the authenticated dependencies' resolved paths, device/inode identities, sizes, modification times and expected content hashes. Every new-study tensor retirement, native-feature retirement and compact archive deletion batch rechecks those stat identities, including partial-failure paths. Qualification cleanup uses the same guarded retirement wrapper. A missing keeper receipt is a production failure. Candidates cannot leave the study root or name a protected path/inode, including a hard-linked alias.

These per-batch checks deliberately do not reread entire model weights after each of22,272feature batches. They are recorded as stat-identity checks, not complete byte authentication; the postflight API performs full hashes again. A CPU test explicitly demonstrates the distinction by altering bytes while restoring the same modification time: routine stat checks pass, final content hashing fails. All test fixtures establish tiny authenticated keeper receipts rather than bypassing the production requirement.

Latest combined model-free CPU suite: **81passed**. No GPU, qualification optimizer steps, feedback collection or scientific generations were added by this review.
