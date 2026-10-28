# Independent KRC2_OFF implementation review

**Verdict: the reviewed trainer and two-panel campaign match the authorized correction-removal control. Native exact restoration and first-A reproduction remain execution gates.** No model-scale or GPU experiment was performed by this reviewer. Native equivalence and performance are not established by CPU tests.

Material Passport: ARS academic-research-suite / experiment-agent validation; source and CPU review by a separate same-team agent with shared project context. This is not a blinded external or cross-model review. Scope is one ordinary-K event resumed from KRC1 followed by 768 paired generations; Stage 1 and the scientific design remain unchanged.

## Restoration and ordinary training

`online_feedback/krc_off.py` subclasses the qualified strict continuation loader. It reconstructs KRC1's saved contract before the unchanged loader checks configuration, optimizer recipe, horizon, accumulation and event history. The campaign checks the full saved snapshot against KRC1's corrected export and then compares the live weights, Adam state, Python/NumPy/Torch/CUDA RNG, counters and configuration with that snapshot. Only after that exact restoration does `enable_off_continuation` replace the recorded reference-correction contract. Its receipt preserves both contracts; ordinary K settings, optimizer and RNG are unchanged.

OFF calls the inherited ordinary training loop. It neither creates the reference-correction controller nor attaches a prefix guard. Its wrappers observe existing forwards, backwards and native Adam calls; they do not calculate a reference loss or change gradients. The normal shuffle and anchor sampler must match all retained event-2 records. No sampled anchors are substituted, and no RNG reset is used to conceal disagreement. The original capped response and all feature metadata remain intact.

Each response keeps its own K loss normalization, and the original accumulation-2 path averages the two responses before clipping and one Adam step. The same optimizer continues from 128 to 256. The new arm records local work separately from global counters; its one-event limits remain intact. Final state/export are written to a new directory. Existing KRC1/KRC2 and earlier experiments remain read-only.

## Included first-A and accounting checks

The retained KRC2 `RESULT.json` contains the first two ordinary A records before its first correction: `train_00140` and `train_00194`, with global counters 128 and 129. The OFF gate compares those actual records' anchors, losses, accuracy/selection metrics, learning rate and counters exactly. Their metric scopes are compared directly: the first record is one presentation and the second represents the accumulated pair. They are not added together as independent exposure totals. No unavailable historical weight tensor or later-trajectory equivalence is required.

This gate uses the first real update, without another forward or Adam step. Failure records retain the completed native work even if the parent trainer has not yet advanced its outer bookkeeping. All reference/protected-prefix/correction/probe counts must remain zero. Final work is 256 ordinary forwards, 256 backwards and 128 Adam steps. The qualified disposable startup is separate: two forwards/backwards, zero Adam steps. Training closes before serving begins.

## Evaluation and interpretation

Only KRC2_OFF and retained KRC2 are evaluated. Both unchanged 64-request panels use three balanced repeats; the first eight development requests supply the included 48-execution gate, with no second gate on the additional panel. Both panels require the retained canonical tokens and termination, repeat-consistent counts, and exact retained **per-request KRC2 counts**. Historical K1/K2 values do not become correctness gates.

Raw returned records are saved before validation. Failed, unknown and unattempted executions remain visible; rejected measurements are independently revalidated when reporting. A harmful but valid OFF checkpoint does not stop the planned evaluation. Complete-request, native-generation and cache-maintenance times remain separate, with native/cache time nested inside request time. Counts use one validated trajectory per request, timing uses all three repeats, and panels are never pooled.

The primary sign is correct: `1 − KRC2_calls / KRC2_OFF_calls`, so positive values favor retaining correction in event 2. OFF-versus-KRC1 call changes separately answer whether ordinary continuation learned from this initial state. This experiment does not establish equivalence from a tiny difference, equal adaptation cost, seed robustness or a universal event-1-only policy.

## CPU evidence and resolved integration finding

The independent review ran all 18 then-current focused tests successfully in 11.009 seconds (`CPU_INDEPENDENT_REVIEW.log`). They include an actual tiny BF16 KRC1 serialization and exact OFF-versus-uninterrupted-plain-K weights, Adam, RNG, sampled anchors and loss metrics; zero reference work; first-A failure preservation; schedule mismatch rejection; both KRC2 historical count gates; balanced 768-slot evaluation; invalid/unknown accounting; and report revalidation. A subsequently added nonfinite-evidence fixture also passed independently in 0.818 seconds (`CPU_INDEPENDENT_NONFINITE.log`), bringing focused coverage to 19 distinct tests. The reviewer additionally compared all 191 inherited Python/YAML/Markdown files with the completed KRC2 worktree: no changed or missing files.

During development, the campaign initially expected a `work` summary and global-counter aliases that the trainer had not yet provided. The core report now provides that schema, and the integrated tests pass. The fix changes reporting integration, not learning. Nonfinite diagnostics are encoded explicitly before writing failure receipts so the original scientific failure is not hidden by JSON serialization.

An independent post-run auditor lives outside executed code. Its four arithmetic/provenance fixtures pass (`CPU_AUDIT.log`), including both-panel KRC2 count binding and the distinction between correction's endpoint benefit and OFF's own learning. It has not yet audited native results.

## Preparation review

The preparation script verifies the completed parent campaign and every retained sealed dependency, then installs only new files. It carries over the exact configuration, model/runtime pins, event-2 receipt and unchanged panel requests. KRC2 control counts come from three valid, count-consistent retained repeats on each panel. It preserves the capped `train_00187` (2,048 output tokens; 2,133 total training tokens). Current source is bound to the qualified parent seal, which already records the two older/current source identities; no runtime rollback or scientific-source replacement is made.

The final preflight and launcher were also reviewed. Preflight checks every sealed dependency and exact file set, both model inventories, recipe and pinned training/native runtimes. Its fresh-output peak budget includes a 3W full recovery state, W export and W potential export temporary overlap, 10% serialization slack, 4 GiB compiler/log/compact-retrieval allowance and an 8 GiB reserve; it also checks 72 GiB host memory and refuses an existing campaign output. The launcher requires idle GPUs, preserves the qualified environments/GPU mapping, runs CPU checks before the campaign and records postflight integrity even after a failed campaign. It contains no automatic retry or extra model qualification. The proposed storage retirement affects only O2’s exact duplicate disposable trainer-state container after tensor/Adam/RNG equality and separate metadata retention; its required initial snapshot and production states are untouched. Actual storage admission and native restoration remain execution checks, not conclusions of source review.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_krc_off_20261028/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
