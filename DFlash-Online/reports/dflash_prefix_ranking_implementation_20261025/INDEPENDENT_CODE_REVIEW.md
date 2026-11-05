# Independent CPU implementation review

## Material Passport

- Scope: read-only review of the U/P implementation against the user's October 25 handoff, with CPU synthetic tests only.
- Implementation: `worktree/online_feedback/{objective,ranking,training,presentation_schedule}.py`.
- Reviewer: separate Codex agent in the same model family; no external service or GPU use.
- Status: no blocking implementation finding in the reviewed source. Native GPU execution, full-size memory fit and serving equivalence remain unqualified.

## Findings

The implementation preserves ordinary forward KL and its existing decay weights, per-response denominator (including its single `1e-6`), response averaging and optimizer accumulation. The new objectives are explicit `osd_uniform` and `osd_prefix` contracts. Lambda zero follows the ordinary O path without constructing a reference.

P uses only the consecutive canonical-matching prefix after each known seed. A mismatch, invalid position or boundary terminates protection; a matching EOS is included once and subsequent positions are excluded. U uses one detached weighted prefix fraction over the entire response presentation, not an independently calculated chunk fraction. Neither auxiliary loss is normalized by protected-position count.

The strongest noncanonical competitor is selected over the full projected vocabulary. Only its discrete index is detached: canonical and rival gathered logits both receive gradients. Subtraction and softplus use FP32 after the qualified projection. Reference masks are computed before the current autograd forward; the vocabulary projection is chunked without reducing full-block attention. Independent frozen draft parameters and preserved Python/NumPy/Torch RNG states prevent reference work from advancing the learning trajectory or changing current model modes.

The retained O schedule is used as an assertion on normal response shuffles and anchor draws, not as a silent replacement sampler. Feature identities/paths, ordering, positions and keep masks are checked. O/P/U checkpoint contracts remain separate; reference parameters do not enter the trainable or exported parameter tree. The inspected diff does not change ordinary O optimizer settings or the completed O artifacts.

## Tests independently executed

With CUDA disabled, the following six modules passed: `test_ranking`, `test_ranking_objective`, `test_ranking_schedule`, `test_ranking_training`, `test_osd_objective`, and `test_osd_bf16_loss`.

**38 tests passed in 3.554 seconds.** These cover direct loss/gradient agreement, U/P weighted coefficient mass, canonical-prefix boundaries and EOS, full-vocabulary rivals, stable FP32 ranking arithmetic, reference isolation and activation release, full-block conditioning, chunk invariance, exact lambda-zero compatibility including optimizer/RNG state, ordinary schedule replay and failure gates.

An initial invocation used a nonexistent `tests` package prefix and ran no tests; the corrected module invocation above is the reported result. An additional overlapping trainer/schedule run passed 13 tests. Neither run is a scientific training experiment; all models/data were small CPU fixtures.

## Limits and reporting reminder

- CPU fixtures do not qualify the actual A800 Flex Attention/FSDP/BF16 path or establish full-size peak memory. Reference cloning and the extra reference forward need the separately proposed GPU qualification.
- U/P coefficient mass equality is mathematical equality subject to FP32 reduction rounding, not a promised bitwise identity between different summation orders.
- Existing accumulated `sum_metrics` must still be interpreted at optimizer boundaries or directly from anchors; naively summing both intermediate and accumulated rows double counts exposure. The new prefix/ranking sum metrics inherit that existing logging convention.
- No GPU qualification, production training, generation, or performance conclusion was made in this review.

## Final report and installation audit

Reviewed `REPORT.md`, the final source/test manifest and installation receipt, `CPU_TESTS.json`, and the final local CPU log. All eight final source/test files match their recorded source digests and sizes. The recorded final suites both passed 369 tests, with zero failures/errors/skips and CUDA uninitialized; these supersede the narrower independent 38-test check above without implying this reviewer reran all 369 tests.

No blocking report issue found. The report correctly distinguishes the installed four production files plus four tests from unchanged baseline artifacts. Its future qualification arithmetic reconciles: five disposable one-step trials, ten current-model presentations, four active-reference forwards and four extra detached forwards, totaling eighteen top-level draft forwards plus checkpoint recomputation, with zero generations. The two future U/P events require 256 main optimizer steps, and the fresh four-arm panel requires 768 evaluation generations. These remain proposals, not authorized/executed GPU work. CPU-only scope, frozen O artifact retention, equal-coefficient-mass limitations and absence of a performance claim are explicit.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_prefix_ranking_implementation_20261025/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
