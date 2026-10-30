# KW neutral-update repair: qualification passed

## Material Passport

ARS-Codex 3.24.0 experiment-agent run/validate. Executed source:
**`708e974e06f5d9f4b186c32607c16b37cbd4b14f`**, [PR #19](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/19).
This task covers debugging and bounded requalification only. Stage 1 / project None
is unchanged. The production events 9–12 campaign remains unstarted.

## Result

**Both zero-allocation KW windows now match ordinary K exactly.** All 31 checks
passed, including initial model/Adam/RNG state, all four backward boundaries,
both normalized pre-clip gradients, preview isolation, both final neutral states,
and nonzero optimizer-preview agreement at allocations −0.25, 0.137 and +0.25.
There was no tolerance relaxation, optimizer substitution or algorithm change.

## What was wrong and what changed

The original test restored model weights, Adam state and RNG, but compared arms
with different process histories. It ran four K responses first, then constructed
KW in that same interpreter. The author runtime retains compiled FlexAttention
and block-mask functions in process-wide singleton objects. Closing a trainer
does not reset those functions. KW therefore inherited K's earlier shape history.
A bound-method reference also kept the old K optimizer tensors alive during KW.

The repair runs K and KW in separate fresh interpreter processes, with the same
retained K8 initialization, RNG, four response identities, settings and shape order.
It adds no model warmup. The two worker records confirm fresh CUDA/compiler state
at entry, identical pinned environments and distinct process IDs. Their logs both
record the same attention recompile guard when key length changes from 8,839 to
7,812. Ordinary K's four complete response records also exactly reproduce those
from the original failed qualification.

This identifies a defective comparison harness and shows that fixing startup
parity resolves the checked neutral mismatch. **It does not identify the specific
compiled kernel or reduction responsible for the old coordinate differences.**
The original run did not retain those gradient tensors or compiler traces. The
fresh-process repair also removes the retained-K-storage asymmetry, so its effects
were not independently isolated. We do not label the old failure harmless or
claim that arbitrary compiler histories are bitwise interchangeable.

A second, separate harness bug was fixed: optimizer-only restoration passed its
saved dictionary directly to PyTorch, which can retain CPU Adam step tensors by
reference. A later temporary step could mutate that supposedly fixed snapshot.
Restoration now deep-copies the payload. This branch was not reached in the
original failure and therefore cannot explain its first neutral mismatch.

The KW numerical core is unchanged: native unweighted gradients are preserved
at `a=0`; nonzero weighting, BF16 arithmetic, clipping, native AdamW previews,
delayed credit and all fixed hyperparameters retain their definitions. The sole
production-path addition is a read-only qualification callback after previews.
Cleanup failures and exact-comparison failures now write durable diagnostics;
complete mismatch paths, magnitudes and bounded coordinate examples replace the
old truncated list. These diagnostics do not change acceptance criteria.

## Exact native checks

| Check group | Checks | Result |
|---|---:|---|
| Ordinary boundary normalization | 2 | Exact |
| Initial K8 model/Adam/RNG/counters/modes | 1 | Exact |
| Per-response backward gradients | 4 | Exact |
| Normalized pre-clip gradients | 2 | Exact |
| Live state and gradient isolation across neutral previews | 4 | Exact |
| Neutral final model/Adam/RNG/counters/modes | 2 | Exact |
| Response metrics and anchors | 1 | Exact |
| Nonzero native preview, Adam and isolation checks | 15 | Exact |
| **Total** | **31** | **No differing paths** |

The first two allocations remain zero. By the end of the second window, the
controller has applied one delayed credit and recorded four previews, as specified.
The credit is applied inside window 2 after its allocation has been frozen; it
does not retroactively change that allocation. This validates the bounded
implementation behavior, not learned-weight usefulness or held-out drafting.

## Work and resources

| Full-model qualification work | Completed | Ceiling |
|---|---:|---:|
| Response forwards | 8 | 8 |
| Response backwards | 8 | 8 |
| Disposable optimizer applications | 14 | 16 |
| Serving generations | 0 | 0 |
| Persistent optimizer updates | 0 | 0 |

All 30 recorded work attempts completed; none is unknown. The 14 optimizer
applications are two ordinary K steps, four KW previews, two neutral KW steps,
and six optimizer-only preview/native checks. The original failed attempt's
six forwards, six backwards and five applications remain separately recorded.
No additional full-model diagnostic passes were executed outside this rerun.
CPU fixture work is separate: 68 combined local tests and 35 A800 tests passed;
the suites overlap. A separate reviewer also ran focused CPU checks.

Qualification took **392.921 seconds (6m33s)**. The successful wrapper, including
read-only admission and CPU tests, ran **01:38:21–01:45:59 UTC: 458 seconds (7m38s)**.
Ordinary K and KW workers took 158.982 and 213.390 seconds respectively, including
instrumentation, state serialization and comparisons. These are not production
training-overhead measurements. Internal checkpoint recomputation is included
in elapsed cost; its individual operator calls were not separately instrumented.

Observed peak GPU allocation was **19.069 GiB for K** and **29.770 GiB for KW**;
KW peak reserved memory was **30.221 GiB**. No out-of-memory error occurred. This
qualifies the specified four retained shapes, not every possible future input.

An initial read-only admission attempt stopped because deployment rewrote the
tracked retrospective `GITHUB_SOURCE.json` after source sealing. The exact
committed file was restored; the failed admission, timestamps and traceback are
preserved in `raw/admission_failure_001/`. That attempt performed zero model
forwards, backwards or optimizer applications. Execution identity is carried by
the sealed PLAN and publication record, rather than rewriting that tracked file.

## Preservation and deliverables

Postflight verified **2,641 source/input files, eight original state/export/receipt
files and four teacher-feature files**, with no model work. Original O8/K8, the
original failed qualification and its source copy remain intact. Both GPUs are
idle. Thirty-nine retrieved compact files were independently hash-verified.

The 27.265 GB of temporary K comparison states/gradients remain on A800 for review;
none is a serving checkpoint or production initialization. A800 has 190.751 GB
free after this qualification. A future production launch must readmit storage;
these temporary files are not silently counted as already retired.

- [Exact checks](analysis/CHECKS.csv), [work/cost summary](analysis/SUMMARY.json),
  [execution source](analysis/SOURCE_PIN.json).
- [Source diagnosis](SOURCE_DIAGNOSIS.md), [code review](CODE_REVIEW.md),
  [separate result review](RESULT_REVIEW.md).
- Compact raw evidence: this directory's `raw/`.
- Full comparison tensors: A800
  `/data/sources/minhao/runs/dflash_kw_v1_20261029/neutral_repair_001/implementation_qualification/transient/`.

The requested repair and bounded qualification are complete. No production event,
serving comparison, algorithm tuning or additional qualification retry follows.
