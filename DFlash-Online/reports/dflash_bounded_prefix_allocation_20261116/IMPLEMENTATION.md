# Bounded prefix allocation: implementation and CPU review

Status: implementation complete; native qualification and production pending.
Scientific base: `9b2d8d8752c5b48e34a3537743ac349ab159b284`.
The actual execution commit is recorded in the sealed source manifest, plan,
qualification receipts and final report. This document makes no GPU result claim.

The new `online_feedback/prefix_allocation.py` is an optional extension after
native K+. It uses the existing BF16 log-softmax, detached FP32 probabilities
and bounded prefix coefficients. The original eleven K+ terms and response-wide
correction are reduced first; the additive prefix adjustment follows, before
the unchanged original denominator. Cap zero or exactly unchanged coefficients
returns the native numerator object. A zero-valued adjustment with nonzero
coefficient movement keeps its gradient.

The runner continues authenticated complete K6 to C7 and then complete own C7
to C8, collecting C7's own feedback between events. N8 and completed restart S8
remain read-only controls. Existing training, feedback preparation and native
serving components are reused. The old evaluator gains optional method/session
arguments while retaining its original defaults.

CPU verification comprises **65 new tests**: 12 objective/property tests and
53 preparation, serialization, scheduling, runner, qualification-orchestration,
staging and analysis tests. An additional 43 inherited objective tests and 11
inherited evaluation tests pass. The final 53-test orchestration suite took
48.819 seconds. These are CPU checks; they do not establish native parity,
resource fit or efficacy.

A subsequent staging review adds an explicit owned-process open-file/mapping
check before retiring byte-equivalent new feature duplicates, plus the fixed
C7/N7/S7 secondary count reducer. Thirteen additional CPU tests pass (four
retirement checks and nine secondary-analysis tests), bringing the new-test
total to **78**. These changes add no model work or algorithm adjustment.

Review caught and repaired three execution issues before GPU work: a final
qualification restore drew another permutation after restoring RNG; one generic
reducer still used the old E arm name; and failure cleanup could obscure the
last qualification receipt. The full-prefix diagnostic table was also extended
to include its boundary row. No allocation formula, scientific tolerance or
experiment setting changed. Initial test invocations with a Python lacking
Torch or an incorrect test-module name are retained as local invocation errors;
the correct environment and complete test paths pass.

The planned qualification uses five disposable Adam applications, ten response
presentations plus two separately counted startup forwards/backwards, and eight
native generations. It checks exact native/cap-zero gradients and complete
state, real active coefficient/parameter movement, original-forward membership,
own-state save/reload and output preservation. All failed work remains charged.

The final package contains every tracked source file and all 128 existing
canonical references before worker construction. The same unpacked reader is
tested locally and on A800. Full recoveries, source/runtime/data identities,
compiler paths, current host/VRAM and peak storage are checked before the first
model load. Planned new peak allocation is 58.34 GB within the 80 GB cap and
400 GB free reserve. Initial execution estimate is about one to two hours,
updated from actual qualification measurements before production admission.

Only the fixed handoff is authorized. No new reference generation, N/S training,
coefficient tuning, favorable-subset selection or automatic confirmation is
implemented. Negative valid outcomes complete the experiment; invalid work is
preserved and cannot be counted as a negative efficacy result.

Administrative staging now reuses byte-verified unchanged source from the retained PR49 checkout and transfers only changed source plus every sealed input/reference. The complete assembled source is verified before any model work. Nineteen staging CPU tests pass, including exact source partitioning, stale or altered base rejection, and a local-only dry-run override forbidden for production. This replaces seven earlier staging tests (90 unique new CPU checks total). No scientific setting or execution budget changed. The failed rsync attempt and stopped slow full-source transfer consumed zero model work and remain recorded.

A final read-only review found that the compact coefficient table used repeated-index CUDA floating accumulation, which could falsely fail exact replay of diagnostics. It now uses a fixed-layout masked sum. This is a diagnostic-only fix: coefficients, loss graph, gradients and numerical gates are unchanged. The independent table oracle and 56 core/inherited CPU checks pass (91 unique new checks in total, plus54 inherited). Native qualification still checks exact actual replay.
