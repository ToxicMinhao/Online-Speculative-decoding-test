# Independent source and diagnostic review

## Material Passport

ARS experiment-agent, validate mode; October 26, 2026. Verification status:
**ANALYZED**. This reviewer read local source and retained evidence, and reviewed
the standalone diagnostic. No GPU execution, remote operation, production edit,
or clean-base edit was performed. The authorized scope is three one-step
O1/O2/U0 trials and the separate 16-generation diagnostic in PLAN.md. The failed
original exact gate is unchanged; no main U/P training is authorized by a
diagnostic result alone.

## Source conclusions

Lambda zero takes the ordinary OSD execution branch: `ranking_enabled` is false,
`begin_event` creates no reference, and `forward` never calls `_forward_ranking`.
The ordinary chunk loss receives no ranking coefficients. This excludes active
ranking/reference arithmetic as an explanation of the previous O/U0 failure.
U0 still loads EOS metadata under `preserve_rng` and carries different objective
metadata; initial-state comparison is therefore necessary rather than assuming
all setup is identical. Sources: `worktree/online_feedback/objective.py:94,108,262,294`
and `training.py:235` in the preceding one-event directory.

The training runtime is Torch **2.13.0+cu130** according to the coordinator's
current runtime inspection, matching the local inspected Torch version. The
2.9.1 value in the previous PREFLIGHT.json belongs to the separate serving
runtime; it must not be used to identify training kernels. New `runtime_flags`
records the training flags directly without changing them.

The baseline calls `set_seed(17)` without enabling deterministic algorithms.
That does not itself prove any participating kernel is nonrepeatable. In
particular, a blanket assertion that FlexAttention backward uses nondeterministic
atomics is unsupported here: the forced Triton path in
`training/specforge/algorithms/common/dflash_family_model.py:660–678` leads to
local Torch's `flex_backwards.py.jinja`, where dQ and dK/dV are accumulated within
programs and stored (`:218–224`, `:359–381`). Its backward lowering selects kernel
configurations through `autotune_select_algorithm`
(`torch/_inductor/kernel/flex/flex_attention.py:936,1062`). Actual generated
kernel identity would be needed to attribute a difference to a particular kernel.

The Flex wrappers are process-wide compiled singletons
(`/home/minhao2/DFlash-Research/training/specforge/modeling/draft/flex_attention.py:14–43`).
Closing and recreating the trainer does not reset these wrappers. O1 is first in
the process; O2 and U0 can reuse compiled kernels. The three pairwise comparisons
are therefore useful: O1 differs from both while O2 equals U0 would show an
order-associated pattern in this run, not establish an objective-specific bug or
prove a compiler cause.

## Bounded hypotheses and discriminating evidence

| Candidate boundary | Code basis | Smallest check already in the diagnostic | Interpretation limit |
|---|---|---|---|
| Fresh initialization or state isolation | Model construction follows reseeding; U0 has additional guarded metadata setup. Optimizer and DDP are constructed afresh. | Compare initial draft tensors, optimizer state/options, RNG, module modes, parameter order/stride and gradient-None masks across all three trials. | If they differ, post-step differences cannot be assigned to the loss or backward. |
| Native backward execution | Same BF16 model and chunked KL graph feed the baseline backward. Multiple graph paths share trainable tensors. | Compare the full first-response gradients and RNG after backward, with identical inputs, anchors and initial state. | Equal scalar loss and scalar gradient norm do not imply equal gradient coordinates; disagreement localizes before the optimizer but does not name a kernel. |
| Second response and accumulation | TrainerCore divides each loss by two and accumulates two responses (`controller.py:448–467`). NO_SHARD is DDP with bucket views (`backend.py:258–277`), not an unwrapped single model. | Compare first and second backward snapshots, then normalized pre-optimizer gradients. | If only the second snapshot differs, this includes both the second response's backward and accumulation; snapshots alone do not distinguish those two. |
| Gradient scaling, clipping or AdamW | TrainerCore normalizes before the adapter; DirectBF16AdamW clips once, updates BF16 parameters/moments, then clears gradients (`optimizer.py:40–50`). | Compare raw accumulated gradients, normalized pre-clip gradients, recorded norm, final weights and every Adam tensor. | Equal pre-clip gradients but unequal final state localizes to clipping/update or intervening state; no post-clip snapshot is present to separate clipping from Adam. |

Shared-gradient arithmetic is a plausible investigation boundary, not a proven
cause. The model uses the same K/V projections for context and noisy blocks
(`draft/dflash.py:386–389`) and shares projected target context across layers
(`:872,889`). BF16 additions can depend on summation order, and native
`AccumulateGrad` adds into an existing gradient tensor
(`torch/include/torch/csrc/autograd/functions/accumulate_grad.h:85–89`). However,
the engine also explicitly orders ready nodes by reentrant depth and sequence
number (`engine.h:91–104`); this review has not demonstrated that those orders
differ between the trials.

Chunking alone is a weak explanation. `checkpointed_chunk_reduce` uses fixed,
disjoint slices, non-reentrant checkpointing and a fixed Python sum order
(`training/specforge/core/chunking.py:60–89,102`). Its slices share the upstream
hidden tensor, but there is no source evidence here that their accumulation
order changed. Keep checkpointing and all numerical settings unchanged during
the initial three-trial diagnostic.

## Runner and evaluation review

The reviewed final runner retains initial state, both raw backward-gradient
states, normalized pre-clip gradients, final weights/Adam/RNG, restart state and
export **before** numerical comparison. It compares O1/O2, O1/U0 and O2/U0 without
stopping at the first tensor mismatch. Native-dtype inequality determines exact
equality; FP64 is used only for bounded discrepancy statistics. It preserves the
full 128-response shuffle and the original first-two-presentation/anchor checks.
Ranking/reference inactivity and the three-step/six-forward budget are checked.
Numerical differences are recorded as observations, not relabelled as passing
the original qualification.

`evaluate.py` loads only the separate diagnostic exports after the training
process exits, checks native publication and tensor equality, and validates
the first four saved requests against canonical outputs/termination/native
traces. F additionally checks historical verification-call counts. O1/O2/U0
are intentionally not checked against the full 128-step O checkpoint's counts.
There is no timing-speedup claim and no training feedback capture.

Two preparation findings were sent to the coordinator and corrected in source:
the storage gate now admits 100 GiB planned writes plus 8 GiB reserve, and
preflight now checks hashes for the new diagnostic driver files in addition to
the retained production source. The coordinator subsequently refreshed the
diagnostic scope and sealed six diagnostic files; this reviewer independently
checked their local hashes against that final seal. No remaining
runner/evaluation blocker was identified.

Instrumentation copies gradients to CPU and persists them between backwards.
It therefore adds synchronization and changes timing relative to the failed
qualification. Agreement in all three instrumented trials would establish this
bounded diagnostic's agreement, not explain the earlier discrepancy or certify
128-step equivalence. O/O disagreement would show that the observed mismatch is
not uniquely associated with the U0 label in these trials, without proving that
the same mechanism caused the earlier failure.

The validation makes no efficacy, significance, population-generalization or
performance claim. Statistical fallacies involving p-values, confidence
intervals, multiple testing, sample representativeness and causal effects are
not applicable to this source audit. The relevant checks are explicit: do not
infer coordinate equality from rounded summaries; do not infer cause from
branch labels or a single O/O discrepancy; do not infer full-run equivalence
from one step; and do not treat added diagnostic synchronization as invisible.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_prefix_ranking_gpu_diagnostic_20261026/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
