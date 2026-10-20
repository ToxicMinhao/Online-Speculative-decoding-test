# Direct-full-gradient proposal: numerical protocol v2

## Status and authorized scope

The previous native qualification failed because separately computed BF16 A/B
gradients did not satisfy the declared comparison with a direct full-loss
gradient. The [bounded diagnostic](GRADIENT_DIAGNOSTIC_RESULTS.md) preserves that
failure and its measured magnitude. Repeated compiled backwards and reversing
their order were bit-identical in the inspected case; equivalent eager attention
also failed the sum comparison. These observations support finite-precision
non-additivity, without locating its first operation or establishing filtering
efficacy.

The user authorized this narrow repair and correctness qualification. The
[native result](NUMERICAL_REPAIR_QUALIFICATION.md) now shows exact disabled-filter
equivalence to unchanged H, while N fails its unchanged 1% norm control. **Full
qualification remains unsuccessful and the main pilot remains paused.**
Historical results and their source versions remain intact; no retuning follows.

## The change

At each G/N pass, use one sampled response objective at the current weights:

1. Compute the full forward-KL loss and its chronological A/B partial losses from
   the same stochastic forward, anchors, positions, target conditioning and full
   denominator.
2. Compute raw A and B gradients, retaining the graph. Keep their existing FP32
   copies of native autograd outputs.
3. Compute the direct full-loss gradient on that same graph. Retain its native
   dtype, BF16 in the qualified runtime.
4. Clone this direct full gradient into the parameter gradient buffers. Apply the
   qualified global clipping operation once and advance ordinary AdamW once.
5. Use A/B only to filter the resulting signed displacement or to calculate N's
   own hypothetical filter and diagnostics. Never use A+B to construct the
   ordinary proposal.

The clone matters because clipping mutates `.grad`: the saved full-loss gradient
must remain unchanged for diagnostics. There is no optional A+B fallback.
Both group backwards retain the graph; the final full-loss backward releases it.
Normal G/N training now costs **three backwards per pass**, rather than two.
Each pass has one optimizer step, so the two-pass response event has **six
backwards and two optimizer steps**. H/L remain on the unchanged ordinary
single-backward-per-pass trainer: **two backwards and two optimizer steps per
response event**.

G keeps the exact ordinary proposed native weight or restores the exact previous
weight coordinate. N still uniformly scales its own ordinary proposal to its own
hypothetical filtered norm and stores the resulting native BF16 weights. Neither
operation filters, resets or substitutes the optimizer's moments or step counter.
Even a zero applied displacement advances AdamW exactly once.

## What remains fixed

- Released DFlash, full-weight trainable inventory, native BF16 parameters and
  Adam moments; no FP32 master weights.
- Forward KL, target temperature, position weighting, full weighted denominator,
  newest-response loss mask and full causal history.
- Fixed chronological group membership by absolute output position, including
  overlapping occurrences; normal per-pass anchor sampling and seed 17.
- The existing optimizer, clipping, two passes/two steps, and learning rates:
  H/G/N `3e-4`, L `1e-4`.
- Product-free sign comparisons on the raw A/B gradient buffers.
- `torch._functorch.config.donated_buffer=False` for all learning arms, with
  effective configuration recorded and drift rejected.
- Native output preservation, checkpoint/cache handling, common benchmark inputs,
  independent canonical histories and no learning from benchmark replays.
- N's existing **1% per-step relative actual/intended norm bound**. An intended
  zero requires exact actual zero. Every ancestral N update must qualify;
  missing or failed control evidence blocks selective-filter attribution.

There is no precision change, learning-rate adjustment or tolerance widening.
This is nevertheless a **numerical algorithm and qualification-protocol change**:
the ordinary proposal now uses the direct full gradient, and agreement between
A+B and that gradient is no longer a condition for accepting the update.

## Qualification and historical diagnostics

The old comparison remains numerically unchanged:

```text
abs(float32(gA) + float32(gB) - float32(gFull))
    <= 8 * eps(parameter_dtype) * (abs(gA) + abs(gB) + abs(gFull)) + 1e-8
```

Every grouped pass retains its compact gradient-sum summary, including failures,
per-parameter/global norms, violation counts and bounded failing-coordinate
samples. A failed sum comparison remains labeled failed; it is not relabeled as
a pass. The receipt states that it is diagnostic rather than an admission gate.
Nonfinite gradients remain fatal, with durable evidence.

The relevant qualification now compares two complete events from the exact same
initial model, optimizer, feedback and RNG state:

- The unchanged ordinary OSD event: two passes and two optimizer steps.
- The repaired event with filtering disabled: two passes and two optimizer steps,
  using the direct full gradient for both proposals.

The existing equivalence criteria remain: parameter differences within one native
ULP plus `1e-7`; total displacement error no more than 10% of the reference norm;
no missing nonzero reference update; separately pinned moment tolerances; and
exact counters, anchors, exposure, sampler/global RNG and final restoration.
Passing CPU fixtures does not establish native GPU qualification.

The first-stream qualification retains its existing H/L/G/N sequence:
three requests per learning arm, 12 canonical generations, eight persistent
qualification events/16 steps and ten scored replays. G's isolated comparison
adds four temporary optimizer steps, restored afterward. These are correctness
checks and qualification costs, not campaign performance results.

## Cost and memory evidence

Per-pass records separate A-backward, B-backward, full-backward and gradient-sum
diagnostic time. They also record native full-gradient storage bytes and CUDA
live allocated/reserved memory plus the running event peak before/after these
phases. Complete training time includes teacher preparation, state handling,
diagnostics, gradient-buffer cloning and optimizer work; nested timers must not
be added again.

The change in the running event peak after the full backward is an observation
within this execution. It is not a counterfactual measurement of how much memory
a two-backward implementation would use: earlier work may already have set a
higher peak. Native buffer bytes describe stored tensors, not total allocator or
driver memory. The additional backward cost cannot be hidden by matching only
optimizer-step counts.

## Execution and artifact compatibility

Use `run --qualification-only` with the reviewed source/reference hashes and all
normal runtime arguments. On success, the controller records:

```json
{
  "status": "qualification_complete",
  "campaign": null,
  "campaign_started": false
}
```

The execution-scope receipt forbids campaign execution in this mode. Failed
correctness or numerical-control checks preserve evidence and stop. A passing
qualification also stops for review; it does not authorize the bounded pilot.

Numerical/group protocol metadata are versioned in `core.NUMERICS` and
`training.GROUP_PROTOCOL`, and recovery checks require exact agreement.
Old snapshots are intentionally incompatible with the new rule. Reproduce old
qualification or diagnostic states with their archived source version rather
than silently migrating metadata or continuing their learning under v2.
Private responses, raw runtime artifacts and model states remain outside Git.
