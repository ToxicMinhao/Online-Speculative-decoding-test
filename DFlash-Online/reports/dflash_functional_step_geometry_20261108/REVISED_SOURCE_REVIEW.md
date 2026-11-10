# SCALAR ordinary-gradient qualification failure: source and evidence review

## Material Passport

- Date: 2026-11-09 UTC.
- ARS experiment-agent validation; **ANALYZED from retained evidence**, not reproduced by this reviewer.
- Scope: execution `1c72903`, retained GSM ordinary recapture, passed auxiliary checks, stopped SCALAR case, and their source/restoration paths.
- Same-team reviewer with authorship overlap in the qualification helpers. No new GPU operation, optimizer application, generation, source edit or tolerance change.
- Machine-readable companion: `SCALAR_GRADIENT_FAILURE_SOURCE_REVIEW.json`.

## Finding

**No concrete source defect has been demonstrated that explains the ordinary-gradient mismatch. Qualification remains incomplete.** The failure is before SCALAR's Adam application and before SCALAR's geometry is computed, so it cannot be attributed to its correction or scalar-matching algorithm having changed this update.

The preserved exact comparison finds 49 differing parameter tensors out of 58, containing 155,364,420 differing coordinates out of 1,048,626,432. Maximum absolute error is 0.00048828125 and total discrepancy L2 is 0.0016444962594889566. No nonfinite coordinates are reported. This is an exact equality gate; these numbers do not turn it into a tolerance-based pass. A relative gradient L2 value cannot be recovered from the discrepancy-only receipt and is not invented here.

## Confirmed controls

- Both complete ordinary response-observation rows are identical after excluding only the event label, including sampled anchors, losses and all recorded loss/exposure metrics.
- The ordinary recapture also matches the retained historical response observations and presentation metrics. That establishes those scalar/input comparisons, not historical raw-gradient tensor equality: the historical raw tensors were not supplied as the new reference.
- The same-layout independent numerical oracle passed with exact native gradients on its checked fixture. The maximum-shape expanded resource check passed. These are different checks from the later ordinary-gradient identity failure.
- Exact scientific-state restoration passed after both auxiliary checks. It compares the recorded weights, Adam state, counters, RNG, gradients, modes and runtime buffers. SCALAR then restores the original continuation boundary before its ordinary forwards.
- No new Adam application occurred in this attempt. The failing observer runs before the optimizer work counter increments.

## Where differences become visible

Nine parameter gradients match exactly:

- `layers.4.mlp.up_proj.weight`
- `layers.4.mlp.down_proj.weight`
- `layers.4.mlp.gate_proj.weight`
- `layers.4.post_attention_layernorm.weight`
- `layers.4.self_attn.o_proj.weight`
- `layers.4.self_attn.v_proj.weight`
- `layers.4.self_attn.q_norm.weight`
- `layers.4.self_attn.k_norm.weight`
- `norm.weight`

Within the last layer, `q_proj.weight` differs at 3,191 coordinates, `k_proj.weight` at 469, and `input_layernorm.weight` at two. Differences then appear more broadly in earlier layers. This pattern is consistent with a perturbation entering through the later attention backward path, but parameter-gradient aggregates do not locate the first differing operator. Activations and operator adjoints were not preserved at those boundaries. The pattern is not proof of a FlexAttention defect or of harmless variation.

## Source inspection

`recover_ordinary_reference` captures cloned CPU raw gradients before clipping, then performs native clipping/norm measurement and clears gradients without Adam. It restores the exact saved endpoint for export. `run_case` captures its raw gradients at the corresponding pre-optimizer observer. Both use the same response-mean TrainerCore path and ordinary sampler. The source contains no extra denominator, second clipping before the comparison, alternate loss or gradient filtering on one side.

`restore_origin` clears accumulated metrics, observation lists and gradients, reloads the complete original state, restores runtime buffers/modes and checks the original RNG/counters. The helper's isolated evaluation context restores registered buffers, caches, modes and RNG. No leaked global precision setting or installed FP32 head helper was found in this path.

There is nevertheless a **confirmed procedural asymmetry**: the new ordinary reference gradients are captured during the fresh process's first two training backwards, which are also designated startup work. SCALAR runs later, after the independent numerical oracle and the maximum-shape auxiliary computations. The earlier qualification had a separate discarded warm-up before its ordinary reference. Matching the stored scientific state does not reset process-global compiler/autotuner caches, allocator state or all kernel-selection state.

This makes startup or intervening compiled-work effects plausible hypotheses. It does not identify which effect caused the difference, and it does not demonstrate that the source implementation is correct at every unrecorded runtime boundary. No speculative source repair is justified yet.

## Smallest useful next discriminator, if separately authorized

Keep exact gates, weights, two response presentations, anchors and RNG fixed. Compare plain K+ and capture-enabled SCALAR **before Adam**, after the same explicit startup procedure, initially without an intervening resource check. Reuse the already prepared feature packets. If that exact comparison passes, an otherwise identical comparison after the auxiliary resource work can test whether the intervening workload matters. Retain intermediate final-hidden or attention-adjoint comparisons only where needed to locate the first divergence.

This is a proposed zero-Adam diagnosis, not permission to retry qualification or to weaken its equality requirement. It distinguishes a changed measurement procedure from a mechanism effect without adding training or serving.

## Work boundary

The failed attempt consumed two ordinary forwards/backwards, sixteen additional forwards and eight additional backwards, with zero Adam. Cumulative qualification work is therefore three Adam applications, eight ordinary forwards/backwards and 34/20 additional forwards/backwards. The original ordinary-presentation allowance now has eight remaining, enough for the four untouched MathQA cases but not an additional complete SCALAR rerun as well. Compact failure records do not contain a resumable live graph or the full failed gradients. Any additional diagnostic or retry must have explicit accounting and authorization; failed work must not be hidden or reclassified away.

The passed numerical/resource checks remain usable evidence. SCALAR, MathQA and native serving qualification remain incomplete, and the main pilot stays paused. No conclusion about FSG serving efficacy follows from this failure.
