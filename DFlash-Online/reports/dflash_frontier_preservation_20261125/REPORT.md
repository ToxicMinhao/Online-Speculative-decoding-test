# Frontier preservation: native derivative feasibility

**Status: blocked before production by unsupported native derivatives.** Ordinary native BF16 first-order backward passed, but the pinned compiled FlexAttention path rejected both routes for the parameter-to-hidden directional derivative required by the candidate. No continuation event, new feedback collection, optimizer step or serving evaluation ran; there is no serving result or confidence interval. Preserve the candidate mathematics and this implementation, but do not launch the event until a compatible derivative backend has been implemented and qualified. This result does not establish that FRONTIER_P would help or fail to help drafting.

## Material Passport and source

- ARS experiment-agent run/validate workflow; source, state and work receipts independently reviewed within the same shared-context agent team. This is not an independent GPU replication.
- GPU execution: **`da4f4cfe28844141a5548401ca86e0ec08e73abf`**, `execution_001`, physical B `GPU-c6943f33-bcd0-a724-0b35-f77af6b0a3f1`. All **5,104 staged files** were checked against the execution manifest immediately before launch.
- Parent: PR59 publication `a780c94fbac2c4e5a2d31b22cb99d0c380ec85b2`, scientific execution `416cb7931028621c09b8fa983e53fc8a7dcebaab`; original PR56 recipe `1551e5e5d0dd7736d8441977aa3dea250535da7b`.
- Parent recovery: 6,296,473,051 bytes, SHA256 `728770a7abf0ba3ef96805ef760997976ad7bedbbee1d8debe165e785a4d98a2`. Loaded and restored **step128, 256 presentations, event1, 58 warm Adam state entries**. Parent recovery/export, full original input manifest and both selected packets hash identically before and after this task. [Identities](IDENTITIES.json), [state checks](STATE_CHECKS.json).
- Code follow-up `fd2bb9e56b32a63b5c62563d533512cb53e3d7d9` includes a subsequent CPU-only invariant-check repair and evidence exporter. Neither was invoked in the native prerequisite. The original executed source and failure records remain intact; see `SOURCE_CHANGES.md` for that distinction.

## Intended comparison and implemented scope

The agreed event would fork CONTROL and FRONTIER_P from the same complete PAIR1 state and share 128 parent-generated responses from original PR56 training indices128–255. Both retain K+PAIR, native BF16/FlexAttention, constant LR3e-4, warm AdamW, two passes and128 new updates. CONTROL retains C; FRONTIER_P replaces only its prefix-preservation allocation using the signed sensitivity of reachable-progress assessment through an FP32 optimizer shadow. Both would finish at cumulative step256/presentations512 and receive768 contemporary serving scores plus24 declared warmups.

Implemented components are frozen L/e/H/K roles and boundaries, fresh-rival assessment, signed mass-conserving allocation, globally coupled clipping/Adam-shadow VJP, coefficient-gradient contraction, exact parent-import/new-binding helpers, CPU tests, resource/source admission, native prerequisite runner and evidence export. The native optimizer, K+PAIR objective and attention implementation were not changed. Graduate was untouched.

**These components are not a completed production trainer.** Full native score composition, final replay/RNG integration, disabled-path equivalence, beta-reference comparison, active BF16 intervention, event_002 serialization, collection, witnesses and endpoint serving remain unqualified or unimplemented. The handoff explicitly permits a useful implementation plus precise blocker report when the required native derivative is unsupported. We stopped at that boundary rather than implementing an unexecutable production coordinator or substituting a token-local score.

## What the native gate established

Selected the first two ordered authenticated PR59 PAIR training packets, independent of outputs or proposed improvement. Their native anchor sets were sampled once from the restored parent RNG and frozen: **512 and269 anchors**. No evaluation data entered the check. Each derivative route stopped at its first response failure; its second response was unattempted.

| Check | Outcome | Interpretation |
|---|---|---|
| Original complete-state import | Passed | Exact warm parent, original contract and clean accumulation boundary |
| Ordinary native hidden forward and first-order parameter VJP | Passed;58 finite BF16 gradient tensors | Native forward/backward capability works; this hidden-output control is not a new ordinary-loss qualification |
| Forward-mode parameter-to-hidden JVP | Failed,0.114s | Compiled attention's custom autograd function lacks a `jvp` implementation |
| Reverse-over-reverse equivalent JVP | Failed,0.281s | Forward and first parameter VJP complete; AOTAutograd rejects differentiation of that backward with respect to its input cotangent |
| Final live-state restoration | Passed, zero mismatches | Weights, Adam state, counters, history and RNG preserved; no diagnostic state becomes a training origin |

The exact errors are:

- `NotImplementedError: You must implement the jvp function for custom autograd.Function to use it with forward mode AD.`
- `RuntimeError: torch.compile with aot_autograd does not currently support double backward`

The first trace reaches the unchanged author wrapper at `training/specforge/modeling/draft/flex_attention.py:57`, then the generated AOTAutograd function. The second fails at the final cotangent derivative in `native.py:217`, after its first parameter VJP succeeds. This is a missing derivative operation, not an Adam-state mismatch, loss discrepancy, nonfinite gradient or BF16 tolerance failure. [Forward trace](forward_ad_jvp_0_TRACEBACK.txt), [reverse trace](transposed_reverse_jvp_0_TRACEBACK.txt), [machine-readable results](QUALIFICATION.json).

Pinned Torch is **2.13.0+cu130**. Its actual Flex operator source is51,950 bytes, SHA256 `0c5bb2a496f06df71b3e203923ef34fdb65c6a309ba966f091555736582c5471`. Source inspection also finds no JVP on its underlying `FlexAttentionAutogradOp`, and an explicit deferred `autograd_not_implemented` registration for `flex_attention_backward` at line1481. Merely removing the compile wrapper is therefore not an evidenced fix. `autograd.functional.jvp` uses the same double-backward route; `torch.func.jvp` still requires forward-mode support. Custom attention derivatives would require a separately qualified backend. These observations do not prove that every conceivable implementation is impossible.

## CPU qualification and numerical detail

**33 focused CPU tests pass**: role/EOS/gap/empty/tie cases, shortfall bound, current rival gradients, direct-loss support, signed/constant/singleton allocation, mass/cap/variance checks, warm Adam shadow versus independent FP64/native Adam references, clipping branches, cross-position sensitivities on a shared-parameter model, complete parent import, original-validator guards, failure persistence, source authentication and resource admission. [Test output](CPU_TESTS.log).

A review fixture found a CPU-only false rejection when adding a large common score offset. Allocation itself was unchanged, but the raw dot-product invariant amplified the small FP32 mass residual. The follow-up checks the algebraically equivalent centered dot product with the **same tolerances**, while retaining the raw dot and its discrepancy. No cap, coefficient, score floor or scientific formula changed. The regression now passes; this helper was not invoked during the native gate. [Preserved failing fixture](CPU_MATH_COMMON_OFFSET_BEFORE_FIX.log), [passing tests](CPU_MATH_AFTER_CENTERED_GATE.log).

An initial local CPU test command omitted `CUDA_VISIBLE_DEVICES=''`: the RNG utility initialized a Fenrir CUDA context and four CPU-isolation teardown checks failed. No model training occurred. The command was corrected to mask CUDA, with the failure log retained privately; all final tests pass. This was separate from the single A800 attempt, which had no retry.

## Work, duration and resources

| Work | Completed / disposition |
|---|---:|
| Parent runtime construction/import | 1 |
| Frozen training responses | 2, anchors sampled once |
| Full native draft forwards | 2 |
| First-order parameter VJPs | 2 |
| Forward-AD attempt | 1 failed before a full forward completed |
| Cotangent second-derivative attempt | 1 failed |
| Second-response derivative forwards | 2 unattempted, one per route |
| Vocabulary-head calls / ordinary-loss presentations | 0 /0 |
| Adam updates / shadow optimizer previews | 0 /0 |
| New feedback / serving generations | 0 /0 |
| Production events / endpoint checkpoints | 0 /0 |

The durable ledger retains the two incomplete primitives. Hooks record **3 draft entries and11 layer entries**, consistent with two completed five-layer forwards plus entry into the failed forward-AD path. These are entry counters, not three successful forwards; partial kernel work inside failed calls is not reconstructed. No extra checkpoint-recomputation or optimizer work was claimed. [Work table](WORK.csv), [raw evidence inventory](RAW_EVIDENCE_IDENTITIES.json).

The native probe took **36.928s**; the supervised child occupied one card for **40.939s** (0.01137 card-hours), including construction and compilation/control work. Peak allocated/reserved CUDA memory after parent import was **46.102/46.913GB**; counters were reset after initialization, so initialization peak is not independently measured. Thus this attempt stopped on derivative support, not an observed out-of-memory failure.

A800 source/run allocation was0.329GB; measured Fenrir study allocation0.336GB, approximately0.665GB combined, plus small unmeasured shared-Git bookkeeping. Compiler-cache growth was0. The derivative-only admission was6GB, inside the40GB handoff ceiling. Postflight A800 free space was **431.792GB**, above the400GB reserve; both GPUs were idle. No old artifact or newly produced tensor checkpoint was deleted. [Resource and kernel-source evidence](RESOURCES.json).

**There is no running full event and no supportable full-event ETA.** Candidate update time cannot be extrapolated from a failed prerequisite. The original4-hour/8-card-hour ceilings remain unchanged; they are limits, not a duration forecast.

## Remaining decision

Serving improvement, useful-repair gains and coefficient activity are **unmeasured**. No result supports accepting or rejecting FRONTIER_P on efficacy. The narrow prerequisite is a supported, numerically qualified parameter-direction derivative through the same native attention computation. That would need its own bounded implementation/resource decision before finishing full qualification and the original event. No kernel substitution, tuning, further event or automatic retry follows from this report.

Independent receipt/source audit: [AUDIT.json](AUDIT.json). It checks retained work and provenance, not an independent GPU rerun.
