# AdaFlash C1 implementation and execution plan

The user-approved experiment adds one paper-defined AdaFlash OPD arm in DFlash-Online, with the existing PR25 full GSM8K protocol. The objective is 0.8 teacher-argmax CE plus 0.2 reverse KL with vocabulary-entry upper clipping at 0.01. Negative entries remain unchanged. Ordinary response-only masks, sampler, positional decay, BF16 arithmetic and response averaging are preserved; no K/K+ mask or adaptive-length head is used.

This intentionally follows [AdaFlash v3](https://arxiv.org/html/2607.19223v3), whose clipping order differs from the [pinned public trainer](https://github.com/ZinYY/AdaFlash/blob/478bf0ff7a3a5232b13d0f14e752d5fe1da0b718/pipeline/training/rkl_model.py#L168-L207). It is a matched-protocol paper-loss comparison, not a full AdaFlash reproduction.

A separate same-team source/implementation review found no blocking scientific issue. The reviewer supplied an independent analytic fixture and inspected the shared working tree; this is not external or blind review. The final executor checks passed 40 new CPU tests and 27 existing objective/storage regression tests. A tiny full-event CPU fixture checks 128-step accounting and exact resume. Native GPU checks remain required and must not be inferred from these CPU results.

## Fixed execution

- One released seed-17 A initialization, fresh native BF16 AdamW; 7,473 own-feedback requests, 58 events, 14,848 presentations and 7,424 AdamW steps. Final49 responses do not train.
- Reuse retained O58/K+58 and retime allthree with three balanced repeats on the original1,024requests:9,216 timed completions. Four inherited native correctness generations are separately counted; total16,693complete generations.
- Tiny numerical GPU qualification has12scalar-loss forward/backward evaluations,zero model/optimizer steps, and preserves RNG. Failure diagnostics are written before stopping. No extra model warmups or tuning.
- Exact canonical tokens/termination, checkpoint identities, repeated counts, runtime pins and training-free/transfer-free timing are enforced. Valid unfavorable results continue.
- Primary K+/A complete-request latency uses per-request repeat means and paired bootstrap10,000draws,PCG64seed20261102. Calls use one repeat-stable trajectory. Training costs remain separate.
- Full-cap raw output storage plus rolling checkpoint overlap, feature buffers, compiler/workspace and8GiBreserve requires143.200GB; current post-retirement capacity149.429GB. Retirements are separately documented, preserving compact evidence and selected exports.

Expected execution duration is initially11–15hours from launch, based on comparable retained full-horizon runs. This estimate will be refined from the first actual event without changing settings. The sealed plan and every generated report identify the clean published execution commit. No native result is claimed in this implementation note.
