# AR versus frozen DFlash: mismatch investigation

## Material Passport

ARS experiment-agent debugging and retained-evidence review. Parent failed execution: `93a53f11abbfbf00e36c1bcc309a33943f972099`. Bounded diagnostic code: `7ba747702d5b86628cfed95ac064db4d54469586`. The diagnostic completed successfully; the strict AR/F equality gate still fails and the main CODE experiment remains stopped.

## Result: the first disagreement is an exact tie versus a unique winner

Both instrumented native replays reproduced their own historical outputs exactly. Frozen DFlash also reproduced every field of all30 original verification blocks. At output index279, both predictions condition on the same344-token prefix:

| Native score/decision | AR | Frozen DFlash |
|---|---:|---:|
| Token11 score |32.000 |31.875 |
| Token323 score |32.000 |32.000 |
| Maximizing token IDs |11 and323 |323 only |
| Native greedy choice |11 |323 |
| Final output length |290 |292 |
| Termination |EOS |EOS |

AR's native argmax breaks the exact tie by choosing the lower token ID. Frozen DFlash has a0.125 margin for token323. That directly explains the first output disagreement. The returned logit tensors are FP32 containers, but the pinned implementation computes the head in BF16 and then copies/promotes its results; inference precision was not changed. At this magnitude,0.125 is the spacing immediately below32 in BF16.

The selected comparison is AR target call280 versus F target call29,row0, both absolute input position343. Three earlier F calls also predict at343 using hypothetical suffixes; those are not the committed common-prefix comparison and are excluded. Full151,936-entry vectors are retained privately. Both are finite; the maximum vector difference is0.25, with122,375 differing entries. All322 observed target-forward matrices are finite.

**Supported diagnosis:** native AR and block verification reproduce different finite-precision target scores at the same token prefix; a tie changes the greedy choice. No request-binding, sampling, or token-commit defect was found. The particular upstream source—attention/cache history, head batching, or another native operation—has not been isolated. Parameter identities/versions/layouts remain unchanged throughout both runs; this is not an exhaustive cross-engine bitwise weight check.

This is the same *observed numerical pattern* as the earlier ShareGPT case. It is not evidence that the current OSD/AdaFlash/C loss is wrong: no training or optimizer step has occurred. The actual outputs nevertheless differ, so the current exact-equality contract genuinely remains unsatisfied.

## Earlier evidence and applicability

The September30 AdaFlash C1 reproduction used the same released Qwen3-8B and DFlash revisions, BF16/FA3 and greedy block16 decoding. Its 1,024-request records agree across AR cards and between DFlash/AdaFlash, but AR differs from DFlash on611 requests. Seed, server capacity and memory fraction differed from the current experiment. That protocol explicitly did not require cross-path AR/SD token equality. This was a different prespecified comparison contract, not a code repair.

The October22 ShareGPT diagnostic used one unchanged BF16/SDPA target to reconstruct both saved speculative trajectories. The disputed scores were25.125 versus25.0 on one path and25.125 versus25.125 on the other, producing different greedy tokens. All recorded native predictions and commits were reproduced. That established path-dependent numerical divergence for that case, without finding a verifier defect or isolating a specific kernel. The old failed equality gate remained failed.

These precedents justify reusing the diagnostic method. They do not prove the cause of the new case or authorize adopting an older, less restrictive validity policy.

## Current retained-evidence audit

Both paths have the same65 input tokens and agree for279 output tokens. Their first differing prediction conditions on the same344-token prefix. Frozen DFlash's native target selects token323 at verification block27,row0, absolute input position343; AR emits token11. All30 DFlash blocks pass an independent check of positions, accepted-prefix selection, committed inputs and target predictions. Reassembling them gives the saved292-token output exactly. AR produces the same290-token output on both cards.

The adapters supply identical greedy sampling, cap and stop settings. Actual runtime records have the same target/BF16/FA3/context/seed/graph policy. Expected native differences include one-row AR versus16-row verification, overlap scheduling and draft machinery. No request-binding, sampling-parameter, cache-flush or commit-indexing bug was identified. Original qualification records contain no logit vectors for this prediction, so the retained audit alone cannot establish a tie or numerical cause.

## Bounded diagnostic

Two sequential native replays on the original physical B card, one AR and one frozen DFlash, use the unchanged saved inputs and settings. Passive instrumentation records native target logits and row identities after the existing forward and before sampling/verification; it adds no model forward. Each entire output and termination must reproduce its own retained trajectory, and F must reproduce the full verification trace. This is diagnostic timing, not a serving benchmark.

Five focused CPU checks and an independent implementation review passed. They cover row mapping and replay acceptance, not a complete native qualification. The source closure was verified:5,002 tracked entries, with all4,998 entries of the prior execution source preserved. The diagnostic retains existing graph-initialization accounting. It checks target parameter identity, version and layout stability, not cross-engine bitwise weight equality.

The two completed replays count against the existing64-generation qualification ceiling. There are now14 attempts and13 returned completions, with the original interrupted request still unknown. This leaves50 ceiling units while52 original qualification slots remain unexecuted. No old record is replaced or reset; a later continuation must explicitly reconcile this additional diagnostic work. No training, teacher preparation, optimizer operation, altered setting or campaign retry is included.

## Costs and audit

The diagnostic completed in50.997s launcher wall time and44.664 occupied-card seconds. AR made291 target forwards:290 emitted-token predictions plus one normal overlapping forward after EOS. F made31 target and30 draft forwards. Existing graph initialization added54 forwards (AR3; F48 target and3 draft), counted separately. The observer requested zero extra model forwards. Training, teacher preparation and optimizer work remain zero.

Peak allocated/reserved GPU memory was57.216/57.451GB. Both GPUs are now idle; A800 has491.909GB free, above the400GB reserve. No artifact was deleted. Original clock and all failures remain: cumulative occupied-card time237.903s. The original failed piecewise initialization still has an unknown additional forward count; this diagnostic does not fill that historical gap.

The1,321,900-byte private evidence archive was verified after transfer (SHA256 `0bfa4ef6c39adb666dd4a141ccae55c9ff3783bbcbc5569bf9b4bd0b08d96b52`). A separate agent independently checked the original/fresh outputs, full F traces, native row identities, full vectors, settings and costs. [CPU audit](audit_mismatch.py) reproduces the compact [numeric evidence](MISMATCH_DIAGNOSTIC.json). Raw input/output text, token streams, native vectors and logs remain outside Git.

## Can the earlier solution be reused?

**The earlier diagnostic method worked here; there is no demonstrated equality-restoring code fix to reuse.** The older AdaFlash reproduction allowed AR/SD differences under its original protocol, while the earlier ShareGPT investigation preserved its failed strict gate. Neither can silently replace this handoff's stronger requirement.

Changing tie-breaking, forcing canonical tokens, enabling a different precision/kernel, or removing the equality gate would change the current experiment. No such change was made. The useful next decision is a reviewer-approved numerical-validity contract, rather than altering the learning objective or blindly retrying unchanged qualification. If the strict AR/F contract is retained, this numerical mismatch remains a blocker. Any continuation must also account explicitly for the diagnostic's two generations and the old unknown request; the existing recovery path is not authorized to erase or redo them automatically.

## Source reports

- Fenrir: `research/adaflash_reproduction_20260930/SERVING_C1_RESULT_REVIEW.md`, lines42–93; independently rechecked its four raw REQUESTS.jsonl files.
- Fenrir: `research/adaflash_reproduction_20260930/EXECUTION_PROTOCOL.md`, lines45–55.
- Fenrir: `research/projected_target_feasibility_20261022/numerical_check_001/REPORT.md`, particularly lines7–50 and74–82.
- Fenrir: `research/target_variant_specialization_001/qualification/AR_NATIVE_DISCREPANCY_REVIEW.md`, lines78–86: earlier proposed, unexecuted native logit diagnostic.
- Current failed qualification: [QUALIFICATION_REPORT.md](QUALIFICATION_REPORT.md), preserved unchanged.
