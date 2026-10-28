# First-group independent audit

**GSM8K correctness and accounting pass; no correctness blocker found.** This is a bounded ARS review of [the retrieved first-group recovery evidence](retrieved/first_group_recovery/), performed locally on CPU without generation, training, remote operations, or source changes. It does not qualify later groups or establish efficacy. The snapshot was collected while CodeAlpaca was running.

All 48 F/C/P records are present. The 16 reused F records are JSON-identical to [the failed original attempt](retrieved/campaign_001/gsm8k/F/records/). For every request, C and P match F's prompt, entire response, stop-token set, finish reason and 512-token cap. All 16 distinct responses terminate at EOS, without an earlier stop token. Independent reconstruction of all **2,121 verification blocks** confirms anchors, draft-prefix matches, accepted-token counts, committed posterior tokens and aggregate counts. Block size is 16, context limit 2048; training-feature capture is disabled during serving and compact verification traces are enabled in all arms. No paired request is dropped.

The runtime manifest equals the approved 48-request manifest after the runner's one-based per-group indices are added. The approved exception applies to eight historically used GSM8K A questions; it does not make them fresh. All eight installed source-file hashes match the reviewed local repair/runner files, and all 111 retrieved inventory entries match their recorded byte sizes. This inventory check is not a content hash of every artifact.

[Direct tensor extraction](retrieved/first_group_recovery/MASK_EVIDENCE.json) confirms exactly eight A examples: input IDs are prompt plus saved response; loss masks are zero on every prompt token and one on every response token. Shapes, dtypes and recorded finiteness agree with the expected BF16 five-layer features and final hidden states. All saved target labels equal F responses. The eight [teacher receipts](retrieved/first_group_recovery/campaign_002/gsm8k/F/PREPARATION.json) record **372 target forwards**: eight prefills, 356 native verification blocks, and eight final feature forwards. Every one of the **5,696 posterior rows**, including uncommitted rows, matched the original native trace according to the replay receipts. Final feature logits were not used as labels. B and adapted outputs supply no training features.

C and P each present A00–A07 in order twice, with 16 presentations, accumulation two, and exactly eight optimizer steps. All 16 anchor samples and keep masks match exactly. Independent recomputation gives **15,014 supervised position exposures and 6,690.117535430221 total decay weight per arm**; these are repeated training exposures, not unique positions. Stepped-row training metrics aggregate two presentations and must not be summed across every row. Initial and final Python/NumPy/Torch/CUDA RNG states match exactly between arms; compact-state RNG and event records also match their standalone receipts. Configurations and algorithm contracts agree apart from arm paths/name and `hard_ce` versus `projected`.

The eight actual learning rates match the pinned unchanged upstream sequence, from **0.0006237396779379481** to **0.000023739677937948036**; the nominal base is 0.0006 and integer warmup is zero. Both arms start with released weights and a fresh optimizer according to runtime checks. P's fixed reference matches released weights before and after the event. All 58 published tensors pass strict loading, finiteness, configuration and exact publication checks. This compact audit checks those persisted runtime comparisons; it does not reload large weights or repeat GPU comparisons. Target immutability is guarded by parameter identity/version, disabled gradients and optimizer exclusion, not a whole-target bitwise scan. Completed optimizer states were retired only after valid replay, retaining exports and compact metadata; exact Adam/master continuation is explicitly lost.

| GSM8K subset | F calls | C calls | P calls | C/F change | P/F change | P/C change |
|---|---:|---:|---:|---:|---:|---:|
| A: eight training responses | 356 | 351 | 394 | −1.40% | +10.67% | +12.25% |
| B: eight held-out responses | 303 | 367 | 350 | +21.12% | +15.51% | −4.63% |

Changes are `100 × (sum current calls / sum baseline calls − 1)` over all eight valid pairs; lower is better. Output-token totals are identical across arms: 1,998 for A and 1,849 for B. P versus C improves/worsens/ties **1/7/0** A requests and **5/1/2** B requests. The independently recomputed sums, percentages and per-request counts match `SUMMARY.json`. P reduces held-out harm relative to C here, but both worsen held-out calls relative to F, and P worsens source calls. This group does not support a benefit over frozen or a selected mechanism.

The completed group consumes **48 effective generations = 16 original reused + 32 new**, 32 presentations and 16 steps. The unchanged full budget is **144 = 16 original + 128 new generations**, 96 presentations and 48 steps. The later status snapshot reports 56 = 16 + 40 generations because the next group had begun; its partial files are not part of this completed-group audit. Teacher forwards are separate preparation work. Original F serving is charged once; total consumed runner wall should add the original failed attempt's 37.984 s to the eventual recovery wall without adding F latency again. [The separate cost audit](COST_AUDIT.md) explains nested timers and omitted summary components. Single sequential instrumented timings establish no serving-speedup claim. Stage 1 and the previously authorized unchanged continuation remain unchanged.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/projected_target_feasibility_20261022/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
