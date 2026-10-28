# Projected-target feasibility screen: two qualified groups, one incomplete

## Material Passport

ARS bounded experiment execution and evidence audit, October 22, 2026. This report concerns the approved Qwen3-8B / DFlash block-16 F/C/P screen. It does not select a new project direction or resume earlier campaigns.

**The implementation produced valid comparisons on GSM8K and CodeAlpaca, but projected targets did not improve drafting over frozen DFlash.** They reduced held-out harm relative to ordinary hard-label CE while worsening source-response drafting. ShareGPT stopped at the required exact-output check and supplies no qualified gain claim.

## Scope and validity

The sealed manifest contains 48 distinct independent requests: eight feedback A and eight held-out B for each of GSM8K, CodeAlpaca and ShareGPT. The explicitly approved exception reuses eight historical GSM8K train questions for A, with fresh generated responses. Both learning arms independently start from released weights and fresh optimizers. B is excluded from training and passed the retained-project-history exclusion audit. Dataset assistant answers are unused.

F is released DFlash; C is ordinary hard-label CE; P uses the fixed pre-event projected target with c=1.1. This is the current SpecForge foundation, not the earlier Llama/forward-KL OSD recipe. Each learning arm receives the same eight A responses for two passes, with response-mean reduction, accumulation two and eight optimizer steps. BF16 model tensors, FP32 CPU optimizer masters/moments, masks, sampling, clipping and the actual upstream schedule remain unchanged. The nominal LR is 6e-4; the exact used sequence is retained in every TRAINING.json, including the qualified upstream scheduler's initial 6.237396779e-4 value.

Inference uses greedy decoding, block size 16, 512 output tokens or EOS, full independent prompts, fresh request caches and concurrency one. All arms retain compact verification traces without training-feature capture during timed generation. No extra generation warmups or tuning runs were added.

GSM8K and CodeAlpaca each pass all 48 output/termination checks, exact publication, initialization, matched anchor/exposure/RNG checks and fixed-reference checks. The target guard uses parameter identity/version, disabled gradients and optimizer exclusion; it is not a repeated bitwise scan of the whole target. Independent local audits reproduce the saved traces, masks and aggregate arithmetic.

## Native drafting results

The entries below are sums of verification calls on identical output tokens. Lower is better. Percentage change is `100 × (arm calls / baseline calls − 1)`: positive means more work. A measures source-response fitting; B measures held-out transfer.

| Dataset | Subset | Requests | F calls | C calls | P calls | C versus F | P versus F | P versus C |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GSM8K | A, trained source | 8 | 356 | 351 | 394 | −1.40% | +10.67% | +12.25% |
| GSM8K | B, held out | 8 | 303 | 367 | 350 | +21.12% | +15.51% | −4.63% |
| CodeAlpaca | A, trained source | 8 | 456 | 424 | 496 | −7.02% | +8.77% | +16.98% |
| CodeAlpaca | B, held out | 8 | 598 | 656 | 637 | +9.70% | +6.52% | −2.90% |
| Qualified groups only | A | 16 | 812 | 775 | 890 | −4.56% | +9.61% | +14.84% |
| Qualified groups only | B | 16 | 901 | 1,023 | 987 | +13.54% | +9.54% | −3.52% |

These pooled rows cover only the two qualified groups, not the full planned three-group experiment. P versus C improves/worsens/ties 1/15/0 source requests and 11/2/3 held-out requests. Relative to F, P improves/worsens/ties 4/11/1 source requests and 2/12/2 held-out requests. All regressions and cap hits remain in the denominator. Raw losses from different objectives are not compared.

The evidence supports **reduced held-out harm versus C**, not an improvement over F or successful preservation of useful source learning. Under this fixed update recipe, the component's earlier token-level correctness did not translate into a native drafting benefit. This does not establish failure of every projected-target configuration, a cause for the regression, or a replacement mechanism. No margin sweep, learning-rate change or additional arm was run.

## Stopped ShareGPT comparison

ShareGPT P request `sharegpt_A_07` differs from F/C only at response index 511, the last token of the 512-token cap. F/C produce token 9492 (` tags`); P produces token 3847 (` users`). The first 511 response tokens are identical.

The saved target posteriors themselves differ. F/C predict this position from a block starting at output 508 with width four; P uses a block starting at 510 with width two. Saved commit indexing is consistent with each path's target posterior. The trace therefore does not show a rejected proposal being incorrectly committed. Different numerical target execution/cache histories are a possible explanation, but logits and margins were not retained, so BF16 rounding or a specific kernel cause is not established. No extra forwards were run to force a match.

The final group-wide native target version check was not reached after the mismatch. Local frozen head/embedding and optimizer-exclusion checks passed, but this incomplete group does not have the completed whole-target version receipt that the first two groups have. No target mutation is demonstrated.

The strict gate remains unchanged. All ShareGPT records, including the mismatch, are preserved; the group has no completion receipt and is excluded from qualified gain calculations. Its eight P held-out generations were not executed. No later checkpoint, replacement request or regenerated canonical answer was substituted.

## Execution, costs and retention

The run executed **136/144 generations, 96/96 response presentations and 48/48 optimizer steps**. Of the 136 generations, 16 valid frozen GSM8K answers came from the original stopped attempt and 120 were newly executed during recovery. They are counted once. There are 96 qualified records, 40 present ShareGPT records and eight unexecuted ShareGPT P-B records. Neither incomplete execution nor a preserved output mismatch is a successful full campaign.

Before training, the original attempt exposed a teacher replay mismatch when replacing native cached block execution with a full-sequence forward. The narrow repair now reconstructs the exact saved target prefill and verification blocks, checks every posterior row, and retains canonical features. It passed 116 CPU tests in both environments and all 24 source-response GPU reconstructions. This changed feedback collection, not the objective, optimizer, inference verifier, precision or generation budget. The original failure and both source versions remain available.

| Group | Shared teacher preparation | C training | P training | C/P checkpoint save | C/P export | C/P activation |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K | 13.87 s | 31.53 s | 31.69 s | 38.07 / 37.73 s | 18.11 / 18.20 s | 0.37 / 0.41 s |
| CodeAlpaca | 15.61 s | 30.28 s | 32.58 s | 38.07 / 38.12 s | 18.22 / 17.64 s | 0.43 / 0.41 s |
| ShareGPT, incomplete | 48.92 s | 27.77 s | 31.01 s | 37.78 / 37.87 s | 17.50 / 17.32 s | 0.40 / 0.39 s |

Training includes nested reference/head/projection work; do not add those subcosts again. Feature persistence, loading, diagnostic checks and serving timings are retained separately. The table is not an exhaustive end-to-end decomposition. Process-wide training-through-export GPU peaks range from 28.00 to 36.90 GiB allocated; they exclude later serving/activation and CPU optimizer memory. Single instrumented request timings do not establish deployment speedup.

Recovery runner wall was 1,175.149 s. The original stopped attempt consumed 37.984 s, giving **1,213.133 s (20.22 minutes) of executed runner time**. This excludes the repair/development pause, preflight and external retrieval; reused F serving is already included in the original interval and is not added again. The recovery wrapper independently records 1,177.474 s.

All six learned model exports, 24 feedback payloads, exact token outputs, compact traces, configs, anchors, RNG records and source snapshots remain retained. Five completed-arm full optimizer states were retired only after their native comparisons passed, with compact metadata and exact export checks preserved; their exact Adam/master continuation is lost, not archived. The incomplete ShareGPT P full restart state remains intact. Both A800 GPUs were idle after the stop, with 13.48 GB free, above the 8 GiB reserve. No further campaign was launched.

## Review package

- [Raw report and all planned comparisons](retrieved/final_recovery/campaign_002/REPORT.md)
- [Per-request measurements and validity](retrieved/final_recovery/campaign_002/PER_REQUEST.csv)
- [Summed comparisons](retrieved/final_recovery/campaign_002/POOLED.csv)
- [Structured results](retrieved/final_recovery/campaign_002/SUMMARY.json)
- [Cost receipts](retrieved/final_recovery/campaign_002/COSTS.json), [complete timing boundaries](COST_BOUNDARIES.csv), and [cost interpretation](COST_AUDIT.md)
- [GSM8K independent audit](FIRST_GROUP_AUDIT.md) and [CodeAlpaca independent audit](SECOND_GROUP_AUDIT.md)
- [Final stopped-run audit](FINAL_AUDIT.md) and [mismatch evidence](MISMATCH.json)
- [Sealed manifest](data/approved/manifest.json), [GSM8K exception](data/approved/AUTHORIZED_EXCEPTION.json), and [executed source bindings](retrieved/final_recovery/BINDINGS.json)
- [Feedback compatibility repair](repair_001/REPORT.md) and [original failure analysis](repair_001/FAILURE_ANALYSIS.md)

The immediate review question is whether these two valid negative groups justify further work on this exact objective, while separately resolving the native output-preservation limitation. The missing ShareGPT comparisons do not justify an automatic retry or retuning.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/projected_target_feasibility_20261022/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
