# Second-group independent audit

**CodeAlpaca correctness and accounting pass; no correctness blocker found.** This extends the [first-group audit](FIRST_GROUP_AUDIT.md) to [the second recovery snapshot](retrieved/second_group_recovery/), using the same local CPU artifact checks. No new unit tests, model execution, remote operations or source changes were performed. ShareGPT was still running at retrieval.

All 48 F/C/P outputs exactly match in prompt, response tokens, stop-token set, finish reason and cap. All **3,267 verification blocks** reconstruct their saved outputs and accepted-token counts. Eleven distinct responses terminate at EOS; **five reach the 512-token cap: two A and three B**. These capped requests remain in every comparison. Near-cap verification widths correctly shrink to `min(16, 512 − output_start)`; all posterior rows and accepted commits remain consistent. No response continues past an EOS token. Source and held-out output-token totals are identical across arms: **2,253 A and 3,282 B**.

Exactly eight A examples appear in [the direct mask extraction](retrieved/second_group_recovery/MASK_EVIDENCE.json) and training receipts. Input IDs equal prompt plus response, prompt masks are zero, response masks are one, and all recorded tensor shapes/dtypes/finiteness match. Teacher labels equal F outputs. Preparation records **472 target forwards**: eight prefills, 456 verification blocks and eight final feature forwards. All **7,236 posterior rows** match according to saved replay receipts, including uncommitted and shortened near-cap blocks; final feature logits never supply labels. No B or adapted-response features train either arm.

C and P each train on the same ordered eight A responses twice, with accumulation two and exactly eight optimizer steps. Independent recomputation confirms identical anchors, keep masks, **14,914 supervised position exposures** and **6,661.177241408448 total decay weight**. Initial and final Python/NumPy/Torch/CUDA RNG states match exactly; compact metadata matches the event and final RNG receipts. The pinned eight-rate schedule, response-mean reduction, seed17 and other training settings remain unchanged. Configurations and algorithm contracts agree except arm identity/path and objective name. Runtime receipts confirm released-weight initialization, fresh optimizers, fixed-reference equality before/after P, and validated 58-tensor publication. Completed optimizer-state retirement preserves exports/metadata and explicitly loses exact continuation. The target guard and compact-evidence limits in the first audit still apply.

| CodeAlpaca subset | F calls | C calls | P calls | C/F change | P/F change | P/C change |
|---|---:|---:|---:|---:|---:|---:|
| A: eight training responses | 456 | 424 | 496 | −7.02% | +8.77% | +16.98% |
| B: eight held-out responses | 598 | 656 | 637 | +9.70% | +6.52% | −2.90% |

Ratios use all eight paired call sums, with no dropped requests; lower is better. All recomputed report values agree. P versus C improves/worsens/ties **0/8/0** A requests and **6/1/1** B requests. P again reduces held-out harm relative to C but worsens both source and held-out calls versus frozen. The result supports no frozen-baseline benefit or serving-speedup claim.

The released F reset has a valid strict 58-tensor activation receipt. The manifest and installed-source receipt remain unchanged; all **82 previously retrieved GSM8K files are byte-identical**, and all **193 inventory entries** match their recorded sizes. These two completed groups account for **96 generations = 16 reused + 80 new**, 64 presentations and 32 steps. Snapshot status is later, at **100 = 16 + 84 generations**, because ShareGPT F generation had begun. The full 144/96/48 budget remains unchanged; later partial artifacts are not treated as completed results. Cost nesting and failed-attempt/reused-F accounting follow [COST_AUDIT.md](COST_AUDIT.md). Stage 1 is unchanged.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/projected_target_feasibility_20261022/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
