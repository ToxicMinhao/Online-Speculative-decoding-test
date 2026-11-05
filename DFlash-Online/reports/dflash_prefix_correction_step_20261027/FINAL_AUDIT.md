# Independent final audit: single-step post-K correction

**Passed with no discrepancies. The prespecified local criterion is met: lost old-prefix tokens decrease from 153 to 113, and net prefix gain increases from 1,740 to 1,807. All 527 useful repairs made by ordinary K remain; the correction obtains six more.** This is a result on the two training responses, not evidence of held-out or serving improvement.

## Material Passport

- Workflow: ARS experiment-agent, reproducibility validation.
- Evidence: [raw diagnostic result](raw/execution_001/RESULT.json), exact state/reproduction receipts, model metadata, sealed inputs and retained first-step reference.
- Independent checks: [auditor](audit.py), [machine-readable audit](FINAL_AUDIT.json), [774-block table](analysis/PER_BLOCK.csv).
- Scope: local CPU analysis only; no model forward, optimizer operation, generation, source change or follow-up experiment was performed by this audit.

All 53 retrieved evidence files and the 565-file preflight/postflight seal pass verification. The audit derives EOS IDs from pinned generation/tokenizer metadata rather than inferring them from the outputs. It reconstructs every original, post-K and corrected prefix from saved proposal IDs, validity masks and EOS boundaries; verifies triangular protection weights, per-response metrics, global normalization, wrong ties and block-identity repair retention. Saved-margin hinge numerators match independent FP64 reconstruction exactly in both responses and states. The global protected-weight denominator is **30,693**, covering **5,236 protected position occurrences**.

| Local measurement | Ordinary K | Corrected K |
| --- | ---: | ---: |
| Sampled blocks | 774 | 774 |
| New earlier-error blocks | 47 | 41 |
| Lost original-prefix tokens | 153 | 113 |
| Useful repairs, out of 632 original first errors | 527 | 533 |
| Net matching-prefix token gain | +1,740 | +1,807 |
| Strict negative-margin protected positions | 50 | 40 |
| Wrong-at-tie protected positions | 1 | 3 |
| Weighted hinge V | 0.003567589 | 0.002353957 |

The original K metrics reproduce the retained reference separately for each response before the correction is applied. The correction retains all 527 useful repairs, loses none and adds six. It fully restores six K-damaged original prefixes and introduces no new damage to previously intact original prefixes.

The full block comparison still contains regressions: corrected prefixes are longer than post-K prefixes in **17 blocks**, shorter in **8**, and unchanged in **749**. Those changes add 77 and lose 10 matching-prefix tokens, respectively, yielding the net +67. Retaining the binary useful-repair event does not mean preserving every additional token K had gained. The block CSV preserves these regressions. The 774 overlapping sampled blocks are not 774 independent requests.

The single correction uses the declared FP32 coefficient **−0.9369601**, from V divided by global gradient norm squared **0.0038076208**, without an epsilon or additional step. Its proposed norm is **0.0578160** and its BF16-realized norm **0.0366449**, compared with K's update norm **10.0516004**: **0.5752%** and **0.3646%** of K's norm. BF16 rounding changes the realized displacement; these measurements do not establish an exact nonlinear constraint guarantee. Wrong-at-tie errors remain outside the hinge's direct gradient coverage and increase here from one to three.

Execution remains within scope: **one ordinary Adam step, two training presentations, six core response forwards, four backwards and one direct correction**. The separate qualified startup uses two forward/backwards and zero Adam steps. All exact initialization, schedule, parameter-preservation and optimizer/RNG/mode/cache/counter receipts pass. There are no complete-response generations, exported checkpoints, repeated corrections or retries.

The added post-K measurement/gradient work takes **2.492 s**, direct correction **0.157 s**, and corrected-state evaluation **2.450 s**: **5.100 s** total for these three components. Chunk-only pre-K capture adds a recorded 0.059 s; its label correctly excludes copies/snapshots included in the outer runtime. Ordinary K plus its observation costs 5.584 s. The diagnostic driver takes 97.812 s; the enclosing supervisor including checks takes 163 s. Component timings are nested and must not be added twice. Ordinary-K-phase peak allocation is 25.274 GB; correction-phase peak allocation/reservation is **29.469/35.830 GB**, including the resident model. These are diagnostic costs, not a deployment-overhead estimate.

The result supports the handoff's stated next decision: this correction deserves consideration for a separately authorized one-event performance test. It does not establish useful transfer, serving acceleration, robustness, or that this direction outperforms a trained shrinkage control. Execution has stopped at the agreed local comparison.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_prefix_correction_step_20261027/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
