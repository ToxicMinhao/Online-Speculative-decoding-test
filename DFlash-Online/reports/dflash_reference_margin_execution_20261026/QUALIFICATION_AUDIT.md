# Independent R qualification audit

**Passed, with no audit issues.** This local review reads the retrieved GPU qualification JSON and ledger. It does not execute a model, access A800, or recompare tensor files. Detailed checks and artifact hashes are in [QUALIFICATION_AUDIT.json](QUALIFICATION_AUDIT.json).

The 366.870457 s qualification contains the ordered warmup, Oa, R0 and R1 trials. The first three each complete one accumulation window; R1 completes two. All schedules, anchors and optimizer boundaries match the corresponding retained O presentations, and all four trials use the complete original 128-response first-pass shuffle. Each ledger presentation and gradient snapshot matches its trial receipt; logical work is checked by trial and presentation as well as in total.

| Work | Attempted | Completed | Failed |
|---|---:|---:|---:|
| Current training forwards | 10 | 10 | 0 |
| Reference training forwards | 4 | 4 | 0 |
| Detached comparison forwards | 4 | 4 | 0 |
| Backwards | 10 | 10 | 0 |
| Temporary optimizer steps | 5 | 5 | 0 |
| Detached logit gradient calls | 20 | 20 | 0 |

There are exactly ten completed presentations and zero qualification generations, main steps or teacher-model forwards in the fixed plan. Eleven exact-equality receipts pass: four initial states, six R0/Oa boundary/final/record comparisons, and R1's first-window RNG comparison. Oa and R0 scalar metrics also match retained O's first window. Two detached full-block alignment receipts and four prefix, own-rival, gradient and direct loss/logging checks pass. The active sampled-logit counts are **0, 0, 1, 2** across R1's four presentations; the second window exercises the updated model. Synthetic changed-rival and equality-boundary checks remain separately identified.

R1 preserves all six distinct gradient snapshots, including both pre-optimizer boundaries. Frozen target-table identity/layout, reference RNG isolation and detached-reference checks pass in the receipts. Maximum allocated GPU memory is 24,693,195,264 bytes. Qualification storage admission satisfies the recorded 38W + 10 GiB budget; this audit does not measure server free space anew.

All six qualification input hashes match their retained local input or sealed source/dependency bindings. Fresh main R has zero counters, no completed event or recovery-state path, and a twelfth exact initial-state receipt matching the pinned state. Its initialization receipt records restored full RNG, no reference yet and no inherited warmup state.

Auditor validation is retained under [audit_validation](audit_validation/): eight CPU tests pass, including hidden longest-prefix corruption, EOS-clipped native/scientific counter differences, pooled ratios, duplicate/missing evaluation slots, boundary-only diagnostics, and relabelled qualification work. All 768 retained prior native records (37,098 blocks) also passed the independent reconstruction and added identity/capture checks during auditor development. These are auditor tests, not new scientific generations.

This establishes compact-evidence consistency for the bounded qualification and fresh main initialization. It does not establish R efficacy, complete the main-event/evaluation audit, or prove numerical equivalence of historical O's entire 128-step trajectory. The authorized main campaign continues under the existing supervisor; no scientific settings or sealed sources were changed by this audit.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_reference_margin_execution_20261026/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
