# KRC2 independent implementation review checklist

Material Passport: ARS academic-research-suite / experiment-agent validation; local source and retained-artifact review by a separate project agent. No GPU, SSH, scientific execution or inferred performance. Implementation review remains pending until the additive files are complete.

## Exact authorized scope

Resume KRC1's corrected weights, native Adam state/counters, full own RNG and completed event history at step 128. Reuse the original O/K event-2 pool, ordering and sampled anchors. Run one additional 128-step KRC event with all optional probes disabled. Evaluate KRC2 against retained K2 on the two unchanged 64-request panels, three repeats each (768 generations total). Stop after reporting, without tuning or subsequent events.

## Findings from inherited code

1. `online_feedback/training.py:_checkpoint_state` serializes the current backend model state, native optimizer payload and all RNG components. Because the correction completes before the second `train_step` returns, the KRC1 checkpoint contains corrected parameters with ordinary-K Adam moments. The exporter preserves RNG. The retained KRC1 receipt/status records step128, presentations256, event_001, zero accumulation remainder and horizon7424.
2. The saved KRC1 algorithm contract includes `reference_correction_event`, including the old optional probe schedule. Shared `_restore` requires an exact contract. Disabling probes must be a narrow, documented diagnostic-only continuation accommodation, validated against the original saved contract; it must not weaken the shared loader or accept changed scientific hyperparameters.
3. `build_reference_schedule` expects event-local counters0→128, whereas original O2 records contain global128→256. A validated event-local view may support this unchanged builder. Preserve original global counters in all actual training and audit records. `PresentationSchedule` already supports normal sampling with assertion-only order/anchor checks; do not use KR's forced-anchor replay path.
4. `ReferenceCorrectionEvent._begin_window` already subtracts its start step, so its windows1→128 can run at native steps129→256. With `probe_steps=[]` and no old reproduction callback, it needs no A-prefix capture or post-correction diagnostic forward.
5. Pre-/post-reference replay preserves RNG/modes/caches/ordinary gradients. These existing checks must remain enabled. Exact restoration alone is insufficient if any startup or schedule-construction work later advances the resumed RNG.

## Release checks

- [ ] KRC1 complete recovery state and corrected export identified and sealed; no weights-only substitution or later checkpoint.
- [ ] Exact resumed parameters, Adam moments/counters, full RNG, event history, configuration and accumulation checked before the first actual window.
- [ ] No reseeding or reset to released/ordinary K state after restore. Existing 7,424-step horizon, constant3e-4 LR, BF16 and qualified optimizer unchanged.
- [ ] Disposable startup separately counted, zero Adam; no duplicate qualification steps or extra event-2 prediction/loss reproduction gate.
- [ ] Exactly128 immutable event-2 features retained, including capped train_00187. Same prompts, actual O-generated responses, canonical conditioning and teacher features; evaluation requests excluded.
- [ ] Normal ordinary shuffle/anchor RNG draws match original O2 schedule. Mismatch stops with evidence; never replace a draw or reseed to force agreement.
- [ ] A current pair and R next disjoint pair use their own same-pass anchors with cyclic wrap. Schedule local/global indices remain distinct.
- [ ] Unchanged tested correction formula/FP32 decisions/BF16 cast and no-op/nonfinite behavior. Ordinary Adam moments remain unchanged by all reference operations and corrections.
- [ ] Exactly256 K forwards +256 pre-R +256 post-R =768 core response forwards; 256 K backwards plus at most256 violation backwards;128 native Adam steps; at most128 direct corrections. All probe/A-capture/post-C counts zero.
- [ ] First actual window checks include true counters, current record/anchor bindings and state integrity; no K2 loss or weight equality requirement.
- [ ] One final KRC2 export and complete recovery state at step256/presentations512/events1–2. Source KRC1 artifacts unchanged.
- [ ] Actual two-arm K2/KRC2 evaluation: two fixed panels, exactly768 generations, balanced32 starts per arm/repeat, only first-eight development48 executions as included native gate.
- [ ] Exact retained panel token IDs and termination for every execution; repeat-consistent counts. Historical K2 count gate only on development, never K1 and never invented for additional.
- [ ] Native cache validity, runtime/precision/block16/cap2048/context4096/captureoff/concurrency1 unchanged; no generation warmups, concurrent training or transfers during timing.
- [ ] Raw-before-validation, finite-work checks and invalid/unknown/unattempted accounting; no automatic retry or continuation after correctness failure.
- [ ] Report each panel separately; KRC2/K2 advantage distinct from KRC2/KRC1 improvement; K2/K1 context is count-only. No historical-versus-fresh timing learning curve or equal-compute/independent-deployment claim.
- [ ] Parent storage admission covers live restore/model/Adam/checkpoint overlap and unchanged reserve, with original evidence preserved.

## Initial verdict

No scientific redesign is needed. The two concrete integration risks are the saved diagnostic probe contract and the event-local/global schedule counters. Both have been communicated to the implementation agents. Final release verdict awaits the additive implementation and focused tests.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_krc_continuation_20261028/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
