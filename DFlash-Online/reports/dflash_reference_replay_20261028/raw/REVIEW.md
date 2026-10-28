# Independent KR implementation review

## Material Passport

- Origin Skill: ARS academic-research-suite, experiment-agent validate.
- Scope: one authorized ordinary-reference replay event, followed by the two unchanged KR/KRC panels.
- Provenance: separate agent code and contract review within the same project team; not a blinded external review.
- Verification: source inspection and **23 independently executed CPU tests**, including real tiny-model TrainerCore/Adam integration. No GPU, remote command, model generation or scientific run by this reviewer.

**Verdict: no blocking implementation finding.** Correctness gates remain active during the bounded main execution; CPU tests do not establish GPU correctness or efficacy.

## Training and the first-A gate

The additive `ReferenceReplayEventTrainer` subclass changes the presentation schedule while reusing the existing K loss, per-response normalization, two-response accumulation, clipping, BF16 AdamW and export path. Existing K/KRC source and accounting are not weakened. Each sealed outer window contains two distinct ordinary optimizer steps: A, then R from the next pair in the same pass with cyclic wrap. Every role uses that response's own original same-pass anchors. The same optimizer, moments and counters continue through all 256 updates, and current-model first-error selection is recomputed during every presentation.

The RNG policy is explicit: consume one normal shuffle per pass and one author anchor draw per actual presentation, then use the sealed order and anchors. Initial shuffle and first-A draws must reproduce the retained trajectory. Later discarded draws advance the RNG; no restoration or reseeding occurs around a real R update. Each response receives two A and two R presentations over the event.

The qualified process startup consists of two separately accounted disposable ordinary forward/backwards, with zero Adam steps. `prepare_main` checks the new trainer against the exact pinned fresh model, optimizer and full RNG state. The first-A gate then compares the two actual training records against retained K1: IDs/pass, sampled anchors and keep masks, all ordinary K metrics, learning rate, accumulation scope and optimizer boundaries. This reviewer independently checked that these retained records also exactly match the later passed ordinary-K qualification's first two records.

The prior KRC post-update prefix-probe gate would require extra forwards and is deliberately not reused. The new gate claims exact ordinary first-A record equivalence, **not historical post-update tensor equivalence**. The CPU integration additionally compares first-A weights, gradients, Adam state and RNG against an ordinary K execution. `FIRST_AR_WINDOW.json` records the actual first four presentations and first two updates without duplicating work.

The event requires exactly **512 response forwards, 512 backwards and 256 Adam steps**, grouped into 128 outer windows. No protected-prefix capture, hinge, correction, reference model, additional optimizer or post-update scoring forward is introduced. Hooks count actual work; per-role presentation costs, feature loading, update wall time and journaling are explicitly separated. The journal and failure receipt preserve partial work and the active/last update identity. All completed-event counters and bindings are checked before serving.

## Evaluation and sealing

The two panels retain their original order and inputs. Development uses its retained F canonical tokens/termination; the additional panel uses its retained first-K1 canonical tokens/termination. KRC counts are explicitly checked per request against the correct panel's retained values (3,005 and 3,125 pooled calls), in addition to repeat consistency. Equal `original_index` numbers across panels are not treated as source identity: those fields originate in different indexing namespaces; panel IDs, exact tokens and existing provenance bind the inputs.

Only KR/KRC generate outputs: two panels ×64 requests ×2 arms ×3 balanced repeats = **768 generations**. Only the first eight development requests form the included 48-execution gate. The second panel adds no qualification generations. Trainer closure precedes native backend loading. Raw records are written before validity checks, harmful valid results continue, and a failure preserves invalid/unknown/unattempted accounting for both panels without retry.

Native publication validates caches and loaded tensor identity. Reports pool one count trajectory per request and three-repeat mean request times, keep panels separate, retain regressions, and expose native-generation/cache-maintenance timers as nested components of complete request latency. They do not combine those components additively or pool historical timings into fresh comparisons.

The parent seal binds both prior campaigns, retained canonical/count sources, exact checkpoint exports, model/runtime/config and new additive files. The peak-storage calculation is supported for the required fresh KR output: one atomic 3W model/Adam checkpoint plus W export and W possible export temporary overlap, with 10% tensor slack, 4GiB auxiliary/staging allowance and the unchanged 8GiB reserve. This admission does not assume deletion of any earlier state or weight-sized retrieval staging.

## Checks and interpretation

[CPU_INDEPENDENT_REVIEW.log](CPU_INDEPENDENT_REVIEW.log) records all 23 passing tests. Coverage includes within-pass/cyclic bindings, current selection, separate response normalization, continuous Adam/RNG, exact first-A CPU state, complete work counts, first-gate failure before R, actual 768-slot two-panel CPU evaluation, retained-output/count rejection, nonfinite/unknown work and nested timers.

KR intentionally changes exposure and optimizer-step/state evolution. It is a practical replay alternative, not an equal-compute control or an isolated test of correction direction. Earlier KRC training times are historical measurements. The result must be interpreted separately on each panel and must not authorize tuning or another run automatically.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_reference_replay_20261028/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
