# Independent implementation and input review

## Material Passport

- Origin Skill: ARS academic-research-suite, experiment-agent validate.
- Scope: the authorized fixed K1/KRC confirmation on 64 additional GSM8K inputs; no mechanism change.
- Review provenance: a separate agent reviewed the source and input evidence within the same project team. This is not a blinded external review.
- Verification: local source review, 17 independent CPU evaluator tests, three independent audit-accounting tests, and question/token reconstruction. No model forward, generation, optimizer, GPU or remote execution by this reviewer.

**Verdict: no blocking implementation finding.** Native output agreement remains the included first-eight-request gate; CPU tests do not establish that gate.

The evaluator uses genuine K1/KRC checkpoints and the unchanged qualified backend. Its exact schedule is 64 requests × two arms × three repeats, with 32 first positions for each arm in every repeat. The first eight requests consume 48 of the same 384 executions. There is no training, feedback, KL evaluation, correction, generation warmup, historical call-count gate or automatic retry.

Each request's first fresh K1 execution supplies its canonical tokens and termination. KRC-first output is saved pending that reference and then validated. Raw records are preserved before rejection. Repeat-count failures, malformed measurements, interrupted worker replies and final weight-integrity failures retain explicit invalid/pending/unknown/unattempted accounting. Native loaded weights are checked before each arm's first and after its final execution; these four checks require no additional generation. File sealing and postflight checks protect both existing exports.

The primary metric pools one validated count trajectory per request. Latency uses three-repeat request means. Request gains/regressions/ties and gross savings/additions are retained. The leave-one-request-out calculation leaves the primary pool unchanged and is identified as descriptive sensitivity. Loading, publication, weight validation and campaign time remain separate from request latency.

## Input evidence

The independent reconstruction reproduced selection seed 20261028 from the pinned question-only dataset: 1,191 eligible test requests after 128 prior test exclusions; 64 selected. Exclusion text also covers all 512 original/continuation training requests. All 64 new token-ID sequences exactly match the pinned tokenizer and author rendering helper. There is no source-ID or normalized-question overlap with the current-study inventories. This reviewer rehashed 35 training/qualification/probe receipts, four early native-probe bindings and the cited inventory dependencies. The earlier native probes map to excluded development inputs.

All 64 selected requests have documented exposure in older unrelated project experiments. Thus the panel is new to the current study, not globally unseen project data. Selection occurs before these exposure annotations and does not read answers, output quality, call counts or latency.

Evidence: [PANEL_INDEPENDENT_CHECK.json](PANEL_INDEPENDENT_CHECK.json), [CPU_INDEPENDENT_REVIEW.log](CPU_INDEPENDENT_REVIEW.log), and `inventory_audit/SELECTION_AUDIT.json`. The selected panel digest is `e5684b9ec7b7f9e1321e007ae1f2d282980f23da688c2de017d7fdf7cc15c07c`.

Reviewed source: the additive `experiments/krc_confirmation` namespace, inherited arm-agnostic scalar/native validators, parent `campaign.py`, `seal_remote.py`, `preflight.py`, launch/supervisor scripts and inventory/selection scripts. No existing training/inference implementation was modified by this reviewer. The campaign's meaningful remaining checks are actual storage/source admission and native execution correctness; valid null or harmful outcomes remain authorized to complete.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_krc_confirmation_20261028/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
