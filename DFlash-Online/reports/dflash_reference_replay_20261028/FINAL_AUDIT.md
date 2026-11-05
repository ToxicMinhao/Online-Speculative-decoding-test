# Independent KR replay audit

**Passed: zero unresolved audit issues. KR performs worse than retained KRC on both prespecified panels.**

This is an ARS experiment-agent validation of retained raw evidence by a separate agent in the same project team. The auditor reconstructed the results before reading the new report. This is not a blinded external review. No model execution, GPU work, network access, training or generation was performed by this audit.

## Raw result reconstruction

| Panel | KRC calls | KR calls | KR reduction versus KRC | Improved / regressed / tied | Gross saved / added |
|---|---:|---:|---:|---:|---:|
| Development | 3,005 | 3,109 | −3.460899% | 14 / 44 / 6 | 44 / 148 |
| Additional | 3,125 | 3,255 | −4.160000% | 9 / 44 / 11 | 23 / 153 |

Counts use one validated trajectory per request. The two panels are not pooled. The auditor reconstructed **37,482 native verification blocks across all 768 generations**, checked every native commit against canonical token IDs and termination, and confirmed exact count consistency across repeats. Fresh KRC counts reproduce the retained per-request values. Canonical outputs, prompts and retained counts were independently linked to the earlier raw panel artifacts, rather than relying solely on the new sealed plan.

| Panel | KRC summed mean request time | KR summed mean request time | KR time reduction |
|---|---:|---:|---:|
| Development | 51.864162 s | 53.764256 s | −3.663597% |
| Additional | 53.695815 s | 55.876409 s | −4.061012% |

Each request contributes the mean of three fresh repeats. KR is slower in every repeat: development reductions are −4.013548%, −3.497834% and −3.479306%; additional reductions are −4.094477%, −4.052487% and −4.036074%. Native generation and cache-maintenance timers are nested within request latency and were not added again. Each panel contains 193 checkpoint publications, accounted separately.

All 384 executions per panel are valid, with zero invalid, unknown or unattempted generations. Every repeat has 32 requests starting with each arm. Only the first eight development requests formed the included 48-generation native gate; the additional panel did not repeat qualification. No extra native warmup or feedback generation occurred.

## Training and scientific controls

The actual event contains **512 presentations, 256 native AdamW steps, 512 response forwards and 512 backwards**. Native Adam counters advance continuously from 1 to 256. Each of the 128 retained responses appears twice in role A and twice in role R. The audit independently rebuilt the cyclic next-pair schedule from the original O records and matched every response, pass, anchor and update boundary.

There are **130,192 active sampled-anchor occurrences**, equally split between A and R. Each role supplies 949,560 available proposal occurrences; the current first-error selection retains 558,183 A occurrences and 528,755 R occurrences. Exposure is counted at optimizer boundaries, avoiding duplicate accumulation or checkpoint-recomputation counts. The fixed training pool has no ID or exact prompt-token overlap with either evaluation panel.

The fresh released model, empty optimizer, full pinned RNG and clean accumulation state pass exact initialization checks. The actual first A update matches the retained K training records exactly, including its anchors, losses/exposure metrics, learning rate, gradient-norm metric and counters. This establishes the declared first-A check; it does **not** claim equality with unretained historical post-update tensors. The first actual A/R window passes its four-presentation/two-step check without additional work.

The sealed source and native receipts retain one optimizer and one advancing RNG stream. They consume ordinary sampling draws before using the sealed order/anchors; the initial shuffle and first two A anchor draws match the retained path. There is no protected-prefix capture, reference model, hinge, correction or per-R RNG reset. The original 7,424-step horizon remains unchanged. Training closes with zero accumulation remainder before timed native evaluation.

Startup is separate: **two disposable forwards/backwards, zero Adam steps**, followed by exact fresh-state validation. No extra qualification optimizer step was added.

## Costs and evidence integrity

- KR complete event: **691.503 s**; A/R training calls: 326.687/326.907 s, nested within the event. Checkpoint persistence and export cost 18.619/14.737 s and are also nested.
- Startup: 38.677 s; fresh loading/initialization: 20.703 s; native backend loading: 21.507 s.
- Peak allocated/reserved training memory: **27.157/30.906 GB**.
- Campaign process: 2,160.595 s. Supervisor: **00:42:52–01:20:45 UTC, 37 min 53 s, exit 0**.
- All **722 sealed source/input files** pass remote preflight and postflight. The auditor rehashed **509 available local counterparts**, including the unchanged author-training tree. The other 213 remain covered by the full remote seal checks; model tensors were not reloaded locally.
- All **2,415 retrieved files / 132,364,277 bytes** were independently rehashed. Native loaded-tensor equality checks pass for both checkpoints on both panels.
- The report's result tables and nested cost accounting match the reconstructed evidence. Three focused audit-accounting tests pass.

The initial audit could not resolve the original O receipt because that older run stores it under `final_review/raw`. That audit-only path failure is preserved in `INITIAL_AUDIT_001.json`. The resolver was corrected to read the existing retained file; no scientific source, raw result or experiment was changed.

## Supported conclusion

**The tested ordinary-replay alternative does not reproduce KRC's advantage: it worsens drafting and serving time on both panels.** KRC therefore wins this practical comparison against spending additional updates on the same reference responses.

This does not isolate why KRC wins. KR changes the number of ordinary optimizer steps, Adam-state evolution and supervised exposure; it is not an equal-compute or direction-only control. The result does not establish seed/window robustness, a universal preservation guarantee or adaptation-inclusive speedup. Historical KRC training time is not a fresh paired cost measurement. The additional panel remains the previously selected confirmation panel with disclosed older-project exposure; neither panel was newly selected for this test.

Evidence: `FINAL_AUDIT.json`, `audit_results.py`, `raw/campaign_001/`, `raw/sealed/`, and the original retained panel records linked by the audit. No further experiment or tuning follows from this audit.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_reference_replay_20261028/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
