# Independent continuation audit preparation

The local standard-library auditor is ready for the completed or partial run's compact evidence. It changes no production source and performs no model load, GPU work, inference, optimizer update or remote operation. Sixteen focused CPU tests pass in `AUDIT_CPU_TESTS_REVIEW_FIX.log`; the earlier14-test log is retained. Retained completed event-1 and native records are immutable test fixtures. Current-run data has not been analyzed during preparation. Independent review fixes are documented in `AUDIT_REVIEW_FIXES.md`.

After retrieval, run from this directory:

```sh
CUDA_VISIBLE_DEVICES='' python audit_results.py --raw-root raw --output .
```

Outputs are `FINAL_AUDIT.json` and `FINAL_AUDIT.md`. Exit zero requires a completed campaign, supervisor exit zero, all required evidence present, and no audit issues. Partial evidence remains visible and cannot pass. Missing files and malformed records are listed. Interrupted optimizer-call counts are described as observed completed calls rather than proof of no in-flight work.

Retrieve these compact files while preserving relative paths:

- `sealed/PLAN.json`, `SEAL.json`, `RUNTIME_PIN.json`, `config.yaml`; `PREFLIGHT.json`, `POSTFLIGHT.json`, `EXIT_CODE`, supervisor timestamps/logs; prepared manifests and selection audit.
- `campaign_001/STATUS.json`, `COSTS.json`; all six `resume_*` directories' JSON receipts, including failure or cleanup records.
- Both arms' event-2/3/4 `RESULT.json`, `PENDING.json`, `FAILED.json` when present.
- All three feedback windows' `POOL.json`, `PREPARATIONS.json`, `PUBLICATION.json`, `COSTS.json`, storage-admission receipts, response JSON, feature JSON headers and terminal-feature JSON receipts.
- Complete compact `evaluation_002` and `evaluation_004` trees, including raw generation JSON, measurements, validation events, native weight checks, activations, status/protocol and reports.
- `native/request_*.json`, `native_status.json`, `engine_settings.json` and failure records. The auditor maps remote native paths by basename without modifying records.
- Optional final new-state/export file-identity inventory before retention, and the retrieval hash manifest. These supplement the initial seal but are not claimed to be a fresh local tensor comparison.

Bulky `.pt` or safetensors are not required locally. Full tensor/Adam/RNG restore claims depend on the sealed runtime's saved/live receipts. The auditor independently reconstructs native token commits and counters, exposure and denominator diagnostics, schedules and file identities. Natural training logits were not retained; aggregate diagnostics do not replace the earlier bounded native mask qualification. Sequential source and trainer closures support phase ordering, without an independent activity-overlap clock trace.

The separate `analyze_results.py` report owns efficacy interpretation and same-event O/K summaries. This execution audit keeps the original development panel separate from the later fixed-checkpoint confirmation and discloses older-project exposure of all384 continuation training inputs. Its methodological scan covers all11 ARS fallacy categories without claiming a scientific rerun.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_ok_continuation_20261027/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.

Historical-source gap: this run archive contains 2 unresolved source-pin occurrences; the missing versions remain explicitly identified.
