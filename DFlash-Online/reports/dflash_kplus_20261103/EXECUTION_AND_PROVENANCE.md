# Execution and reporting provenance

Scientific model execution used commit `e0a9d3c235e1efd15404c549282d21c99243a21c` from `codex/kplus-full-gsm8k`, based on retained K report `4d94f30c42701d0f86ac04e2829c834593c16d8e`. CPU report recovery used `181bf736aee231e36bd82a3bc1f73aa9f1eeb0f2`. The final report publication is a later documentation/analysis commit; it did not execute the experiment.

The original A800 source `/data/sources/minhao/DFlash-Online` and run `/data/sources/minhao/runs/dflash_kplus_20261103/production_001` remain unchanged. A separate source tree at sibling `report_recovery_001/source` ran the CPU-only report with CUDA hidden:

```text
python -m experiments.kplus_full_gsm8k.report <production_001> \
  --recover-completed-audit --output <report_recovery_001/report>
```

The original wrapper/campaign/report failed with exits 1/1/1 because both report callers omitted the production-root argument when resolving native sidecars. Postflight passed. Original failure records remain, bound by `REPORTING_RECOVERY.json`. The repair passes `evidence_root=root` and permits explicit recovery only for this completed missing-path failure after the full saved-evidence audit succeeds. No original status is rewritten; no inference, feedback, teacher preparation, training, optimizer step, checkpoint or tensor reconstruction was added.

All 50 focused K+ CPU tests passed in 12.972 s; the separate path/recovery review exercised 20 focused cases. Scientific execution had already passed 76 local CPU tests, 40 server tests and its four included native correctness generations. These counts describe distinct test runs, not additive scientific sample sizes.

Configuration/model/runtime pins and the original execution seal are under `configs/`. Raw private prompts, output token IDs, native records, anchor bindings and per-event scientific receipts remain in the private retained evidence tree and on A800; they are not uploaded to Git. Public CSVs contain measurements and request identities. The code does not reconstruct the retired intermediate tensors.
