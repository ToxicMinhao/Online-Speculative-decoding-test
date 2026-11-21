# CODE qualification: AR startup failure and narrow repair

Execution commit: `9a7eabe7e3539410ce995a17d7407ea4f8fd495c`. Repair/report commit: the Git commit containing this document. **The campaign has not started and GPU qualification has not passed.**

## Material Passport

ARS experiment-agent run supervision. Scope: the prescribed CODE comparison, its startup evidence, and a CPU-tested configuration repair. No scientific result is available.

## What happened

CPU preflight passed in 24.354 seconds. Both A800 AR engines loaded the pinned target and completed ordinary batch-size-one CUDA-graph capture. Each then failed with an illegal-memory-access error while initializing the runtime's experimental piecewise prefill graphs. Both cards were released and returned to zero allocated MiB. Launcher wall time was58.544 seconds; occupied-card time was56.0318 seconds in total.

This was an engine-startup failure, before any complete-response generation, teacher preparation, training presentation or Adam step. The scientific ledger has zero attempted and completed operations and no pending scientific operations. Six ordinary graph-initialization forwards completed across both cards. Additional piecewise initialization executed but was not fully instrumented: its exact forward count is **unknown**, not zero. Original logs, clocks and receipts remain intact.

## Confirmed mismatch and repair

The exact installed SGLang `server_args.py` defaults `disable_piecewise_cuda_graph=False` (line629), but automatically sets itTrue when `speculative_algorithm` is non-null (lines1044–1046). The qualified DFlash path sets `DFLASH`; genuine AR correctly setsNone but had omitted an explicit piecewise policy. Consequently AR unintentionally enabled an additional experimental compiler path absent from the DFlash configuration.

The narrow repair explicitly sets `disable_piecewise_cuda_graph=True` in the AR adapter. Ordinary CUDA graphs remain enabled (`disable_cuda_graph=False`). BF16, FA3, greedy decoding, seed17, physical/logical context4112/4096,8192 prefill chunking, cache handling, models, training objectives and all experiment budgets remain unchanged. The measurement manifest and actual runtime status expose both graph settings.

The logs show an8192-row piecewise compilation followed by failure around the7680 shape, while physical context is4112. This suggests a possible shape-related trigger; it **does not prove the underlying CUDA kernel cause**. The operational configuration mismatch is established independently of that hypothesis.

44 focused AR, lifecycle and launcher CPU tests pass. Independent source review confirmed the runtime's effective DFlash policy and the supported status field. These checks do not establish that the native retry will succeed. No retry has run.

## Storage and authorization

The user approved90 GB additional storage, retaining the400 GB reserve,48 charged elapsed hours and90 occupied-card hours. The sealed peak forecast is86,530,781,582 bytes. Free space after source staging was493,257,908,224 bytes, before native startup. The dynamic reserve and component checks remain mandatory.

Five obsolete source-transport archives were retired after content and no-live-use checks:194,314,240 allocated bytes. The new source-transfer archive was also removed after all4,993 staged source entries authenticated. No models, recovery states, scientific results or useful caches were retired.

The provisional full-run estimate remains35–45 hours, conditional on successful qualification, all1,024 AR/F parity checks and measured admission. This failed startup provides no reliable campaign calibration.

## Next bounded action

Explicit retry authorization is required under the requested ARS execution workflow. The proposed retry uses the repaired published source, preserves the original failure records and time/cost ledger, and repeats the uncompleted qualification before any production. The ordinary scientific work budgets remain unchanged because none of their slots executed. Failed engine initialization stays separately charged, including the unknown piecewise-forward total. All other correctness and resource gates remain required; there is no automatic restart, tuning or performance claim.

Private source evidence is retained under the study's `failure_001` and `monitor_90gb` directories and at the A800 study root: `CPU_PREFLIGHT.json`, `LAUNCH.json`, `WORK.json`, `CLOCK.json`, both `qualification/{A,B}/{A,B}_AR/native/worker.log` files and their `PHYSICAL_WORK.json` receipts. The public machine-readable summary is [STARTUP_FAILURE.json](STARTUP_FAILURE.json).
