# Pre-launch review

## Material Passport

ARS experiment-agent run mode, October 24, 2026. User authorized review and the
fixed experiment. One additional independent internal agent reviewed the current
snapshot while the root checked the actual A800 allocation/storage and launch
configuration. This is not external peer review.

No additional implementation blocker was found. The review checked:

- Seven sealed scientific source files and the locally available input receipts.
- Exact seed-20261024 sampling of 64/1024 requests, sorted original indices and
  consecutive selected-panel ordinals; unique request IDs.
- 768 scheduled generations, with each arm in each order position 48 times.
- First-eight correctness gate, zero performance gate, no training/capture,
  structural/reproduction failure stops and coherent later partial-cohort handling.
- Native generate/publication/equality APIs, cache invalidation, and the declared
  timing boundaries; common all-arm/all-repeat cohort and denominator handling.

The locally missing copy of SOURCE_TRAINING_CONFIG.yaml has been retrieved. The
remote launch rechecks all INPUT_SEAL entries, actual model/target/hybrid/source
files, historical base source pins and native runtime before engine startup.
No scientific source change or tolerance adjustment was needed. The prior actual
A800 322-test result remains applicable; unchanged tests were not repeated.

A800 had 48.58 GB free and both GPUs idle. The remaining-write admission is 2 GB
plus 8 GiB reserve; existing hybrids are reused. No storage deletion occurred.
GPU 0 is selected through the existing worker API; no concurrent trainer starts.

Launch command (detached supervisor, existing wrapper):

```bash
bash experiments/osd_components/run_campaign.sh \
  /data/sources/minhao/runs/osd_checkpoint_components_20261024/prepared \
  /data/sources/minhao/runs/osd_checkpoint_components_20261024/campaign_001
```

Supervisor PID 238144; start 2026-10-25T02:18:56Z (October 24 in Edmonton).
The 7200-second timeout plus 120-second cleanup grace remains unchanged. The run
writes raw records, qualification, progress and partial/final reports itself.
No automatic retry, extra warmup or follow-up mechanism experiment is configured.
Native qualification is pending at the time of this review record.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/osd_checkpoint_components_20261024/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
