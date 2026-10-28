# KRC events 3–4 evidence package

The experiment is complete and stopped at event 4. [REPORT.md](REPORT.md) gives the result and interpretation; [FINAL_AUDIT.md](FINAL_AUDIT.md) and [FINAL_REPORT_REVIEW.md](FINAL_REPORT_REVIEW.md) document the evidence checks.

The scientific execution used commit `11b7718a04416a89025a6df69c0c33e558e5fe1b`. The independent audit and report builder used `ea4f289c6e18383b7560c8c4103a9d91ca3943bf`. Later publication helpers extract existing records; they do not change the experimental code or run additional model passes. File identities and publication transformations are recorded in [PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json) and [SUPPLEMENTAL_PUBLICATION_MANIFEST.json](SUPPLEMENTAL_PUBLICATION_MANIFEST.json).

- [SUMMARY.json](SUMMARY.json): results, validity, costs, source identity and retained state paths.
- [COMPLETION.json](COMPLETION.json): completed budget, checkpoints and stopping point.
- [TRAINING.csv](TRAINING.csv): event-level training, reference exposure and costs.
- [CORRECTIONS.csv](CORRECTIONS.csv): all 256 update windows, including zero-violation no-ops, correction norms, pre-correction reference outcomes and work/cost accounting. Post-K reference metrics are not measurements of the corrected model; optional post-correction probes remained disabled.
- [Development tables](analysis/development/): per-request results, all 576 executions, three-repeat totals, comparisons and event-1/2/4 count curve.
- [Additional-panel tables](analysis/additional/): the corresponding separate tables for the seed-20261028 panel.
- [POSTRUN_STORAGE.json](POSTRUN_STORAGE.json): final idle GPU/storage check and removal of the verified redundant transport archive.

Complete raw validity records, sampled anchors, feedback and canonical token sequences are retained outside Git at `/home/minhao2/EAGLE_SPEC/research/dflash_krc_events_3_4_20261028/raw/` and `/data/sources/minhao/runs/dflash_krc_events_3_4_20261028/`. Complete KRC3/KRC4 optimizer recovery states and model exports remain at the A800 paths listed in the report. These paths are evidence locators, not public download URLs.

Public copies preserve numerical results. Only the report's private-evidence links and availability note are adapted for publication; the original local report remains unchanged. No tuning, additional arm or continuation beyond event 4 is included.
