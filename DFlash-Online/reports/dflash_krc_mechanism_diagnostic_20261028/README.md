# Completed KRC mechanism diagnostic

Start with [REPORT.md](REPORT.md), the final interpreted result. Execution commit:
`c9461345189b91d241ff21590027500e6ac7dab4`; main numerical analysis commit:
`220c62e2c52489cf114125b679ae7188eea09218`.

The experiment is complete and stopped. Both historical scientific endpoints
reproduce exactly; all 20 windows and 80 temporary candidates are valid. There
were no serving generations or learned candidate arms.

- [INTERPRETATION.json](INTERPRETATION.json): final decision, distinct from the
  descriptive builder's deliberately non-decisional summary.
- [analysis/](analysis/): every candidate/window, event/pass/role aggregate,
  prediction, oracle and cost table; the numerical supplement adds distributions
  and timer/recomputation details. All denominators and regressions remain.
- [WINDOW_PLAN_PUBLIC.json](WINDOW_PLAN_PUBLIC.json): fixed request identities,
  roles, windows and hashes of the retained private anchor bindings.
- [INDEPENDENT_FINAL_AUDIT.md](INDEPENDENT_FINAL_AUDIT.md): completed same-team
  compact-evidence reconstruction. It is not an independent GPU replication.
- `INDEPENDENT_CODE_REVIEW.json` and `ANALYSIS_SOURCE_REVIEW.json`: historical
  source-review snapshots. Any pending execution/reconstruction fields describe
  that earlier review stage, superseded by the final audit and completion receipt.
- [REPORT_PROVENANCE.json](REPORT_PROVENANCE.json): execution/main-analysis
  identities plus hashes of the separately published supplemental/audit scripts.
- `pins/` and `validity/`: inherited configuration/model/runtime identities,
  exact endpoint checks, work checks and duplicate-artifact retirement receipt.
- [POSTRUN_STORAGE.json](POSTRUN_STORAGE.json): verified transport cleanup and
  remaining storage; original scientific artifacts remain preserved.

The supplement script expects `raw/` and `analysis/` beside it. Public source and
summaries do not include private responses, tokens, teacher features, native
per-position arrays, anchor arrays, model weights or optimizer tensors. Those
remain in the verified private run. The audit's detailed raw-reconstruction
companions are also retained there. Reproducing the analysis requires those
source artifacts; this public package is not a fabricated substitute for them.

Only the original full R correction continued in the replay. No candidate was
selected based on its local result. The report proposes no automatically executed
follow-up or replacement mechanism.
