# Cost scope

All clocks are observed host-wall measurements at the named level. Nested rows are already included in their parents; do not sum the table.

| Phase | Clock | Seconds | Included in | Scope |
|---|---|---:|---|---|
| run | runtime wall | 3330.467275 | none | All current execution phases; excludes prior staging, transfer and CPU review. |
| qualification | three sequential case wrappers | 432.147379 | runtime wall | Includes3disposableupdates and all qualification startup/serialization. |
| training K | worker subprocess wrapper | 480.478028 | runtime wall | 128updates/256presentations plus initialization, startup, export and shutdown. |
| training K | intrinsic training | 328.064734 | worker subprocess wrapper | Recorded native training clock; excludes separate startup and checkpoint/export clocks. |
| training SHUFFLED | worker subprocess wrapper | 554.585870 | runtime wall | 128updates/256presentations plus initialization, startup, export and shutdown. |
| training SHUFFLED | intrinsic training | 400.984953 | worker subprocess wrapper | Recorded native training clock; excludes separate startup and checkpoint/export clocks. |
| training TRUE | fresh intrinsic training | not measured | not applicable | Authenticated reuse; zero new main updates. No fresh comparable TRUE training clock. |
| training SHUFFLED | extra detached student projections | 50.372672 | SHUFFLED intrinsic training | 7994extra projections; nested timer, not an additional run wall. |
| training SHUFFLED | assignment | 8.139749 | SHUFFLED intrinsic training | Fixed hash mapping and mask assignment; nested timer. |
| references | preparation wrapper | 66.623824 | runtime wall | 35new references;221authenticated references reused. |
| references | loading | 28.258641 | reference preparation wrapper | Separate serving backend initialization. |
| references | generation | 28.134607 | reference preparation wrapper | 35 canonical-reference generations using retained TRUE through the native SGLangDFlashBackend verifier. |
| diagnostic | parent block wrapper | 26.438241 | runtime wall | 64untimed one-block parent probes, no full parent completions. |
| serving | worker wall | 1669.717516 | runtime wall | Includes loading, evaluation and closure. |
| serving | evaluation wall | 1633.331483 | serving worker wall | 1536timed completions plus161publications and evaluation overhead. |
| serving | loading | 20.756898 | serving worker wall | Backend initialization; no extra generation warmup. |
| serving | outer publications | 302.128923 | serving evaluation wall | Sum of161publication wrappers; includes native activation work. |
| serving | native activations | 55.846632 | outer publications | Nested native activation timers; not additive to wrapper time. |
| serving | all timed request clocks | 1254.390692 | serving evaluation wall | Sum across all3arms and both repeats; differs from per-arm mean-repeat metric totals. |

Observed fresh intrinsic training difference: SHUFFLED − K = 72.920219s (+22.227387%). This includes K+ reduction and other arm work as well as the detached projection/assignment overhead; it is not a pure assignment-cost estimate. TRUE has no fresh training clock in this run.

Current runtime3330.467275s =55.507788min. No prior failed-launch receipt was supplied to this audit, and missing historical costs are not imputed. Qualification and final main-training costs are separate. No training-inclusive acceleration claim.

Exact values, source paths and nesting are also retained in COST_SCOPE.csv and NUMERICAL_RESULT_NOTE.json.
