# Native qualification and launch report

Execution: `cdb640543d8fe3a43c8e63af4ccaaa002e09a599`, [PR #32](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/32).

All six disposable native updates passed. Qualification took **377.465 seconds**. Each case began from the same authenticated released weights, fresh BF16 Adam and seed-17 RNG; two fixed response pairs were checked independently for each arm. Losses and gradients were finite before Adam, all 58 Adam entries advanced exactly once, updated weights/moments remained finite BF16, and the frozen target-derived parameters remained outside optimization. Actual updated state was saved/decoded, the trainer was reset to its initial state, and decoded weights, moments, counters, accumulation and RNG restored exactly. The original initial state was then restored again.

| Arm | Pair | Validity | Changed BF16 parameter coordinates | Peak allocated GPU memory |
|---|---:|---|---:|---:|
| KPLUS | 1 | passed | 717,960,027 | 15.359 GiB |
| KPLUS | 2 | passed | 720,361,705 | 19.048 GiB |
| MIX | 1 | passed | 721,489,086 | 21.701 GiB |
| MIX | 2 | passed | 721,745,538 | 18.329 GiB |
| ROLE | 1 | passed | 719,456,360 | 21.585 GiB |
| ROLE | 2 | passed | 720,722,589 | 19.048 GiB |

The memory column measures allocated memory after each fresh trainer construction and counter reset; it is not a separate construction or reserved-memory peak.

These coordinate counts verify a nonzero native update; they are not efficacy measurements or cross-arm quality scores. Losses differ by definition and are not directly compared.

Accounted work: **6 disposable Adam steps, 12 response forwards and 12 backwards**, zero complete generations and zero additional startup response forwards/backwards. Native instrumentation records 60 layer forwards and 1,872 head calls, including any checkpoint recomputation. Six temporary decoded tensor files were retired only after successful comparison/restoration; compact receipts and identities remain. No temporary state enters production. The retained input files were rehashed unchanged.

A800 repeated all **77 new CPU tests** and 41 explicitly invoked objective/regression checks (20 overlap the 77; 21 are retained baseline tests). Exact source/input verification passed for 3,847 files. Model, runtime and initial-state validation passed. Full storage admission passed with **149.583 GB free** against **149.062 GB required additional allocation plus reserve**, including the unchanged 8 GiB reserve. Historical project artifacts remain intact.

The authorized main pilot has started. The first GSM8K KPLUS trajectory reconstructed all 16 fixed diagnostic contexts and began its own fresh feedback collection. All six trajectories and event-2/event-4 endpoint comparisons remain pending at this report's publication. Expected duration is approximately **6–9 hours from the main launch**, subject to measured progress and technical failures. No endpoint benefit is claimed.

Native qualification is a correctness/resource check on these fixed cases. It does not demonstrate that ROLE or MIX improves drafting, establish all future cases finite, or replace the prescribed complete pilot. All algorithms, coefficients, panels and endpoints remain fixed; failure preserves evidence without automatic retries.

A separate same-team receipt audit passed 90 consistency checks across 46 files, including the raw 60-event work ledger and all six case/restoration/retirement receipts. The numerical tensors were retired after successful checks, so this is an evidence audit, not independent GPU numerical replication. [Audit](QUALIFICATION_AUDIT.md).
