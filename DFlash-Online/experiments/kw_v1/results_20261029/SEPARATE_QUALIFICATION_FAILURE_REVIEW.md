# Separate KW-v1 qualification failure review

Material Passport: ARS experiment-agent `validate`; verification status **ANALYZED**; local retained-receipt and published-source audit; 2026-10-30 UTC. Reviewer: separate same-team `kw_eval` agent. No rerun, GPU access, model forward/backward, optimizer application, remote operation or source edit was performed by this reviewer. Reproducibility rerun was explicitly excluded. The companion JSON records source bindings, exact counts, all retained bounded difference paths and raw-file hashes.

**The implementation qualification failed at `neutral_window_1`; production did not start.** This is a correctness blocker, with no efficacy interpretation. All 512 production feedback requests and 4,032 evaluations remain unattempted. No automatic retry occurred. Stage 1 / project None is unchanged.

## Failure evidence

Published execution source is `9fd9fe6f7db9796a49e2a6879b5a79e7f01fbd22`. The qualification first ran two ordinary K windows on four preselected retained responses, then reached the first KW window on the first two responses. The exact comparison at cumulative step 1,025 raised an exception before the second KW window or subsequent optimizer-only qualification checks.

I independently compared the saved response rows. Both complete KW rows equal the first two K rows exactly, including response IDs, all 31/32 recorded scalar metric fields respectively, anchors, optimizer-step indicator and cumulative step. The inputs are `train_00947` and `train_00916`. The KW window receipt records allocation `a=0`, weights `[0.5, 0.5]`, zero controller parameters before/after and no applied credit.

Despite those matching observations, the saved native-state comparison reports differing weights and Adam state. Its **100 saved paths are capped evidence**: 31 weight paths, 42 `exp_avg` paths and 27 `exp_avg_sq` paths. The comparison computes native-bit differences and then writes only the first 100 names. These are not the full number of differing paths or parameter coordinates. Neutral-comparison tensors and coordinate difference magnitudes were not retained, so those quantities cannot be reconstructed. No claim that RNG, modes, counters or other state fields match at the failed boundary follows from this truncated list.

The window's changed-coordinate and norm statistics describe its positive/negative preview endpoints. They do not quantify the failed neutral K/KW comparison. Root cause is not established by this receipt audit; matching scalar loss/anchor records alone do not prove equality of every backward contribution or optimizer effect.

## Work accounting

The qualification receipt and independently parsed work ledger agree. All 17 recorded attempts have completed status; none remains recorded as unknown.

| Qualification work | Actual | Ceiling |
|---|---:|---:|
| Response forwards | 6 | 8 |
| Response backwards | 6 | 8 |
| Disposable optimizer applications | 5 | 16 |
| Serving generations | 0 | 0 |
| Persistent optimizer updates | 0 | 0 |

The six response passes are four K and two KW. The five disposable optimizer applications are two ordinary K updates, two KW previews at allocations `+0.0625` and `-0.0625`, and one KW live update at `0`. The KW work log independently records two completed previews and one completed live update, with no credit attempt. These temporary qualification updates are not production optimizer steps.

Production remains at zero feedback/evaluation generations, zero committed optimizer steps, zero response presentations, zero previews and zero credit opportunities. The corresponding unattempted budgets are 512 feedback, 4,032 evaluations, 1,536 optimizer steps, 3,072 presentations, 1,024 previews and 504 credits. The retained wrapper invokes only preflight and qualification. Source inspection confirms that production requires passed qualification before creating `campaign_001`; no such production directory appears in the retained raw tree. The failure receipt has no passed native-neutral, preview-isolation or full-model-memory gate.

## Source and identity checks

All **2,598 sealed repository files** match the bytes at published commit `9fd9fe6`, and all **19 sealed run inputs** match their retrieved copies: 2,617 final-seal entries in total. The final seal predates qualification, at `2026-10-30T00:33:04.997040+00:00`. The qualification receipt, PLAN and SEAL agree on execution commit. The qualification-input digest independently recomputes to `403896b58e38146f15087ffeb3a8b68cbaf519cb6ec6e52968b9c5bbfc62fc7b` using sorted compact JSON.

The source K8 digest agrees across qualification receipt, sealed PLAN, qualification selection and preflight: `1b1a646eb1dac0b61629a0eb3586b424dea847f487a849f63ec800489de6e9ed`. The four selected input identities/order agree with the recorded K response rows and preflight feature checks. This binds the failed test to the intended original state and inputs, rather than to a newly trained production checkpoint.

The subsequently retained **postflight passed** in 54.503220 seconds. It checked all 2,617 sealed files and eight original O8/K8 state/export/receipt files, with zero model passes, optimizer steps or generations and CUDA uninitialized. Original O8/K8 state digests match preflight; both serving exports remain exactly equal to their original states, and the four qualification feature identities are unchanged. This establishes preservation of the retained originals; it does not establish equality of the failed temporary neutral-boundary states.

## Timing and memory

The qualification receipt reports **106.590378942 seconds**. The enclosing wrapper ran from `2026-10-30T00:42:37Z` to `00:45:43Z`, 186 seconds, and exited with code 1. That wrapper interval includes the 73.689185-second read-only preflight and process startup/overhead; it is not the qualification function's wall time.

| Observed phase | Peak allocated | Peak reserved |
|---|---:|---:|
| Ordinary K qualification | 21,607,704,064 bytes = 20.123743 GiB | 23,282,581,504 bytes = 21.683594 GiB |
| Failed KW qualification | 42,752,323,584 bytes = **39.816204 GiB** | 43,947,917,312 bytes = 40.929688 GiB |

These are observed peaks before the correctness failure. They do not establish a passed full qualification or complete production memory bound. The recorded exception is a neutral-state mismatch, not an out-of-memory error.

## Limits and disposition

The second neutral KW window, later nonzero optimizer-only comparisons and final qualification success checks were not reached. There is no held-out serving measurement, final-panel comparison, bootstrap interval or production-speed result to interpret. Statistical efficacy and related significance/fallacy claims are inapplicable to this failure audit. The original failure, work ledger, response records, window record, source/input seal and read-only postflight are retained. Production remains stopped; this review requests no rerun or additional model/optimizer work.

## Final report and analysis cross-check

The final `REPORT.md`, `SOURCE_FAILURE_ANALYSIS.md` and seven `analysis/` JSON/CSV artifacts were checked against the retained receipts on 2026-10-30 UTC. All reported analysis-table values match their raw sources, including both CSV rows, first-window metrics, source bindings, timing/memory bytes, work totals and failure paths. All 55 entries in the retrieved compact-file identity table rehash exactly. No scientific arithmetic error or unsupported efficacy/qualification-pass claim was found.

The report's more precise statement of exactly 31 differing weight tensors is supported by source traversal order: the complete weight mapping is traversed before any optimizer fields, and the retained list already includes optimizer fields. The 69 recorded Adam-moment paths remain only a lower bound. This refinement does not establish coordinate magnitudes, coordinate counts, gradient equality, or complete RNG/mode/counter equality. The report correctly limits paired equality to saved response records and separates 17 completed qualification attempts from zero production work. Its partial memory observations are explicitly not a qualification pass or production cost estimate.

Retirement accounting also matches the compact evidence: 77 retired paths, 59 unique containers, 169,880,600,576 unique allocated bytes. I rehashed all 118 retained metadata/statistics artifacts successfully. The report discloses lost exact historical tensor reinspection and re-serving of the three disposable diagnostic exports, and does not call retired tensors archived. One wording precision was sent to the report author: 169.88 GB is the verified unique allocated size retired, whereas the immediate `STORAGE_REMOVED.json` free-space delta is 167.778807808 GB; those measures should be named separately. This does not change the qualified retirement scope or the failure outcome. This cross-check performed no model tests, reruns, remote actions or further scientific work.
