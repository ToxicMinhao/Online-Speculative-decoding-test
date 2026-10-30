# Separate bounded requalification result review

ARS experiment-agent validation; **ANALYZED** from retained receipts and published source, 2026-10-30 UTC. Separate same-team reviewer: `kw_eval`. No rerun, model execution, GPU/remote operation or source edit was performed for this result audit.

**The repaired qualification passed its exact checks within the fixed budget. Production remains unstarted.** The result establishes neutral equivalence and preview isolation under the tested fresh-process startup. It provides no drafting-efficacy result or authorization to resume production.

## Independent checks

- All **39 retrieved compact-file identities** match. All **2,641 seal entries** independently match available committed/retrieved bytes: 2,615 repository files at execution commit `708e974e06f5d9f4b186c32607c16b37cbd4b14f`, seven new repair inputs and nineteen retained original-run inputs. Qualification receipt, PLAN and SEAL bind the same revision. Original K8 and canonical qualification-input digests match the sealed sources.
- The **31 unique exact checks** agree across `CHECKS.json`, worker and final receipts, all with native-bit criterion, empty differences and zero differing paths. These include two K backward/pre-clip consistency checks, exact initial state, all four K/KW backward gradients, both pre-clip gradients, both neutral preview state/gradient isolation checks, both neutral endpoints, complete response observations, and fifteen nonzero preview/native checks at allocations −0.25, 0.137 and 0.25.
- All four complete new K response rows exactly equal both the new KW rows **and all four retained K rows from the original failed qualification**. This covers recorded IDs, metrics, anchors, stepped flags and reported steps. It does not retroactively prove equality of unretained old gradient/state tensors.
- K and KW ran in distinct children, PIDs **1083298** and **1085007**, under coordinator **1083238**. Both recorded uninitialized CUDA and absent compiled Flex/block-mask singletons before model construction, identical recorded environments and runtime versions, and actual GPU initialization afterward. The coordinator remained CPU-only. Both neutral allocations are exactly zero with weights `(0.5, 0.5)`; two windows/four previews and one delayed credit are recorded.

## Work and costs

The shared ledger has **30/30 completed attempts**, consecutive per-kind ordinals and no unknown work. Final counts are **8 response forwards, 8 backwards and 14 disposable optimizer applications**, within ceilings 8/8/16. The applications comprise two K updates, two KW updates, four KW previews and six nonzero optimizer-only applications. Serving generations and persistent updates are zero. The earlier failed 6/6/5 qualification remains separate historical work; no additional full-model debugging sequence is recorded between the two qualifications.

Production remains at zero: all 512 feedback generations, 1,536 optimizer steps, 3,072 presentations, 1,024 previews, 504 credits and 4,032 evaluations are unattempted. The executed repair launcher contains no production invocation.

| Measurement | Observed |
|---|---:|
| Qualification coordinator wall time | 392.920741 s |
| K worker wall time | 158.981513 s |
| KW worker wall time | 213.390347 s |
| Wrapper, including admission/CPU checks/startup | 458 s, exit 0 |
| K peak allocated / reserved | 19.068843 / 19.939453 GiB |
| KW peak allocated / reserved | 29.769989 / 30.220703 GiB |

The coordinator time encloses child execution and orchestration; these times must not be added together. Qualification includes tensor persistence, hashing and exact comparisons and is not a serving-latency or production-controller-overhead estimate. The seven unique retained comparison tensor files total **27,264,636,789 bytes**, matching the postflight transient total. They remain on A800; this reviewer did not retrieve or independently recompute their tensor comparisons.

## Compiler-history evidence and limits

Each fresh worker log contains two matching recompilation events: `create_block_mask` after the `Q_LEN == 8192` guard fails, and `flex_attention` after key dimension 2 changes from **8,839 to 7,812**. Startup also records `assume_static_by_default=true` and `automatic_dynamic_shapes=true`. This is actual evidence that the selected response sequence changes compilation specialization within each process, consistent with the source-identified asymmetry of running cold K before KW in one shared process.

The fresh-process comparison now passes, but the original failed log did not retain kernel/recompilation identities or gradient boundaries. The repair also adds observation/synchronization and snapshot instrumentation. Consequently, these records do not isolate one specific kernel, compiler transformation or sole causal mechanism for the old mismatch. The optimizer-snapshot alias repair affects the later optimizer-only checks, which the original failure never reached; it cannot explain the original first neutral-window failure.

## Admission history and preservation

The first repair admission rejected a changed tracked `GITHUB_SOURCE.json` before qualification. Its preserved receipt reports zero model passes, optimizer steps and generations; its exit code is 1. `admission_failure_001` retains its timestamps, receipt and logs. `ADMISSION_REPAIR.json` records restoring the exact committed metadata and explicitly resuming admission, with unchanged scientific source, algorithm and tolerances. This was a zero-work admission repair, not a hidden repeated scientific run.

The final postflight reports all **2,641 source/input entries**, **eight original O8/K8 state/export/receipt identities** and **four qualification feature identities** intact, with no model/optimizer work and no active GPU compute process. I also independently rehashed all **55 original failed-run raw files** against the earlier review manifest; every byte identity is unchanged. The first failure and its interpretation limits therefore remain preserved.

No discrepancy was found in this bounded receipt/source audit. Qualification success is scoped to the sealed four responses and tested optimizer allocations under the repaired startup, not arbitrary process-history reproducibility, a complete production campaign or KW efficacy. Stage 1 / project None and the production stop remain unchanged.

Final reporting cross-check: all 31 rows in `analysis/CHECKS.csv`, the numeric/work/source fields in `analysis/SUMMARY.json`, and `analysis/SOURCE_PIN.json` agree with the retained receipts. The report correctly retains the zero-work admission rejection, the original failed qualification, the limited causal claim, and the 27.265 GB of comparison tensors as still retained. One wording clarification was sent to the author: the delayed credit is applied during the second window after its allocation has already been fixed; “by the end of the second window” accurately describes the final controller receipt.
