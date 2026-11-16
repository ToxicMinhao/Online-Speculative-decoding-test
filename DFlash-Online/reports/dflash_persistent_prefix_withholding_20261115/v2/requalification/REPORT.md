# Persistent prefix withholding: restoration fixed and qualification passed

## Material passport

- Workflow: ARS experiment execution and retained-evidence audit; Stage 1 / project None unchanged.
- Scope: repaired native qualification, followed by the explicitly authorized serving-only continuation. No production training, feedback collection or final campaign evaluation.
- Status: **QUALIFICATION_PASSED_NO_PRODUCTION**. This is implementation evidence, not an E-versus-N/S performance result.
- Training execution: `eb0aa345671086651254d5f79092809ea9b406b7`.
- Serving execution: `c471708969cf8c131c0c965b9fc7a801b63cc55a`.
- Restoration repair: `e50d185dc36efb6fa390026251046e30bef1033d`; compiler-path repair: `1d16ab1`.
- The [original failure report](../REPORT.md) and all failed-attempt records remain intact. This report supersedes their pending-qualification status without relabeling the failed checks as passes.

## The restoration error and its verification

The qualification loader shared the saved Adam step counters between disposable trials. PyTorch keeps these scalar counters on CPU even when the model and moment tensors move to CUDA. Consequently, the first trial changed the reusable restore source: the next trial reached Adam step 770 instead of 769. Identical gradients and moments then produced different weights through Adam's step-dependent bias correction.

The repair gives every restored optimizer its own counter tensors. It also checks the original counters after each trial and captures the final restoration oracle before any disposable update. It changes no learning objective, numerical tolerance, BF16 arithmetic, optimizer formula, anchors or scientific setting.

All **eight native training cases passed** on the two physical A800s:

| Check | Result |
|---|---|
| Legacy native K+ versus the ALL adapter | Exact forward values, preclip gradients, weights, optimizer state and RNG |
| Reusable source counters | All 58 remain at 768 after every case |
| Disposable optimizer endpoints | All 58 reach 769 in every case |
| E role gradients | All 2,937 correct-prefix occurrences have zero direct prediction-loss gradient; all 291 error occurrences have nonzero active gradient |
| Repeated E | Exact forward, gradient, optimizer, weight, RNG and role receipts |
| Fixed lower-LR S | Native forward/preclip gradients and Adam moments/counters match; declared LR remains 0.96 × 3e-4 |
| Physical-card comparison | N/ALL, E and S scientific receipts match exactly across cards |
| Final restoration | Exact original K6 weights, optimizer, counters, RNG and empty accumulation on both cards |

E differs from N in 41 of 58 endpoint weight tensors, and S differs in 40. These expected changes show that the tested paths are active; they do not establish useful drafting. See [training cases](TRAINING_CHECKS.csv), [cross-card receipt](TRAINING_CROSS_CARD.json) and [independent training audit](TRAINING_AUDIT.json).

## Serving qualification and the separate launch repair

The first serving startup omitted the qualified `CC`/`CXX` exports. SGLang's initial CUDA graph attempted to link its RoPE kernel using the unavailable system command `c++`. The retained compiler remains installed; its exact paths were recovered from the earlier qualified serving launcher. No compiler installation, precision change, kernel substitution or CUDA-graph disabling was used.

That failed startup attempted one target graph-initialization forward and completed none. It never entered request evaluation. Its raw generic accounting initially marked the absent request receipt as unresolved; the authenticated constructor traceback and physical-work receipt independently establish zero request-generation attempts. Both original records remain unchanged.

After the user explicitly authorized serving-only continuation, **all eight fixed generations passed**: two old requests × two repeats × two physical cards. Every execution matches its authenticated canonical token IDs and termination; per-request verification counts and acceptance signatures agree across repeats and cards. No training checks were repeated.

| Qualification request | Output tokens | Verification calls | Native accepted draft tokens | Valid executions |
|---|---:|---:|---:|---:|
| `eval_00515` | 226 | 30 | 195 | 4 / 4 |
| `eval_00491` | 435 | 66 | 368 | 4 / 4 |

The native committed-progress counters are225 and434: the full output includes one initial target-generated token outside these verification rounds. The retained legacy aggregate `output_tokens − verification_calls` is therefore one higher than the native accepted-draft count; the CSV preserves both fields.

These reused runtime checks establish no speedup or held-out efficacy. All individual measurements are in [SERVING_CHECKS.csv](SERVING_CHECKS.csv); see [cross-card validity](SERVING_CROSS_CARD.json) and [compiler diagnosis](COMPILER_DIAGNOSIS.json).

The independent final serving audit passed **93,040 assertions with zero errors**, while preserving the earlier81,258-check training audit unchanged. See [serving audit](SERVING_AUDIT.json) and [combined cost records](COMBINED_COSTS.json). All186files in the two compact result deliveries were byte-verified.

## Complete work and costs

| Work | Original failed GPU attempt | Repaired training attempt | Serving-only continuation | Cumulative |
|---|---:|---:|---:|---:|
| Disposable Adam steps | 2 | 8 | 0 | **10** |
| Response forwards / backwards | 6 / 6 | 20 / 20 | 0 / 0 | **26 / 26** |
| Training initializations | 1 | 2 | 0 | **3** |
| Complete serving generations | 0 | 0 | 8 | **8** |
| Occupied-card seconds | 89.016 | 278.060 | 68.019 | **435.095** |

Training response counts include the qualified disposable startup: two startup responses per initialization, with zero startup Adam steps. The repaired attempt's 278.060 seconds includes its failed serving startup (24.005 seconds). Serving initialization separately records 102 completed graph forwards and the earlier one failed graph forward; these are not response-level training forwards or extra complete-response warmups. New production steps, feedback generations, teacher preparation and exports are all zero.

The serving-only wrapper took 120.958 seconds, including CPU preflight. The original shared clock was never reset: elapsed time at completion was 3,131.714 seconds, including transfers, debugging, preparation and the earlier attempts. This is not GPU compute time. [WORK.json](WORK.json) preserves cumulative accounting; raw measurements remain outside Git under the task's `delivery` directory.

CPU evidence includes the earlier 133-test restoration suite, 20 retry/launch tests, the added final-status regression, 23 compiler/runner checks, six serving-continuation checks and two tests for propagating the same compiler settings to future production workers. These suites overlap; their counts are not summed into a unique-test total. The final production-launch propagation change (`8558c7a174a4d6b44f357ace3ee0e668119d255b`) is CPU-tested and was not a production GPU execution.

## Storage and readiness

Both A800s are idle after qualification. Training peak reserved VRAM was 18.581 GB; serving peak reserved VRAM was 57.594 GB, with a 58.630 GB device-wide memory sample after loading. Training's sampled process-tree RSS peaked at 34.399 GB. These are different measurements, not interchangeable peak definitions.

A800 retained **703.72 GB free** after evidence collection. The sealed peak-write estimate is 100.705 GB, including prior attempts, below the unchanged 110 GB cap with the 400 GB reserve. The original compiler-cache baseline remains 5.022 GB; observed growth is 0.0165 GB within its existing 4 GiB allowance.

A CPU-only admission attempt also exposed one obsolete dangling documentation link in the first transfer's staging directory. Only that verified redundant staging link was removed; its complete published-source counterpart and original archive remain. It occupied zero allocated data blocks, so this was not bulk storage reclamation. Exact path and before/after capacity are retained in `delivery/qualification_retry_001/DISPOSABLE_LINK_RETIREMENT.json`. No results, checkpoints or recovery state were deleted.

[Resource admission](RESOURCE_ADMISSION.json) passes for the measured model working sets and fixed work estimate. At qualification completion, its conservative estimated remaining elapsed runtime was approximately **2.11 hours**, including the declared contingency. This is an estimate, not a scheduled launch or guaranteed duration. The remaining elapsed-time cap at that moment was3.13hours, a different quantity.

The **main campaign remains stopped**. Retaining its original 640 production steps after the recorded qualification attempts would require cumulative totals of 650 Adam steps, 1,316 response forwards/backwards and eight training initializations. The original campaign ceilings of 648 / 1,310 / seven have not been silently increased. Full execution requires explicit budget reconciliation and a fresh check of the unchanged wall-clock/storage limits. There is still no E8/N8/S8 efficacy result.
