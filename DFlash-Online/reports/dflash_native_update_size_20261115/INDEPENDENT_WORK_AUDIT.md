# Independent execution/accounting audit — PR48

**Passed:** 621,322 CPU receipt/ledger assertions. This audit performed no model execution or remote operation. Actual scientific execution commit: `5d4e0ff65c5976318a50ebb181923447068169c8`.

All 16 windows have qualified lower/upper brackets, producing 32 selected exports from 149 scalar trials. The largest relative bracket discrepancy is 0.244534%, within 0.25%. Every trial preserves the recorded clipped-gradient signatures, native clipping semantics and restored state; recorded moment checks pass. Four boundary P/N states were compared directly with original tensors. All 28 reconstructed intermediate P/N exports reproduced their original file hash and byte count exactly. The six CPU serialization checks and both retained E norms pass. Final native weights, Adam state, counters and RNG match retained NATIVE7 at step 896 and 1,792 presentations.

| Training work | Response forwards | Response backwards | Adam applications |
|---|---:|---:|---:|
| Native trajectory | 256 | 256 | 128 |
| Disposable GPU qualification | 2 | 2 | 6 |
| Qualified startup | 2 | 2 | 0 |
| Scalar calibration | 0 | 0 | 149 |
| **Total** | **260** | **260** | **283** |
| Prescribed maximum | 260 | 260 | 294 |

Physical instrumentation counters are cumulative snapshots. This audit checks monotonicity and final values rather than summing snapshots.

The 408,990-row native ledger has matched attempt/completion tickets and no failed, unknown or pending operations: 11,050 draft blocks, 11,111 target calls, 182,295 context forwards and 37 model loads. These include parent-history reconstruction and runtime startup, not merely the 8,192 scored control blocks. Eight neutral and 24 reuse sentinel blocks total the 32-block extra allowance. All 64 parents reconstructed successfully; all 8,192 new outcomes are complete, with zero invalid, unknown or unattempted outcomes. The complete calibration seal precedes scorer entry. The first native draft ticket occurs at 6,000.866 occupied-card seconds, after the 5,819.077-second qualification/calibration phase.

Occupied-card accounting is **7,147.993 seconds (119.133 minutes)**, below the 180-minute cap. Campaign wall time is 7,211.199 seconds (120.187 minutes). Full CPU preparation took 499 seconds, containing the 121.649-second serializer Gate0. The earlier 76-second CPU administrative failure consumed no GPU execution or Adam applications; its staging allocation remains in the budget roots. These clocks are nested and must not be summed as independent phases.

All 69 storage admission samples pass the 110 GB additional-allocation cap and 400 GB free-space reserve. The largest admission projection was 95.555 GB. Postflight observed allocation was **76.360 GB new and 629.333 GB free**. This is sampled admission/postflight evidence, not a continuously measured storage peak. It must not be reported as an exact peak.

[CONTROL_RETIREMENT_CANDIDATES.json](CONTROL_RETIREMENT_CANDIDATES.json) lists exactly 32 new selected scalar exports, totaling **67,112,290,304 logical bytes**. They may be retired only after the final raw-token/statistical audits and report are verified, while retaining identities, alphas, regeneration inputs and compact outcomes. Recheck that these are unique files with no aliases into KEEP assets first. This audit deleted nothing. The two framework-produced native model/recovery copies are excluded from this recommendation pending separate proof that they are no longer needed.

Limits: this CPU audit verifies retained parity receipts and their consistency; it does not rerun GPU-state comparisons. Scientific result interpretation belongs to the separate statistics audit. A preliminary audit-only check initially summed cumulative physical counters. It was corrected to monotone snapshot/final-counter validation without changing any execution artifact or scientific measurement.
