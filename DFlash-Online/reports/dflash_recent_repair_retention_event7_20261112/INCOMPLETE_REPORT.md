## Material Passport

- Status: **training and qualification complete; evaluation incomplete, with evaluation-only recovery authorized**.
- Scientific execution: `562f63fdff6955d5ffcf6086bf9b2f8f948dba8b` (PR44).
- First replay-accounting repair: `8fc2f68afc28b6f6891814f180eb257dca7153cf`; this was not the failed execution's code.
- Report publication commit: to be recorded at publication.
- This is a same-team source/receipt review, including overlap with implementation authorship. It is not an independent GPU replication.

**There is no result yet about whether RECENT improves native drafting.** All three final checkpoints exist, but no held-out parent trajectory or timed endpoint comparison completed. The interruption is a replay-budget implementation error, not an observed failure of the learning mechanism.

## Completed work and remaining scope

| Work | Completed | Remaining / status |
|---|---:|---|
| Frozen K6 preparation | 256 response forwards; 8,170 head chunks | Complete; zero backward/Adam work, caller RNG/modes restored |
| Disposable qualification | 3 Adam steps; 14 response forwards/backwards including startup | All four cases passed |
| Main NATIVE / RECENT / CONTROL | 256 presentations and 128 Adam steps each | All advance their own exact parent state from step 768 to 896 |
| First-natural-firing comparisons | 4 temporary Adam steps; 8 forwards/backwards total | ACTIVE replay and restoration exact for both active arms |
| Additional canonical references | 2 generated and valid; 254 reused | Complete |
| Held-out parent generations | 0 begun/completed | All 64 remain pending |
| Fixed-start endpoint comparisons | 0 completed | Cannot count required starts until parent paths are collected |
| Timed serving | 0 of 1,536 completions | All 1,536 remain pending |

The main event contains **384 scientific Adam steps and 768 presentations**. The seven disposable steps are separate and never become serving checkpoints. Three main-trajectory startups add six further discarded forwards/backwards and zero Adam steps, separate from qualification and training.

## Qualification and actual treatment

| Check | Result | Observed work / evidence |
|---|---|---|
| Ordinary NATIVE | Passed | One disposable Adam step, two response presentations |
| Disabled extension | Passed | Exact native gradients, updated weights, Adam state/counters, RNG and selections |
| Zero coefficient | Passed | Same exact native and selection comparisons |
| Active synthetic fixture | Passed | One forward/backward; nonzero draft gradients in all 58 trainable tensors |
| Longest retained response | Passed | 638-token sequence, active extension, one forward/backward; exact state restoration |
| First natural firing, each active arm | Passed | ACTIVE reproduces its production endpoint; OFF changes gradients/moments/BF16 weights; final scientific state restored exactly |
| Final state/export integrity | Passed, all three arms | Step 896, 1,792 cumulative presentations, 58 Adam-state entries; no extra update during boundary checks |

Prospective training eligibility retains **815 distinct recent slots / 1,629 occurrences across 128 requests**. The placement control switches **114 slots across 70 requests**, covering **16.4569%** of assigned weight. Absolute margin SMD is **0.012456**. All prespecified placement screens pass; the arms are not identical placements.

| Active arm | Assigned occurrences | Reachable | Hinge fires | Fires while wrong / still correct | Firing presentations / windows |
|---|---:|---:|---:|---:|---:|
| RECENT | 1,629 | 1,538 | 341 | 249 / 92 | 160 / 107 |
| CONTROL | 1,629 | 1,543 | 322 | 243 / 79 | 164 / 107 |

At each arm's first natural firing, switching the term off changes the pre-clipping gradient by L2 **0.00117054** and changes **297,041 BF16 parameter entries** after the update. Both arms happen to share this initial diagnostic result; they diverge later. Final RECENT and CONTROL differ from NATIVE in **296,587,104** and **297,981,102** of **1,048,626,432** parameter coordinates. These observations establish that the implementation applies a nonzero treatment. They do not establish useful repair retention or serving benefit.

Frozen assigned weight is **0.335772%** of frozen selected weight before the 0.10 coefficient. This is an exposure measure, not gradient influence. Counts use each original response observation once, never duplicated accumulation rows or checkpoint recomputation. The raw assignment is equal between active arms; tiny differences in reported FP32 reductions across their different chunk placements are not additional assigned exposure.

## Measured cost so far

| Arm | Main training s | Loading s | Disposable startup s | Persistence s | Export s | Diagnostic state capture s | First-firing replay s | Worker wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NATIVE | 333.391 | 32.273 | 19.199 | 18.603 | 14.505 | 0 | 0 | 442.578 |
| RECENT | 339.372 | 31.229 | 19.576 | 19.301 | 14.505 | 55.720 | 107.531 | 627.868 |
| CONTROL | 338.977 | 32.965 | 18.431 | 20.818 | 15.707 | 54.989 | 104.772 | 612.502 |

The observed main-training clock is **5.981 s / 1.794%** higher for RECENT than NATIVE. This compares the actual diverging trajectories and their logging, not an isolated kernel overhead measurement. Diagnostic capture/replays are explicitly separate. Worker wall clocks include the listed components and other lifecycle checks; they must not be added to their subcomponents.

- Frozen preparation: **103.909 s**.
- Qualification worker walls: NATIVE **93.180 s**, DISABLED **95.714 s**, ZERO **97.177 s**, ACTIVE/long **68.616 s**; startup is included in these workers.
- Canonical preparation: **49.965 s** wall, including **42.012 s** loading and **1.678 s** generation calls.
- Failed fixed-start phase: **24.235622 occupied GPU seconds**, including startup work described below.
- Campaign wall before interruption: **2,325.631 s (38.7605 min)**. Recorded allocated phase windows total **0.624344 card-hours**; this includes loading/CPU lifecycle time and is not a kernel-busy measurement.
- Peak main-training allocation: **20,514,508,800 bytes**; peak reserved memory: **25,811,746,816 bytes**. Both cards were idle at the post-exit check.
- A800 disk free: **618,310,000,640 bytes** at preflight and **504,902,623,232 bytes** at the post-exit check. No artifact was deleted. This observed change is not a measured decomposition of per-method storage cost.

**Reused-artifact scope:** feedback responses, teacher packets and K5/K6 label trajectories come from retained PR42 evidence. This run does not newly time their collection, teacher preparation or historical-checkpoint label replays. Future online events need those labels and their replay work is not free. Full online deployment costs and training-inclusive speedup remain unestablished.

## Why evaluation stopped

The fixed-start worker's `model_load` allowance was **1**. Engine construction consumed it, but the inherited `capture()` path first publishes the requested checkpoint before starting `transition_begin` or generation. That legitimate publication uses the same counted operation and was rejected as `model_load allowance exhausted`.

The original work ledger is retained exactly. It contains **one failed `parent_walk` wrapper and 63 unattempted wrappers**, while **no actual parent generation began**. Its **51 completed draft blocks and 48 completed target calls are all explicitly marked startup**. There are no saved request captures and `STARTS.json`/`OUTCOMES.json` are empty. Accordingly, all 64 scientific parent generations remain pending. Startup work must not be erased or described as zero GPU work.

The automatic report correctly withholds all gain claims. Its empty per-request and repeat tables mean evaluation was unexecuted, not that methods tied. Its generic incomplete summary does not, by itself, expose the failed-wrapper/startup distinction; this report and the retained ledger provide it.

## Evaluation-only recovery

The user has authorized evaluation-only recovery. It must use a new output directory and source commit, preserving the original failure and the three unchanged trained exports.

1. Fix publication/load accounting and test engine initialization, capture publication, historical publication and three endpoint publications. This changes infrastructure accounting, not the algorithm or scientific cases.
2. Authenticate the exact exports, original input panel, assignments and canonical references. Reuse the completed qualification/training/reference evidence; perform **zero new training, optimizer steps or canonical-reference generations**.
3. Execute the pending 64 parent generations, their prescribed parity/historical/endpoint block replays, then the fixed 1,536 timing completions. Keep the original panel, order, settings and scientific gates. The failed startup remains separately charged; the remaining fixed-start GPU allowance is **14,375.764378017 s** from the original 14,400-second cap.
4. Record both training execution `562f63f…` and the new evaluation execution commit. Report prior plus recovery wall time and retain the first failed startup once; do not duplicate the linked training/qualification costs.

**Decision pending.** The current evidence supports completed implementation/qualification and an applied treatment. It cannot answer whether RECENT protects held-out repairs, beats either comparator, improves acceptance length or reduces serving time. That decision requires the unchanged pending evaluation.

## Evidence package

The generated `SUMMARY.json`, scalar CSVs, `AUTOMATED_INCOMPLETE_REPORT.md`, qualification/provenance tables and publication metadata accompany this report. Compact private receipts remain outside Git. A total of **67 files / 24,457,160 bytes** was retrieved and verified against remote per-file SHA256 identities; original remote evidence remains intact. No model tensors were retrieved for this report.
