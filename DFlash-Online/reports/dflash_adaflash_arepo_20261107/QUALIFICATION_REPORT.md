# Released AdaFlash loss: qualification and campaign readiness

## Material Passport

**Status: native numerical qualification passed; full campaign admitted and initializing.** This report summarizes retained qualification and operational evidence. It is not a serving-performance result or an independent GPU replication.

Scientific execution commit: [`7dac02b0fc6686d7129c5d29699ee9bfe5be2e7a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/7dac02b0fc6686d7129c5d29699ee9bfe5be2e7a), [PR #31](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/31). The approved experiment is **released AdaFlash loss under our matched online protocol**. The source loss is pinned to AdaFlash `478bf0ff7a3a5232b13d0f14e752d5fe1da0b718`, `pipeline/training/rkl_model.py`, lines 168–207.

## What passed

All **47 new CPU tests and 41 retained regression tests passed**. The native check compared the literal source expression with the new port on three fixed pairs of retained responses. Every branch restored the same released weights, fresh optimizer and recorded RNG before its disposable one-step update.

For each of the three pairs, recorded gradients and actual BF16 AdamW displacements agree exactly across **1,048,626,432 parameter coordinates**: zero differing coordinates and zero discrepancy norm. Recorded response losses, clipping decisions, Adam state and final RNG also match. All six initial-state restorations passed. A separate same-team audit recomputed **54 aggregate checks** from the per-parameter records; all passed. That audit has disclosed implementation authorship overlap.

The port preserves the literal CE/softmax/teacher-log-softmax-cast/`F.kl_div` arithmetic, full-response in-place normalization, position clipping and final CE/RKL mixture. This qualifies the prescribed implementation on these cases; it does not establish a training or serving benefit.

| Fixed pair | Eligible position occurrences | Capped occurrences | Smallest positive draft probability | Zero probabilities |
|---|---:|---:|---:|---:|
| 1 | 11,400 | 1,478 | 1.1258623e-26 | 0 |
| 2 | 6,630 | 2,588 | 5.4283728e-25 | 0 |
| 3 | 7,125 | 2,226 | 4.4428646e-26 | 0 |
| Total | 25,155 | 6,292 | 1.1258623e-26 | 0 |

These are sampled training-position occurrences, counted once per pair rather than twice across the matching implementations. They are not independent served tokens. Native zero probabilities were not observed here. CPU boundary tests still show that the literal source can produce nonfinite backward values if softmax underflows to zero; qualification does not rule out that later boundary. The campaign must preserve its stop-on-nonfinite policy, without stabilization or automatic retry.

**Legacy-label clarification:** the reused comparison helper writes `paper_entry` for its left side and `repo_position` for its right side. In these qualification records those aliases mean **literal_source** and **ported_loss**, respectively, as established by branch order and directory identities. This is not a new comparison with the earlier paper-clipping arm. The original records remain unchanged.

## Qualification work and cost

| Work | Completed |
|---|---:|
| Disposable AdamW steps | 6 |
| Core response forwards / backwards | 12 / 12 |
| Startup response forwards / backwards | 2 / 2 |
| Startup AdamW steps | 0 |
| Small numerical-fixture forwards / backwards | 6 / 6 |
| Complete-response generations | 0 |
| Permanent serving exports | 0 |
| Uncompleted qualification branches | 0 |

Qualification wall time was **242.275886 seconds**. Peak allocated GPU memory was **23,290,610,176 bytes**. Recorded execution included 14 model/draft forwards, 70 layer forwards including recomputation, and 2,136 vocabulary-head forwards including recomputation. These detailed counters describe the same work and must not be added to the response-level totals. Disposable scientific states are not reused for main training.

## Storage and launch status

The full conservative requirement remains **159,565,098,291 bytes**, including the 8 GiB reserve. Qualification had its separate 24 GiB admission; passing that gate did not admit the full campaign.

Reviewed retirement operation 002 removed exactly **12 hardlink paths representing six completed recovery-state inodes**, retiring **37,826,387,968 allocated bytes**. Recorded free space afterward was **166,391,746,560 bytes**, above the planned requirement. The retired states are the final Track2 K/U/R event-8 recovery states, O_HALF seed-17/29 event-4 states, and K_HALF event-58 state. Their serving exports, initial states, compact results/validity evidence and current experiment dependencies were preserved and rechecked. Exact optimizer/RNG continuation from those retired boundaries is lost: these tensors were **retired, not archived**.

Two earlier operational stops remain preserved. The initial pre-model admission encountered a research-record transfer and performed no scientific work. Retirement attempt 001 stopped on a `/proc/1` permission check before any unlink. The repaired storage inspection explicitly scopes process checks to the execution cgroup namespace and records external host processes as uninspected; it does not claim complete host visibility. No scientific setting or numerical tolerance changed, and native qualification ran only once.

Full admission passed at launch. The main campaign supervisor started at **2026-11-07 01:02:22 UTC** (PID2893749); the training worker is PID2894161. At this snapshot it is initializing the native runtime, with no completed event or endpoint result yet. The continuation reuses the passed qualification and adds no disposable optimizer steps. Operational status is under `operational_continuation_001`; the original root-level exit1 remains preserved as the earlier storage-admission outcome, not the status of the new main run. The final generated report will be written to `results_after_admission_001`.

## Unchanged remaining experiment

One new AREPO trajectory starts cleanly from released DFlash with fresh seed-17 BF16 AdamW: **7,473 own-feedback requests, 58 events, 7,424 optimizer steps and 14,848 response presentations**, with the last 49 requests served without another update. Four inherited correctness generations are accounted separately. Existing O58, K+58 and paper-A58 remain unchanged.

The final endpoint measures **O58 / K+58 / paper-A58 / AREPO58** on the same 1,024 requests with three balanced repeats: **12,288 timed completions**. Primary analysis remains K+58 versus AREPO58 complete-request latency, using the fixed paired 10,000-draw bootstrap with seed 20261102. Calls, both acceptance counters, committed progress and adaptation costs remain separate. No tuning, new panel or automatic follow-up is authorized.

The estimated remaining campaign duration is **12–15 hours after admitted main launch**, based on the earlier matched trajectory and the larger four-arm endpoint. This is provisional and will be updated from actual progress. No endpoint efficacy claim is available yet.

Evidence: `qualification_evidence_001/qualification/QUALIFICATION.json`, `BATCH_*_PARITY.json`, branch `RESULT.json` files, `reviews/QUALIFICATION_AUDIT.md` and `.json`, plus the storage audit, visibility addendum and both retirement receipts. Raw response/token evidence remains private; this report contains only aggregate qualification results.
