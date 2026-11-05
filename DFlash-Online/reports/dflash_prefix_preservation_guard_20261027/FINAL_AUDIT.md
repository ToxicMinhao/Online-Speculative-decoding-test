# Independent audit of the stopped guard qualification

The evidence supports the current report: **guard-disabled G matched ordinary K in the disposable one-step GPU control; enabled G did not qualify. No main training, local probes or serving evaluations ran.** This is an implementation/runtime failure and supplies no evidence about G's drafting benefit. No correction to the current `REPORT.md` or derived accounting is required.

The audit independently checked the original work log, both completed trial records, equality receipts, failure traceback, counters, costs, CPU logs and pre/postflight records. All **48 retrieved evidence files, 780,772 bytes**, match their recorded sizes and SHA-256 values. The separately retrieved postflight record reports all **438 sealed source/dependency files unchanged**. The audit made no source changes and launched no computation or retry.

## What passed

Ordinary K and disabled G each completed two response presentations and one native AdamW step. Their records agree exactly on response IDs, anchor positions/masks, losses, exposure, selection statistics and optimizer counters; only elapsed training times differ. The two responses were `train_00072` and `train_00121`, with 363 and 411 sampled anchors.

Six disabled-G comparison receipts report exact equality against K: after each accumulated K backward, before clipping, after clipping, after native AdamW, and at the final state. They cover gradients, weights, Adam moments/counters, RNG and mode metadata. The corresponding six K receipts establish the stored reference's finiteness/self-consistency; they are not six additional independent equivalence tests. The disabled guard's work-count equality also passes.

Startup restored its pinned initial state exactly. All three trials began with the pinned released parameters, fresh optimizer/counters and RNG/modes. Enabled G has only these initial-state checks; its K-gradient isolation, native pre-correction equivalence and final projection are **not** qualified.

Sources: [K trial](raw/campaign_001/qualification/K/TRIAL.json), [disabled G trial](raw/campaign_001/qualification/G_disabled/TRIAL.json), [disabled final equality](raw/campaign_001/qualification/G_disabled/final.json), [startup restoration](raw/campaign_001/qualification/startup/RESTORATION.json).

## Failure and exact work accounting

Enabled G completed its first response forward, then failed inside `torch.autograd.grad(..., retain_graph=True)` for H at `online_feedback/prefix_guard.py:258`. PyTorch's compiled backward explicitly rejected retained-graph execution with donated buffers. The training environment is Torch **2.13.0+cu130**; the separate, unused native serving environment is Torch 2.9.1. The traceback establishes this compiler/graph-retention incompatibility. It does not identify the exact compiled subgraph, nor establish numerical disagreement, a verifier defect, an out-of-memory condition or an adverse guard result.

| Work | Completed | Partial / failed | Unattempted |
|---|---:|---:|---:|
| Separate startup response forward/backward evaluations | 2 | 0 | 0 |
| Qualification response presentations, planned 6 | 4 | 1 | 1 |
| Qualification native AdamW steps, planned 3 | 2 | 0 | 1 |
| Enabled H backwards, planned 2 | 0 | 1 | 1 |
| Main G optimizer steps, planned 128 | 0 | 0 | 128 |
| Main G presentations, planned 256 | 0 | 0 | 256 |
| Local draft-only response probes, planned 18 | 0 | 0 | 18 |
| Native K/G generations, planned 384 | 0 | 0 | 384 |

There were **four completed qualification K backwards**, plus two completed startup backwards. The fifth `qualification_k_backward_attempted` is entry into the enclosing wrapper; its H calculation failed before the underlying K backward. It must not be reported as a fifth actual K backward. No enabled AdamW step or projection ran.

Hooks record **7 response/draft forwards, 35 layer forwards and 1,360 vocabulary-head calls**, including recomputation. These are instrumentation counts, not additional response presentations or generations. New feedback collection and teacher-model execution were zero.

Sources: [original work log](raw/campaign_001/qualification/WORK.jsonl), [qualification result](raw/campaign_001/qualification/RESULT.json), [failure traceback](raw/launch.log), [derived accounting](analysis/QUALIFICATION_ACCOUNTING.json).

## Costs, gaps and conclusion

Qualification lasted **195.462 s**; the enclosing campaign function lasted **196.163 s**. The supervisor elapsed **291 s** including preflight and CPU tests. These nested times must not be added. K's peak allocated/reserved CUDA memory was **21,075,324,416 / 23,741,857,792 bytes**; disabled G's was **21,075,979,776 / 23,683,137,536 bytes**. Enabled G's failed-trial peak and guard-specific work/cost summary were not persisted. Completed-trial peaks do not establish enabled-G memory feasibility or overhead.

The original enabled-trial `running` label is stale: the global failed result, exception and supervisor exit code 1 establish termination. The audit preserves that inconsistency rather than rewriting the record. Remote CPU logs record 483 regression and 31 focused tests passing; overlapping test sets must not be summed as distinct tests. Passing CPU tests did not cover this native compiled-backward limitation.

The report's proposed `donated_buffer=False` repair is a plausible compiler compatibility change, **not a demonstrated repair**. Any later repair needs the unchanged exact controls and enabled qualification again; its memory/cost effect remains unmeasured. The current experiment correctly stopped. The original 128-step G event and 384-generation comparison remain outstanding, and the guard's scientific value remains unknown.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_prefix_preservation_guard_20261027/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
