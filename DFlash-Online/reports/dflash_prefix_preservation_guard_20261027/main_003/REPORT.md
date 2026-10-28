# Prefix-preservation guard on K: one-event screen

**The compiler issue was fixed without changing the algorithm, and the complete experiment finished. The guard provided no benefit in this screen: it activated on 0/128 updates, the final G and retained K1 weight files are byte-for-byte identical, and all 64 requests tie on verification calls.** Both methods use 3,027 calls. The 0.0579% timing difference is descriptive measurement variation between identical models, not a gain attributable to the guard.

The original failure was PyTorch compiled backward buffer donation conflicting with the retained graph needed to differentiate H and K separately. Pinning `donated_buffer=False` before startup resolved it. Exact GPU controls preserved K gradients and the native AdamW result. This main run additionally establishes identical **final** weights to retained K1; it does not claim every historical intermediate gradient or optimizer state was compared. Original failure and repair evidence remain in the parent and `repair_002` directories. [Final weight comparison](raw/FINAL_WEIGHT_COMPARISON.json), [repair report](../repair_002/REPORT.md).

The three local probe steps each reduce mean H and extend the total matching prefix, yet introduce 47, 2 and 1 new earlier errors, respectively. Their denominators and all candidates are retained below and in the CSVs. Ordinary K, G and the norm-scaled candidate agree in these probes because no projection was applied. **An improving aggregate H can coexist with damage to some individual prefixes.** This is local training-conditioned evidence; it does not explain particular later-request errors or prove all preservation guards ineffective. No replacement mechanism or tuning was implemented.

Supervisor runtime: **22m21s**, 09:12:31–09:34:52 UTC, exit 0, including preflight/CPU/postflight. Main campaign timer: 1,280.990s. Training peak GPU memory: **38.435GB allocated / 41.970GB reserved**. All 384 generations are valid, with zero unknown or unattempted work. All 542 sealed files passed postflight, and all 1,357 compact retrieved files were verified. Both GPUs are idle; about 37.8GB disk space remains before negligible report packaging.

## Setup and validity

Status: **completed**. Reused GPU qualification: **passed**. ARS experiment-agent execution and evidence review; project stage unchanged.

The pinned pair is Qwen3-8B (`b968826d…`) with z-lab/Qwen3-8B-DFlash-b16 (`9b41424b…`). Training uses BF16 AdamW, constant learning rate 3e-4, weight decay 0.01, clipping 1, response accumulation 2, up to 512 anchors, Flex attention and the unchanged K objective. The original 7,424-step scheduler horizon remains configured, but this event stops at step 128. Serving uses the pinned SGLang runtime, greedy block-16 decoding, 2,048 tokens or EOS, context limit 4,096 and concurrency 1. [Exact configuration](raw/sealed/config.yaml), [model pins](raw/sealed/MODEL_PIN.json), [execution plan](raw/sealed/PLAN.json).

G starts from released DFlash with a fresh seed-17 optimizer and reuses the original 128 feedback responses and anchor/presentation schedule. K is the retained event-1 checkpoint. The K objective and native AdamW step remain unchanged; G projects only the component of the proposed displacement predicted to increase the current strict-prefix protection quantity.

The compiler repair is fixed before startup. The successful three-trial qualification is reused; the current run performs no additional qualification optimizer steps. Its fresh process performs two declared startup forward/backward evaluations with zero Adam steps, then restores the approved RNG and constructs the fresh G trainer.

**This development panel is held out from feedback training but was already used in development.** It is not a fresh confirmation panel. No feedback is collected or learned from during replay.

## Native serving comparison

Valid common requests: **64/64**; valid timed generations: **384**. Each count uses one validated trajectory; timings use the mean of three fresh repeats per request. Exact output tokens and termination are checked against retained F.

| Arm | Verification calls | Sum of mean request times (s) | Accepted draft tokens |
|---|---:|---:|---:|
| K | 3027 | 52.179745 | 16513 |
| G | 3027 | 52.149549 | 16513 |

**G versus K:** pooled verification-call reduction **0.0000%**; request-time reduction **0.0579%**. Requests improved/regressed/tied: **0/0/64**.

Positive reductions favor G. Ratios use summed paired measurements, not mean request percentages. Serving time includes prefill, drafting, verification and the qualified request cache work, with the checkpoint available; training, probes, loading and publication are excluded.

## Guard behavior and local measurements

Observed main steps: **128**; active projections: **0** (0.00%). Active steps with positive realized directional derivative: **0**.

Median realized/native update norm: **1.000000** (range 1.000000–1.000000). All analytic, FP32-proposal and realized BF16 derivatives/norms are retained in `guard_steps.csv`. Positive realized derivatives are recorded without rollback; the first-order projection is not a finite-step guarantee.

| Step | Candidate | Mean H before → after | Useful first-error repairs | New earlier errors | Lost old-prefix tokens | Net prefix tokens |
|---:|---|---:|---:|---:|---:|---:|
| 1 | ordinary_K | 0.063069 → 0.021695 | 527 | 47 | 153 | +1740 |
| 1 | projected_G | 0.063069 → 0.021695 | 527 | 47 | 153 | +1740 |
| 1 | norm_scaled_K | 0.063069 → 0.021695 | 527 | 47 | 153 | +1740 |
| 64 | ordinary_K | 0.066645 → 0.056188 | 70 | 2 | 3 | +201 |
| 64 | projected_G | 0.066645 → 0.056188 | 70 | 2 | 3 | +201 |
| 64 | norm_scaled_K | 0.066645 → 0.056188 | 70 | 2 | 3 | +201 |
| 128 | ordinary_K | 0.043910 → 0.036832 | 66 | 1 | 2 | +178 |
| 128 | projected_G | 0.043910 → 0.036832 | 66 | 1 | 2 | +178 |
| 128 | norm_scaled_K | 0.043910 → 0.036832 | 66 | 1 | 2 | +178 |

These are same-step local counterfactuals on training-conditioned blocks with fixed pre-update prefix membership and weights. They are not native serving measurements or independently trained arms. The norm-scaled candidate targets the projected FP32 norm; its realized BF16 norm is reported separately. Lower H alone is insufficient. Native benefit alongside retained repair progress supports the candidate; benefit dominated by near-zero updates may instead reflect suppressed learning. A local norm-scaled comparison does not replace a trained shrinkage control for a later causal claim.

## Costs and scope accounting

| Measurement | Seconds | Scope |
|---|---:|---|
| Reused successful qualification (historical) | 242.096 | Historical work; not part of this campaign wall time |
| New process startup | 39.583 | Two disposable response evaluations; zero optimizer steps |
| G loading and initial-state check | 19.724 | Separate from training and serving |
| Complete G event | 610.048 | Contains training, local probes, persistence/export and event overhead |
| Training windows including local probes | 577.763 | Nested inside complete event |
| Local probes | 10.280 | Diagnostic work nested inside training windows |
| Training windows excluding local probes | 567.483 | Subtraction only; not a second cost |
| H backward | 197.631 | Nested component timer |
| Projection | 34.386 | Nested component timer |
| Checkpoint persistence | 17.561 | Nested inside complete event |
| Export | 12.079 | Nested inside complete event |
| Native backend loading | 21.258 | Separate benchmark setup |
| Benchmark publications | 264.623 | Benchmark-only switches, excluded from serving latency |
| Actual campaign wall time | 1280.990 | Includes new startup, event, benchmark setup and all repetitions; excludes historical qualification |

Nested timers must not be added again to event wall time. H backward plus projection measures only those components; it does not capture all added forward/snapshot/bookkeeping overhead or establish a fresh matched training-cost difference against K. Retained feedback/features have zero new collection or preparation in this run, not zero historical cost.

Completed main training: 256 presentations / 128 steps. Recorded local probe responses: 18 of 18 planned. Historical temporary optimizer steps: 5 across the failed and successful qualification attempts; new temporary optimizer steps: 0. New startup completed 2 response evaluations / 2 backwards, with 0 optimizer steps. Actual forward, chunk, backward and startup counters remain in `analysis/SUMMARY.json` and the raw startup records.

The qualification established exact K-gradient/Adam isolation under the repaired compiler policy. Its single enabled decision was inactive. After the complete event, the exported G and retained K1 files have the same byte count and SHA-256 digest, establishing identical final weights. No intermediate-history equivalence claim is made. All 128 main decisions were inactive; the active GPU correction branch therefore remained unexercised. No retuning was performed.

Independent review checked the response-wide normalization, separate H-gradient buffer, unchanged K/Adam state, global projection and probe restoration. CPU tests exercise those contracts; only the native qualification receipts establish GPU integration. Raw invalid/incomplete work is retained.

No tuning, additional arm, feedback window or later event is authorized by this report.

Files: `analysis/guard_steps.csv`, `analysis/guard_responses.csv`, `analysis/local_probes.csv`, `analysis/local_probe_summary.csv`, `analysis/SUMMARY.json`; serving per-request and repeat records remain under the raw evaluation report directory.

Reviewable data: [per-request results](raw/campaign_001/evaluation/report/PER_REQUEST.csv), [all timing repeats](raw/campaign_001/evaluation/report/ALL_REPEATS.csv), [guard decisions](analysis/guard_steps.csv), [local probes](analysis/local_probe_summary.csv), [independent audit](FINAL_AUDIT.json).


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_prefix_preservation_guard_20261027/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
